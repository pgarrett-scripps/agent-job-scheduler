"""The daemon's brain: owns state, runs the scheduling tick, supervises jobs.

Single-threaded asyncio. Because every mutation happens in one task on one event loop,
there is no locking anywhere in the system -- concurrency bugs are designed out rather
than guarded against.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import secrets
import time
from pathlib import Path
from typing import Any, cast

from . import sysinfo
from .config import Config, db_path, log_dir
from .contention import ContentionMonitor, ExternalLoad, ForeignProcesses
from .db import Store
from .executor import Executor, RunningProcess, read_exit_file
from .models import Job, JobClass, JobState, ResourceRequest, short_actor
from .scheduler import Capacity, Decision, effective_request, plan

log = logging.getLogger("ajs.engine")


class Engine:
    """Scheduler state machine."""

    def __init__(self, cfg: Config, *, store: Store | None = None, executor: Executor | None = None) -> None:
        self.cfg = cfg
        self.cap = Capacity.from_config(cfg)
        self.store = store or Store(db_path())
        self.executor = executor or Executor(cfg, log_dir())
        self.running: dict[int, RunningProcess] = {}
        self.monitors: dict[int, ContentionMonitor] = {}
        self.last_decision = Decision()
        self.last_finish_at: float = 0.0
        self.external = ExternalLoad(half_life_s=cfg.external_load_half_life_s)
        self.foreign = ForeignProcesses()
        self.quiet_since: float | None = None
        """When load outside ajs last fell within the quiet limits; None while it is over."""
        self.noise = ""
        """What is over the quiet limits right now, for blocked reasons and job notes."""
        self.settling_since: dict[int, float] = {}
        """Exclusive jobs holding the machine while it goes quiet, and since when."""
        self.interference: dict[int, list[str]] = {}
        """Timing jobs -> interference seen while they ran, one line per episode."""
        self._interfering: set[int] = set()
        self._timing: set[int] = set()
        self._mem_warned: set[int] = set()
        """Running exclusive (CPU) jobs; a GPU-exclusive run is not spoiled by CPU load."""
        self._runtime_warned: set[tuple[int, int]] = set()
        """(job, max_runtime) pairs already warned, so an extension earns a fresh warning."""
        self._own_cgroups: set[Path] = set()
        self.paused = False
        self.pause_until: float | None = None
        self.dep_waiting: dict[int, str] = {}
        """Queued jobs whose dependencies have not finished yet, and what they wait on."""
        """Set by `ajs pause --for`: the tick resumes by itself at this epoch time."""
        self.draining = False
        self._waiters: dict[int, list[asyncio.Future[None]]] = {}
        self._wake = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()

    # --- lifecycle --------------------------------------------------------

    async def recover(self) -> None:
        """Reconcile after a daemon restart.

        A job still alive in its own scope is adopted: supervised, monitored and charged
        exactly as if this daemon had started it, so restarting ajs costs no work. A job
        that ended while no daemon was watching is recorded from its exit file. Anything
        else marked RUNNING is gone and is marked failed, and scopes no job claims are
        stopped rather than left holding capacity the scheduler does not know about.
        """
        now = time.time()
        adopted: set[str] = set()
        for job in self.store.jobs_in_state(JobState.RUNNING):
            rp = self.executor.adopt(job)
            if rp is not None:
                self._adopt(job, rp, now)
                if rp.unit:
                    adopted.add(rp.unit)
                continue
            exit_file = self.executor.exit_file(job.id)
            code = read_exit_file(exit_file)
            if code is not None:
                ended = exit_file.stat().st_mtime
                state = JobState.DONE if code == 0 else JobState.FAILED
                if job.cancel_reason:
                    state = JobState.TIMEOUT if "max_runtime" in job.cancel_reason else JobState.CANCELLED
                log.info("job %s finished while the daemon was down: %s (exit=%s)", job.id, state, code)
                self.store.update_job(job.id, state=str(state), finished_at=ended, exit_code=code)
                self.last_finish_at = max(self.last_finish_at, ended)
                continue
            log.warning("job %s was lost while the daemon was down", job.id)
            self.store.update_job(
                job.id,
                state=str(JobState.FAILED),
                finished_at=now,
                exit_code=None,
                cancel_reason="daemon restarted and the job's process was gone, with no exit status",
            )
        for unit in self.executor.orphan_units():
            if unit in adopted:
                continue
            log.warning("stopping orphaned unit %s", unit)
            await Executor.stop_unit(unit)

    def _adopt(self, job: Job, rp: RunningProcess, now: float) -> None:
        """Supervise a job a previous daemon started. Contention is measured from now
        on; what happened before the restart went unwatched, and the job's history says so."""
        monitor = ContentionMonitor(
            self.cfg.contention_threshold,
            assess=job.resources.exclusive or job.resources.gpu_exclusive,
        )
        monitor.start(rp.proc.pid, now, cgroup=rp.cgroup)
        self.monitors[job.id] = monitor
        if job.resources.exclusive:
            self._timing.add(job.id)
        self.running[job.id] = rp
        detail = f"daemon restarted; job kept running, watched again from {_clock(now)}"
        self.store.add_event(job.id, actor="ajs", action="adopt", detail=detail)
        log.info("adopted running job %s (%s) in %s", job.id, job.project, rp.unit)
        task = asyncio.create_task(self._supervise(job.id, rp))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run(self) -> None:
        """Main loop. Ticks on a timer or whenever something interesting happens."""
        await self.recover()
        while True:
            try:
                await self.tick()
            except Exception:  # pragma: no cover - keep the daemon alive
                log.exception("scheduler tick failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self.cfg.tick_seconds)
            self._wake.clear()

    def wake(self) -> None:
        """Ask for a scheduling pass now rather than at the next tick."""
        self._wake.set()

    async def shutdown(self) -> None:
        """Stop the daemon, leaving jobs in their own scopes running for the next one to
        adopt. Jobs without a scope share the daemon's cgroup and would die with it
        anyway, so those are stopped cleanly."""
        for task in list(self._tasks):
            task.cancel()
        for rp in list(self.running.values()):
            if rp.unit:
                log.info("leaving job %s running for the next daemon", rp.job_id)
                if rp.log_file is not None:
                    rp.log_file.close()
                continue
            await self.executor.stop(rp)
        self.store.close()

    # --- the tick ---------------------------------------------------------

    async def tick(self) -> None:
        now = time.time()
        self._sample_external(now)
        self._warn_mem_underuse(now)
        self._warn_runtime(now)

        expired = self.store.expire_leases(now - self.cfg.lease_timeout_s)
        for lease_id in expired:
            log.warning("reclaimed lease %s: holder stopped heartbeating", lease_id)

        self._enforce_runtime_limits(now)
        self._enforce_disk_floor()
        self._expire_timed_holds(now)
        self._resolve_dependencies(now)

        if self.paused or self.draining:
            self.last_decision = Decision(
                blocked={
                    j.id: "scheduler is paused" if self.paused else "scheduler is draining"
                    for j in self.store.jobs_in_state(JobState.QUEUED)
                    if not j.held and j.id not in self.dep_waiting
                }
            )
            return

        # Held jobs, and jobs still waiting on dependencies, are invisible to the planner:
        # they neither start nor claim the reservation, so holding a starved giant also
        # stops it vetoing backfill.
        queued = [j for j in self.store.jobs_in_state(JobState.QUEUED) if not j.held and j.id not in self.dep_waiting]
        running = self.store.jobs_in_state(JobState.RUNNING)
        if not queued:
            self.last_decision = Decision()
            return

        decision = plan(
            queued=queued,
            running=running,
            cap=self.cap,
            cfg=self.cfg,
            now=now,
            last_start=self.store.project_last_start(),
            lease_usage=self._lease_usage(),
            external_usage=self._external_usage(),
            free_disk_mb=sysinfo.free_disk_mb(self.cfg.disk_watch_path),
            last_finish_at=self.last_finish_at,
            quiet_since=self.quiet_since if self.cfg.track_external_load else 0.0,
            noise=self.noise,
            settling_since=self.settling_since,
            mem_headroom_mb=self._mem_headroom(running),
        )
        self.last_decision = decision
        self.settling_since = {j: self.settling_since.get(j, now) for j in decision.settling}
        for job_id, why in decision.unquiet_start.items():
            self.interference.setdefault(job_id, []).append(why)
            self.store.add_event(job_id, actor="ajs", action="unquiet-start", detail=why)

        for job_id in decision.start:
            job = self.store.get_job(job_id)
            if job is not None and job.state is JobState.QUEUED:
                await self._start(job, now)

        if decision.reservation is not None:
            self.store.update_job(decision.reservation.job_id, reserved_until=decision.reservation.start_at)

    def _sample_external(self, now: float) -> None:
        """Measure what is running on this machine that ajs did not start.

        Summing the live job cgroups gives ajs's own share; whatever else the kernel
        counts as busy belongs to somebody else.

        Each cgroup is counted once. Without systemd every job shares the daemon's own
        cgroup, and summing it per job would multiply ajs's share by the job count.

        Monitors are polled even when external-load tracking is off: a job's cgroup is
        removed the moment it exits, so the last reading taken here is what the
        contention verdict falls back on.
        """
        own_cpu = 0.0
        own_mem = 0
        own_cgroups: set[Path] = set()
        for monitor in self.monitors.values():
            cpu, mem = monitor.poll(now)
            if monitor.cgroup is None or monitor.cgroup in own_cgroups:
                continue
            own_cgroups.add(monitor.cgroup)
            if cpu is not None:
                own_cpu += cpu
            if mem is not None:
                own_mem += mem // (1024 * 1024)
        self._own_cgroups = own_cgroups
        if self.cfg.track_external_load:
            self.external.sample(now, own_cpu, own_mem, own_cgroups)
            if now - (self.foreign.sampled_at or 0.0) >= 5.0:
                self.foreign.sample(now, own_cgroups)
            self._update_quiet(now)

    def _update_quiet(self, now: float) -> None:
        """Track whether the machine is quiet enough to time on, and flag timing runs
        that something outside ajs is disturbing.

        This is the one place that decides "quiet", so benchmark scripts do not each need
        their own busy-wait: an exclusive job does not start until the machine has been
        quiet for ``quiet_seconds``, and its max_runtime does not run while it waits.
        """
        if not self.external.ready:
            return
        cores = max(0.0, self.external.cpu_cores - self.cfg.external_cpu_allowance)
        loud_cpu = cores > self.cfg.quiet_cpu_cores
        # Once a timing job runs, its own I/O shows up as iowait, so only the CPU part
        # can be held against the machine. Before the start, iowait is someone else's.
        timing = [j for j in self.running if self._is_timing(j)]
        loud_io = not timing and self.external.iowait_pct > self.cfg.quiet_iowait_pct
        parts = []
        if loud_cpu:
            parts.append(f"{cores:.1f} cores outside ajs above the desktop allowance")
        if loud_io:
            parts.append(f"iowait {self.external.iowait_pct:.0f}%")
        self.noise = ", ".join(parts)
        if parts:
            self.quiet_since = None
        elif self.quiet_since is None:
            self.quiet_since = now

        for job_id in timing:
            if loud_cpu and job_id not in self._interfering:
                self._interfering.add(job_id)
                who = self.foreign.top_cpu()
                line = time.strftime("%H:%M:%S", time.localtime(now)) + f" {self.noise}"
                if who:
                    line += ": " + "; ".join(p.describe() for p in who)
                self.interference.setdefault(job_id, []).append(line)
                self.store.add_event(job_id, actor="ajs", action="interference", detail=line)
                log.warning("timing job %s disturbed: %s", job_id, line)
            elif not loud_cpu:
                self._interfering.discard(job_id)

    def _is_timing(self, job_id: int) -> bool:
        return job_id in self._timing

    def _external_usage(self) -> dict[str, int] | None:
        if not self.cfg.track_external_load:
            return None
        return self.external.usage(
            self.cap.cpu,
            cpu_allowance=self.cfg.external_cpu_allowance,
            mem_allowance_mb=self.cfg.mem_reserve_mb,
        )

    def _lease_usage(self) -> dict[str, int]:
        usage = {"cpu": 0, "mem_mb": 0, "gpu": 0, "gpu_mem_mb": 0}
        for row in self.store.active_leases():
            req = ResourceRequest.from_json(row["resources"])
            usage["cpu"] += req.cpu
            usage["mem_mb"] += req.mem_mb
            usage["gpu"] += req.gpu
            usage["gpu_mem_mb"] += req.gpu_mem_mb
        return usage

    def _expire_timed_holds(self, now: float) -> None:
        if self.paused and self.pause_until is not None and now >= self.pause_until:
            log.info("timed pause over; resuming")
            self.paused = False
            self.pause_until = None
        for job in self.store.jobs_in_state(JobState.QUEUED):
            if job.held and job.hold_until is not None and now >= job.hold_until:
                self.store.update_job(job.id, held=0, hold_until=None)
                self.store.add_event(job.id, actor="ajs", action="release", reason="timed hold expired")

    def _resolve_dependencies(self, now: float) -> None:
        """Work out which queued jobs are still waiting on others, and cancel the ones
        whose `after_ok` dependency failed. Jobs are visited in id order and dependencies
        always have lower ids, so a failure cascades down a whole chain in one pass."""
        waiting: dict[int, str] = {}
        for job in self.store.jobs_in_state(JobState.QUEUED):
            pending, doomed = [], None
            for dep_id, must_succeed in [(i, True) for i in job.after_ok] + [(i, False) for i in job.after_any]:
                dep = self.store.get_job(dep_id)
                if dep is None:
                    doomed = f"dependency {dep_id} no longer exists"
                    break
                if not dep.state.is_terminal:
                    pending.append(f"{dep_id} ({dep.state})")
                elif must_succeed and dep.state is not JobState.DONE:
                    doomed = f"dependency {dep_id} ended {dep.state}"
                    break
            if doomed is not None:
                log.info("cancelling job %s: %s", job.id, doomed)
                self.store.update_job(job.id, state=str(JobState.CANCELLED), finished_at=now, cancel_reason=doomed)
                self.store.add_event(job.id, actor="ajs", action="cancel", reason=doomed)
                self._notify(job.id)
            elif pending:
                waiting[job.id] = "waiting on job " + ", ".join(pending)
        self.dep_waiting = waiting

    def _enforce_runtime_limits(self, now: float) -> None:
        """Kill jobs that blew past the max_runtime they declared.

        This is not just hygiene: backfill decisions were made assuming that ceiling, so
        letting a job overrun would silently break the reservation guarantee.
        """
        for job in self.store.jobs_in_state(JobState.RUNNING):
            if job.started_at is None:
                continue
            if now - job.started_at > job.max_runtime_s:
                if job.cancel_reason:
                    continue  # already being stopped
                rp = self.running.get(job.id)
                log.warning("job %s exceeded max_runtime %ss; killing", job.id, job.max_runtime_s)
                self.store.update_job(job.id, cancel_reason=f"exceeded max_runtime of {job.max_runtime_s}s")
                if rp is not None:
                    # In the background: the grace period must not stall the tick.
                    task = asyncio.create_task(self.executor.stop(rp))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

    def _warn_mem_underuse(self, now: float) -> None:
        for job in self.store.jobs_in_state(JobState.RUNNING):
            monitor = self.monitors.get(job.id)
            if monitor is None:
                continue
            try:
                self._check_mem_underuse(job, monitor, now, final=False)
            except Exception:  # a warning must never stall the tick
                log.exception("memory underuse check failed for job %s", job.id)

    def _check_mem_underuse(self, job: Job, monitor: ContentionMonitor, now: float, *, final: bool) -> None:
        """Tell the owner, once, when a job holds far more memory than it ever uses.

        The event lands in the owner's `ajs inbox`. Judged on the job's peak, not its
        current use, so a job between phases is not flagged; judged at the job's
        ``mem_check`` age, or at its end if it finishes sooner."""
        reserved = job.resources.mem_mb
        peak = monitor.mem_peak_mb
        if job.started_at is None or peak is None or reserved < self.cfg.mem_underuse_min_mb:
            return
        after = mem_check_after(job.meta.get("mem_check"), self.cfg.mem_underuse_after_s)
        # The peak covers only what this daemon has watched: after a restart that starts
        # at the adoption, and one sample taken between two steps of a script reads zero.
        watched = now - max(job.started_at, monitor.watched_since)
        if after is None or watched < (MEM_UNDERUSE_MIN_WATCH_S if final else after):
            return
        if peak >= reserved * self.cfg.mem_underuse_ratio:
            return
        if job.id in self._mem_warned:
            return
        self._mem_warned.add(job.id)
        if self.store.last_event(job.id, "mem-underuse") is not None:
            return  # warned before a daemon restart
        suggest_gb = max(1, -(-int(peak * 1.25) // 1024))
        minutes = (now - job.started_at) / 60
        self.store.add_event(
            job.id,
            actor="ajs",
            action="mem-underuse",
            detail=f"peak {peak / 1024:.1f} of {reserved / 1024:.0f} GB reserved after {minutes:.0f} min",
            reason=(
                f"reserve about {suggest_gb}G next time; the unused reservation keeps other jobs "
                "queued (move this check with --meta mem_check=20m, or =off)"
            ),
        )
        log.info("job %s uses %s of %s MB reserved", job.id, peak, reserved)

    def _warn_runtime(self, now: float) -> None:
        for job in self.store.jobs_in_state(JobState.RUNNING):
            try:
                self._check_runtime(job, now)
            except Exception:  # a warning must never stall the tick
                log.exception("runtime warning check failed for job %s", job.id)

    def _check_runtime(self, job: Job, now: float) -> None:
        """Tell the owner, once per max_runtime, that the job is about to be killed.

        The event lands in the owner's `ajs inbox`, which it sees on its next turn, so the
        notice comes early enough to act on: ``runtime_warn_s`` before the kill, or at 80%
        for a job too short for that to mean anything."""
        if job.started_at is None or job.cancel_reason:
            return
        limit = job.max_runtime_s
        key = (job.id, limit)
        if key in self._runtime_warned:
            return
        remaining = job.started_at + limit - now
        if remaining > min(self.cfg.runtime_warn_s, limit * 0.2) or remaining <= 0:
            return
        self._runtime_warned.add(key)
        warned = self.store.last_event(job.id, "runtime-warning")
        changed = self.store.last_event(job.id, "runtime")
        if warned is not None and (changed is None or cast(float, warned["at"]) > cast(float, changed["at"])):
            return  # warned before a daemon restart
        if job.resources.exclusive or job.resources.gpu_exclusive:
            how = "a timing run cannot be extended; let it finish or resubmit with a longer max_runtime"
        else:
            cap = int((job.orig_max_runtime_s or limit) * self.cfg.runtime_extend_factor)
            how = (
                f"extend with `ajs extend {job.id} --by 30m -r WHY` (up to {cap // 60} min in total), "
                "or save its work now"
            )
        self.store.add_event(
            job.id,
            actor="ajs",
            action="runtime-warning",
            detail=f"killed in {remaining / 60:.0f} min: used {(now - job.started_at) / 60:.0f} of {limit // 60} min",
            reason=how,
        )
        log.info("job %s is %.0fs from its max_runtime", job.id, remaining)

    def _enforce_disk_floor(self) -> None:
        free = sysinfo.free_disk_mb(self.cfg.disk_watch_path)
        if free >= self.cfg.disk_floor_mb:
            return
        log.error("free disk %s MB is below floor %s MB", free, self.cfg.disk_floor_mb)
        # Admission control already refuses new jobs; running jobs are left alone rather
        # than killed, because killing mid-write is at least as likely to corrupt output
        # as running out of space. Surfaced loudly instead.

    # --- job lifecycle ----------------------------------------------------

    async def _start(self, job: Job, now: float) -> None:
        need = effective_request(job, self.cap)
        load = sysinfo.load_average()[0]

        try:
            rp = await self.executor.start(job, need)
        except OSError as exc:
            log.error("failed to start job %s: %s", job.id, exc)
            self.store.update_job(
                job.id,
                state=str(JobState.FAILED),
                started_at=now,
                finished_at=time.time(),
                cancel_reason=f"could not launch: {exc}",
            )
            self._notify(job.id)
            return

        monitor = ContentionMonitor(
            self.cfg.contention_threshold,
            assess=job.resources.exclusive or job.resources.gpu_exclusive,
        )
        monitor.start(rp.proc.pid, now, cgroup=rp.cgroup)
        self.monitors[job.id] = monitor
        if job.resources.exclusive:
            self._timing.add(job.id)
        self.running[job.id] = rp

        self.store.update_job(
            job.id,
            state=str(JobState.RUNNING),
            started_at=now,
            pid=rp.proc.pid,
            unit=rp.unit,
            log_path=str(rp.log_path),
            load_before=load,
            reserved_until=None,
        )
        self.store.note_project_start(job.project, now)
        log.info(
            "started job %s (%s) cpu=%s mem=%sMB%s cgroup=%s",
            job.id,
            job.project,
            need["cpu"],
            need["mem_mb"],
            " EXCLUSIVE" if job.resources.exclusive else "",
            rp.cgroup.name if rp.cgroup is not None else "unknown",
        )

        task = asyncio.create_task(self._supervise(job.id, rp))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _supervise(self, job_id: int, rp: RunningProcess) -> None:
        """Await a job's exit and record how it went."""
        try:
            exit_code = await rp.proc.wait()
        except asyncio.CancelledError:  # pragma: no cover
            raise
        finally:
            if rp.log_file is not None:
                with contextlib.suppress(Exception):
                    rp.log_file.close()

        exited = time.time()
        # Keep the job's slots until its whole process tree is gone, not just the main
        # process; otherwise the next job starts on top of the dying workers.
        with contextlib.suppress(Exception):
            await self.executor.drain(rp)
        self.running.pop(job_id, None)
        # Stamped after the drain, which can take many seconds: an `ajs inbox` answered
        # meanwhile moves its cursor past the exit time and would never see this job.
        now = self.last_finish_at = time.time()

        job = self.store.get_job(job_id)
        monitor = self.monitors.pop(job_id, None)

        fields: dict[str, Any] = {
            "finished_at": now,
            "exit_code": exit_code,
            "load_after": sysinfo.load_average()[0],
        }

        if monitor is not None and job is not None:
            try:
                self._check_mem_underuse(job, monitor, now, final=True)
            except Exception:  # a warning must never stop a job being recorded
                log.exception("memory underuse check failed for job %s", job_id)
        if monitor is not None:
            report = monitor.finish(exited)
            fields["contended"] = 1 if report.contended else 0
            fields["contention_note"] = report.note
            seen = self.interference.get(job_id)
            if seen:
                # Any episode spoils a reported timing, even if the run-long average
                # stays under the threshold; the owner decides what to redo.
                fields["contended"] = 1
                fields["contention_note"] = f"{report.note}; interference: {' | '.join(seen)}"
            if fields["contended"] and job is not None and (job.resources.exclusive or job.resources.gpu_exclusive):
                log.warning("job %s ran CONTENDED: %s", job_id, fields["contention_note"])
        self.interference.pop(job_id, None)
        self._interfering.discard(job_id)
        self._timing.discard(job_id)
        self._mem_warned.discard(job_id)
        self._runtime_warned = {w for w in self._runtime_warned if w[0] != job_id}

        if job is not None and job.cancel_reason and "max_runtime" in job.cancel_reason:
            fields["state"] = str(JobState.TIMEOUT)
        elif job is not None and job.cancel_reason:
            fields["state"] = str(JobState.CANCELLED)
        else:
            fields["state"] = str(JobState.DONE if exit_code == 0 else JobState.FAILED)

        self.store.update_job(job_id, **fields)
        log.info("job %s finished: %s (exit=%s)", job_id, fields["state"], exit_code)
        self._notify(job_id)
        self.wake()

    # --- API surface used by the server ------------------------------------

    def dependency_warnings(self, after_ok: list[int] | None, after_any: list[int] | None) -> list[str]:
        """Dependencies that will not make the new job wait the way its owner expects.

        A finished dependency releases the job at once, so a chain built on it runs out
        of order; a held one keeps it queued until someone releases the hold."""
        warnings = []
        for dep_id in [int(i) for i in (after_ok or []) + (after_any or [])]:
            dep = self.store.get_job(dep_id)
            if dep is None:
                continue  # submit() refuses these
            if dep.state.is_terminal:
                warnings.append(f"dependency {dep_id} already ended {dep.state}, so this job does not wait for it")
            elif dep.held:
                warnings.append(f"dependency {dep_id} is held, so this job waits until someone releases it")
        return warnings

    def submit(
        self,
        *,
        project: str,
        session_id: str,
        cmd: list[str],
        cwd: str,
        env: dict[str, str] | None = None,
        cpu: int = 1,
        mem_mb: int = 512,
        gpu: int = 0,
        gpu_mem_mb: int = 0,
        disk_mb: int = 0,
        exclusive: bool = False,
        gpu_exclusive: bool = False,
        locks: list[str] | None = None,
        max_runtime_s: int | None = None,
        job_class: str = "batch",
        title: str | None = None,
        description: str | None = None,
        meta: dict[str, str] | None = None,
        note: str | None = None,
        held: bool = False,
        hold_until: float | None = None,
        after_ok: list[int] | None = None,
        after_any: list[int] | None = None,
    ) -> Job:
        after_ok = [int(i) for i in after_ok or []]
        after_any = [int(i) for i in after_any or []]
        for dep_id in after_ok + after_any:
            dep = self.store.get_job(dep_id)
            if dep is None:
                raise ValueError(f"no such job to depend on: {dep_id}")
            if dep_id in after_ok and dep.state.is_terminal and dep.state is not JobState.DONE:
                raise ValueError(f"job {dep_id} already ended {dep.state}; --after-ok on it would never run")
        resources = ResourceRequest(
            cpu=cpu,
            mem_mb=mem_mb,
            gpu=gpu,
            gpu_mem_mb=gpu_mem_mb,
            disk_mb=disk_mb,
            exclusive=exclusive,
            gpu_exclusive=gpu_exclusive,
            locks=list(locks or []),
        )
        job = self.store.add_job(
            project=project,
            session_id=session_id,
            cmd=cmd,
            cwd=cwd,
            env=env or {},
            resources=resources,
            max_runtime_s=max_runtime_s or self.cfg.default_max_runtime_s,
            job_class=JobClass(job_class),
            title=title or None,
            # `note` was the first name for the description; still accepted from old clients.
            description=description or note or None,
            meta={str(k): str(v) for k, v in (meta or {}).items()},
            held=held or hold_until is not None,
            hold_until=hold_until,
            after_ok=after_ok,
            after_any=after_any,
        )
        if held or hold_until is not None:
            detail = f"until {_clock(hold_until)}" if hold_until is not None else ""
            self.store.add_event(job.id, actor=session_id, action="hold", detail=detail, reason="submitted held")
        self.wake()
        return job

    # --- queue management ---------------------------------------------------

    def _queued_job(self, job_id: int) -> Job:
        job = self.store.get_job(job_id)
        if job is None:
            raise ValueError(f"no such job: {job_id}")
        if job.state is not JobState.QUEUED:
            raise ValueError(f"job {job_id} is {job.state}; only queued jobs can be managed")
        return job

    def hold(self, job_id: int, *, actor: str, reason: str, until: float | None = None) -> Job:
        """Keep a queued job out of scheduling until released, or until ``until``."""
        job = self._queued_job(job_id)
        if not job.held or until != job.hold_until:
            self.store.update_job(job_id, held=1, hold_until=until, reserved_until=None)
            detail = f"until {_clock(until)}" if until is not None else ""
            self.store.add_event(job_id, actor=actor, action="hold", detail=detail, reason=reason)
            self.wake()
        return self._queued_job(job_id)

    def release(self, job_id: int, *, actor: str, reason: str = "") -> Job:
        job = self._queued_job(job_id)
        if job.held:
            self.store.update_job(job_id, held=0, hold_until=None)
            self.store.add_event(job_id, actor=actor, action="release", reason=reason)
            self.wake()
        return self._queued_job(job_id)

    def set_priority(self, job_id: int, job_class: str, *, actor: str, reason: str) -> Job:
        """Move a queued job to another priority band."""
        new = JobClass(job_class)
        job = self._queued_job(job_id)
        if job.job_class is not new:
            self.store.update_job(job_id, job_class=str(new))
            self.store.add_event(
                job_id, actor=actor, action="priority", detail=f"{job.job_class} -> {new}", reason=reason
            )
            self.wake()
        return self._queued_job(job_id)

    def set_runtime(self, job_id: int, max_runtime_s: int, *, actor: str, reason: str) -> Job:
        """Change a queued or running job's max_runtime. Only its owner, or a person, may.

        Shortening is always fine: it frees the slot sooner. Lengthening is capped at
        ``runtime_extend_factor`` times the submitted value and refused for timing runs.
        Backfill admitted this job on its old ceiling, so an extension past the start
        promised to a reserved job goes ahead but delays it, and that job's owner is told.
        """
        job = self.store.get_job(job_id)
        if job is None:
            raise ValueError(f"no such job: {job_id}")
        if job.state not in (JobState.QUEUED, JobState.RUNNING):
            raise ValueError(f"job {job_id} is {job.state}; only queued or running jobs can change their runtime")
        if actor != job.session_id and not actor.startswith("user:"):
            raise ValueError(
                f"job {job_id} belongs to {short_actor(job.session_id)}; "
                "only its owner or a person at a terminal can change its runtime"
            )
        old = job.max_runtime_s
        if max_runtime_s == old:
            return job
        now = time.time()
        elapsed = now - job.started_at if job.started_at is not None else 0.0
        if max_runtime_s < elapsed + 60:
            raise ValueError(
                f"job {job_id} has already run {elapsed / 60:.0f} min; a max_runtime of "
                f"{max_runtime_s // 60} min would kill it now (use `ajs cancel` for that)"
            )
        orig = job.orig_max_runtime_s or old
        if max_runtime_s > old:
            if job.resources.exclusive or job.resources.gpu_exclusive:
                raise ValueError(f"job {job_id} is a timing run; timing runs cannot be extended")
            cap = int(orig * self.cfg.runtime_extend_factor)
            if max_runtime_s > cap:
                raise ValueError(
                    f"job {job_id} was submitted with {orig // 60} min; it can be extended to at most "
                    f"{cap // 60} min. For longer, resubmit it with an honest max_runtime"
                )
        self.store.update_job(job_id, max_runtime_s=max_runtime_s, orig_max_runtime_s=orig)
        self.store.add_event(
            job_id, actor=actor, action="runtime", detail=f"{old // 60} -> {max_runtime_s // 60} min", reason=reason
        )
        res = self.last_decision.reservation
        if (
            job.started_at is not None
            and res is not None
            and not res.external
            and res.job_id != job_id
            and max_runtime_s > old
        ):
            delay = job.started_at + max_runtime_s - max(job.started_at + old, res.start_at)
            if delay > 0:
                self.store.add_event(
                    res.job_id,
                    actor="ajs",
                    action="delayed",
                    detail=f"up to {delay / 60:.0f} min",
                    reason=f"job {job_id} was extended to {max_runtime_s // 60} min: {reason}",
                )
        self.wake()
        job = self.store.get_job(job_id)
        assert job is not None
        return job

    async def cancel(self, job_id: int, reason: str = "cancelled by user") -> bool:
        job = self.store.get_job(job_id)
        if job is None or job.state.is_terminal:
            return False
        self.store.update_job(job_id, cancel_reason=reason)
        rp = self.running.get(job_id)
        if rp is not None:
            await self.executor.stop(rp)
        else:
            self.store.update_job(job_id, state=str(JobState.CANCELLED), finished_at=time.time())
            self._notify(job_id)
        self.wake()
        return True

    async def wait(self, job_id: int, timeout: float) -> Job | None:
        """Block until ``job_id`` reaches a terminal state.

        Long-polling rather than making callers poll: an agent that busy-waits on job
        status burns tokens doing nothing, so the daemon holds the request instead.
        """
        job = self.store.get_job(job_id)
        if job is None:
            return None
        if job.state.is_terminal:
            return job

        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(job_id, []).append(fut)
        try:
            await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError:
            pass
        finally:
            waiters = self._waiters.get(job_id, [])
            if fut in waiters:
                waiters.remove(fut)
        return self.store.get_job(job_id)

    def _notify(self, job_id: int) -> None:
        for fut in self._waiters.pop(job_id, []):
            if not fut.done():
                fut.set_result(None)

    def blocked_reason(self, job_id: int) -> str | None:
        reason = self.last_decision.blocked.get(job_id)
        if reason is not None:
            return reason
        job = self.store.get_job(job_id)
        if job is not None and not job.held and job_id in self.dep_waiting:
            return self.dep_waiting[job_id]
        if job is not None and job.held:
            event = self.store.last_event(job_id, "hold")
            if event is None:
                return "held"
            why = f": {event['reason']}" if event.get("reason") else ""
            if job.hold_until is not None:
                why = f" until {_clock(job.hold_until)}{why}"
            return f"held by {short_actor(str(event.get('actor') or 'unknown'))}{why}"
        return None

    def status(self) -> dict[str, Any]:
        running = self.store.jobs_in_state(JobState.RUNNING)
        queued = self.store.jobs_in_state(JobState.QUEUED)
        used = {"cpu": 0, "mem_mb": 0, "gpu": 0, "gpu_mem_mb": 0}
        for job in running:
            need = effective_request(job, self.cap)
            for key in used:
                used[key] += need[key]
        leases = self._lease_usage()
        for key in used:
            used[key] += leases[key]
        external = self._external_usage() or {"cpu": 0, "mem_mb": 0, "gpu": 0, "gpu_mem_mb": 0}
        cap = self.cap.as_dict()
        return {
            "capacity": cap,
            "used": used,
            "external": external,
            # Free is what the scheduler will actually hand out, so foreign load is
            # subtracted here too -- reporting it as free is the very confusion this
            # measurement exists to remove.
            "free": {k: max(0, cap[k] - used[k] - external.get(k, 0)) for k in used},
            "running": [self._with_usage(j) for j in running],
            "queued": [{**j.to_dict(), "blocked_reason": self.blocked_reason(j.id)} for j in queued],
            "reservation": (
                {
                    "job_id": self.last_decision.reservation.job_id,
                    "start_at": self.last_decision.reservation.start_at,
                    "in_seconds": self.last_decision.reservation.start_at - time.time(),
                    "external": self.last_decision.reservation.external,
                }
                if self.last_decision.reservation
                else None
            ),
            "paused": self.paused,
            "pause_until": self.pause_until,
            "draining": self.draining,
            "free_disk_mb": sysinfo.free_disk_mb(self.cfg.disk_watch_path),
            "disk_floor_mb": self.cfg.disk_floor_mb,
            "load": sysinfo.load_average(),
            "active_leases": len(self.store.active_leases()),
            "measured": self._measured(running),
        }

    def inbox(self, session_id: str, since: float, stall_s: float = 900.0) -> dict[str, Any]:
        """What changed on one session's jobs since ``since``, for that session's agent.

        Finished jobs, anything done to its jobs by someone else (holds, cancels, priority
        changes, interference, adoption after a restart), and running jobs whose log has
        been silent for ``stall_s``. The caller keeps the cursor: pass back ``now``.
        """
        now = time.time()
        finished, stalled = [], []
        for job in self.store.session_jobs(session_id, since=since):
            if job.state.is_terminal:
                finished.append(job.to_dict())
            elif job.state is JobState.RUNNING and job.log_path:
                try:
                    quiet = now - Path(job.log_path).stat().st_mtime
                except OSError:
                    continue
                if quiet >= stall_s:
                    stalled.append({**job.to_dict(), "log_quiet_s": quiet})
        return {
            "now": now,
            "finished": finished,
            "events": self.store.session_events(session_id, since=since),
            "stalled": stalled,
        }

    def _mem_headroom(self, running: list[Job]) -> int | None:
        """Memory a new job can really have: MemAvailable, less what running jobs may
        still grow into up to their declared limits, less the guard. A job whose usage
        is unknown is assumed to still have all of its declared memory to come."""
        available = sysinfo.available_mem_mb()
        if available is None:
            return None
        growth = 0
        for job in running:
            monitor = self.monitors.get(job.id)
            # Page cache is left out: MemAvailable already counts it as free, so
            # treating it as used would hide the job's real room to grow.
            now_mb = 0
            if monitor is not None:
                now_mb = next((v for v in (monitor.mem_anon_now_mb, monitor.mem_now_mb) if v is not None), 0)
            growth += max(0, job.resources.mem_mb - now_mb)
        return available - growth - self.cfg.mem_guard_mb

    def _measured(self, running: list[Job]) -> dict[str, Any]:
        """What the machine is really doing, split into ajs jobs and everything else.

        Unlike ``used`` (declared reservations) and ``external`` (rounded, minus the
        desktop allowance), these are raw measurements, so the two parts add up to what
        `top` would show.
        """
        ajs_cpu = sum(m.cpu_cores_now or 0.0 for m in self.monitors.values())
        ajs_mem = sum(m.mem_now_mb or 0 for m in self.monitors.values())
        total_mem = sysinfo.total_mem_mb()
        available = sysinfo.available_mem_mb()
        return {
            "cpu_ajs": ajs_cpu,
            "cpu_outside": self.external.cpu_cores if self.external.ready else None,
            "cpu_allowance": self.cfg.external_cpu_allowance,
            "iowait_pct": self.external.iowait_pct if self.external.ready else None,
            "mem_total_mb": total_mem,
            "mem_used_mb": None if available is None else total_mem - available,
            "mem_ajs_mb": ajs_mem,
            "mem_outside_mb": self.external.mem_mb,
            "mem_reserve_mb": self.cfg.mem_reserve_mb,
            "mem_headroom_mb": self._mem_headroom(running),
            "quiet": self.quiet_since is not None,
            "noise": self.noise,
            "top_cpu": [p.to_dict() for p in self.foreign.top_cpu()],
            "top_mem": [p.to_dict() for p in self.foreign.top_mem()],
        }

    def _with_usage(self, job: Job) -> dict[str, Any]:
        """A running job's record plus what its cgroup is actually consuming right now,
        so declared and real usage can be shown side by side."""
        data = job.to_dict()
        monitor = self.monitors.get(job.id)
        data["cpu_now"] = None if monitor is None else monitor.cpu_cores_now
        data["mem_now_mb"] = None if monitor is None else monitor.mem_now_mb
        data["interference"] = list(self.interference.get(job.id, []))
        return data

    # --- leases -----------------------------------------------------------

    def acquire_lease(self, project: str, session_id: str, resources: ResourceRequest, reason: str) -> str | None:
        """Grant resources to a caller that will run the work itself.

        Only granted if free right now -- leases do not queue, because a caller holding a
        half-satisfied lease while waiting is exactly how deadlock gets in.
        """
        status = self.status()
        free = status["free"]
        if (
            resources.cpu > free["cpu"]
            or resources.mem_mb > free["mem_mb"]
            or resources.gpu > free["gpu"]
            or resources.gpu_mem_mb > free.get("gpu_mem_mb", 0)
        ):
            return None
        lease_id = secrets.token_hex(8)
        self.store.add_lease(lease_id, project, session_id, resources, reason)
        return lease_id

    def log_tail(self, job_id: int, lines: int = 50) -> str:
        job = self.store.get_job(job_id)
        if job is None or not job.log_path:
            return ""
        path = Path(job.log_path)
        if not path.exists():
            return ""
        with path.open("rb") as fh:
            data = fh.read()
        text = data.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])


def _clock(ts: float | None) -> str:
    """Local wall-clock time for a hold's expiry, with the date only if not today."""
    if ts is None:
        return "-"
    fmt = "%H:%M" if time.localtime(ts)[:3] == time.localtime()[:3] else "%a %d %b %H:%M"
    return time.strftime(fmt, time.localtime(ts))


MEM_UNDERUSE_MIN_WATCH_S = 60.0
"""A job watched for less than this at its end has too few samples to judge."""


def mem_check_after(value: str | None, default_s: int) -> int | None:
    """Seconds into a run to judge its memory use, from ``--meta mem_check``.

    ``off`` disables the check; anything unreadable falls back to the default rather
    than failing a job over a typo in its metadata."""
    if value is None:
        return default_s
    value = value.strip().lower()
    if value in ("off", "no", "false", "0"):
        return None
    units = {"s": 1, "m": 60, "h": 3600}
    scale = units.get(value[-1:], 1)
    try:
        seconds = float(value[:-1] if value[-1:] in units else value) * scale
    except ValueError:
        return default_s
    if not math.isfinite(seconds) or seconds < 0:
        return default_s
    return int(seconds)
