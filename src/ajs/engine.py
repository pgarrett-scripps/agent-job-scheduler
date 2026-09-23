"""The daemon's brain: owns state, runs the scheduling tick, supervises jobs.

Single-threaded asyncio. Because every mutation happens in one task on one event loop,
there is no locking anywhere in the system -- concurrency bugs are designed out rather
than guarded against.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from pathlib import Path
from typing import Any

from . import sysinfo
from .config import Config, db_path, log_dir
from .contention import ContentionMonitor, ExternalLoad
from .db import Store
from .executor import Executor, RunningProcess
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
        self.paused = False
        self.pause_until: float | None = None
        """Set by `ajs pause --for`: the tick resumes by itself at this epoch time."""
        self.draining = False
        self._waiters: dict[int, list[asyncio.Future[None]]] = {}
        self._wake = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()

    # --- lifecycle --------------------------------------------------------

    async def recover(self) -> None:
        """Reconcile after a daemon restart.

        Jobs marked RUNNING in the database have no supervising task any more, and their
        cgroups may still be alive holding resources the scheduler no longer knows about.
        Stop them and mark them failed rather than leaving phantom capacity consumed.
        """
        stale = self.store.jobs_in_state(JobState.RUNNING)
        for job in stale:
            log.warning("recovering orphaned job %s from previous daemon life", job.id)
            self.store.update_job(
                job.id,
                state=str(JobState.FAILED),
                finished_at=time.time(),
                exit_code=None,
                cancel_reason="daemon restarted while job was running",
            )
        for unit in self.executor.orphan_units():
            log.warning("stopping orphaned unit %s", unit)
            await Executor.stop_unit(unit)

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
        for task in list(self._tasks):
            task.cancel()
        for rp in list(self.running.values()):
            await self.executor.stop(rp)
        self.store.close()

    # --- the tick ---------------------------------------------------------

    async def tick(self) -> None:
        now = time.time()
        self._sample_external(now)

        expired = self.store.expire_leases(now - self.cfg.lease_timeout_s)
        for lease_id in expired:
            log.warning("reclaimed lease %s: holder stopped heartbeating", lease_id)

        self._enforce_runtime_limits(now)
        self._enforce_disk_floor()
        self._expire_timed_holds(now)

        if self.paused or self.draining:
            self.last_decision = Decision(
                blocked={
                    j.id: "scheduler is paused" if self.paused else "scheduler is draining"
                    for j in self.store.jobs_in_state(JobState.QUEUED)
                    if not j.held
                }
            )
            return

        # Held jobs are invisible to the planner: they neither start nor claim the
        # reservation, so holding a starved giant also stops it vetoing backfill.
        queued = [j for j in self.store.jobs_in_state(JobState.QUEUED) if not j.held]
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
        )
        self.last_decision = decision

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
        if self.cfg.track_external_load:
            self.external.sample(now, own_cpu, own_mem, own_cgroups)

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
            with contextlib.suppress(Exception):
                rp.log_file.close()

        now = time.time()
        # Keep the job's slots until its whole process tree is gone, not just the main
        # process; otherwise the next job starts on top of the dying workers.
        with contextlib.suppress(Exception):
            await self.executor.drain(rp)
        self.running.pop(job_id, None)
        self.last_finish_at = time.time()

        job = self.store.get_job(job_id)
        monitor = self.monitors.pop(job_id, None)

        fields: dict[str, Any] = {
            "finished_at": now,
            "exit_code": exit_code,
            "load_after": sysinfo.load_average()[0],
        }

        if monitor is not None:
            report = monitor.finish(now)
            fields["contended"] = 1 if report.contended else 0
            fields["contention_note"] = report.note
            if report.contended and job is not None and (job.resources.exclusive or job.resources.gpu_exclusive):
                log.warning("job %s ran CONTENDED: %s", job_id, report.note)

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
    ) -> Job:
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
        }

    def _with_usage(self, job: Job) -> dict[str, Any]:
        """A running job's record plus what its cgroup is actually consuming right now,
        so declared and real usage can be shown side by side."""
        data = job.to_dict()
        monitor = self.monitors.get(job.id)
        data["cpu_now"] = None if monitor is None else monitor.cpu_cores_now
        data["mem_now_mb"] = None if monitor is None else monitor.mem_now_mb
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
