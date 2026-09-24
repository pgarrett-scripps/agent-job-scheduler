"""Launching, supervising and killing jobs.

Jobs run inside transient systemd scopes when available, which gives real enforcement
(a job that declares 4 GB cannot quietly take 40) rather than mere bookkeeping. Without
systemd we fall back to a plain process group, which still allows clean termination of
the whole subtree.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from . import sysinfo
from .config import Config, instance_tag
from .contention import cgroup_path_for_pid
from .models import Job

log = logging.getLogger("ajs.executor")

#: How long to wait for systemd to move a freshly spawned job into its scope. On this
#: hardware it takes ~60 ms; the ceiling is generous because guessing wrong is worse
#: than a short delay (see ``_await_cgroup``).
CGROUP_SETTLE_TIMEOUT_S = 3.0

#: How often to check on a job when the kernel cannot tell us it exited (no pidfd).
EXIT_POLL_S = 0.2

#: Runs the job and records its exit status in a file, so a daemon restarted while the job
#: ran can still say how it ended: the job is no longer its child, so it cannot wait on it.
EXIT_WRAPPER = 'rc=0; "$@" || rc=$?; printf "%s\\n" "$rc" > "$AJS_EXIT_FILE"; exit "$rc"'

#: How long a job's leftover processes get to exit after its main process has, first on
#: their own and then after SIGTERM, before they are SIGKILLed.
DRAIN_GRACE_S = 10.0


class JobProcess:
    """The main process of a job, whether this daemon started it or inherited it.

    Deliberately not an asyncio subprocess: asyncio kills a child that is still running
    when its transport closes, which is every job the moment the daemon exits. A plain
    Popen is left alone, so jobs in their own systemd scopes outlive a restart and the
    next daemon adopts them.
    """

    def __init__(self, pid: int, *, popen: subprocess.Popen[bytes] | None = None, exit_file: Path | None = None):
        self.pid = pid
        self.returncode: int | None = None
        self._popen = popen
        self._exit_file = exit_file
        self._done: asyncio.Future[int | None] | None = None

    @property
    def adopted(self) -> bool:
        return self._popen is None

    def poll(self) -> bool:
        """True once the process has gone; sets ``returncode`` where it can be known."""
        if self._popen is not None:
            code = self._popen.poll()
            if code is None:
                return False
            self.returncode = code
            return True
        if _alive(self.pid):
            return False
        self.returncode = read_exit_file(self._exit_file) if self._exit_file else None
        return True

    async def wait(self) -> int | None:
        """Wait for the process to exit. For an adopted job the code comes from its exit
        file, and is None if it died without writing one (killed by a signal)."""
        if self._done is None:
            self._done = asyncio.ensure_future(self._wait())
        return await asyncio.shield(self._done)

    async def _wait(self) -> int | None:
        loop = asyncio.get_running_loop()
        fd = _pidfd_open(self.pid)
        if fd is None:
            while not self.poll():
                await asyncio.sleep(EXIT_POLL_S)
            return self.returncode
        exited = loop.create_future()
        loop.add_reader(fd, lambda: exited.done() or exited.set_result(None))
        try:
            if not self.poll():
                await exited
            # The pidfd turns readable at exit; a child may need a moment to be reapable.
            while not self.poll():
                await asyncio.sleep(0.01)
        finally:
            loop.remove_reader(fd)
            os.close(fd)
        return self.returncode


def _pidfd_open(pid: int) -> int | None:
    """A file descriptor that turns readable when ``pid`` exits, or None if unsupported.

    ``os.pidfd_open`` is missing from some Python builds even on kernels that have the
    syscall, so fall back to calling it directly (434 on every Linux architecture)."""
    opener = getattr(os, "pidfd_open", None)
    if opener is not None:
        try:
            return opener(pid)
        except OSError:
            return None
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))
    except (OSError, AttributeError):
        return None
    return fd if fd >= 0 else None


def _alive(pid: int) -> bool:
    """True while ``pid`` runs. A zombie has exited; it only waits for its parent."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


def read_exit_file(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


@dataclass(slots=True)
class RunningProcess:
    """Handle on a live job."""

    job_id: int
    proc: JobProcess
    unit: str | None
    log_path: Path
    log_file: IO[bytes] | None
    cgroup: Path | None = None
    """The cgroup the job actually runs in, resolved once systemd has moved it there.
    None means unknown, and callers must treat unknown as "cannot account for this job"
    rather than as "uses nothing"."""


class Executor:
    """Starts jobs and reaps them."""

    def __init__(self, cfg: Config, log_dir: Path, *, use_systemd: bool | None = None, tag: str | None = None) -> None:
        self.cfg = cfg
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag if tag is not None else instance_tag()
        self.unit_prefix = f"ajs-{self.tag}-job-"
        self.drain_grace_s = DRAIN_GRACE_S
        if use_systemd is None:
            use_systemd = cfg.use_systemd and sysinfo.has_systemd_run()
            if cfg.use_systemd and not use_systemd:
                log.warning("systemd-run is unavailable; jobs will run in plain process groups without limits")
        self._systemd = use_systemd

    def _wrap(self, job: Job, need: dict[str, int]) -> tuple[list[str], str | None]:
        """Build the argv that actually gets executed, plus the cgroup unit name."""
        if not self._systemd:
            return list(job.cmd), None

        unit = f"{self.unit_prefix}{job.id}"
        argv = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            f"--unit={unit}",
            f"--description=ajs job {job.id} ({job.project})",
        ]
        if self.cfg.enforce_limits:
            # CPUQuota is expressed as a percentage of one CPU, so 4 slots == 400%.
            argv += [
                "-p",
                f"CPUQuota={need['cpu'] * 100}%",
                "-p",
                f"MemoryMax={need['mem_mb']}M",
                # Let a job that blows its memory budget be killed rather than dragging
                # the whole machine into swap.
                "-p",
                "MemorySwapMax=0",
            ]
        argv += ["/bin/sh", "-c", EXIT_WRAPPER, "ajs-job", *job.cmd]
        return argv, f"{unit}.scope"

    def exit_file(self, job_id: int) -> Path:
        return self.log_dir / f"job-{job_id}.exit"

    async def start(self, job: Job, need: dict[str, int]) -> RunningProcess:
        """Launch ``job``. Raises OSError if the command cannot be spawned."""
        log_path = self.log_dir / f"job-{job.id}.log"
        log_file = log_path.open("wb")

        argv, unit = self._wrap(job, need)
        env = {**os.environ, **job.env}
        # Let a job discover its own scheduling context, e.g. to size a thread pool to
        # the slots it was actually granted instead of to nproc.
        env["AJS_JOB_ID"] = str(job.id)
        env["AJS_CPU"] = str(need["cpu"])
        env["AJS_MEM_MB"] = str(need["mem_mb"])
        env["AJS_GPU_MEM_MB"] = str(need.get("gpu_mem_mb", 0))
        env["AJS_EXCLUSIVE"] = "1" if job.resources.exclusive else "0"
        env["AJS_GPU_EXCLUSIVE"] = "1" if job.resources.gpu_exclusive else "0"
        env["AJS_EXIT_FILE"] = str(self.exit_file(job.id))
        self.exit_file(job.id).unlink(missing_ok=True)
        if not (job.resources.gpu or job.resources.gpu_exclusive):
            # A job that did not ask for the GPU must not be able to take it by accident;
            # the scheduler's VRAM arithmetic is only true if this holds.
            env["CUDA_VISIBLE_DEVICES"] = ""

        cwd = job.cwd if Path(job.cwd).is_dir() else str(Path.home())

        try:
            popen = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            log_file.close()
            raise

        proc = JobProcess(popen.pid, popen=popen)
        rp = RunningProcess(job_id=job.id, proc=proc, unit=unit, log_path=log_path, log_file=log_file)
        rp.cgroup = await self._await_cgroup(rp)
        return rp

    def adopt(self, job: Job) -> RunningProcess | None:
        """Take over a job a previous daemon started, if it is still running.

        Only a job in its own scope can have survived: without systemd it lived in the
        daemon's cgroup and went down with it. The PID must still sit in the job's scope,
        which also rules out the PID having been reused by something else.

        This reads /proc and never asks systemd, so it works even when this daemon could
        not reach systemd at startup. Failing the job then would free its resources while
        it kept running, and the scheduler would pile new jobs on top of it.
        """
        if not job.unit or not job.pid:
            return None
        cgroup = cgroup_path_for_pid(job.pid)
        if cgroup is None or cgroup.name != job.unit:
            return None
        proc = JobProcess(job.pid, exit_file=self.exit_file(job.id))
        log_path = Path(job.log_path) if job.log_path else self.log_dir / f"job-{job.id}.log"
        return RunningProcess(job_id=job.id, proc=proc, unit=job.unit, log_path=log_path, log_file=None, cgroup=cgroup)

    @staticmethod
    async def _await_cgroup(rp: RunningProcess) -> Path | None:
        """Resolve the cgroup the job runs in.

        ``systemd-run --scope`` registers the scope and only *then* moves itself into it,
        tens of milliseconds after we get the PID back. Reading ``/proc/<pid>/cgroup``
        immediately returns the daemon's own cgroup, and a monitor pointed at that
        attributes the job's every CPU-second to "someone else": timing runs get stamped
        contended by their own work, and the foreign-load tracker charges each running
        job against free capacity a second time. So poll until the PID is in a cgroup
        named after its unit.
        """
        if rp.unit is None:
            return cgroup_path_for_pid(rp.proc.pid)
        deadline = time.monotonic() + CGROUP_SETTLE_TIMEOUT_S
        while True:
            path = cgroup_path_for_pid(rp.proc.pid)
            if path is not None and path.name == rp.unit:
                return path
            if rp.proc.poll() or time.monotonic() >= deadline:
                # Exited before the move completed, or systemd never delivered. Better
                # to report "unknown" than to measure the wrong cgroup as if it were ours.
                if rp.proc.returncode is None:
                    log.warning("job %s: cgroup did not appear within %.1fs", rp.job_id, CGROUP_SETTLE_TIMEOUT_S)
                return None
            await asyncio.sleep(0.01)

    async def stop(self, rp: RunningProcess, *, grace_s: float = 10.0) -> None:
        """Terminate a job and everything it spawned, escalating to SIGKILL."""
        if rp.unit:
            # --no-block: a plain `systemctl stop` waits for the unit to go down, up to
            # systemd's stop timeout, and this is awaited from the scheduler tick. The
            # grace period and SIGKILL escalation below are our own.
            await self._systemctl("stop", "--no-block", rp.unit)
        else:
            self._signal_group(rp, signal.SIGTERM)

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(rp.proc.wait(), timeout=grace_s)
            return

        if rp.unit:
            await self._systemctl("kill", "--signal=SIGKILL", rp.unit)
        else:
            self._signal_group(rp, signal.SIGKILL)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(rp.proc.wait(), timeout=5.0)

    @staticmethod
    def is_empty(rp: RunningProcess) -> bool:
        """True once nothing the job started is still alive.

        The main process exiting is not enough: a shell wrapper dies on SIGTERM at once
        while the workers it spawned keep running, and a job can leave background children
        behind on a normal exit. Their cores are still in use until they are gone.
        """
        # Only a scope's cgroup is the job's alone; without systemd, rp.cgroup is the
        # daemon's own, which is never empty.
        if rp.unit and rp.cgroup is not None:
            try:
                return not (rp.cgroup / "cgroup.procs").read_text().strip()
            except OSError:
                return True  # systemd removes the cgroup once the scope is empty
        try:
            os.killpg(rp.proc.pid, 0)  # start_new_session: the job's pgid is its pid
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    async def drain(self, rp: RunningProcess, *, grace_s: float | None = None) -> bool:
        """After the main process exits, make sure everything it started is gone too.

        Waits ``grace_s`` for leftovers to finish, then SIGTERMs them, waits again, then
        SIGKILLs. Returns False if something survived even that (it escaped the process
        group, or is stuck in the kernel); the caller frees the slots anyway, since a job
        that never drains must not wedge the queue forever.
        """
        if grace_s is None:
            grace_s = self.drain_grace_s
        if await self._wait_empty(rp, grace_s):
            return True
        log.warning("job %s: main process exited but its children are still running; stopping them", rp.job_id)
        if rp.unit:
            await self._systemctl("stop", "--no-block", rp.unit)
        else:
            self._signal_group(rp, signal.SIGTERM)
        if await self._wait_empty(rp, grace_s):
            return True
        if rp.unit:
            await self._systemctl("kill", "--signal=SIGKILL", rp.unit)
        else:
            self._signal_group(rp, signal.SIGKILL)
        if await self._wait_empty(rp, 5.0):
            return True
        log.error("job %s: processes survived SIGKILL; freeing its slots anyway", rp.job_id)
        return False

    async def _wait_empty(self, rp: RunningProcess, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not self.is_empty(rp):
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.1)
        return True

    def _signal_group(self, rp: RunningProcess, sig: int) -> None:
        # The pgid is the main pid (start_new_session), and stays valid after the main
        # process is reaped as long as any member lives: os.getpgid(pid) would not.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(rp.proc.pid, sig)

    def set_mem_limit(self, unit: str, mem_mb: int) -> bool:
        """Lower a running job's MemoryMax in place. False if systemd refused."""
        try:
            done = subprocess.run(
                ["systemctl", "--user", "set-property", "--runtime", unit, f"MemoryMax={mem_mb}M"],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return done.returncode == 0

    async def _systemctl(self, *args: str) -> None:
        with contextlib.suppress(OSError):
            proc = await asyncio.create_subprocess_exec(
                "systemctl",
                "--user",
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

    def orphan_units(self) -> list[str]:
        """Job scopes left behind by a previous life of *this* daemon.

        The daemon reconciles against these on startup so a crash does not leak cgroups
        that still hold real resources the scheduler has forgotten about. Matched by
        this instance's tag: another daemon's jobs are not orphans, they are its jobs.
        """
        if not self._systemd:
            return []
        try:
            out = subprocess.run(
                ["systemctl", "--user", "list-units", "--all", "--no-legend", "--plain", f"{self.unit_prefix}*.scope"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return []
        units = []
        for line in out.stdout.splitlines():
            parts = line.split()
            if parts and parts[0].startswith(self.unit_prefix):
                units.append(parts[0])
        return units

    @staticmethod
    async def stop_unit(unit: str) -> None:
        with contextlib.suppress(OSError):
            proc = await asyncio.create_subprocess_exec(
                "systemctl",
                "--user",
                "stop",
                unit,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
