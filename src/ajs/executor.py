"""Launching, supervising and killing jobs.

Jobs run inside transient systemd scopes when available, which gives real enforcement
(a job that declares 4 GB cannot quietly take 40) rather than mere bookkeeping. Without
systemd we fall back to a plain process group, which still allows clean termination of
the whole subtree.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import subprocess
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


@dataclass(slots=True)
class RunningProcess:
    """Handle on a live job."""

    job_id: int
    proc: asyncio.subprocess.Process
    unit: str | None
    log_path: Path
    log_file: IO[bytes]
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
        argv += list(job.cmd)
        return argv, f"{unit}.scope"

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
        if not (job.resources.gpu or job.resources.gpu_exclusive):
            # A job that did not ask for the GPU must not be able to take it by accident;
            # the scheduler's VRAM arithmetic is only true if this holds.
            env["CUDA_VISIBLE_DEVICES"] = ""

        cwd = job.cwd if Path(job.cwd).is_dir() else str(Path.home())

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            log_file.close()
            raise

        rp = RunningProcess(job_id=job.id, proc=proc, unit=unit, log_path=log_path, log_file=log_file)
        rp.cgroup = await self._await_cgroup(rp)
        return rp

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
            if rp.proc.returncode is not None or time.monotonic() >= deadline:
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

    def _signal_group(self, rp: RunningProcess, sig: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(rp.proc.pid), sig)

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
