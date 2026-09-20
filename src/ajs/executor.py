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
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .config import Config
from .models import Job

log = logging.getLogger("ajs.executor")


@dataclass(slots=True)
class RunningProcess:
    """Handle on a live job."""

    job_id: int
    proc: asyncio.subprocess.Process
    unit: str | None
    log_path: Path
    log_file: IO[bytes]


class Executor:
    """Starts jobs and reaps them."""

    def __init__(self, cfg: Config, log_dir: Path, *, use_systemd: bool | None = None) -> None:
        self.cfg = cfg
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._systemd = cfg.use_systemd if use_systemd is None else use_systemd

    def _wrap(self, job: Job, need: dict[str, int]) -> tuple[list[str], str | None]:
        """Build the argv that actually gets executed, plus the cgroup unit name."""
        if not self._systemd:
            return list(job.cmd), None

        unit = f"ajs-job-{job.id}"
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

        return RunningProcess(job_id=job.id, proc=proc, unit=unit, log_path=log_path, log_file=log_file)

    async def stop(self, rp: RunningProcess, *, grace_s: float = 10.0) -> None:
        """Terminate a job and everything it spawned, escalating to SIGKILL."""
        if rp.unit:
            await self._systemctl("stop", rp.unit)
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

    @staticmethod
    def orphan_units() -> list[str]:
        """Job scopes left behind by a previous daemon life.

        The daemon reconciles against these on startup so a crash does not leak cgroups
        that still hold real resources the scheduler has forgotten about.
        """
        try:
            out = subprocess.run(
                ["systemctl", "--user", "list-units", "--all", "--no-legend", "--plain", "ajs-job-*.scope"],
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
            if parts and parts[0].startswith("ajs-job-"):
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
