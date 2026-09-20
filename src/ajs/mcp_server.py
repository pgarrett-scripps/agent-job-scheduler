"""MCP server exposing the scheduler to Claude Code and Codex.

A deliberately thin wrapper over `client.Client`. It holds no scheduler state, because
every agent session spawns its own copy of this process -- state here would fragment into
N schedulers that each believe they own the whole machine.

The tool descriptions carry the policy. An agent learns *when* to schedule from the text
below and nowhere else, so it is written as instructions rather than as documentation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .cli import detect_project, parse_duration, parse_mem
from .client import Client
from .protocol import SchedulerError

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None  # type: ignore[assignment]

INSTRUCTIONS = """
Shared-machine job scheduler. Several coding agents work on this one computer at the
same time, so expensive commands must be queued here instead of run directly.

Use `submit_job` (or `run_job`) for any command that will take more than ~30 seconds or
use more than one core: test suites, builds, data processing, searches, benchmarks.
Cheap commands -- ls, cat, git status, a quick grep -- should just be run directly; they
are not worth queueing.

If you are measuring how long something takes, you MUST pass exclusive=true. That makes
the scheduler drain the machine, wait for it to go quiet, and run your job alone, then
report whether anything else interfered. A timing run without it is unreliable and the
number should not be trusted.

For GPU work, pass gpu=1 and declare `gpu_mem`. The card has 4 GB total, so a job that
does not say how much VRAM it needs is charged for the whole thing and will serialise
against every other GPU job. GPU timing runs use gpu_exclusive=true, which is a separate
switch from `exclusive`: a CPU benchmark does not need the card idle, and a GPU benchmark
does not need all 20 cores. Set both only if the job genuinely needs both.

Always declare `cpu` and `mem` honestly. The scheduler hands out slots based on what you
claim, so under-declaring causes the overloading this exists to prevent.
"""


def _client() -> Client:
    return Client()


def _err(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(exc)}


def build_server() -> Any:
    if FastMCP is None:  # pragma: no cover
        raise SystemExit(
            "fastmcp is not installed. Install the MCP extra:\n  uv pip install 'agent-job-scheduler[mcp]'"
        )

    mcp = FastMCP(name="ajs", instructions=INSTRUCTIONS)
    session_id = os.environ.get("AJS_SESSION", f"pid-{os.getpid()}")

    @mcp.tool
    def submit_job(
        command: list[str],
        cwd: str | None = None,
        cpu: int = 1,
        mem: str = "512M",
        gpu: int = 0,
        gpu_mem: str = "0M",
        disk: str = "0M",
        exclusive: bool = False,
        gpu_exclusive: bool = False,
        locks: list[str] | None = None,
        max_runtime: str = "1h",
        job_class: str = "batch",
        project: str | None = None,
    ) -> dict[str, Any]:
        """Queue a command and return immediately with a job id.

        Use this for anything expensive (>30s or >1 core). It returns right away, so you
        can do other work and call `wait_for_job` later. For a command you need the
        result of before continuing, `run_job` is simpler.

        Args:
            command: argv list, e.g. ["cargo", "test", "--release"]. Not a shell string.
            cwd: working directory. Defaults to where the daemon was started, so pass it.
            cpu: CPU slots to reserve. Declare honestly -- the scheduler trusts this.
            mem: memory reservation, e.g. "8G". The job is killed if it exceeds this.
            gpu: set to 1 for any job that touches CUDA. Jobs that leave this at 0 get
                CUDA_VISIBLE_DEVICES="" and cannot see the card at all.
            gpu_mem: VRAM to reserve, e.g. "2G". The card has 4 GB. Omitting this on a
                GPU job reserves the whole card, which is safe but serialises.
            disk: how much disk the job will write, e.g. "20G". Checked against the
                free-space floor before admission.
            exclusive: CPU TIMING RUNS ONLY. Takes every core, waits for the machine to
                settle, and records whether anything else interfered. Required for any
                benchmark whose number you intend to report.
            gpu_exclusive: GPU TIMING RUNS ONLY. Takes the whole card. Independent of
                `exclusive` -- set both only if the measurement depends on both.
            locks: named locks for logical conflicts, e.g. ["sage-index"]. Two jobs
                naming the same lock never run together.
            max_runtime: hard ceiling, e.g. "30m". The job is killed past it. Keep it
                honest: short ceilings let your job backfill ahead of big queued ones.
            job_class: "interactive" if you are blocked waiting, else "batch", or
                "background" for work nobody is waiting on.
            project: defaults to the git repo name at cwd.
        """
        try:
            work_dir = cwd or os.getcwd()
            job = _client().submit(
                project=project or detect_project(Path(work_dir)),
                session_id=session_id,
                cmd=command,
                cwd=work_dir,
                cpu=cpu,
                mem_mb=parse_mem(mem),
                gpu=gpu,
                gpu_mem_mb=parse_mem(gpu_mem),
                disk_mb=parse_mem(disk),
                exclusive=exclusive,
                gpu_exclusive=gpu_exclusive,
                locks=list(locks or []),
                max_runtime_s=parse_duration(max_runtime),
                job_class=job_class,
            )
            return {"ok": True, "job_id": job["id"], "state": job["state"]}
        except (SchedulerError, ValueError) as exc:
            return _err(exc)

    @mcp.tool
    def wait_for_job(job_id: int, timeout_seconds: int = 300) -> dict[str, Any]:
        """Block until a job finishes, then return its result.

        This long-polls: the daemon holds the connection until the job is done, so it
        costs one tool call rather than a polling loop. If it returns with
        timed_out=true the job is still going -- just call it again.

        The result includes `contended` and `contention_note`. For an exclusive timing
        run, contended=true means something else was using the machine and the
        measurement should be discarded and rerun.
        """
        try:
            result = _client().wait(job_id, timeout=float(timeout_seconds))
            return {"ok": True, **_summarise(result)}
        except SchedulerError as exc:
            return _err(exc)

    @mcp.tool
    def run_job(
        command: list[str],
        cwd: str | None = None,
        cpu: int = 1,
        mem: str = "512M",
        gpu: int = 0,
        gpu_mem: str = "0M",
        exclusive: bool = False,
        gpu_exclusive: bool = False,
        max_runtime: str = "1h",
        timeout_seconds: int = 600,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Queue a command and wait for it, returning output. The common case.

        Behaves like running the command directly, except it waits its turn instead of
        piling onto a busy machine. Prefer this when you need the result before you can
        continue; use `submit_job` when you would rather get on with something else.
        """
        submitted = submit_job(
            command=command,
            cwd=cwd,
            cpu=cpu,
            mem=mem,
            gpu=gpu,
            gpu_mem=gpu_mem,
            exclusive=exclusive,
            gpu_exclusive=gpu_exclusive,
            max_runtime=max_runtime,
            job_class="interactive",
            project=project,
        )
        if not submitted.get("ok"):
            return submitted
        waited = wait_for_job(submitted["job_id"], timeout_seconds=timeout_seconds)
        if waited.get("ok") and not waited.get("timed_out"):
            waited["output"] = get_job_logs(submitted["job_id"], lines=200).get("output", "")
        return waited

    @mcp.tool
    def job_status(job_id: int) -> dict[str, Any]:
        """Check a job without blocking.

        If it is still queued, `blocked_reason` says exactly what it is waiting for --
        a busy machine, a lock, the disk floor, or the settle period before a timing run.
        """
        try:
            return {"ok": True, **_summarise(_client().job(job_id))}
        except SchedulerError as exc:
            return _err(exc)

    @mcp.tool
    def get_job_logs(job_id: int, lines: int = 100) -> dict[str, Any]:
        """Return the tail of a job's combined stdout/stderr."""
        try:
            return {"ok": True, "output": _client().logs(job_id, lines)}
        except SchedulerError as exc:
            return _err(exc)

    @mcp.tool
    def cancel_job(job_id: int, reason: str = "cancelled by agent") -> dict[str, Any]:
        """Cancel a queued or running job, killing it and everything it spawned."""
        try:
            return {"ok": True, "cancelled": _client().cancel(job_id, reason)}
        except SchedulerError as exc:
            return _err(exc)

    @mcp.tool
    def list_jobs(project: str | None = None, include_finished: bool = False, limit: int = 20) -> dict[str, Any]:
        """List jobs, most recent first. Includes other agents' jobs."""
        try:
            states = None if include_finished else ["queued", "running"]
            jobs = _client().jobs(project=project, states=states, limit=limit)
            return {"ok": True, "jobs": [_summarise(j) for j in jobs]}
        except SchedulerError as exc:
            return _err(exc)

    @mcp.tool
    def scheduler_status() -> dict[str, Any]:
        """Current machine load: capacity, what is running, what is queued and why.

        Worth checking before submitting something large, to see what else is going on.
        """
        try:
            data = _client().status()
            return {
                "ok": True,
                "capacity": data["capacity"],
                "used": data["used"],
                "free": data["free"],
                "load_1min": data["load"][0],
                "free_disk_mb": data["free_disk_mb"],
                "disk_floor_mb": data["disk_floor_mb"],
                "running": [_summarise(j) for j in data["running"]],
                "queued": [_summarise(j) for j in data["queued"]],
                "paused": data["paused"],
            }
        except SchedulerError as exc:
            return _err(exc)

    return mcp


def _summarise(job: dict[str, Any]) -> dict[str, Any]:
    """Trim a job record to what an agent needs, to keep tool results small."""
    keys = (
        "id",
        "project",
        "state",
        "cmd_str",
        "exit_code",
        "runtime_s",
        "exclusive",
        "gpu_exclusive",
        "gpu_mem_mb",
        "contended",
        "contention_note",
        "blocked_reason",
        "timed_out",
    )
    return {k: job[k] for k in keys if k in job and job[k] is not None}


def main() -> None:
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
