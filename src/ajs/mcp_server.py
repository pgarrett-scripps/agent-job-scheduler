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

from .advice import submission_warnings
from .cli import detect_project, forwarded_env, parse_duration, parse_mem, parse_when, session_identity
from .client import Client
from .protocol import SchedulerError

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None  # type: ignore[assignment]

INSTRUCTIONS = """
Shared-machine job scheduler. Several coding agents work on this one computer at the
same time. This exists to stop genuinely heavy jobs colliding. It is not a gate on
ordinary work.

NEVER refuse, delay, or ask permission for a task because the machine is busy. The
scheduler queues jobs, it does not veto them, and most commands should not come here at
all. Do not tell the user you are waiting for resources to free up. Do not treat a
`contended` result as a reason to stop doing what was asked. If you are unsure whether
something is heavy enough to queue, just run it directly.

Queue a command only if it will use SEVERAL CORES AT ONCE, MANY GB OF RAM, OR THE GPU:
parallel builds and test suites (cargo build/test, make -j, pytest -n), data processing
over large files, proteomics searches, parameter sweeps, model training or inference,
and benchmarks whose numbers will be reported.

Run everything else directly, without comment: document builds (latexmk, tectonic,
pandoc, quarto, sphinx), git, ls, cat, grep, rg, find, linters and formatters, package
installs, single-threaded scripts, short test runs.

Wall-clock time is NOT the test. A five-minute single-threaded LaTeX build is not worth
queueing; a ten-second 20-core compile is.

If you are measuring how long something takes, you MUST pass exclusive=true. That makes
the scheduler take the whole machine, wait for it to go quiet, run your job alone, and
report whether anything else interfered. A timing run without it is unreliable and the
number should not be trusted. This is the one case where waiting is correct.
"Quiet" is ajs's job, not your script's: it waits until CPU and iowait outside ajs have
stayed low for a minute before starting, and your max_runtime does not run meanwhile. Do
not add your own busy-wait. While the job runs, ajs names any process that disturbs it
(`interference` in job_status); afterwards `contended` says whether to rerun that chunk.

For GPU work, pass gpu=1 and declare `gpu_mem`. The card has 4 GB total, so a job that
does not say how much VRAM it needs is charged for the whole thing and will serialise
against every other GPU job. GPU timing runs use gpu_exclusive=true, which is a separate
switch from `exclusive`: a CPU benchmark does not need the card idle, and a GPU benchmark
does not need all 20 cores. Set both only if the job genuinely needs both.

Split long runs. If a job would run over about 30 minutes and the work divides (per
file, per sample, per parameter set), submit resumable chunks that each write their own
output, rather than one multi-hour job. Other agents' jobs can then start between your
chunks instead of waiting hours. When chunks must run in order, chain them with
`after_ok`: each waits for the previous one, and a failure cancels the rest. Never use
exclusive=true for throughput: it drains the whole machine first. Submissions that look
like either pattern come back with `warnings`.

Pass `title` and `description` with every submission: a short name, and one line on
what the job is for and what it unblocks. Add `meta` for anything else worth knowing,
especially `est` (expected runtime). The queue is ordered by people reading these.

Do not reorder the queue (hold_job, release_job, set_job_priority) unless the user has
explicitly asked you to in this conversation. Cancelling your own job stays fine.

When you do queue something, declare `cpu` and `mem` honestly. The scheduler hands out
slots based on what you claim, so under-declaring causes the overloading this exists to
prevent. Over-declaring wastes it the other way: memory you reserve and never touch
keeps other jobs queued on an idle machine. CPU is overbooked for ordinary jobs, so
memory is what decides how many run at once: declare your real peak plus about 20%.
If a job's peak stays under 60% of its reservation (with 4 GB or more unused), ajs tells
you in your inbox after 5 minutes (or at the end), and again if that unused memory is
what keeps another job queued; size your queued and next submissions from that. A job
that loads slowly can move the check with meta `mem_check=20m`, or turn it off with
`mem_check=off`.

A job past its max_runtime is killed. About 10 minutes before (at 80% for short jobs)
your inbox gets a runtime-warning; if the job needs longer, call `set_job_runtime` with
a reason, up to twice what you submitted. Timing runs cannot be extended. If a job will
clearly finish early, shorten it the same way: that frees the slot sooner.
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
    session_id = session_identity()

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
        env: dict[str, str] | None = None,
        title: str = "",
        description: str = "",
        meta: dict[str, str] | None = None,
        note: str = "",
        start_after: str = "",
        after_ok: list[int] | None = None,
        after_any: list[int] | None = None,
    ) -> dict[str, Any]:
        """Queue a command and return immediately with a job id.

        For heavy work only: several cores at once, many GB of RAM, or the GPU. Document
        builds, linters, git, and single-threaded scripts should be run directly instead
        -- queueing them adds latency and buys nothing.

        It returns right away, so you can do other work and call `wait_for_job` later.
        For a command you need the result of before continuing, `run_job` is simpler.

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
            env: extra environment variables for the job. PATH and the usual toolchain
                variables are forwarded from this session automatically.
            title: short name shown in the queue, under 60 characters, e.g.
                "spectrl Table 2 timings". Shown instead of the command.
            description: one or two lines on what the job is for and what it unblocks,
                e.g. "last missing number in the spectrl paper; blocks submission".
                Whoever manages the queue uses this to decide what goes first.
            meta: optional key/value extras, e.g. {"paper": "spectrl", "est": "40m"}.
            note: older name for `description`; still accepted.
            start_after: do not start before this: "5h", "22:30" or "2026-09-23 08:00".
                The job is queued held and released by the daemon at that time.
            after_ok: job ids that must finish successfully first. If one fails or is
                cancelled, this job is cancelled too. Use it to chain the chunks of a
                split-up long run, so a failure stops the rest instead of burning them.
            after_any: job ids that must finish first, however they end.
        """
        try:
            work_dir = cwd or os.getcwd()
            job = _client().submit(
                project=project or detect_project(Path(work_dir)),
                session_id=session_id,
                cmd=command,
                cwd=work_dir,
                env={**forwarded_env(), **(env or {})},
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
                **({"title": title} if title else {}),
                **({"description": description or note} if description or note else {}),
                **({"meta": {str(k): str(v) for k, v in meta.items()}} if meta else {}),
                **({"hold_until": parse_when(start_after)} if start_after else {}),
                **({"after_ok": list(after_ok)} if after_ok else {}),
                **({"after_any": list(after_any)} if after_any else {}),
            )
            result: dict[str, Any] = {"ok": True, "job_id": job["id"], "state": job["state"]}
            warnings = submission_warnings(
                exclusive=exclusive,
                gpu_exclusive=gpu_exclusive,
                max_runtime_s=parse_duration(max_runtime),
                title=title,
            ) + list(job.get("warnings") or [])
            if warnings:
                result["warnings"] = warnings
            return result
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
        """Queue a command and wait for it, returning output. The common case for heavy work.

        Behaves like running the command directly, except it waits its turn instead of
        piling onto a busy machine. Prefer this when you need the result before you can
        continue; use `submit_job` when you would rather get on with something else.

        Only for work that is actually heavy (several cores, many GB, or the GPU). Run
        light commands with your normal shell tool -- that is faster and always correct.
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
        if submitted.get("warnings"):
            waited["warnings"] = submitted["warnings"]
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

    def _manage(action: str, job_id: int, reason: str, **extra: Any) -> dict[str, Any]:
        if not reason.strip():
            return {"ok": False, "error": "reason is required: say what the user asked for"}
        try:
            job = getattr(_client(), action)(job_id, **extra, actor=session_id, reason=reason)
            return {"ok": True, "job": _summarise(job)}
        except SchedulerError as exc:
            return _err(exc)

    @mcp.tool
    def hold_job(job_id: int, reason: str, until: str = "") -> dict[str, Any]:
        """Keep a queued job from starting until `release_job`. It keeps its place in line.

        `until` makes it a timed hold that releases by itself: "5h", "22:30" or
        "2026-09-23 08:00".

        QUEUE MANAGEMENT: call this ONLY when the user has explicitly asked, in this
        conversation, for this change to the queue. Never to get your own job ahead and
        never on your own judgement: it reorders other agents' work. `reason` is required
        and is recorded with your session in `ajs events`.
        """
        try:
            extra = {"until": parse_when(until)} if until else {}
        except ValueError as exc:
            return _err(exc)
        return _manage("hold", job_id, reason, **extra)

    @mcp.tool
    def release_job(job_id: int, reason: str) -> dict[str, Any]:
        """Let a held job be scheduled again.

        QUEUE MANAGEMENT: call this ONLY when the user has explicitly asked, in this
        conversation, for this change to the queue. Never to get your own job ahead and
        never on your own judgement: it reorders other agents' work. `reason` is required
        and is recorded with your session in `ajs events`.
        """
        return _manage("release", job_id, reason)

    @mcp.tool
    def set_job_priority(job_id: int, job_class: str, reason: str) -> dict[str, Any]:
        """Move a queued job to another band: "interactive", "batch" or "background".

        QUEUE MANAGEMENT: call this ONLY when the user has explicitly asked, in this
        conversation, for this change to the queue. Never to get your own job ahead and
        never on your own judgement: it reorders other agents' work. `reason` is required
        and is recorded with your session in `ajs events`.
        """
        return _manage("set_priority", job_id, reason, job_class=job_class)

    @mcp.tool
    def set_job_runtime(job_id: int, max_runtime: str, reason: str) -> dict[str, Any]:
        """Change YOUR OWN queued or running job's max_runtime, e.g. "2h".

        Use it when a runtime-warning says the job is about to be killed and it needs
        longer, or to shorten a job that will finish early. It can go up to twice the
        submitted value; timing (exclusive) runs cannot be extended. `reason` is
        required and recorded in `ajs events`; an extension that delays a reserved job
        tells that job's owner.
        """
        if not reason.strip():
            return {"ok": False, "error": "reason is required: say why the job needs a different ceiling"}
        try:
            job = _client().set_runtime(job_id, parse_duration(max_runtime), actor=session_id, reason=reason)
            return {"ok": True, "job": _summarise(job)}
        except (SchedulerError, ValueError) as exc:
            return _err(exc)

    @mcp.tool
    def queue_events(job_id: int | None = None, limit: int = 30) -> dict[str, Any]:
        """Audit trail of holds, releases and priority changes: who, when, why."""
        try:
            return {"ok": True, "events": _client().events(job_id, limit)}
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
        "cpu_now",
        "mem_now_mb",
        "exclusive",
        "gpu_exclusive",
        "gpu_mem_mb",
        "contended",
        "contention_note",
        "blocked_reason",
        "timed_out",
        "class",
        "held",
        "hold_until",
        "after_ok",
        "after_any",
        "title",
        "description",
        "meta",
        "session_id",
    )
    return {k: job[k] for k in keys if k in job and job[k] is not None}


def main() -> None:
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
