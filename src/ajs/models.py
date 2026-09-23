"""Core data types shared by the daemon, the client, and the CLI."""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class JobState(StrEnum):
    """Lifecycle of a job.

    Terminal states are DONE, FAILED, CANCELLED and TIMEOUT.
    """

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATES


_TERMINAL_STATES = frozenset({JobState.DONE, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT})


class JobClass(StrEnum):
    """Scheduling priority band.

    INTERACTIVE means an agent is blocked waiting on the result, so it jumps ahead of
    work that nobody is watching.
    """

    INTERACTIVE = "interactive"
    BATCH = "batch"
    BACKGROUND = "background"

    @property
    def rank(self) -> int:
        """Lower sorts first."""
        return _CLASS_RANK[self]


_CLASS_RANK = {JobClass.INTERACTIVE: 0, JobClass.BATCH: 1, JobClass.BACKGROUND: 2}


@dataclass(slots=True)
class ResourceRequest:
    """What a job needs before it may start.

    ``exclusive`` is not a separate mechanism: it is expanded at admission time into a
    request for the machine's entire CPU capacity, which makes the existing counted
    semaphores guarantee that nothing else runs alongside it. ``gpu_exclusive`` does the
    same for GPU memory.

    The two are deliberately independent axes. A CPU benchmark does not care what the GPU
    is doing, and making it wait for an idle GPU would cost throughput for nothing; a GPU
    benchmark usually wants both, and can ask for both.
    """

    cpu: int = 1
    mem_mb: int = 512
    gpu: int = 0
    gpu_mem_mb: int = 0
    """VRAM to reserve. The real constraint on a 4 GB card -- `gpu: 1` says *a* GPU is
    needed but not how much of it, so two jobs both assuming they have the whole card
    OOM each other."""
    disk_mb: int = 0
    exclusive: bool = False
    gpu_exclusive: bool = False
    locks: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "cpu": self.cpu,
                "mem_mb": self.mem_mb,
                "gpu": self.gpu,
                "gpu_mem_mb": self.gpu_mem_mb,
                "disk_mb": self.disk_mb,
                "exclusive": self.exclusive,
                "gpu_exclusive": self.gpu_exclusive,
                "locks": self.locks,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> ResourceRequest:
        data = json.loads(raw)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class Job:
    """A unit of work the daemon owns from submission through to exit."""

    id: int
    project: str
    session_id: str
    cmd: list[str]
    cwd: str
    env: dict[str, str]
    resources: ResourceRequest
    max_runtime_s: int
    job_class: JobClass
    state: JobState
    submitted_at: float
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    pid: int | None = None
    unit: str | None = None
    log_path: str | None = None
    # Timing-run provenance. Populated for every job, but only meaningful for exclusive ones.
    load_before: float | None = None
    load_after: float | None = None
    contended: bool = False
    contention_note: str | None = None
    # Set when the scheduler promises a starved job a start time (see scheduler.plan).
    reserved_until: float | None = None
    cancel_reason: str | None = None
    # A held job stays queued but is never admitted until released. Queue management,
    # not a lifecycle state: it can be flipped back and forth while the job waits.
    held: bool = False
    hold_until: float | None = None
    """A timed hold: the daemon releases the job by itself at this epoch time."""
    # Submitter-supplied description, so whoever manages the queue can tell a paper's last
    # missing figure from an exploratory sweep. All optional and never interpreted.
    title: str | None = None
    description: str | None = None
    meta: dict[str, str] = field(default_factory=dict)
    """Free-form key=value pairs, e.g. {"paper": "uno", "est": "40m"}."""

    @property
    def cmd_str(self) -> str:
        return shlex.join(self.cmd)

    @property
    def runtime_s(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """Wire representation. Keep flat and JSON-native for easy consumption by agents."""
        return {
            "id": self.id,
            "project": self.project,
            "session_id": self.session_id,
            "cmd": self.cmd,
            "cmd_str": self.cmd_str,
            "cwd": self.cwd,
            "cpu": self.resources.cpu,
            "mem_mb": self.resources.mem_mb,
            "gpu": self.resources.gpu,
            "gpu_mem_mb": self.resources.gpu_mem_mb,
            "exclusive": self.resources.exclusive,
            "gpu_exclusive": self.resources.gpu_exclusive,
            "locks": self.resources.locks,
            "max_runtime_s": self.max_runtime_s,
            "class": str(self.job_class),
            "state": str(self.state),
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_s": self.runtime_s,
            "exit_code": self.exit_code,
            "pid": self.pid,
            "log_path": self.log_path,
            "load_before": self.load_before,
            "load_after": self.load_after,
            "contended": self.contended,
            "contention_note": self.contention_note,
            "reserved_until": self.reserved_until,
            "cancel_reason": self.cancel_reason,
            "held": self.held,
            "hold_until": self.hold_until,
            "title": self.title,
            "description": self.description,
            "meta": self.meta,
            "note": self.description,  # pre-title name, kept for older clients
        }


def display_name(job: dict[str, Any]) -> str:
    """What to show for a job in a table: its title if the submitter gave one."""
    return job.get("title") or job.get("cmd_str") or ""


def short_actor(actor: str) -> str:
    """`claude:<uuid>` trimmed to its first block: enough to find the session in a list of
    agents, short enough to fit a table column. Full ids stay in the database."""
    kind, sep, rest = actor.partition(":")
    if sep and kind == "claude":
        return f"claude:{rest.split('-')[0]}"
    return actor
