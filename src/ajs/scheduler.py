"""Admission control: who runs, who waits, and who gets promised a start time.

`plan()` is a pure function of (queue, running set, capacity, clock). It performs no I/O
and mutates nothing, so the interesting behaviour -- backfill, starvation avoidance,
fair share -- is testable without spawning a process or a daemon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .models import Job, ResourceRequest

# Resource keys tracked as counted semaphores.
_COUNTED = ("cpu", "mem_mb", "gpu")


@dataclass(slots=True)
class Capacity:
    """Total schedulable resources on this machine."""

    cpu: int
    mem_mb: int
    gpu: int

    @classmethod
    def from_config(cls, cfg: Config) -> Capacity:
        return cls(cpu=cfg.cpu, mem_mb=cfg.mem_mb, gpu=cfg.gpu)

    def as_dict(self) -> dict[str, int]:
        return {"cpu": self.cpu, "mem_mb": self.mem_mb, "gpu": self.gpu}


@dataclass(slots=True)
class Reservation:
    """A promise that a starved job will start no later than ``start_at``.

    Without this, a job requesting most of the machine never runs: small jobs keep
    trickling in and there is never an instant when enough resources are simultaneously
    free. Holding a reservation and only letting jobs backfill *around* it bounds the
    wait.
    """

    job_id: int
    start_at: float
    needs: dict[str, int]
    external: bool = False
    """True when foreign load -- not ajs's own jobs -- is what blocks the job. Such a
    reservation carries no honest start time (nothing tells us when a browser will
    close), so it must not be used to veto backfill."""


@dataclass(slots=True)
class Decision:
    """Output of one scheduling pass."""

    start: list[int] = field(default_factory=list)
    reservation: Reservation | None = None
    blocked: dict[int, str] = field(default_factory=dict)
    """job id -> why it did not start, surfaced verbatim to agents via `ajs status`."""


def effective_request(job: Job, cap: Capacity) -> dict[str, int]:
    """Resolve a job's declared needs against real capacity.

    This is where `exclusive` stops being a special case: it expands into a request for
    every CPU slot on the machine, after which the ordinary counted semaphores guarantee
    nothing else can be running. There is no separate drain mechanism.
    """
    r: ResourceRequest = job.resources
    if r.exclusive:
        return {"cpu": cap.cpu, "mem_mb": cap.mem_mb, "gpu": r.gpu}
    # Deliberately *not* clamped to capacity. Clamping would turn a mis-declared
    # `--cpu 999` into a silent grant of the whole machine; leaving it oversized lets
    # plan() reject it as impossible and tell the submitter what they got wrong.
    return {"cpu": max(1, r.cpu), "mem_mb": max(0, r.mem_mb), "gpu": max(0, r.gpu)}


def _locks_of(job: Job) -> set[str]:
    locks = set(job.resources.locks)
    if job.resources.exclusive:
        # A machine-wide name, so `ajs status` can show *why* everything is waiting.
        locks.add("machine")
    return locks


def _fits(need: dict[str, int], free: dict[str, int]) -> bool:
    return all(need[k] <= free[k] for k in _COUNTED)


def _deduct(free: dict[str, int], need: dict[str, int]) -> None:
    for k in _COUNTED:
        free[k] -= need[k]


def _projected_end(job: Job, now: float) -> float:
    """When a running job must be finished by, per its declared max_runtime.

    Backfill correctness rests on this bound, which is why max_runtime is mandatory at
    submission: a job with no declared ceiling could invalidate every reservation.
    """
    start = job.started_at if job.started_at is not None else now
    return start + job.max_runtime_s


def order_queue(queued: list[Job], last_start: dict[str, float]) -> list[Job]:
    """Priority band first, then fair share between projects, then FIFO.

    Fair share uses each project's most recent start, so a project that just launched
    something drops behind one that has been waiting -- this is what stops a single agent
    with fifty queued jobs from starving everyone else.
    """
    return sorted(
        queued,
        key=lambda j: (j.job_class.rank, last_start.get(j.project, 0.0), j.submitted_at, j.id),
    )


def plan(
    *,
    queued: list[Job],
    running: list[Job],
    cap: Capacity,
    cfg: Config,
    now: float,
    last_start: dict[str, float],
    held_locks: set[str] | None = None,
    lease_usage: dict[str, int] | None = None,
    external_usage: dict[str, int] | None = None,
    free_disk_mb: int = 1 << 30,
    last_finish_at: float = 0.0,
) -> Decision:
    """Decide which queued jobs may start right now.

    ``last_finish_at`` is when the most recent job exited; it gates the settle period
    before an exclusive run, so writeback and dying processes are not still perturbing
    the machine when the measurement starts.

    ``external_usage`` is resource consumed by processes ajs did not start. It is
    subtracted from what is free but *not* from ``cap``, so a busy machine delays jobs
    instead of declaring them impossible.
    """
    decision = Decision()

    free = cap.as_dict()
    for job in running:
        _deduct(free, effective_request(job, cap))
    for usage in (lease_usage, external_usage):
        for key, amount in (usage or {}).items():
            if key in free:
                free[key] -= amount

    locks: set[str] = set(held_locks or set())
    for job in running:
        locks |= _locks_of(job)

    per_project = {j.project: 0 for j in running}
    for job in running:
        per_project[job.project] += 1

    reservation: Reservation | None = None

    for job in order_queue(queued, last_start):
        need = effective_request(job, cap)

        # --- checks that do not depend on current availability -------------
        if need["cpu"] > cap.cpu or need["mem_mb"] > cap.mem_mb or need["gpu"] > cap.gpu:
            decision.blocked[job.id] = f"impossible: needs {need} but machine has {cap.as_dict()}"
            continue

        required_disk = cfg.disk_floor_mb + job.resources.disk_mb
        if free_disk_mb < required_disk:
            # Admitting here risks filling the root filesystem and taking the desktop
            # down with it, so this is a hard stop rather than a delay.
            decision.blocked[job.id] = (
                f"disk guard: {free_disk_mb} MB free, needs {required_disk} MB "
                f"(floor {cfg.disk_floor_mb} MB + job {job.resources.disk_mb} MB)"
            )
            continue

        if per_project.get(job.project, 0) >= cfg.max_jobs_per_project:
            decision.blocked[job.id] = f"project cap: {job.project} already running {cfg.max_jobs_per_project}"
            continue

        job_locks = _locks_of(job)
        conflicting = job_locks & locks
        if conflicting:
            decision.blocked[job.id] = f"waiting on lock(s): {', '.join(sorted(conflicting))}"
            continue

        # --- availability --------------------------------------------------
        if not _fits(need, free):
            outside = _external_note(external_usage)
            if reservation is None:
                reservation = _reserve(job, need, running, cap, now, external_usage)
                decision.reservation = reservation
                decision.blocked[job.id] = (
                    f"waiting for load outside ajs to drop{outside}; no reservation possible"
                    if reservation.external
                    else f"waiting for resources{outside}; "
                    f"reserved to start by {reservation.start_at - now:.0f}s from now"
                )
            else:
                decision.blocked[job.id] = f"waiting for resources{outside}"
            continue

        if reservation is not None and not reservation.external:
            # A starved job holds a reservation. Only let this one jump the queue if it
            # provably finishes before that promised start time (conservative EASY
            # backfill) -- otherwise it would push the reservation back indefinitely.
            if now + job.max_runtime_s > reservation.start_at:
                decision.blocked[job.id] = (
                    f"would delay reserved job #{reservation.job_id}; "
                    f"max_runtime {job.max_runtime_s}s exceeds the "
                    f"{reservation.start_at - now:.0f}s window"
                )
                continue

        if job.resources.exclusive:
            quiet_for = now - last_finish_at if last_finish_at else float("inf")
            if quiet_for < cfg.settle_seconds:
                decision.blocked[job.id] = (
                    f"settling: {cfg.settle_seconds - quiet_for:.0f}s left before the "
                    "machine is quiet enough to time on"
                )
                # Hold the resources rather than letting something else grab them, or we
                # would never get through the settle period.
                _deduct(free, need)
                locks |= job_locks
                continue

        decision.start.append(job.id)
        _deduct(free, need)
        locks |= job_locks
        per_project[job.project] = per_project.get(job.project, 0) + 1

    return decision


def _external_note(external_usage: dict[str, int] | None) -> str:
    """Name foreign load in a blocked message, so `waiting for resources` on an
    apparently idle queue is explicable rather than mysterious."""
    if not external_usage:
        return ""
    parts = []
    if external_usage.get("cpu"):
        parts.append(f"{external_usage['cpu']} cpu")
    if external_usage.get("mem_mb"):
        parts.append(f"{external_usage['mem_mb']} MB")
    return f" ({', '.join(parts)} in use outside ajs)" if parts else ""


def _reserve(
    job: Job,
    need: dict[str, int],
    running: list[Job],
    cap: Capacity,
    now: float,
    external_usage: dict[str, int] | None = None,
) -> Reservation:
    """Earliest time ``job`` can start, assuming running jobs last their full max_runtime.

    Walks the running set in projected-completion order, accumulating freed resources
    until the job fits. Pessimistic by construction: jobs usually finish early, so the
    real start is typically sooner than promised.

    Foreign load is held constant throughout, since nothing declares when it will end.
    If the job still does not fit once every ajs job has drained, foreign load is the
    binding constraint and the returned reservation is marked ``external``.
    """
    free = cap.as_dict()
    for key, amount in (external_usage or {}).items():
        if key in free:
            free[key] -= amount
    drained = dict(free)
    for other in running:
        _deduct(free, effective_request(other, cap))

    if _fits(need, free):
        return Reservation(job_id=job.id, start_at=now, needs=need)

    for other in sorted(running, key=lambda j: _projected_end(j, now)):
        _deduct(free, {k: -v for k, v in effective_request(other, cap).items()})
        if _fits(need, free):
            return Reservation(job_id=job.id, start_at=_projected_end(other, now), needs=need)

    # Nothing running frees enough. Either the machine is simply full of ajs work, or
    # foreign load alone already exceeds what the job needs -- distinguishable by asking
    # whether it would fit on a fully drained scheduler.
    latest = max((_projected_end(j, now) for j in running), default=now)
    return Reservation(job_id=job.id, start_at=latest, needs=need, external=not _fits(need, drained))
