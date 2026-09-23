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
_COUNTED = ("cpu", "mem_mb", "gpu", "gpu_mem_mb")


@dataclass(slots=True)
class Capacity:
    """Total schedulable resources on this machine."""

    cpu: int
    mem_mb: int
    gpu: int
    gpu_mem_mb: int = 0

    @classmethod
    def from_config(cls, cfg: Config) -> Capacity:
        return cls(cpu=cfg.cpu, mem_mb=cfg.mem_mb, gpu=cfg.gpu, gpu_mem_mb=cfg.gpu_mem_mb)

    def as_dict(self) -> dict[str, int]:
        return {"cpu": self.cpu, "mem_mb": self.mem_mb, "gpu": self.gpu, "gpu_mem_mb": self.gpu_mem_mb}


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
    settling: list[int] = field(default_factory=list)
    """Exclusive jobs holding the machine while it goes quiet."""
    unquiet_start: dict[int, str] = field(default_factory=dict)
    """Exclusive jobs started after ``quiet_max_wait_s`` without a quiet window, and why."""


def effective_request(job: Job, cap: Capacity) -> dict[str, int]:
    """Resolve a job's declared needs against real capacity.

    This is where `exclusive` stops being a special case: it expands into a request for
    every CPU slot on the machine, after which the ordinary counted semaphores guarantee
    nothing else can be running. There is no separate drain mechanism.

    ``gpu_exclusive`` does the same for the GPU, on a **separate axis**. A CPU benchmark
    holding all 20 cores does not need the card idle, and a model that owns all 4 GB of
    VRAM barely touches the CPU -- coupling them would idle one resource whenever the
    other was being measured, which on a single-GPU laptop is most of the time.
    """
    r: ResourceRequest = job.resources
    # Deliberately *not* clamped to capacity. Clamping would turn a mis-declared
    # `--cpu 999` into a silent grant of the whole machine; leaving it oversized lets
    # plan() reject it as impossible and tell the submitter what they got wrong.
    need = {
        "cpu": cap.cpu if r.exclusive else max(1, r.cpu),
        "mem_mb": cap.mem_mb if r.exclusive else max(0, r.mem_mb),
        "gpu": max(0, r.gpu),
        "gpu_mem_mb": max(0, r.gpu_mem_mb),
    }
    if r.gpu_exclusive:
        need["gpu"] = max(cap.gpu, need["gpu"])
        need["gpu_mem_mb"] = cap.gpu_mem_mb
    elif need["gpu"]:
        if not need["gpu_mem_mb"]:
            # `--gpu 1` with no VRAM figure is the common case and the dangerous one: two
            # such jobs would each assume the whole card and OOM each other. Charge them
            # the whole device unless they say otherwise.
            need["gpu_mem_mb"] = cap.gpu_mem_mb
        else:
            # A job that named a VRAM slice is gated by VRAM, not by device count.
            # Keeping the device as a counted semaphore here would cap a single-GPU
            # machine at one GPU job no matter how little memory each wanted, which is
            # precisely the sharing that declaring `gpu_mem` is supposed to buy. A
            # machine with no card has no VRAM capacity, so plan() still rejects the
            # request as impossible rather than leaving it queued forever.
            need["gpu"] = 0
    return need


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


def _external_for(job: Job, external_usage: dict[str, int] | None) -> dict[str, int]:
    """The slice of foreign load that is charged against ``job``.

    An exclusive job asks for the entire machine, so if foreign CPU and memory were
    charged against it too, it could never start on a laptop that always has a desktop
    session -- the request would exceed what is free by definition, forever. What
    exclusivity can actually guarantee here is that no *other ajs job* runs alongside;
    whether the desktop interfered is then measured and reported honestly by the
    contention monitor rather than pretended away in advance.

    Foreign VRAM is charged to everyone, exclusive or not. The display reserve already
    covers the compositor; anything else holding the card is a compute process that
    would OOM a job admitted on top of it, and a timing run is not made trustworthy by
    ignoring that.
    """
    usage = {k: v for k, v in (external_usage or {}).items() if k in _COUNTED}
    if job.resources.exclusive:
        usage.pop("cpu", None)
        usage.pop("mem_mb", None)
    return usage


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
    quiet_since: float | None = 0.0,
    noise: str = "",
    settling_since: dict[int, float] | None = None,
) -> Decision:
    """Decide which queued jobs may start right now.

    ``last_finish_at`` is when the most recent job exited; it gates the settle period
    before an exclusive run, so writeback and dying processes are not still perturbing
    the machine when the measurement starts.

    ``external_usage`` is resource consumed by processes ajs did not start. It is
    subtracted from what is free but *not* from ``cap``, so a busy machine delays jobs
    instead of declaring them impossible.

    ``quiet_since`` is when load outside ajs last dropped below the quiet limits, or None
    while it is still above them; ``noise`` says what is over. ``settling_since`` is when
    each exclusive job began holding the machine, to cap how long it waits for quiet.
    """
    decision = Decision()

    free = cap.as_dict()
    for job in running:
        _deduct(free, effective_request(job, cap))
    leases = {k: v for k, v in (lease_usage or {}).items() if k in free}
    for key, amount in leases.items():
        free[key] -= amount

    # Two pools, differing only in whether foreign CPU and memory are charged: exclusive
    # jobs draw from the one that ignores them (see _external_for).
    free_exclusive = dict(free)
    for key, amount in (external_usage or {}).items():
        if key in free:
            free[key] -= amount
            if key not in ("cpu", "mem_mb"):
                free_exclusive[key] -= amount

    # Everything holding resources that will be given back at a known time: running jobs
    # plus, later in the pass, exclusive jobs parked in their settle period. Reservations
    # are projected from this list, so a hold that is missing from it produces a promise
    # the scheduler cannot keep.
    holds: list[tuple[dict[str, int], float]] = [
        (effective_request(job, cap), _projected_end(job, now)) for job in running
    ]

    # Lock name -> holders. A plain lock admits one; a name in cfg.extra_semaphores
    # admits that many, e.g. {"api:anthropic": 3} caps concurrent API-hammering jobs.
    locks: dict[str, int] = dict.fromkeys(held_locks or (), 1)
    for job in running:
        for name in _locks_of(job):
            locks[name] = locks.get(name, 0) + 1

    def lock_capacity(name: str) -> int:
        return max(1, int(cfg.extra_semaphores.get(name, 1)))

    def take_locks(names: set[str]) -> None:
        for name in names:
            locks[name] = locks.get(name, 0) + 1

    per_project = {j.project: 0 for j in running}
    for job in running:
        per_project[job.project] += 1

    reservation: Reservation | None = None

    # A running exclusive job is charged the whole machine, so every job behind it
    # fails the fit test. Say so, rather than blaming whatever small foreign load
    # happens to be present.
    holder = next((j for j in running if j.resources.exclusive), None)

    for job in order_queue(queued, last_start):
        need = effective_request(job, cap)

        # --- checks that do not depend on current availability -------------
        if any(need[k] > cap.as_dict()[k] for k in _COUNTED):
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
        conflicting = {name for name in job_locks if locks.get(name, 0) >= lock_capacity(name)}
        if conflicting:
            decision.blocked[job.id] = f"waiting on lock(s): {', '.join(sorted(conflicting))}"
            continue

        # --- availability --------------------------------------------------
        external = _external_for(job, external_usage)
        pool = free_exclusive if job.resources.exclusive else free
        if not _fits(need, pool):
            outside = (
                f" (machine held by exclusive job #{holder.id}, {holder.project})"
                if holder is not None
                else _external_note(external)
            )
            if reservation is None:
                reservation = _reserve(job, need, cap, now, holds=holds, static={**leases, **external})
                decision.reservation = reservation
                if not reservation.external:
                    decision.blocked[job.id] = (
                        f"waiting for resources{outside}; "
                        f"reserved to start by {reservation.start_at - now:.0f}s from now"
                    )
                elif holder is None and outside:
                    decision.blocked[job.id] = f"waiting for load outside ajs to drop{outside}; no reservation possible"
                elif holder is not None:
                    decision.blocked[job.id] = f"waiting for resources{outside}"
                else:
                    decision.blocked[job.id] = "waiting for a lease to be released; no reservation possible"
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
            since_finish = now - last_finish_at if last_finish_at else float("inf")
            since_quiet = now - quiet_since if quiet_since is not None else 0.0
            waited = now - (settling_since or {}).get(job.id, now)
            reason = ""
            if since_finish < cfg.settle_seconds:
                left = cfg.settle_seconds - since_finish
                reason = f"settling: {left:.0f}s left before the machine is quiet enough to time on"
            elif since_quiet < cfg.quiet_seconds:
                if waited >= cfg.quiet_max_wait_s:
                    decision.unquiet_start[job.id] = (
                        f"started after waiting {waited / 60:.0f} min without a quiet window"
                        + (f" ({noise})" if noise else "")
                    )
                elif quiet_since is None:
                    reason = f"settling: waiting for the machine to go quiet ({noise or 'load outside ajs'})"
                else:
                    reason = f"settling: quiet for {since_quiet:.0f}s of the {cfg.quiet_seconds:.0f}s needed"
            if reason:
                remaining = max(cfg.settle_seconds - since_finish, cfg.quiet_seconds - since_quiet, 1.0)
                decision.blocked[job.id] = reason
                decision.settling.append(job.id)
                # Hold the resources rather than letting something else grab them, or we
                # would never get through the settle period.
                _deduct(free, need)
                _deduct(free_exclusive, need)
                take_locks(job_locks)
                per_project[job.project] = per_project.get(job.project, 0) + 1
                holds.append((need, now + remaining + job.max_runtime_s))
                continue

        decision.start.append(job.id)
        _deduct(free, need)
        _deduct(free_exclusive, need)
        take_locks(job_locks)
        per_project[job.project] = per_project.get(job.project, 0) + 1
        holds.append((need, now + job.max_runtime_s))

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
    if external_usage.get("gpu_mem_mb"):
        parts.append(f"{external_usage['gpu_mem_mb']} MB VRAM")
    return f" ({', '.join(parts)} in use outside ajs)" if parts else ""


def _reserve(
    job: Job,
    need: dict[str, int],
    cap: Capacity,
    now: float,
    *,
    holds: list[tuple[dict[str, int], float]],
    static: dict[str, int],
) -> Reservation:
    """Earliest time ``job`` can start, assuming every hold lasts its full term.

    ``holds`` are (resources, release time) pairs: running jobs at their declared
    max_runtime, plus anything else the pass has already committed. Walking them in
    release order and accumulating what each gives back finds the first instant the job
    fits. Pessimistic by construction: jobs usually finish early, so the real start is
    typically sooner than promised.

    ``static`` is usage with no declared end -- foreign load and leases -- and is held
    constant throughout. If the job still does not fit once every hold has released,
    that is the binding constraint and the reservation is marked ``external``: it
    carries no honest start time and must not be used to veto backfill.
    """
    free = cap.as_dict()
    for key, amount in static.items():
        if key in free:
            free[key] -= amount
    drained = dict(free)
    for held, _ in holds:
        _deduct(free, held)

    if _fits(need, free):
        return Reservation(job_id=job.id, start_at=now, needs=need)

    for held, release_at in sorted(holds, key=lambda h: h[1]):
        _deduct(free, {k: -v for k, v in held.items()})
        if _fits(need, free):
            return Reservation(job_id=job.id, start_at=release_at, needs=need)

    # Nothing releasing frees enough. Either the machine is simply full of ajs work, or
    # static usage alone already exceeds what the job needs -- distinguishable by asking
    # whether it would fit on a fully drained scheduler.
    latest = max((release_at for _, release_at in holds), default=now)
    return Reservation(job_id=job.id, start_at=latest, needs=need, external=not _fits(need, drained))
