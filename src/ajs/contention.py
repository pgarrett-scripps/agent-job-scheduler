"""Deciding whether a timing run can be trusted.

The naive check -- sample load average before and after -- does not work. At the end of
a run the load is dominated by *the job itself*, so every busy benchmark would look
contaminated.

What actually matters is **foreign** CPU: work done on this machine that was neither the
job nor anything the scheduler knows about. We get it by differencing two counters over
the run:

    foreign_cpu_seconds = (system-wide busy CPU time) - (the job cgroup's own CPU time)

Divided by wall-clock, that is "how many cores something else was using while we timed",
which is exactly the quantity that invalidates a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import sysinfo

CGROUP_ROOT = Path("/sys/fs/cgroup")


def system_busy_seconds() -> float | None:
    """Total non-idle CPU seconds across all cores since boot."""
    try:
        with open("/proc/stat") as fh:
            line = fh.readline()
    except OSError:  # pragma: no cover - non-Linux
        return None
    if not line.startswith("cpu "):  # pragma: no cover
        return None
    fields = [float(x) for x in line.split()[1:]]
    if len(fields) < 5:  # pragma: no cover
        return None
    hz = 100.0  # USER_HZ; fixed at 100 on Linux
    user, nice, system, idle, iowait = fields[:5]
    irq = sum(fields[5:8]) if len(fields) >= 8 else 0.0
    # iowait is not CPU work, so it is excluded from "busy" deliberately: a job blocked
    # on disk is not competing for cores.
    return (user + nice + system + irq) / hz


def cgroup_path_for_pid(pid: int) -> Path | None:
    """Resolve a PID's cgroup v2 directory.

    Done by PID rather than by constructing the unit path by hand, because the systemd
    slice layout varies between distributions and session types.
    """
    try:
        content = Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return None
    for line in content.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            rel = parts[2].lstrip("/")
            path = CGROUP_ROOT / rel
            return path if path.is_dir() else None
    return None


def cgroup_cpu_seconds(path: Path) -> float | None:
    """CPU seconds consumed by everything in ``path``, including exited children."""
    try:
        content = (path / "cpu.stat").read_text()
    except OSError:
        return None
    for line in content.splitlines():
        if line.startswith("usage_usec"):
            return int(line.split()[1]) / 1_000_000
    return None


def cgroup_mem_bytes(path: Path) -> int | None:
    """Current anonymous+page-cache memory charged to ``path``."""
    try:
        return int((path / "memory.current").read_text().strip())
    except (OSError, ValueError):
        return None


@dataclass(slots=True)
class ContentionReport:
    """What else was happening while the job ran."""

    foreign_cpu_seconds: float | None
    elapsed_seconds: float
    foreign_cores: float | None
    contended: bool
    note: str

    @property
    def trustworthy(self) -> bool:
        return not self.contended


class ContentionMonitor:
    """Samples the counters that bracket a job's run."""

    #: Below this, the verdict is noise: /proc/stat counts in 10 ms ticks, so a
    #: sub-second run divides a handful of jittery ticks by a tiny elapsed time and
    #: produces an alarming number that means nothing.
    MIN_MEANINGFUL_SECONDS = 1.0

    def __init__(self, threshold_cores: float, *, assess: bool = True) -> None:
        self.threshold = threshold_cores
        self.assess = assess
        """Whether to deliver a trust verdict. Only timing runs get one -- for an
        ordinary job, competing load is expected and saying so would be noise."""
        self._busy_start: float | None = None
        self._cgroup: Path | None = None
        self._cgroup_start: float | None = None
        self._t_start: float = 0.0

    def start(self, pid: int | None, now: float) -> None:
        self._t_start = now
        self._busy_start = system_busy_seconds()
        if pid is not None:
            self._cgroup = cgroup_path_for_pid(pid)
            if self._cgroup is not None:
                self._cgroup_start = cgroup_cpu_seconds(self._cgroup)

    @property
    def cgroup(self) -> Path | None:
        """The job's cgroup, used to tell its GPU processes apart from everyone else's."""
        return self._cgroup

    def current_cpu_seconds(self) -> float | None:
        """CPU seconds this job has used so far, or None if unaccounted."""
        return cgroup_cpu_seconds(self._cgroup) if self._cgroup is not None else None

    def current_mem_bytes(self) -> int | None:
        """Memory this job is using right now, or None if unaccounted."""
        return cgroup_mem_bytes(self._cgroup) if self._cgroup is not None else None

    def finish(self, now: float) -> ContentionReport:
        elapsed = max(now - self._t_start, 1e-6)
        busy_end = system_busy_seconds()

        if self._busy_start is None or busy_end is None:
            return ContentionReport(None, elapsed, None, False, "cpu accounting unavailable")

        busy_delta = max(0.0, busy_end - self._busy_start)

        own = 0.0
        measured_own = False
        if self._cgroup is not None and self._cgroup_start is not None:
            end = cgroup_cpu_seconds(self._cgroup)
            if end is not None:
                own = max(0.0, end - self._cgroup_start)
                measured_own = True

        if not measured_own:
            # Without the job's own usage we cannot separate it from foreign load, and
            # guessing would be worse than admitting we do not know.
            return ContentionReport(None, elapsed, None, False, "job cpu accounting unavailable; contention unknown")

        foreign = max(0.0, busy_delta - own)
        cores = foreign / elapsed

        if elapsed < self.MIN_MEANINGFUL_SECONDS:
            return ContentionReport(
                foreign,
                elapsed,
                cores,
                False,
                f"run too short ({elapsed:.2f}s) to assess contention reliably",
            )

        contended = self.assess and cores > self.threshold
        note = (
            f"foreign load {cores:.2f} cores over {elapsed:.1f}s "
            f"({foreign:.1f} CPU-s not attributable to this job; "
            f"threshold {self.threshold:.2f})"
        )
        return ContentionReport(foreign, elapsed, cores, contended, note)


class ExternalLoad:
    """Tracks resources consumed by processes the scheduler did not start.

    The scheduler's capacity arithmetic silently assumes ajs is the only thing on the
    machine. It is not: a hand-run search, a browser, another agent's stray subprocess
    all take cores that ajs still counts as free, so it happily admits a job onto a box
    that is already saturated -- and an exclusive timing run gets contaminated by
    exactly the load it was supposed to exclude.

    This closes that gap by measuring what ajs cannot account for:

        external_cpu = (system-wide busy delta) - (delta across ajs job cgroups)
        external_mem = (total - MemAvailable) - (sum of ajs job cgroup memory)

    The result is reported as *usage*, never as a smaller machine. That distinction
    matters: shrinking capacity would make a 20-core exclusive job "impossible" the
    moment a browser opened, whereas treating foreign work as usage merely makes it
    wait for the machine to quieten.
    """

    def __init__(self, *, half_life_s: float = 15.0) -> None:
        self.half_life = half_life_s
        self.gpu_mem_mb: int = 0
        """VRAM held by compute processes outside ajs. Not smoothed and not inferred:
        nvidia-smi reports it directly per PID, so there is nothing to estimate."""
        self.gpu_known = False
        """False when nvidia-smi could not be queried. `unknown` must not be read as
        `idle` -- that is the mistake this whole class exists to prevent."""
        """CPU is smoothed because a raw one-second sample is spiky enough that a single
        compile would evict a queued job. Memory is not smoothed -- it is a level, not a
        rate, and reacting late to it risks an OOM."""
        self.cpu_cores: float = 0.0
        self.mem_mb: int = 0
        self.ready = False
        """False until two samples exist. Until then callers should fall back to a
        conservative prior rather than trusting a 0."""
        self._busy: float | None = None
        self._own_cpu: float = 0.0
        self._t: float | None = None

    def sample(
        self,
        now: float,
        own_cpu_seconds: float,
        own_mem_mb: int,
        own_cgroups: set[Path] | None = None,
    ) -> None:
        """Fold one observation in. ``own_*`` are the totals across all ajs jobs.

        ``own_cgroups`` is how GPU processes get attributed: nvidia-smi reports PIDs, and
        a PID belongs to ajs exactly when its cgroup is one of the job scopes.
        """
        self._sample_gpu(own_cgroups or set())
        total = sysinfo.total_mem_mb()
        available = sysinfo.available_mem_mb()
        if available is not None:
            self.mem_mb = max(0, (total - available) - own_mem_mb)

        busy = system_busy_seconds()
        if busy is None:  # pragma: no cover - non-Linux
            return

        if self._busy is None or self._t is None:
            self._busy, self._own_cpu, self._t = busy, own_cpu_seconds, now
            return

        dt = now - self._t
        if dt < 1e-3:
            return

        busy_delta = max(0.0, busy - self._busy)
        # Counters only rise while a job lives; when one exits its cgroup total vanishes
        # from the sum, so a negative delta means "a job ended", not "foreign work".
        own_delta = max(0.0, own_cpu_seconds - self._own_cpu) if own_cpu_seconds >= self._own_cpu else 0.0
        cores = max(0.0, (busy_delta - own_delta) / dt)

        alpha = 1.0 - 0.5 ** (dt / self.half_life) if self.half_life > 0 else 1.0
        self.cpu_cores = cores if not self.ready else self.cpu_cores + alpha * (cores - self.cpu_cores)

        self._busy, self._own_cpu, self._t = busy, own_cpu_seconds, now
        self.ready = True

    def _sample_gpu(self, own_cgroups: set[Path]) -> None:
        apps = sysinfo.gpu_compute_apps()
        if apps is None:
            self.gpu_known = False
            return
        self.gpu_known = True
        foreign = 0
        for pid, mem_mb in apps.items():
            cgroup = cgroup_path_for_pid(pid)
            if cgroup is not None and cgroup in own_cgroups:
                continue
            # A PID whose cgroup we cannot read is not ours to begin with (ajs job
            # cgroups are always readable by the daemon), so count it as foreign.
            foreign += mem_mb
        self.gpu_mem_mb = foreign

    def usage(self, cap_cpu: int, *, cpu_allowance: float = 0.0, mem_allowance_mb: int = 0) -> dict[str, int]:
        """Foreign usage as whole units, for subtraction from free capacity.

        The allowances exist because this runs on a laptop, where a browser and a
        desktop session are permanent facts rather than transient interference. Capacity
        already holds memory back for the desktop via ``mem_reserve_mb``; charging the
        desktop's *measured* usage on top of that bills it twice and shrinks the queue
        for nothing. Only load above the allowance is treated as real contention.

        Rounded *down* so that rounding never invents contention out of a fractional
        core, and clamped to ``cap_cpu - 1`` so foreign load can throttle the queue but
        can never wedge it completely.
        """
        if not self.ready:
            # No differenced sample yet (daemon just started). Load average is a decent
            # stand-in here precisely because ajs has not run anything to pollute it.
            cores = sysinfo.load_average()[0]
        else:
            cores = self.cpu_cores
        cores = max(0.0, cores - cpu_allowance)
        mem = max(0, self.mem_mb - mem_allowance_mb)
        return {
            "cpu": min(max(0, int(cores)), max(0, cap_cpu - 1)),
            "mem_mb": mem,
            "gpu": 0,
            # Deliberately charged as VRAM rather than as a whole GPU: a 300 MB browser
            # compositor should shrink what a job may allocate, not make the card look
            # fully taken.
            "gpu_mem_mb": self.gpu_mem_mb,
        }
