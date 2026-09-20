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
