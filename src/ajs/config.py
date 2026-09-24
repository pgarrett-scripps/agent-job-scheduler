"""Daemon configuration and filesystem layout.

Capacities default to what the host actually has, minus a reserve so the desktop stays
responsive while jobs run.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from . import sysinfo

APP_NAME = "ajs"


def state_dir() -> Path:
    """Durable state: the job database and captured logs."""
    base = os.environ.get("AJS_STATE_DIR")
    if base:
        return Path(base)
    xdg = os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))
    return Path(xdg) / APP_NAME


def runtime_dir() -> Path:
    """Ephemeral state: the control socket. Cleared on reboot."""
    base = os.environ.get("AJS_RUNTIME_DIR")
    if base:
        return Path(base)
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / APP_NAME
    return state_dir() / "run"


# AF_UNIX paths are capped at ~108 bytes by the kernel. A deep XDG_RUNTIME_DIR (or a
# sandboxed one) silently blows past that, so leave headroom and fall back below.
_SOCKET_PATH_LIMIT = 100


def socket_path() -> Path:
    """Path to the control socket.

    Falls back to a short, deterministic path under /tmp when the natural location is too
    long for AF_UNIX. The fallback is derived purely from the intended directory, so the
    daemon and every client independently compute the same answer without coordinating.
    """
    natural = runtime_dir() / "ajsd.sock"
    if len(str(natural)) <= _SOCKET_PATH_LIMIT:
        return natural

    digest = hashlib.sha256(str(runtime_dir()).encode()).hexdigest()[:12]
    short = Path(tempfile.gettempdir()) / f"ajs-{os.getuid()}-{digest}"
    short.mkdir(parents=True, exist_ok=True)
    return short / "ajsd.sock"


def instance_tag() -> str:
    """Short identifier for this daemon instance, derived from its state directory.

    Job scopes are named with it, so two daemons on one machine (say, the real one and
    a throwaway under a scratch AJS_STATE_DIR) cannot mistake each other's jobs for
    orphans and stop them on startup.
    """
    return hashlib.sha256(str(state_dir()).encode()).hexdigest()[:6]


def db_path() -> Path:
    return state_dir() / "jobs.db"


def log_dir() -> Path:
    return state_dir() / "logs"


def config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(os.environ.get("AJS_CONFIG", Path(xdg) / APP_NAME / "config.json"))


@dataclass(slots=True)
class Config:
    """Tunables. Written to config.json on first run so they are easy to inspect and edit."""

    cpu: int = 0
    """Schedulable CPU slots. 0 means autodetect."""

    mem_mb: int = 0
    """Schedulable memory. 0 means autodetect (total minus mem_reserve_mb)."""

    gpu: int = 0
    """Schedulable GPUs. 0 means autodetect."""

    gpu_mem_mb: int = 0
    """Schedulable VRAM. 0 means autodetect (total minus gpu_mem_reserve_mb)."""

    gpu_mem_reserve_mb: int = 512
    """Held back from scheduling for the display. On a laptop GPU the compositor and
    browser share the card with compute, and a job that takes literally all of it will
    either fail to allocate or stall the desktop."""

    mem_reserve_mb: int = 6144
    """Held back from scheduling so the desktop does not swap while jobs run."""

    mem_guard_mb: int = 2048
    """Real free memory (MemAvailable) a job must leave untouched when it starts, after
    setting aside what running jobs may still grow into up to their declared limits.

    Declared reservations alone miss memory that is really gone: a process outside ajs
    that grew since the last smoothed sample, or an exclusive job, which ignores foreign
    load. This check uses the kernel's own number, so a start cannot push the machine
    into the OOM killer however the bookkeeping looks."""

    mem_underuse_ratio: float = 0.25
    """Warn a job's owner when its peak memory stays under this share of what it
    reserved. A reservation it never uses keeps other jobs queued on an idle machine."""

    mem_underuse_min_mb: int = 4096
    """Only reservations at least this large are worth a warning."""

    mem_underuse_after_s: int = 300
    """How long a job runs before its peak is judged, so a job still loading its input
    is not flagged. A job can move its own check with ``--meta mem_check=20m`` (or
    ``off``); jobs that end sooner are judged when they finish."""

    disk_floor_mb: int = 5120
    """Hard floor on free disk. Jobs are refused admission below this. Protects the root
    filesystem from a job that writes large output.

    Kept modest on purpose: a floor larger than the disk's usual free space refuses every
    job and makes the scheduler look broken. Raise it if you routinely run jobs that
    write tens of GB, and declare `--disk` on those jobs so they are checked individually."""

    disk_watch_path: str = "/"

    settle_seconds: float = 10.0
    """Quiet period between the last job exiting and an exclusive job starting, so page
    cache writeback and dying processes stop perturbing the measurement."""

    quiet_seconds: float = 60.0
    """How long the machine must stay quiet before an exclusive job starts. Quiet means
    foreign CPU at most ``quiet_cpu_cores`` above the desktop allowance and iowait at
    most ``quiet_iowait_pct``. The job's max_runtime does not run while it waits, which is
    why this lives here and not in each benchmark script."""

    quiet_cpu_cores: float = 1.0
    """Foreign cores above ``external_cpu_allowance`` that count as noise, both for the
    quiet gate and for flagging interference while a timing run is going."""

    quiet_iowait_pct: float = 5.0
    """Share of CPU time spent waiting on disk above which the machine is not quiet.
    Only checked before the start: once the job runs, its own I/O shows up here too."""

    quiet_max_wait_s: float = 600.0
    """Give up waiting for quiet after this long and start anyway, noting it on the job.
    Without a cap, a permanently busy desktop would wedge the queue behind one job.
    The machine sits idle for as long as this, so it is kept short: a timing run that
    never got its quiet window is flagged, and its owner can rerun it."""

    contention_threshold: float = 4.0
    """Foreign CPU cores (work not attributable to the job's own cgroup) above which an
    exclusive run is stamped `contended`.

    Set for a laptop that always has a desktop session on it. A tighter value flags
    every run for having a browser open, and a flag that fires every time is one you
    stop reading. What this should catch is another *job* interfering, not Chrome."""

    external_cpu_allowance: float = 2.0
    """Ambient desktop load, in cores, that is not charged against capacity.

    This is a laptop with a browser, a compositor and chat apps permanently running.
    Charging that baseline to the queue makes the scheduler feel broken for no benefit:
    the OS timeshares a couple of busy desktop cores against batch work perfectly well.
    Only foreign load *above* this is real contention."""

    track_external_load: bool = True
    """Count CPU and memory used by processes ajs did not start against free capacity.

    Disable only if ajs is the sole workload on the machine; otherwise the scheduler
    treats cores that a hand-launched process is already using as available."""

    external_load_half_life_s: float = 15.0
    """Smoothing for the foreign-CPU estimate. Short enough to notice a big process
    starting, long enough that a one-second spike does not evict a queued job."""

    tick_seconds: float = 1.0
    """Scheduler loop period."""

    default_max_runtime_s: int = 3600
    """Applied when a submission omits max_runtime. Required for backfill to work."""

    max_jobs_per_project: int = 4
    """Concurrency cap per project, so one agent cannot monopolise the queue."""

    use_systemd: bool = True
    """Run jobs in transient cgroup scopes when systemd-run is available."""

    enforce_limits: bool = True
    """Apply CPUQuota/MemoryMax to job cgroups, rather than only accounting."""

    lease_timeout_s: int = 120
    """A lease whose holder stops heartbeating for this long is reclaimed."""

    extra_semaphores: dict[str, int] = field(default_factory=dict)
    """Named counted resources, e.g. {"api:anthropic": 3}. Jobs request them via locks."""

    def resolved(self) -> Config:
        """Fill in autodetected capacities."""
        cpu = self.cpu or sysinfo.cpu_count()
        mem = self.mem_mb or max(1024, sysinfo.total_mem_mb() - self.mem_reserve_mb)
        gpu = self.gpu if self.gpu else sysinfo.gpu_count()
        total_vram = sysinfo.gpu_total_mem_mb()
        gpu_mem = self.gpu_mem_mb or max(0, total_vram - self.gpu_mem_reserve_mb)
        return replace(self, cpu=cpu, mem_mb=mem, gpu=gpu, gpu_mem_mb=gpu_mem)

    @classmethod
    def load(cls) -> Config:
        path = config_path()
        if not path.exists():
            return cls().resolved()
        data = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known}).resolved()

    def save(self) -> None:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def ensure_dirs() -> None:
    for d in (state_dir(), runtime_dir(), log_dir()):
        d.mkdir(parents=True, exist_ok=True)
