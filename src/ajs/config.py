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

    mem_reserve_mb: int = 6144
    """Held back from scheduling so the desktop does not swap while jobs run."""

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

    contention_threshold: float = 2.0
    """1-minute load average above which an exclusive run is stamped `contended`."""

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
        return replace(self, cpu=cpu, mem_mb=mem, gpu=gpu)

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
