"""Thin wrappers over the bits of the host we need to observe.

Isolated here so tests can monkeypatch load and disk readings without touching /proc.
"""

from __future__ import annotations

import os
import shutil
import subprocess


def cpu_count() -> int:
    """Logical CPUs available to this process (respects taskset/cgroup affinity)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def total_mem_mb() -> int:
    """Total physical memory in MB."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:  # pragma: no cover - non-Linux
        pass
    return 4096


def available_mem_mb() -> int | None:
    """Memory the kernel thinks can be handed out without swapping.

    MemAvailable rather than MemFree: page cache is reclaimable, and counting it as
    used would make a machine that has merely read a large file look full.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:  # pragma: no cover - non-Linux
        pass
    return None


def gpu_count() -> int:
    """Number of NVIDIA GPUs, or 0 if nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return 0
    if out.returncode != 0:
        return 0
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def load_average() -> tuple[float, float, float]:
    """1/5/15-minute load average."""
    try:
        return os.getloadavg()
    except OSError:  # pragma: no cover
        return (0.0, 0.0, 0.0)


def free_disk_mb(path: str = "/") -> int:
    """Free space in MB on the filesystem containing ``path``."""
    try:
        return shutil.disk_usage(path).free // (1024 * 1024)
    except OSError:  # pragma: no cover
        return 0


def has_systemd_run() -> bool:
    """Whether transient cgroup scopes are available for resource enforcement."""
    if shutil.which("systemd-run") is None:
        return False
    try:
        out = subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet", "/bin/true"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return False
    return out.returncode == 0
