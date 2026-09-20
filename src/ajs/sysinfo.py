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


def _nvidia_smi(query: str, *, gpu: bool) -> list[str] | None:
    """Run one nvidia-smi CSV query, or None if the tool is missing or fails."""
    if shutil.which("nvidia-smi") is None:
        return None
    flag = "--query-gpu" if gpu else "--query-compute-apps"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if out.returncode != 0:
        return None
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def gpu_total_mem_mb() -> int:
    """Total VRAM across all GPUs, in MB."""
    rows = _nvidia_smi("memory.total", gpu=True)
    if not rows:
        return 0
    total = 0
    for row in rows:
        try:
            total += int(float(row))
        except ValueError:  # pragma: no cover
            continue
    return total


def gpu_compute_apps() -> dict[int, int] | None:
    """PID -> VRAM MB for every process currently holding GPU memory.

    None means we could not ask (no nvidia-smi, or it failed), which is different from
    an empty dict: callers must not read "unknown" as "nothing is on the GPU".
    """
    rows = _nvidia_smi("pid,used_gpu_memory", gpu=False)
    if rows is None:
        return None
    apps: dict[int, int] = {}
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 2:
            continue
        try:
            apps[int(parts[0])] = int(float(parts[1]))
        except ValueError:
            # "[N/A]" shows up for processes in other containers or MIG instances.
            continue
    return apps


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
