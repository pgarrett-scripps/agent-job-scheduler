"""Docker containers started by jobs.

`docker run` only asks dockerd to start the container, which then runs as root in
Docker's own cgroup, outside the job's scope. Left alone, ajs would see the job idle,
charge the container's work to "load outside ajs", and leave the container running
after the job is cancelled.

So each job gets a `docker` shim first on its PATH that labels every container it
creates with the job id. The engine finds labelled containers, counts their cgroups as
the job's own, and kills them when the job ends.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

LABEL = "ajs.job"

_CGROUP_ROOT = Path("/sys/fs/cgroup")

_SHIM = """#!/bin/sh
# Written by ajsd. Labels containers with the ajs job that started them, so their CPU
# and memory count as the job's and they are stopped when it ends.
real={real}
if [ -n "$AJS_JOB_ID" ]; then
  case "$1" in
    run|create)
      sub=$1; shift
      exec "$real" "$sub" --label "{label}=$AJS_JOB_ID" "$@" ;;
    container)
      case "$2" in
        run|create)
          sub=$2; shift 2
          exec "$real" container "$sub" --label "{label}=$AJS_JOB_ID" "$@" ;;
      esac ;;
  esac
fi
exec "$real" "$@"
"""


def install_shim(bin_dir: Path, real: str | None = None) -> Path | None:
    """Write the docker shim into ``bin_dir``; None if docker is not installed."""
    real = real or shutil.which("docker", path=_path_without(bin_dir))
    if real is None:
        return None
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "docker"
    shim.write_text(_SHIM.format(real=real, label=LABEL))
    shim.chmod(0o755)
    return bin_dir


def _path_without(bin_dir: Path) -> str:
    return os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep) if Path(p) != bin_dir)


def container_cgroup(container_id: str) -> Path | None:
    """The cgroup a container runs in, under either of Docker's cgroup drivers."""
    for path in (
        _CGROUP_ROOT / "system.slice" / f"docker-{container_id}.scope",
        _CGROUP_ROOT / "docker" / container_id,
    ):
        if path.is_dir():
            return path
    return None


async def _docker(*args: str, timeout: float = 10.0) -> str | None:
    docker = shutil.which("docker")
    if docker is None:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            docker, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
    except OSError:
        return None
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return None
    return out.decode() if proc.returncode == 0 else None


def parse_ps(out: str) -> list[tuple[str, int]]:
    """``docker ps`` lines of "<full id> <job id>" -> [(container id, job id)]."""
    found = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            found.append((parts[0], int(parts[1])))
    return found


async def labelled() -> list[tuple[str, int]]:
    """Running containers that ajs jobs started, as (container id, job id)."""
    out = await _docker(
        "ps", "--no-trunc", "--filter", f"label={LABEL}", "--format", f'{{{{.ID}}}} {{{{.Label "{LABEL}"}}}}'
    )
    return parse_ps(out) if out else []


async def kill_for_job(job_id: int) -> int:
    """Kill the containers a job started, which outlive it otherwise. Returns how many."""
    out = await _docker("ps", "-q", "--no-trunc", "--filter", f"label={LABEL}={job_id}")
    ids = out.split() if out else []
    if ids:
        await _docker("kill", *ids)
        log.info("killed %d container(s) left by job %s", len(ids), job_id)
    return len(ids)
