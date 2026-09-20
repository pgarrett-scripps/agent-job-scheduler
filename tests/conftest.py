import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ajs.config import Config  # noqa: E402
from ajs.models import Job, JobClass, JobState, ResourceRequest  # noqa: E402
from ajs.scheduler import Capacity  # noqa: E402

NOW = 1_000_000.0


@pytest.fixture
def cap():
    return Capacity(cpu=20, mem_mb=56000, gpu=1)


@pytest.fixture
def cfg():
    return Config(
        cpu=20,
        mem_mb=56000,
        gpu=1,
        disk_floor_mb=20480,
        settle_seconds=10.0,
        max_jobs_per_project=4,
        contention_threshold=2.0,
    )


def make_job(
    job_id: int,
    *,
    project: str = "proj",
    cpu: int = 1,
    mem_mb: int = 512,
    gpu: int = 0,
    disk_mb: int = 0,
    exclusive: bool = False,
    locks: list[str] | None = None,
    max_runtime_s: int = 3600,
    job_class: JobClass = JobClass.BATCH,
    state: JobState = JobState.QUEUED,
    submitted_at: float = NOW,
    started_at: float | None = None,
) -> Job:
    return Job(
        id=job_id,
        project=project,
        session_id="s",
        cmd=["true"],
        cwd="/tmp",
        env={},
        resources=ResourceRequest(
            cpu=cpu, mem_mb=mem_mb, gpu=gpu, disk_mb=disk_mb, exclusive=exclusive, locks=list(locks or [])
        ),
        max_runtime_s=max_runtime_s,
        job_class=job_class,
        state=state,
        submitted_at=submitted_at,
        started_at=started_at,
    )


@pytest.fixture
def clock():
    return time.time
