"""Launch plumbing that the end-to-end engine tests cannot see, because they run without
systemd: resolving the job's cgroup once systemd has moved it into its scope."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ajs import executor as executor_mod
from ajs.contention import ContentionMonitor
from ajs.executor import Executor, RunningProcess


class FakeProc:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.returncode: int | None = None


def _rp(unit: str | None) -> RunningProcess:
    return RunningProcess(job_id=7, proc=FakeProc(), unit=unit, log_path=Path("/dev/null"), log_file=None)  # type: ignore[arg-type]


class TestAwaitCgroup:
    async def test_waits_until_the_pid_lands_in_its_scope(self, monkeypatch):
        """Right after spawn the PID is still in the daemon's cgroup; systemd moves it a
        few tens of milliseconds later. Reading too early would attribute the job's own
        CPU to 'someone else' for its whole life."""
        daemon_cg = Path("/sys/fs/cgroup/user.slice/ajsd.service")
        scope_cg = Path("/sys/fs/cgroup/user.slice/ajs-job-7.scope")
        readings = iter([daemon_cg, daemon_cg, daemon_cg, scope_cg])
        monkeypatch.setattr(executor_mod, "cgroup_path_for_pid", lambda pid: next(readings, scope_cg))

        assert await Executor._await_cgroup(_rp("ajs-job-7.scope")) == scope_cg

    async def test_gives_up_when_the_process_exits_first(self, monkeypatch):
        daemon_cg = Path("/sys/fs/cgroup/user.slice/ajsd.service")
        monkeypatch.setattr(executor_mod, "cgroup_path_for_pid", lambda pid: daemon_cg)
        rp = _rp("ajs-job-7.scope")
        rp.proc.returncode = 0
        # Unknown is the honest answer here, never the wrong cgroup.
        assert await Executor._await_cgroup(rp) is None

    async def test_gives_up_after_the_timeout(self, monkeypatch):
        monkeypatch.setattr(executor_mod, "CGROUP_SETTLE_TIMEOUT_S", 0.05)
        monkeypatch.setattr(executor_mod, "cgroup_path_for_pid", lambda pid: Path("/sys/fs/cgroup/elsewhere"))
        async with asyncio.timeout(2.0):
            assert await Executor._await_cgroup(_rp("ajs-job-7.scope")) is None

    async def test_without_systemd_takes_the_pid_cgroup_as_is(self, monkeypatch):
        shared = Path("/sys/fs/cgroup/user.slice/session.scope")
        monkeypatch.setattr(executor_mod, "cgroup_path_for_pid", lambda pid: shared)
        assert await Executor._await_cgroup(_rp(None)) == shared


class TestMonitorUsesTheResolvedCgroup:
    def test_explicit_cgroup_wins_over_pid_lookup(self, monkeypatch):
        from ajs import contention

        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 0.0)
        monkeypatch.setattr(contention, "cgroup_path_for_pid", lambda pid: pytest.fail("must not resolve by pid"))
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 1.0)
        monitor = ContentionMonitor(threshold_cores=2.0)
        monitor.start(pid=1, now=0.0, cgroup=Path("/sys/fs/cgroup/ajs-job-1.scope"))
        assert monitor.cgroup == Path("/sys/fs/cgroup/ajs-job-1.scope")


class TestUnitNamespacing:
    def test_units_carry_the_instance_tag(self, tmp_path):
        from ajs.config import Config
        from ajs.models import Job, JobClass, JobState, ResourceRequest

        ex = Executor(Config(), tmp_path, use_systemd=True, tag="abc123")
        job = Job(
            id=5,
            project="p",
            session_id="s",
            cmd=["true"],
            cwd="/tmp",
            env={},
            resources=ResourceRequest(),
            max_runtime_s=10,
            job_class=JobClass.BATCH,
            state=JobState.QUEUED,
            submitted_at=0.0,
        )
        argv, unit = ex._wrap(job, {"cpu": 1, "mem_mb": 512})
        assert unit == "ajs-abc123-job-5.scope"
        assert "--unit=ajs-abc123-job-5" in argv

    def test_orphans_are_matched_by_this_instances_prefix_only(self, tmp_path, monkeypatch):
        """A throwaway daemon under a scratch state dir must never reap the real one's
        jobs on startup -- which is exactly what a global `ajs-job-*` glob did."""
        import subprocess

        from ajs.config import Config

        ex = Executor(Config(), tmp_path, use_systemd=True, tag="abc123")
        listing = (
            "ajs-abc123-job-3.scope loaded active running ajs job 3\najs-ffffff-job-9.scope loaded active running\n"
        )
        calls = []

        def fake_run(argv, **kw):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout=listing, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert ex.orphan_units() == ["ajs-abc123-job-3.scope"]
        assert "ajs-abc123-job-*.scope" in calls[0]

    def test_no_orphans_without_systemd(self, tmp_path):
        from ajs.config import Config

        assert Executor(Config(), tmp_path, use_systemd=False).orphan_units() == []
