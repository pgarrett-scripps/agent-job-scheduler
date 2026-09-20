"""End-to-end: the engine really launches processes and reaps them."""

import asyncio

import pytest

from ajs.config import Config
from ajs.db import Store
from ajs.engine import Engine
from ajs.executor import Executor
from ajs.models import JobState


@pytest.fixture
def engine(tmp_path):
    cfg = Config(
        cpu=4,
        mem_mb=4096,
        gpu=0,
        disk_floor_mb=0,
        settle_seconds=0.0,
        tick_seconds=0.05,
        default_max_runtime_s=30,
        max_jobs_per_project=4,
        use_systemd=False,  # plain process groups: faster and hermetic under pytest
    )
    store = Store(tmp_path / "jobs.db")
    ex = Executor(cfg, tmp_path / "logs", use_systemd=False)
    eng = Engine(cfg, store=store, executor=ex)
    yield eng
    store.close()


async def drive(engine: Engine, job_id: int, timeout: float = 15.0) -> dict:
    """Tick the scheduler until the job reaches a terminal state."""
    async with asyncio.timeout(timeout):
        while True:
            await engine.tick()
            job = engine.store.get_job(job_id)
            if job is not None and job.state.is_terminal:
                return job.to_dict()
            await asyncio.sleep(0.05)


class TestExecution:
    async def test_successful_job(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/echo", "hello"], cwd="/tmp")
        result = await drive(engine, job.id)
        assert result["state"] == "done"
        assert result["exit_code"] == 0
        assert "hello" in engine.log_tail(job.id)

    async def test_failing_job_records_exit_code(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/sh", "-c", "exit 3"], cwd="/tmp")
        result = await drive(engine, job.id)
        assert result["state"] == "failed"
        assert result["exit_code"] == 3

    async def test_stdout_and_stderr_are_both_captured(self, engine):
        job = engine.submit(
            project="p", session_id="s", cmd=["/bin/sh", "-c", "echo out; echo err >&2"], cwd="/tmp"
        )
        await drive(engine, job.id)
        logs = engine.log_tail(job.id)
        assert "out" in logs and "err" in logs

    async def test_cwd_is_honoured(self, engine, tmp_path):
        (tmp_path / "marker.txt").write_text("x")
        job = engine.submit(project="p", session_id="s", cmd=["/bin/ls"], cwd=str(tmp_path))
        await drive(engine, job.id)
        assert "marker.txt" in engine.log_tail(job.id)

    async def test_granted_slots_are_exported_to_the_job(self, engine):
        """So a job can size its own thread pool to what it was granted, not to nproc."""
        job = engine.submit(
            project="p", session_id="s", cmd=["/bin/sh", "-c", "echo $AJS_CPU"], cwd="/tmp", cpu=3
        )
        await drive(engine, job.id)
        assert engine.log_tail(job.id).strip() == "3"

    async def test_unlaunchable_command_fails_cleanly(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/nonexistent/binary"], cwd="/tmp")
        result = await drive(engine, job.id)
        assert result["state"] == "failed"
        assert "could not launch" in (result["cancel_reason"] or "")


class TestCapacity:
    async def test_jobs_beyond_capacity_wait_their_turn(self, engine):
        ids = [
            engine.submit(
                project=f"p{i}", session_id="s", cmd=["/bin/sleep", "0.4"], cwd="/tmp", cpu=3
            ).id
            for i in range(3)
        ]
        await engine.tick()
        running = engine.store.jobs_in_state(JobState.RUNNING)
        assert len(running) == 1  # cpu=3 each, capacity 4

        for job_id in ids:
            await drive(engine, job_id)
        assert all(engine.store.get_job(i).state is JobState.DONE for i in ids)

    async def test_resources_are_released_on_completion(self, engine):
        a = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", cpu=4)
        await drive(engine, a.id)
        assert engine.status()["used"]["cpu"] == 0

        b = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", cpu=4)
        await drive(engine, b.id)
        assert engine.store.get_job(b.id).state is JobState.DONE


class TestCancellation:
    async def test_cancel_a_running_job(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/sleep", "60"], cwd="/tmp")
        await engine.tick()
        assert engine.store.get_job(job.id).state is JobState.RUNNING

        assert await engine.cancel(job.id)
        await asyncio.sleep(0.3)
        assert engine.store.get_job(job.id).state is JobState.CANCELLED

    async def test_cancel_a_queued_job(self, engine):
        blocker = engine.submit(project="a", session_id="s", cmd=["/bin/sleep", "60"], cwd="/tmp", cpu=4)
        waiting = engine.submit(project="b", session_id="s", cmd=["/bin/sleep", "60"], cwd="/tmp", cpu=4)
        await engine.tick()
        assert engine.store.get_job(waiting.id).state is JobState.QUEUED

        assert await engine.cancel(waiting.id)
        assert engine.store.get_job(waiting.id).state is JobState.CANCELLED
        await engine.cancel(blocker.id)

    async def test_cancelling_a_finished_job_is_a_no_op(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        await drive(engine, job.id)
        assert not await engine.cancel(job.id)

    async def test_child_processes_die_with_the_job(self, engine):
        """A killed job must not leave orphans still holding the resources it was granted."""
        job = engine.submit(
            project="p",
            session_id="s",
            cmd=["/bin/sh", "-c", "sleep 60 & echo $!; wait"],
            cwd="/tmp",
        )
        await engine.tick()
        await asyncio.sleep(0.3)
        child_pid = engine.log_tail(job.id).strip()

        await engine.cancel(job.id)
        await asyncio.sleep(0.5)

        if child_pid.isdigit():
            import os

            with pytest.raises(ProcessLookupError):
                os.kill(int(child_pid), 0)


class TestTimeout:
    async def test_job_exceeding_max_runtime_is_killed(self, engine):
        job = engine.submit(
            project="p", session_id="s", cmd=["/bin/sleep", "30"], cwd="/tmp", max_runtime_s=1
        )
        result = await drive(engine, job.id, timeout=20)
        assert result["state"] == "timeout"
        assert "max_runtime" in (result["cancel_reason"] or "")


class TestWait:
    async def test_wait_returns_when_the_job_finishes(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/sleep", "0.2"], cwd="/tmp")

        async def ticker():
            for _ in range(100):
                await engine.tick()
                await asyncio.sleep(0.05)

        task = asyncio.create_task(ticker())
        result = await engine.wait(job.id, timeout=10)
        task.cancel()
        assert result.state is JobState.DONE

    async def test_wait_times_out_without_finishing(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/sleep", "30"], cwd="/tmp")
        await engine.tick()
        result = await engine.wait(job.id, timeout=0.2)
        assert result.state is JobState.RUNNING
        await engine.cancel(job.id)

    async def test_wait_on_an_already_finished_job_returns_at_once(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        await drive(engine, job.id)
        result = await engine.wait(job.id, timeout=0.1)
        assert result.state is JobState.DONE


class TestRecovery:
    async def test_orphaned_running_jobs_are_reconciled(self, engine):
        """A daemon crash must not leave phantom capacity consumed forever."""
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        engine.store.update_job(job.id, state=str(JobState.RUNNING), started_at=0.0)

        await engine.recover()

        recovered = engine.store.get_job(job.id)
        assert recovered.state is JobState.FAILED
        assert "daemon restarted" in (recovered.cancel_reason or "")
        assert engine.status()["used"]["cpu"] == 0


class TestPauseAndDrain:
    async def test_paused_scheduler_starts_nothing(self, engine):
        engine.paused = True
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        await engine.tick()
        assert engine.store.get_job(job.id).state is JobState.QUEUED
        assert "paused" in engine.blocked_reason(job.id)

        engine.paused = False
        await drive(engine, job.id)
        assert engine.store.get_job(job.id).state is JobState.DONE


class TestLeases:
    def test_lease_blocks_capacity_then_releases(self, engine):
        from ajs.models import ResourceRequest

        lease = engine.acquire_lease("p", "s", ResourceRequest(cpu=4, mem_mb=100), "interactive work")
        assert lease is not None
        assert engine.status()["free"]["cpu"] == 0

        assert engine.store.release_lease(lease)
        assert engine.status()["free"]["cpu"] == 4

    def test_lease_refused_when_resources_are_unavailable(self, engine):
        from ajs.models import ResourceRequest

        assert engine.acquire_lease("p", "s", ResourceRequest(cpu=99, mem_mb=1), "too big") is None

    def test_stale_lease_is_reclaimed(self, engine):
        import time

        from ajs.models import ResourceRequest

        lease = engine.acquire_lease("p", "s", ResourceRequest(cpu=4, mem_mb=1), "will die")
        assert engine.status()["free"]["cpu"] == 0

        expired = engine.store.expire_leases(time.time() + 1)
        assert lease in expired
        assert engine.status()["free"]["cpu"] == 4
