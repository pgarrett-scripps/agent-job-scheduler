"""End-to-end: the engine really launches processes and reaps them."""

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

from ajs.config import Config
from ajs.db import Store
from ajs.engine import Engine
from ajs.executor import Executor, JobProcess, RunningProcess
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
        track_external_load=False,  # otherwise the test host's own load decides the outcome
        cpu_overbook=1.0,
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
        job = engine.submit(project="p", session_id="s", cmd=["/bin/sh", "-c", "echo out; echo err >&2"], cwd="/tmp")
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
        job = engine.submit(project="p", session_id="s", cmd=["/bin/sh", "-c", "echo $AJS_CPU"], cwd="/tmp", cpu=3)
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
            engine.submit(project=f"p{i}", session_id="s", cmd=["/bin/sleep", "0.4"], cwd="/tmp", cpu=3).id
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

    async def test_cancel_holds_slots_until_children_die(self, engine, tmp_path):
        # A wrapper that ignores nothing but whose child traps SIGTERM for a moment, like
        # a worker flushing on shutdown: the slot must stay taken until the child is gone.
        pidfile = tmp_path / "child.pid"
        script = f"sh -c 'trap \"sleep 0.5; exit 0\" TERM; echo $$ > {pidfile}; while :; do sleep 0.05; done' & wait"
        job = engine.submit(project="p", session_id="s", cmd=["/bin/sh", "-c", script], cwd="/tmp", cpu=4)
        async with asyncio.timeout(5):
            while not pidfile.exists() or not pidfile.read_text().strip():
                await engine.tick()
                await asyncio.sleep(0.05)
        child = int(pidfile.read_text())
        await engine.cancel(job.id)
        result = await drive(engine, job.id)
        assert result["state"] == "cancelled"
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
        assert job.id not in engine.running

    async def test_leftover_background_children_are_reaped(self, engine):
        engine.executor.drain_grace_s = 0.3
        job = engine.submit(project="p", session_id="s", cmd=["/bin/sh", "-c", "sleep 60 & exit 0"], cwd="/tmp")
        async with asyncio.timeout(5):
            while engine.store.get_job(job.id).state is JobState.QUEUED:
                await engine.tick()
                await asyncio.sleep(0.02)
            rp = engine.running[job.id]
            await rp.proc.wait()
            await asyncio.sleep(0.1)
            assert job.id in engine.running, "slots freed while the background sleep still ran"
        result = await drive(engine, job.id)
        assert result["state"] == "done"
        assert engine.executor.is_empty(rp)

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
        job = engine.submit(project="p", session_id="s", cmd=["/bin/sleep", "30"], cwd="/tmp", max_runtime_s=1)
        result = await drive(engine, job.id, timeout=20)
        assert result["state"] == "timeout"
        assert "max_runtime" in (result["cancel_reason"] or "")

    async def test_killing_a_stubborn_job_does_not_stall_the_tick(self, engine):
        # Ignores SIGTERM, so the kill has to sit out the whole grace period.
        cmd = ["/bin/sh", "-c", "trap '' TERM; while :; do sleep 0.05; done"]
        job = engine.submit(project="p", session_id="s", cmd=cmd, cwd="/tmp", max_runtime_s=1)
        async with asyncio.timeout(5):
            while engine.store.get_job(job.id).state is not JobState.RUNNING:
                await engine.tick()
                await asyncio.sleep(0.02)
        await asyncio.sleep(1.1)
        async with asyncio.timeout(1):
            await engine.tick()
        result = await drive(engine, job.id, timeout=30)
        assert result["state"] == "timeout"


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

    async def test_job_that_ended_while_the_daemon_was_down_keeps_its_exit_code(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        engine.store.update_job(job.id, state=str(JobState.RUNNING), started_at=0.0, pid=999999)
        engine.executor.exit_file(job.id).write_text("3\n")

        await engine.recover()

        recovered = engine.store.get_job(job.id)
        assert recovered.state is JobState.FAILED
        assert recovered.exit_code == 3
        assert not recovered.cancel_reason

    async def test_live_job_is_adopted_and_supervised_to_the_end(self, engine, monkeypatch, tmp_path):
        """A restart must not cost a running job: the new daemon watches it to the end."""
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        exit_file = engine.executor.exit_file(job.id)
        proc = subprocess.Popen(["/bin/sh", "-c", f"sleep 0.3; echo 0 > {exit_file}"])
        engine.store.update_job(job.id, state=str(JobState.RUNNING), started_at=0.0, pid=proc.pid, unit="u.scope")

        def adopt(j):
            return RunningProcess(
                job_id=j.id,
                proc=JobProcess(j.pid, exit_file=exit_file),
                unit=j.unit,
                log_path=tmp_path / "log",
                log_file=None,
            )

        monkeypatch.setattr(engine.executor, "adopt", adopt)
        await engine.recover()
        assert engine.store.get_job(job.id).state is JobState.RUNNING
        assert job.id in engine.running

        async with asyncio.timeout(5):
            await engine.wait(job.id, 5)
        proc.wait()
        done = engine.store.get_job(job.id)
        assert done.state is JobState.DONE
        assert done.exit_code == 0
        assert [e["action"] for e in engine.store.events(job_id=job.id)] == ["adopt"]


class TestInbox:
    async def test_session_sees_its_own_finished_jobs_and_what_others_did(self, engine):
        mine = engine.submit(project="p", session_id="claude:me", cmd=["/bin/false"], cwd="/tmp")
        other = engine.submit(project="p", session_id="claude:you", cmd=["/bin/false"], cwd="/tmp")
        await drive(engine, mine.id)
        await drive(engine, other.id)
        engine.paused = True
        queued = engine.submit(project="p", session_id="claude:me", cmd=["/bin/true"], cwd="/tmp")
        engine.hold(queued.id, actor="claude:me", reason="my own hold is not news")
        engine.release(queued.id, actor="claude:steward", reason="")

        box = engine.inbox("claude:me", since=0.0)

        assert [j["id"] for j in box["finished"]] == [mine.id]
        assert [(e["job_id"], e["action"]) for e in box["events"]] == [(queued.id, "release")]
        assert engine.inbox("claude:me", since=box["now"])["finished"] == []

    async def test_running_job_with_a_silent_log_is_flagged(self, engine):
        job = engine.submit(project="p", session_id="claude:me", cmd=["/bin/sleep", "5"], cwd="/tmp")
        async with asyncio.timeout(5):
            while engine.store.get_job(job.id).state is not JobState.RUNNING:
                await engine.tick()
                await asyncio.sleep(0.02)
        box = engine.inbox("claude:me", since=0.0, stall_s=0.0)
        assert [j["id"] for j in box["stalled"]] == [job.id]
        await engine.cancel(job.id, "done with test")


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


class TestExternalSampling:
    async def test_a_shared_cgroup_is_counted_once(self, engine, monkeypatch):
        """Without systemd every job lives in the daemon's own cgroup. Summing that per
        job would multiply ajs's share by the job count and hide real foreign load."""
        from ajs.contention import ContentionMonitor

        engine.cfg.track_external_load = True
        shared = Path("/sys/fs/cgroup/shared")
        for job_id in (1, 2, 3):
            monitor = ContentionMonitor(2.0)
            monitor._cgroup = shared
            engine.monitors[job_id] = monitor
        monkeypatch.setattr(ContentionMonitor, "current_cpu_seconds", lambda self: 100.0)
        monkeypatch.setattr(ContentionMonitor, "current_mem_bytes", lambda self: 1024 * 1024 * 100)

        seen = {}

        def fake_sample(now, own_cpu, own_mem, own_cgroups=None):
            seen.update(cpu=own_cpu, mem=own_mem, cgroups=own_cgroups)

        monkeypatch.setattr(engine.external, "sample", fake_sample)
        engine._sample_external(now=0.0)
        assert seen == {"cpu": 100.0, "mem": 100, "cgroups": {shared}}


class TestQueueManagement:
    async def test_held_job_never_starts_until_released(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", held=True)
        for _ in range(3):
            await engine.tick()
        assert engine.store.get_job(job.id).state is JobState.QUEUED
        assert engine.blocked_reason(job.id).startswith("held by s")

        engine.release(job.id, actor="me", reason="go")
        result = await drive(engine, job.id)
        assert result["state"] == "done"

    async def test_timed_hold_releases_itself(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        engine.hold(job.id, actor="me", reason="later", until=time.time() + 0.3)
        await engine.tick()
        assert engine.store.get_job(job.id).state is JobState.QUEUED
        assert " until " in engine.blocked_reason(job.id)
        await asyncio.sleep(0.35)
        result = await drive(engine, job.id)
        assert result["state"] == "done"
        [release] = [e for e in engine.store.events(job_id=job.id) if e["action"] == "release"]
        assert release["actor"] == "ajs"

    async def test_submit_with_start_time(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", hold_until=time.time() + 60)
        await engine.tick()
        stored = engine.store.get_job(job.id)
        assert stored.held and stored.hold_until is not None
        engine.release(job.id, actor="me")
        assert engine.store.get_job(job.id).hold_until is None
        assert (await drive(engine, job.id))["state"] == "done"

    async def test_timed_pause_resumes_itself(self, engine):
        engine.paused, engine.pause_until = True, time.time() - 1
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        assert (await drive(engine, job.id))["state"] == "done"
        assert engine.paused is False and engine.pause_until is None

    async def test_after_ok_waits_then_runs(self, engine):
        first = engine.submit(project="p", session_id="s", cmd=["/bin/sleep", "0.3"], cwd="/tmp")
        second = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", after_ok=[first.id])
        await engine.tick()
        assert engine.store.get_job(second.id).state is JobState.QUEUED
        assert engine.blocked_reason(second.id).startswith(f"waiting on job {first.id}")
        assert (await drive(engine, second.id))["state"] == "done"
        a, b = engine.store.get_job(first.id), engine.store.get_job(second.id)
        assert b.started_at >= a.finished_at

    async def test_failed_dependency_cancels_the_whole_chain(self, engine):
        a = engine.submit(project="p", session_id="s", cmd=["/bin/false"], cwd="/tmp")
        b = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", after_ok=[a.id])
        c = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", after_ok=[b.id])
        result = await drive(engine, c.id)
        assert result["state"] == "cancelled"
        assert engine.store.get_job(b.id).cancel_reason == f"dependency {a.id} ended failed"
        assert engine.store.get_job(b.id).started_at is None

    async def test_after_any_runs_even_if_dependency_fails(self, engine):
        a = engine.submit(project="p", session_id="s", cmd=["/bin/false"], cwd="/tmp")
        b = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", after_any=[a.id])
        assert (await drive(engine, b.id))["state"] == "done"

    async def test_dependency_must_exist_and_be_viable(self, engine):
        with pytest.raises(ValueError, match="no such job"):
            engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", after_ok=[999])
        a = engine.submit(project="p", session_id="s", cmd=["/bin/false"], cwd="/tmp")
        await drive(engine, a.id)
        with pytest.raises(ValueError, match="would never run"):
            engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", after_ok=[a.id])

    async def test_dependencies_that_will_not_wait_are_flagged(self, engine):
        done = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        await drive(engine, done.id)
        engine.paused = True
        held = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", held=True)
        live = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        warnings = engine.dependency_warnings([done.id], [held.id, live.id])
        assert len(warnings) == 2
        assert f"{done.id} already ended done" in warnings[0]
        assert f"{held.id} is held" in warnings[1]

    async def test_hold_records_actor_and_reason(self, engine):
        engine.paused = True
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        engine.hold(job.id, actor="manager", reason="long timing run, not now")
        assert engine.store.get_job(job.id).held
        assert engine.blocked_reason(job.id) == "held by manager: long timing run, not now"
        [event] = engine.store.events(job_id=job.id)
        assert (event["actor"], event["action"]) == ("manager", "hold")

    async def test_held_job_does_not_claim_the_reservation(self, engine):
        """A held giant must not veto backfill for everyone else."""
        big = engine.submit(project="a", session_id="s", cmd=["/bin/true"], cwd="/tmp", exclusive=True)
        engine.hold(big.id, actor="m", reason="later")
        await engine.tick()
        assert engine.store.get_job(big.id).state is JobState.QUEUED
        assert engine.last_decision.reservation is None or engine.last_decision.reservation.job_id != big.id

    async def test_set_priority_changes_class_and_logs_it(self, engine):
        engine.paused = True
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        engine.set_priority(job.id, "background", actor="m", reason="4h timing")
        assert str(engine.store.get_job(job.id).job_class) == "background"
        [event] = engine.store.events(job_id=job.id)
        assert event["detail"] == "batch -> background"

    async def test_only_queued_jobs_can_be_managed(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
        await drive(engine, job.id)
        with pytest.raises(ValueError, match="only queued"):
            engine.hold(job.id, actor="m", reason="x")

    async def test_note_is_stored(self, engine):
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", note="fig 3")
        assert engine.store.get_job(job.id).to_dict()["note"] == "fig 3"

    async def test_title_description_meta_are_stored(self, engine):
        job = engine.submit(
            project="p",
            session_id="s",
            cmd=["/bin/true"],
            cwd="/tmp",
            title="uno fig 3",
            description="last figure; blocks submission",
            meta={"est": "40m", "n": 3},
        )
        stored = engine.store.get_job(job.id).to_dict()
        assert stored["title"] == "uno fig 3"
        assert stored["description"] == stored["note"] == "last figure; blocks submission"
        assert stored["meta"] == {"est": "40m", "n": "3"}


def test_old_database_gains_new_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY, project TEXT, session_id TEXT, cmd TEXT, cwd TEXT, env TEXT,"
        " resources TEXT, max_runtime_s INTEGER, job_class TEXT, state TEXT, submitted_at REAL, started_at REAL,"
        " finished_at REAL, exit_code INTEGER, pid INTEGER, unit TEXT, log_path TEXT, load_before REAL,"
        " load_after REAL, contended INTEGER DEFAULT 0, contention_note TEXT, reserved_until REAL,"
        " cancel_reason TEXT)"
    )
    conn.execute(
        "INSERT INTO jobs(project, session_id, cmd, cwd, env, resources, max_runtime_s, job_class, state,"
        " submitted_at) VALUES('p','', '[\"true\"]','/tmp','{}','{\"cpu\":1}',60,'batch','queued',1.0)"
    )
    conn.commit()
    conn.close()
    store = Store(path)
    [job] = store.jobs_in_state(JobState.QUEUED)
    assert job.held is False and job.title is None and job.description is None and job.meta == {}
    store.close()


class TestMemUnderuse:
    """Owners hear, once, when a job reserves far more memory than it ever uses."""

    def _job(self, engine, peak_mb, *, age_s, meta=None):
        from ajs.contention import ContentionMonitor

        engine.cfg.mem_underuse_min_mb = 1000
        job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", mem_mb=4000, meta=meta)
        engine.store.update_job(job.id, started_at=time.time() - age_s)
        monitor = ContentionMonitor(2.0)
        monitor.start(None, time.time() - age_s)
        monitor.mem_peak_mb = peak_mb
        return engine.store.get_job(job.id), monitor

    def _warnings(self, engine, job_id):
        return [e for e in engine.store.events(job_id=job_id) if e["action"] == "mem-underuse"]

    def test_warns_once_after_the_check_age(self, engine):
        job, monitor = self._job(engine, 500, age_s=600)
        engine._check_mem_underuse(job, monitor, time.time(), final=False)
        engine._check_mem_underuse(job, monitor, time.time(), final=False)
        (event,) = self._warnings(engine, job.id)
        assert event["detail"].startswith("peak 0.5 of 4 GB reserved")
        assert "reserve about 1G" in event["reason"]
        assert engine.store.session_events("s", since=0)  # reaches the owner's inbox

    def test_waits_for_the_check_age_but_judges_short_jobs_at_the_end(self, engine):
        job, monitor = self._job(engine, 500, age_s=60)
        engine._check_mem_underuse(job, monitor, time.time(), final=False)
        assert not self._warnings(engine, job.id)
        engine._check_mem_underuse(job, monitor, time.time(), final=True)
        assert self._warnings(engine, job.id)

    def test_a_job_using_its_reservation_is_left_alone(self, engine):
        job, monitor = self._job(engine, 2500, age_s=600)
        engine._check_mem_underuse(job, monitor, time.time(), final=True)
        assert not self._warnings(engine, job.id)

    def test_a_small_unused_remainder_is_left_alone(self, engine):
        job, monitor = self._job(engine, 2000, age_s=600)
        engine.cfg.mem_underuse_min_mb = 2500  # half unused, but only 2 GB of it
        engine._check_mem_underuse(job, monitor, time.time(), final=True)
        assert not self._warnings(engine, job.id)

    def _blocking(self, engine, monkeypatch, peak_mb, waiting_mb):
        from ajs.scheduler import Decision

        monkeypatch.setattr(engine, "_mem_headroom", lambda running: None)
        job, monitor = self._job(engine, peak_mb, age_s=600)
        engine.store.update_job(job.id, state="running")
        engine.monitors[job.id] = monitor
        queued = engine.submit(project="q", session_id="other", cmd=["/bin/true"], cwd="/tmp", mem_mb=waiting_mb)
        running = [engine.store.get_job(job.id)]
        for _ in range(2):
            engine._nudge_mem_blockers(time.time(), [queued], running, Decision(blocked={queued.id: "waiting"}))
        return [e for e in engine.store.events(job_id=job.id) if e["action"] == "mem-blocking"]

    def test_owner_hears_once_when_unused_memory_blocks_a_queued_job(self, engine, monkeypatch):
        (event,) = self._blocking(engine, monkeypatch, 500, 2000)
        assert "job 2 (q, 2 GB) is waiting for memory" in event["detail"]
        assert "ajs resize 1 --mem 1G" in event["reason"]
        assert any(e["action"] == "mem-blocking" for e in engine.store.session_events("s", since=0))

    def test_no_note_when_freeing_the_unused_memory_would_not_be_enough(self, engine, monkeypatch):
        assert not self._blocking(engine, monkeypatch, 3000, 2000)

    def test_an_adopted_job_is_judged_on_the_time_since_adoption(self, engine):
        job, monitor = self._job(engine, 0, age_s=600)
        monitor.start(None, time.time())  # a restarted daemon just took it over
        engine._check_mem_underuse(job, monitor, time.time(), final=False)
        engine._check_mem_underuse(job, monitor, time.time(), final=True)
        assert not self._warnings(engine, job.id)

    def test_meta_moves_or_disables_the_check(self, engine):
        late, monitor = self._job(engine, 500, age_s=600, meta={"mem_check": "20m"})
        engine._check_mem_underuse(late, monitor, time.time(), final=False)
        assert not self._warnings(engine, late.id)
        off, monitor = self._job(engine, 500, age_s=600, meta={"mem_check": "off"})
        engine._check_mem_underuse(off, monitor, time.time(), final=True)
        assert not self._warnings(engine, off.id)


class TestRuntime:
    """Owners hear before max_runtime kills a job, and can move the ceiling themselves."""

    def _running(self, engine, *, max_s=3600, age_s=0, session="s", **kw):
        job = engine.submit(project="p", session_id=session, cmd=["/bin/true"], cwd="/tmp", max_runtime_s=max_s, **kw)
        engine.store.update_job(job.id, state="running", started_at=time.time() - age_s)
        return engine.store.get_job(job.id)

    def _events(self, engine, job_id, action):
        return [e for e in engine.store.events(job_id=job_id) if e["action"] == action]

    def test_warns_once_ten_minutes_before_the_kill(self, engine):
        early = self._running(engine, age_s=2000)
        engine._check_runtime(early, time.time())
        assert not self._events(engine, early.id, "runtime-warning")
        job = self._running(engine, age_s=3100)
        engine._check_runtime(job, time.time())
        engine._check_runtime(job, time.time())
        (event,) = self._events(engine, job.id, "runtime-warning")
        assert event["detail"] == "killed in 8 min: used 52 of 60 min"
        assert f"ajs extend {job.id}" in event["reason"] and "120 min" in event["reason"]
        assert engine.store.session_events("s", since=0)  # reaches the owner's inbox

    def test_short_jobs_are_warned_at_80_percent(self, engine):
        job = self._running(engine, max_s=1000, age_s=700)
        engine._check_runtime(job, time.time())
        assert not self._events(engine, job.id, "runtime-warning")
        engine._check_runtime(job, time.time() + 110)
        assert self._events(engine, job.id, "runtime-warning")

    def test_an_extension_earns_a_fresh_warning(self, engine):
        job = self._running(engine, age_s=3100)
        engine._check_runtime(job, time.time())
        job = engine.set_runtime(job.id, 4000, actor="s", reason="slow sample")
        engine._check_runtime(job, time.time() + 400)
        assert len(self._events(engine, job.id, "runtime-warning")) == 2

    def test_owner_extends_and_shortens_within_the_cap(self, engine):
        job = self._running(engine, max_s=3600, age_s=600)
        job = engine.set_runtime(job.id, 7200, actor="s", reason="needs more")
        assert job.max_runtime_s == 7200 and job.orig_max_runtime_s == 3600
        with pytest.raises(ValueError, match="at most 120 min"):
            engine.set_runtime(job.id, 7300, actor="s", reason="more still")
        job = engine.set_runtime(job.id, 1800, actor="s", reason="finishing early")
        assert job.max_runtime_s == 1800 and job.orig_max_runtime_s == 3600
        (latest, first) = self._events(engine, job.id, "runtime")  # newest first
        assert first["detail"] == "60 -> 120 min" and first["actor"] == "s"

    def test_cannot_shorten_below_what_has_run(self, engine):
        job = self._running(engine, max_s=3600, age_s=1800)
        with pytest.raises(ValueError, match="would kill it now"):
            engine.set_runtime(job.id, 1800, actor="s", reason="x")

    def test_only_the_owner_or_a_person_may_change_it(self, engine):
        job = self._running(engine, session="claude:owner")
        with pytest.raises(ValueError, match="only its owner"):
            engine.set_runtime(job.id, 4000, actor="claude:other", reason="x")
        assert engine.set_runtime(job.id, 4000, actor="user:patrick", reason="x").max_runtime_s == 4000

    def test_timing_runs_cannot_be_extended_but_can_be_shortened(self, engine):
        job = self._running(engine, exclusive=True)
        with pytest.raises(ValueError, match="timing run"):
            engine.set_runtime(job.id, 4000, actor="s", reason="x")
        assert engine.set_runtime(job.id, 3000, actor="s", reason="x").max_runtime_s == 3000
        engine._check_runtime(engine.store.get_job(job.id), time.time() + 2500)
        (event,) = self._events(engine, job.id, "runtime-warning")
        assert "cannot be extended" in event["reason"]

    def test_an_extension_past_a_reservation_tells_the_reserved_owner(self, engine):
        from ajs.scheduler import Decision, Reservation

        job = self._running(engine, max_s=3600, age_s=1800)
        big = engine.submit(project="q", session_id="other", cmd=["/bin/true"], cwd="/tmp", cpu=4)
        engine.last_decision = Decision(
            reservation=Reservation(job_id=big.id, start_at=job.started_at + 3600, needs={})
        )
        engine.set_runtime(job.id, 5400, actor="s", reason="B3 is slow")
        (event,) = self._events(engine, big.id, "delayed")
        assert event["detail"] == "up to 30 min" and "B3 is slow" in event["reason"]
        assert engine.store.session_events("other", since=0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 300), ("20m", 1200), ("90", 90), ("off", None), ("inf", 300), ("1e400m", 300), ("-5m", 300), ("x", 300)],
)
def test_mem_check_after_never_raises(value, expected):
    from ajs.engine import mem_check_after

    assert mem_check_after(value, 300) == expected


def test_headroom_leaves_page_cache_out_of_a_jobs_use(engine, monkeypatch):
    """MemAvailable already counts page cache as free; counting it as used too would
    hide how much a running job can still grow."""
    from ajs import sysinfo
    from ajs.contention import ContentionMonitor

    monkeypatch.setattr(sysinfo, "available_mem_mb", lambda: 20000)
    job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp", mem_mb=4000)
    monitor = ContentionMonitor(2.0)
    monitor.mem_now_mb, monitor.mem_anon_now_mb = 4000, 1000  # 3 GB of it is cache
    engine.monitors[job.id] = monitor
    engine.cfg.mem_guard_mb = 0
    assert engine._mem_headroom([job]) == 20000 - 3000


async def test_finish_time_is_stamped_after_the_process_tree_drains(engine, monkeypatch):
    """An inbox check during the drain must not move its cursor past the job's end."""
    during = {}

    async def slow_drain(rp):
        await asyncio.sleep(0.3)
        during["at"] = time.time()

    monkeypatch.setattr(engine.executor, "drain", slow_drain)
    job = engine.submit(project="p", session_id="s", cmd=["/bin/true"], cwd="/tmp")
    result = await drive(engine, job.id)
    assert result["finished_at"] >= during["at"]


class TestResize:
    """Owners can lower a job's memory reservation in place."""

    def _job(self, engine, *, mem_mb=8000, running=False, peak_mb=None, age_s=600, session="s"):
        from ajs.contention import ContentionMonitor

        job = engine.submit(project="p", session_id=session, cmd=["/bin/true"], cwd="/tmp", mem_mb=mem_mb)
        if running:
            engine.store.update_job(job.id, state="running", started_at=time.time() - age_s)
            monitor = ContentionMonitor(2.0)
            monitor.start(None, time.time() - age_s)
            monitor.mem_peak_mb = peak_mb
            engine.monitors[job.id] = monitor
        return engine.store.get_job(job.id)

    def test_lowers_a_queued_job_and_logs_it(self, engine):
        job = self._job(engine)
        job = engine.set_mem(job.id, 2048, actor="s", reason="peaks at 1.5G")
        assert job.resources.mem_mb == 2048
        (event,) = [e for e in engine.store.events(job_id=job.id) if e["action"] == "mem"]
        assert event["detail"] == "7.8 -> 2.0 GB" and event["actor"] == "s"

    def test_cannot_raise(self, engine):
        job = self._job(engine, mem_mb=2048)
        with pytest.raises(ValueError, match="can only lower"):
            engine.set_mem(job.id, 4096, actor="s", reason="x")

    def test_only_the_owner_or_a_person(self, engine):
        job = self._job(engine, session="claude:owner")
        with pytest.raises(ValueError, match="only its owner"):
            engine.set_mem(job.id, 2048, actor="claude:other", reason="x")
        assert engine.set_mem(job.id, 2048, actor="user:patrick", reason="x").resources.mem_mb == 2048

    def test_running_job_not_below_its_peak_plus_margin(self, engine):
        job = self._job(engine, running=True, peak_mb=3000)
        with pytest.raises(ValueError, match="4112 MB at the least"):
            engine.set_mem(job.id, 4000, actor="s", reason="x")
        assert engine.set_mem(job.id, 4200, actor="s", reason="x").resources.mem_mb == 4200

    def test_running_job_needs_a_minute_of_watching(self, engine):
        job = self._job(engine, running=True, peak_mb=100, age_s=10)
        with pytest.raises(ValueError, match="not been watched long enough"):
            engine.set_mem(job.id, 2048, actor="s", reason="x")

    def test_lowered_limit_reaches_the_cgroup(self, engine, monkeypatch):
        job = self._job(engine, running=True, peak_mb=1000)
        calls = []
        monkeypatch.setattr(engine.executor, "set_mem_limit", lambda unit, mb: calls.append((unit, mb)) or True)
        engine.running[job.id] = RunningProcess(
            job_id=job.id,
            proc=None,  # ty: ignore[invalid-argument-type]  set_mem never touches the process
            unit="ajs-x-job-1.scope",
            log_path=Path("/dev/null"),
            log_file=None,
        )
        engine.set_mem(job.id, 2048, actor="s", reason="x")
        assert calls == [("ajs-x-job-1.scope", 2048)]
        monkeypatch.setattr(engine.executor, "set_mem_limit", lambda unit, mb: False)
        with pytest.raises(ValueError, match="could not lower"):
            engine.set_mem(job.id, 1800, actor="s", reason="x")
        assert engine.store.get_job(job.id).resources.mem_mb == 2048
