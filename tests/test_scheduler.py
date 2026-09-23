"""Scheduling behaviour. `plan` is pure, so these run without a daemon or a subprocess."""

from ajs.models import JobClass, JobState
from ajs.scheduler import effective_request, order_queue, plan

from .conftest import NOW, make_job


def run_plan(queued, running=None, *, cap, cfg, now=NOW, **kwargs):
    return plan(
        queued=queued,
        running=running or [],
        cap=cap,
        cfg=cfg,
        now=now,
        last_start=kwargs.pop("last_start", {}),
        **kwargs,
    )


class TestBasicAdmission:
    def test_job_starts_when_resources_are_free(self, cap, cfg):
        decision = run_plan([make_job(1, cpu=4)], cap=cap, cfg=cfg)
        assert decision.start == [1]

    def test_jobs_queue_once_cpu_is_exhausted(self, cap, cfg):
        jobs = [make_job(i, cpu=8, submitted_at=NOW + i) for i in range(1, 4)]
        decision = run_plan(jobs, cap=cap, cfg=cfg)
        assert decision.start == [1, 2]  # 8 + 8 fits in 20; the third does not
        assert 3 in decision.blocked

    def test_running_jobs_consume_capacity(self, cap, cfg):
        running = [make_job(1, cpu=18, state=JobState.RUNNING, started_at=NOW)]
        decision = run_plan([make_job(2, cpu=4)], running, cap=cap, cfg=cfg)
        assert decision.start == []

    def test_memory_is_counted_separately_from_cpu(self, cap, cfg):
        running = [make_job(1, cpu=1, mem_mb=55000, state=JobState.RUNNING, started_at=NOW)]
        decision = run_plan([make_job(2, cpu=1, mem_mb=4000)], running, cap=cap, cfg=cfg)
        assert decision.start == []
        assert "resources" in decision.blocked[2]

    def test_impossible_job_is_rejected_not_queued_forever(self, cap, cfg):
        decision = run_plan([make_job(1, cpu=999)], cap=cap, cfg=cfg)
        assert decision.start == []
        assert "impossible" in decision.blocked[1]
        # And it must not consume the single reservation slot, which would stall the queue.
        assert decision.reservation is None


class TestExclusive:
    def test_exclusive_expands_to_the_whole_machine(self, cap):
        job = make_job(1, cpu=1, exclusive=True)
        assert effective_request(job, cap)["cpu"] == cap.cpu

    def test_exclusive_waits_for_running_jobs(self, cap, cfg):
        running = [make_job(1, cpu=1, state=JobState.RUNNING, started_at=NOW)]
        decision = run_plan([make_job(2, exclusive=True)], running, cap=cap, cfg=cfg)
        assert decision.start == []

    def test_nothing_starts_alongside_an_exclusive_job(self, cap, cfg):
        jobs = [make_job(1, exclusive=True, submitted_at=NOW), make_job(2, cpu=1, submitted_at=NOW + 1)]
        decision = run_plan(jobs, cap=cap, cfg=cfg, last_finish_at=NOW - 100)
        assert decision.start == [1]
        assert 2 in decision.blocked

    def test_blocked_reason_names_the_running_exclusive_job(self, cap, cfg):
        # Small foreign load must not be blamed when an exclusive job holds everything.
        running = [make_job(1, exclusive=True, state=JobState.RUNNING, started_at=NOW)]
        decision = run_plan([make_job(2, cpu=1)], running, cap=cap, cfg=cfg, external_usage={"mem_mb": 3480})
        assert "machine held by exclusive job #1" in decision.blocked[2]
        assert "outside ajs" not in decision.blocked[2]

    def test_settle_period_delays_the_start(self, cap, cfg):
        # Machine just went quiet, so the measurement would still be perturbed.
        decision = run_plan([make_job(1, exclusive=True)], cap=cap, cfg=cfg, last_finish_at=NOW - 2)
        assert decision.start == []
        assert "settling" in decision.blocked[1]

    def test_starts_once_settled(self, cap, cfg):
        decision = run_plan([make_job(1, exclusive=True)], cap=cap, cfg=cfg, last_finish_at=NOW - 30)
        assert decision.start == [1]

    def test_settling_job_holds_its_resources(self, cap, cfg):
        """Otherwise a small job grabs a slot and the settle period restarts forever."""
        jobs = [make_job(1, exclusive=True, submitted_at=NOW), make_job(2, cpu=1, submitted_at=NOW + 1)]
        decision = run_plan(jobs, cap=cap, cfg=cfg, last_finish_at=NOW - 2)
        assert decision.start == []
        assert "settling" in decision.blocked[1]

    def test_idle_machine_needs_no_settle(self, cap, cfg):
        decision = run_plan([make_job(1, exclusive=True)], cap=cap, cfg=cfg, last_finish_at=0.0)
        assert decision.start == [1]


class TestReservationAndBackfill:
    def test_starved_job_gets_a_reservation(self, cap, cfg):
        running = [make_job(1, cpu=16, state=JobState.RUNNING, started_at=NOW, max_runtime_s=600)]
        decision = run_plan([make_job(2, exclusive=True)], running, cap=cap, cfg=cfg)
        assert decision.reservation is not None
        assert decision.reservation.job_id == 2
        assert decision.reservation.start_at == NOW + 600

    def test_short_job_may_backfill_ahead_of_the_reservation(self, cap, cfg):
        running = [make_job(1, cpu=16, state=JobState.RUNNING, started_at=NOW, max_runtime_s=600)]
        queued = [
            make_job(2, exclusive=True, submitted_at=NOW),
            make_job(3, cpu=2, max_runtime_s=60, submitted_at=NOW + 1),
        ]
        decision = run_plan(queued, running, cap=cap, cfg=cfg)
        assert decision.start == [3]

    def test_long_job_may_not_delay_the_reservation(self, cap, cfg):
        running = [make_job(1, cpu=16, state=JobState.RUNNING, started_at=NOW, max_runtime_s=600)]
        queued = [
            make_job(2, exclusive=True, submitted_at=NOW),
            make_job(3, cpu=2, max_runtime_s=9999, submitted_at=NOW + 1),
        ]
        decision = run_plan(queued, running, cap=cap, cfg=cfg)
        assert decision.start == []
        assert "would delay reserved job #2" in decision.blocked[3]

    def test_exclusive_job_eventually_runs_under_a_stream_of_small_jobs(self, cap, cfg):
        """The starvation scenario this design exists to prevent.

        Small jobs arrive continuously. Without a reservation there is never an instant
        with 20 free slots, so the benchmark would wait forever.
        """
        running = [make_job(1, cpu=20, state=JobState.RUNNING, started_at=NOW, max_runtime_s=300)]
        queued = [make_job(2, exclusive=True, submitted_at=NOW)]
        next_id = 3
        now = NOW
        started_exclusive = False

        for _ in range(60):
            queued.append(make_job(next_id, cpu=2, max_runtime_s=600, submitted_at=now))
            next_id += 1
            decision = run_plan(queued, running, cap=cap, cfg=cfg, now=now, last_finish_at=now - 100)
            if 2 in decision.start:
                started_exclusive = True
                break
            # None of the long trickle may jump the reservation.
            assert all(j != 2 for j in decision.start)
            now += 30
            if now >= NOW + 300:
                running = []  # the big job finished
            queued = [j for j in queued if j.id not in decision.start]

        assert started_exclusive, "exclusive job starved despite the reservation"

    def test_reservation_when_nothing_running_frees_enough(self, cap, cfg):
        running = [
            make_job(1, cpu=10, state=JobState.RUNNING, started_at=NOW, max_runtime_s=100),
            make_job(2, cpu=10, state=JobState.RUNNING, started_at=NOW, max_runtime_s=900),
        ]
        decision = run_plan([make_job(3, exclusive=True)], running, cap=cap, cfg=cfg)
        # Needs the whole machine, so it waits for the *later* of the two.
        assert decision.reservation.start_at == NOW + 900


class TestLocks:
    def test_two_jobs_sharing_a_lock_do_not_overlap(self, cap, cfg):
        jobs = [
            make_job(1, locks=["sage-index"], submitted_at=NOW),
            make_job(2, locks=["sage-index"], submitted_at=NOW + 1),
        ]
        decision = run_plan(jobs, cap=cap, cfg=cfg)
        assert decision.start == [1]
        assert "sage-index" in decision.blocked[2]

    def test_different_locks_run_concurrently(self, cap, cfg):
        jobs = [make_job(1, locks=["a"]), make_job(2, locks=["b"], submitted_at=NOW + 1)]
        decision = run_plan(jobs, cap=cap, cfg=cfg)
        assert decision.start == [1, 2]

    def test_lock_held_by_a_running_job_blocks(self, cap, cfg):
        running = [make_job(1, locks=["gpu-model"], state=JobState.RUNNING, started_at=NOW)]
        decision = run_plan([make_job(2, locks=["gpu-model"])], running, cap=cap, cfg=cfg)
        assert decision.start == []


class TestDiskGuard:
    def test_job_refused_below_the_floor(self, cap, cfg):
        decision = run_plan([make_job(1)], cap=cap, cfg=cfg, free_disk_mb=1000)
        assert decision.start == []
        assert "disk guard" in decision.blocked[1]

    def test_declared_disk_is_added_to_the_floor(self, cap, cfg):
        job = make_job(1, disk_mb=30000)
        decision = run_plan([job], cap=cap, cfg=cfg, free_disk_mb=25000)
        assert decision.start == []
        assert "disk guard" in decision.blocked[1]

    def test_enough_headroom_admits(self, cap, cfg):
        decision = run_plan([make_job(1, disk_mb=1000)], cap=cap, cfg=cfg, free_disk_mb=100000)
        assert decision.start == [1]


class TestFairness:
    def test_interactive_outranks_batch(self, cap, cfg):
        jobs = [
            make_job(1, job_class=JobClass.BATCH, cpu=20, submitted_at=NOW),
            make_job(2, job_class=JobClass.INTERACTIVE, cpu=20, submitted_at=NOW + 5),
        ]
        decision = run_plan(jobs, cap=cap, cfg=cfg)
        assert decision.start == [2]

    def test_project_that_ran_recently_goes_last(self, cap, cfg):
        jobs = [
            make_job(1, project="busy", cpu=20, submitted_at=NOW),
            make_job(2, project="idle", cpu=20, submitted_at=NOW + 1),
        ]
        ordered = order_queue(jobs, {"busy": NOW, "idle": 0.0})
        assert [j.id for j in ordered] == [2, 1]

    def test_project_concurrency_cap(self, cap, cfg):
        running = [make_job(i, project="hog", cpu=1, state=JobState.RUNNING, started_at=NOW) for i in range(1, 5)]
        decision = run_plan([make_job(9, project="hog", cpu=1)], running, cap=cap, cfg=cfg)
        assert decision.start == []
        assert "project cap" in decision.blocked[9]

    def test_cap_is_per_project_not_global(self, cap, cfg):
        running = [make_job(i, project="hog", cpu=1, state=JobState.RUNNING, started_at=NOW) for i in range(1, 5)]
        decision = run_plan([make_job(9, project="other", cpu=1)], running, cap=cap, cfg=cfg)
        assert decision.start == [9]

    def test_fifo_within_a_project(self, cap, cfg):
        jobs = [make_job(2, submitted_at=NOW + 10), make_job(1, submitted_at=NOW)]
        assert [j.id for j in order_queue(jobs, {})] == [1, 2]


class TestLeases:
    def test_lease_usage_reduces_availability(self, cap, cfg):
        decision = run_plan([make_job(1, cpu=8)], cap=cap, cfg=cfg, lease_usage={"cpu": 16, "mem_mb": 0, "gpu": 0})
        assert decision.start == []


class TestHoldsVisibleToReservations:
    """A reservation is a promise. Every hold the pass knows about has to feed into it,
    or the promise is wrong and -- worse -- its backfill veto is applied to a start time
    that means nothing."""

    def test_lease_blocked_head_gets_no_eta_and_does_not_freeze_the_queue(self, cap, cfg):
        """Nothing declares when a lease will be released, so the head job cannot be
        promised a start time -- and a promise of "0s from now" must not veto backfill."""
        head = make_job(1, cpu=16, max_runtime_s=600)
        small = make_job(2, project="other", cpu=1, max_runtime_s=3600, submitted_at=NOW + 1)
        decision = run_plan(
            [head, small],
            cap=cap,
            cfg=cfg,
            lease_usage={"cpu": 8, "mem_mb": 0, "gpu": 0, "gpu_mem_mb": 0},
        )
        assert decision.reservation is not None and decision.reservation.external
        assert "lease" in decision.blocked[1]
        assert decision.start == [2]

    def test_settling_exclusive_job_counts_as_a_hold(self, cap, cfg):
        """While a timing run sits out its settle period it owns the machine. The next
        job's reservation has to be projected past it, not promised for right now."""
        timing = make_job(1, exclusive=True, max_runtime_s=300, job_class=JobClass.INTERACTIVE)
        big = make_job(2, project="other", cpu=14, max_runtime_s=600)
        decision = run_plan([timing, big], cap=cap, cfg=cfg, last_finish_at=NOW - 2)
        assert "settling" in decision.blocked[1]
        assert decision.reservation is not None
        assert decision.reservation.job_id == 2
        assert not decision.reservation.external
        # 8s of settle left, then up to 300s of run.
        assert decision.reservation.start_at == NOW + 8 + 300

    def test_jobs_started_this_pass_count_as_holds(self, cap, cfg):
        first = make_job(1, cpu=12, max_runtime_s=600)
        second = make_job(2, project="other", cpu=12, max_runtime_s=600, submitted_at=NOW + 1)
        decision = run_plan([first, second], cap=cap, cfg=cfg)
        assert decision.start == [1]
        assert decision.reservation is not None
        assert decision.reservation.start_at == NOW + 600


class TestCountedSemaphores:
    def test_a_named_semaphore_admits_its_configured_count(self, cap, cfg):
        cfg.extra_semaphores = {"api:anthropic": 2}
        jobs = [make_job(i, locks=["api:anthropic"], submitted_at=NOW + i) for i in range(1, 4)]
        decision = run_plan(jobs, cap=cap, cfg=cfg)
        assert decision.start == [1, 2]
        assert "api:anthropic" in decision.blocked[3]

    def test_running_holders_count_against_the_semaphore(self, cap, cfg):
        cfg.extra_semaphores = {"api:anthropic": 2}
        running = [
            make_job(i, locks=["api:anthropic"], state=JobState.RUNNING, started_at=NOW, project=f"p{i}")
            for i in (1, 2)
        ]
        decision = run_plan([make_job(3, locks=["api:anthropic"])], running, cap=cap, cfg=cfg)
        assert decision.start == []

    def test_an_unconfigured_lock_is_still_exclusive(self, cap, cfg):
        jobs = [make_job(1, locks=["scratch"]), make_job(2, locks=["scratch"], submitted_at=NOW + 1)]
        decision = run_plan(jobs, cap=cap, cfg=cfg)
        assert decision.start == [1]


class TestMemoryGuard:
    """Real free memory, not just declared reservations, gates a start."""

    def test_blocks_when_really_free_memory_is_short(self, cap, cfg):
        decision = run_plan([make_job(1, mem_mb=8000)], cap=cap, cfg=cfg, mem_headroom_mb=5000)
        assert decision.start == []
        assert "memory guard" in decision.blocked[1]

    def test_each_start_uses_up_headroom(self, cap, cfg):
        jobs = [make_job(i, mem_mb=4000, submitted_at=NOW + i) for i in range(1, 4)]
        decision = run_plan(jobs, cap=cap, cfg=cfg, mem_headroom_mb=9000)
        assert decision.start == [1, 2]
        assert "memory guard" in decision.blocked[3]

    def test_unmeasured_memory_skips_the_check(self, cap, cfg):
        decision = run_plan([make_job(1, mem_mb=8000)], cap=cap, cfg=cfg, mem_headroom_mb=None)
        assert decision.start == [1]


class TestQuietGate:
    """An exclusive job waits for load outside ajs to drop, holding the machine meanwhile."""

    def test_waits_while_the_machine_is_noisy(self, cap, cfg):
        decision = run_plan([make_job(1, exclusive=True)], cap=cap, cfg=cfg, quiet_since=None, noise="iowait 11%")
        assert decision.start == []
        assert "iowait 11%" in decision.blocked[1]
        assert decision.settling == [1]

    def test_waits_for_the_full_quiet_window(self, cap, cfg):
        decision = run_plan([make_job(1, exclusive=True)], cap=cap, cfg=cfg, quiet_since=NOW - 20)
        assert decision.start == []
        assert "quiet for 20s" in decision.blocked[1]

    def test_starts_once_quiet_long_enough(self, cap, cfg):
        decision = run_plan([make_job(1, exclusive=True)], cap=cap, cfg=cfg, quiet_since=NOW - 61)
        assert decision.start == [1]

    def test_waiting_job_blocks_other_jobs(self, cap, cfg):
        jobs = [make_job(1, exclusive=True, submitted_at=NOW), make_job(2, cpu=2, submitted_at=NOW + 1)]
        decision = run_plan(jobs, cap=cap, cfg=cfg, quiet_since=None, noise="x")
        assert decision.start == []

    def test_starts_anyway_after_the_max_wait(self, cap, cfg):
        decision = run_plan(
            [make_job(1, exclusive=True)],
            cap=cap,
            cfg=cfg,
            quiet_since=None,
            noise="iowait 11%",
            settling_since={1: NOW - cfg.quiet_max_wait_s},
        )
        assert decision.start == [1]
        assert "without a quiet window" in decision.unquiet_start[1]

    def test_non_exclusive_jobs_ignore_the_gate(self, cap, cfg):
        decision = run_plan([make_job(1, cpu=2)], cap=cap, cfg=cfg, quiet_since=None, noise="x")
        assert decision.start == [1]
