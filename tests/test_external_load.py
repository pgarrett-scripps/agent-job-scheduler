"""Foreign load -- work ajs did not start -- must consume capacity like anything else."""

from __future__ import annotations

import pytest

from ajs import contention
from ajs.contention import ExternalLoad
from ajs.models import JobState
from ajs.scheduler import plan

from .conftest import NOW, make_job


@pytest.fixture
def fake_machine(monkeypatch):
    """Drive the /proc readers by hand so samples are deterministic."""
    state = {"busy": 0.0, "total": 64000, "available": 64000}
    monkeypatch.setattr(contention, "system_busy_seconds", lambda: state["busy"])
    monkeypatch.setattr(contention.sysinfo, "total_mem_mb", lambda: state["total"])
    monkeypatch.setattr(contention.sysinfo, "available_mem_mb", lambda: state["available"])
    return state


def test_first_sample_only_establishes_a_baseline(fake_machine):
    ext = ExternalLoad()
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0)
    assert not ext.ready


def test_busy_machine_with_no_ajs_jobs_is_all_foreign(fake_machine):
    ext = ExternalLoad(half_life_s=0.0)  # no smoothing, so one sample is the answer
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0)
    fake_machine["busy"] = 40.0  # 40 cpu-seconds over 10s wall = 4 cores
    ext.sample(now=10.0, own_cpu_seconds=0.0, own_mem_mb=0)
    assert ext.ready
    assert ext.cpu_cores == pytest.approx(4.0)


def test_a_jobs_own_usage_is_not_counted_as_foreign(fake_machine):
    ext = ExternalLoad(half_life_s=0.0)
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0)
    fake_machine["busy"] = 40.0
    # The ajs job accounts for 30 of those 40 cpu-seconds; only 1 core is foreign.
    ext.sample(now=10.0, own_cpu_seconds=30.0, own_mem_mb=0)
    assert ext.cpu_cores == pytest.approx(1.0)


def test_a_job_exiting_does_not_register_as_foreign_load(fake_machine):
    """Cgroup totals vanish when a job ends; that must not read as a foreign spike."""
    ext = ExternalLoad(half_life_s=0.0)
    ext.sample(now=0.0, own_cpu_seconds=100.0, own_mem_mb=0)
    fake_machine["busy"] = 5.0
    ext.sample(now=10.0, own_cpu_seconds=0.0, own_mem_mb=0)  # job gone, total dropped
    assert ext.cpu_cores == pytest.approx(0.5)


def test_memory_is_foreign_minus_what_ajs_jobs_hold(fake_machine):
    ext = ExternalLoad()
    fake_machine["available"] = 40000  # 24000 MB in use machine-wide
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=10000)
    assert ext.mem_mb == 14000


def test_usage_never_consumes_the_whole_machine(fake_machine):
    """Foreign load may throttle the queue but must never wedge it completely."""
    ext = ExternalLoad(half_life_s=0.0)
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0)
    fake_machine["busy"] = 1000.0  # 100 cores' worth on a 20-core box
    ext.sample(now=10.0, own_cpu_seconds=0.0, own_mem_mb=0)
    assert ext.usage(cap_cpu=20)["cpu"] == 19


def test_usage_falls_back_to_load_average_before_the_first_delta(monkeypatch, fake_machine):
    """A daemon that just started has no differenced sample, and must not assume idle."""
    monkeypatch.setattr(contention.sysinfo, "load_average", lambda: (7.5, 7.0, 6.0))
    ext = ExternalLoad()
    assert not ext.ready
    assert ext.usage(cap_cpu=20)["cpu"] == 7


def test_fractional_foreign_cores_round_down(fake_machine):
    ext = ExternalLoad(half_life_s=0.0)
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0)
    fake_machine["busy"] = 19.0  # 1.9 cores
    ext.sample(now=10.0, own_cpu_seconds=0.0, own_mem_mb=0)
    assert ext.usage(cap_cpu=20)["cpu"] == 1


# --- how the scheduler reacts to it ---------------------------------------


def test_foreign_load_keeps_a_job_from_starting(cfg, cap):
    job = make_job(1, cpu=16)
    idle = plan(queued=[job], running=[], cap=cap, cfg=cfg, now=NOW, last_start={})
    assert idle.start == [1]

    busy = plan(
        queued=[job],
        running=[],
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
        external_usage={"cpu": 8, "mem_mb": 0, "gpu": 0},
    )
    assert busy.start == []
    assert "outside ajs" in busy.blocked[1]


def test_foreign_load_does_not_make_a_job_impossible(cfg, cap):
    """It is usage, not a smaller machine -- otherwise a browser would permanently
    disqualify every exclusive run."""
    decision = plan(
        queued=[make_job(1, exclusive=True)],
        running=[],
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
        external_usage={"cpu": 19, "mem_mb": 0, "gpu": 0},
    )
    assert "impossible" not in decision.blocked[1]


def test_a_job_blocked_only_by_foreign_load_gets_no_reservation_promise(cfg, cap):
    decision = plan(
        queued=[make_job(1, cpu=16)],
        running=[],
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
        external_usage={"cpu": 8, "mem_mb": 0, "gpu": 0},
    )
    assert decision.reservation is not None
    assert decision.reservation.external
    assert "no reservation possible" in decision.blocked[1]


def test_an_external_reservation_does_not_freeze_the_rest_of_the_queue(cfg, cap):
    """The backfill veto protects a promised start time. With foreign load there is no
    honest promise, so holding cores empty would strand small jobs for nothing."""
    big = make_job(1, cpu=16, max_runtime_s=60)
    small = make_job(2, project="other", cpu=1, max_runtime_s=3600)
    decision = plan(
        queued=[big, small],
        running=[],
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
        external_usage={"cpu": 8, "mem_mb": 0, "gpu": 0},
    )
    assert decision.start == [2]


def test_ajs_reservations_still_veto_backfill(cfg, cap):
    """The pre-existing guarantee must survive the new code path."""
    running = [make_job(9, project="hog", cpu=16, state=JobState.RUNNING, started_at=NOW, max_runtime_s=600)]
    big = make_job(1, cpu=16, max_runtime_s=60)
    small = make_job(2, project="other", cpu=1, max_runtime_s=3600)
    decision = plan(
        queued=[big, small],
        running=running,
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
    )
    assert decision.reservation is not None
    assert not decision.reservation.external
    assert decision.start == []
    assert "would delay reserved job" in decision.blocked[2]
