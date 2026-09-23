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


def test_a_timing_run_still_starts_with_the_desktop_running(cfg, cap):
    """An exclusive job asks for the whole machine. If foreign load were charged against
    it too, it could never start on a laptop that always has a browser open -- the
    request would exceed what is free by definition, forever. Exclusivity here means no
    other *ajs job* runs alongside; the contention monitor reports what the desktop did.
    """
    decision = plan(
        queued=[make_job(1, exclusive=True)],
        running=[],
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
        external_usage={"cpu": 19, "mem_mb": 40000, "gpu": 0},
    )
    assert decision.start == [1]


def test_a_timing_run_still_waits_for_other_ajs_jobs(cfg, cap):
    """The exemption is for foreign load only -- it must not become a way to trample
    another job that the scheduler itself started."""
    running = [make_job(9, project="other", cpu=4, state=JobState.RUNNING, started_at=NOW)]
    decision = plan(
        queued=[make_job(1, exclusive=True)],
        running=running,
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
        external_usage={"cpu": 2, "mem_mb": 0, "gpu": 0},
    )
    assert decision.start == []


def test_an_ordinary_job_is_still_throttled_by_foreign_load(cfg, cap):
    decision = plan(
        queued=[make_job(1, cpu=16)],
        running=[],
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
        external_usage={"cpu": 8, "mem_mb": 0, "gpu": 0},
    )
    assert decision.start == []


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


# --- laptop allowances -----------------------------------------------------


def test_ambient_desktop_cpu_is_not_charged_to_the_queue(fake_machine):
    """A browser and a compositor are permanent facts on a laptop, not interference."""
    ext = ExternalLoad(half_life_s=0.0)
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0)
    fake_machine["busy"] = 18.0  # 1.8 cores
    ext.sample(now=10.0, own_cpu_seconds=0.0, own_mem_mb=0)
    assert ext.usage(cap_cpu=20, cpu_allowance=2.0)["cpu"] == 0


def test_cpu_above_the_allowance_is_still_charged(fake_machine):
    ext = ExternalLoad(half_life_s=0.0)
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0)
    fake_machine["busy"] = 90.0  # 9 cores
    ext.sample(now=10.0, own_cpu_seconds=0.0, own_mem_mb=0)
    assert ext.usage(cap_cpu=20, cpu_allowance=2.0)["cpu"] == 7


def test_desktop_memory_is_not_billed_twice(fake_machine):
    """Capacity already withholds mem_reserve_mb for the desktop. Charging the desktop's
    measured usage on top of that shrinks the queue for no reason."""
    fake_machine["available"] = 64000 - 8000  # 8 GB in use, all of it desktop
    ext = ExternalLoad()
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0)
    assert ext.usage(cap_cpu=20, mem_allowance_mb=6144)["mem_mb"] == 8000 - 6144


def test_memory_allowance_never_goes_negative(fake_machine):
    fake_machine["available"] = 64000 - 1000
    ext = ExternalLoad()
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0)
    assert ext.usage(cap_cpu=20, mem_allowance_mb=6144)["mem_mb"] == 0


def test_a_timing_run_blocked_by_ajs_jobs_still_gets_a_real_reservation(cfg, cap):
    """Foreign load is exempt for exclusive jobs at admission, so it must be exempt when
    projecting their start too. Otherwise the reservation comes back flagged external,
    the backfill veto is dropped, and the timing run is starved by a trickle of small
    jobs -- the exact failure reservations exist to prevent."""
    running = [make_job(9, project="other", cpu=4, state=JobState.RUNNING, started_at=NOW - 100, max_runtime_s=600)]
    timing = make_job(1, project="bench", exclusive=True, max_runtime_s=300)
    long_job = make_job(2, project="other", cpu=2, max_runtime_s=3600, submitted_at=NOW + 1)
    decision = plan(
        queued=[timing, long_job],
        running=running,
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
        external_usage={"cpu": 3, "mem_mb": 0, "gpu": 0, "gpu_mem_mb": 0},
    )
    assert decision.reservation is not None
    assert decision.reservation.job_id == 1
    assert not decision.reservation.external
    assert decision.reservation.start_at == NOW + 500
    assert decision.start == []
    assert "would delay reserved job #1" in decision.blocked[2]


def test_foreign_vram_is_charged_even_to_a_gpu_timing_run(cfg, cap):
    """The desktop exemption is about CPU and memory. A hand-run model holding most of
    the card is not ambient load: admitting a job on top of it means an OOM."""
    decision = plan(
        queued=[make_job(1, cpu=2, gpu_exclusive=True)],
        running=[],
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
        external_usage={"cpu": 0, "mem_mb": 0, "gpu": 0, "gpu_mem_mb": 3000},
    )
    assert decision.start == []
    assert "VRAM" in decision.blocked[1]

    both = plan(
        queued=[make_job(1, exclusive=True, gpu_exclusive=True)],
        running=[],
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
        external_usage={"cpu": 19, "mem_mb": 40000, "gpu": 0, "gpu_mem_mb": 3000},
    )
    assert both.start == []  # exempt from the desktop's cpu/mem, still blocked by VRAM


def test_iowait_share_is_measured(fake_machine, monkeypatch):
    ticks = {"v": (0.0, 0.0)}
    monkeypatch.setattr(contention, "system_cpu_ticks", lambda: ticks["v"])
    ext = ExternalLoad(half_life_s=0.0)
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0)
    ticks["v"] = (100.0, 1000.0)
    fake_machine["busy"] = 1.0
    ext.sample(now=10.0, own_cpu_seconds=0.0, own_mem_mb=0)
    assert ext.iowait_pct == pytest.approx(10.0)


def test_foreign_processes_name_a_busy_process():
    """A real /proc walk: this test's own busy loop must show up by PID."""
    import os
    import time as _time

    from ajs.contention import ForeignProcesses

    fp = ForeignProcesses()
    fp.sample(_time.time(), set(), min_cores=0.0)
    end = _time.time() + 0.3
    while _time.time() < end:
        pass
    busy = fp.sample(_time.time(), set(), top=50, min_cores=0.3)
    assert os.getpid() in {p.pid for p in busy}
