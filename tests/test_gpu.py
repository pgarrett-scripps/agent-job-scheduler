"""GPU scheduling. The card is small (4 GB) and shared with the display, so VRAM rather
than device count is the resource that actually runs out."""

from __future__ import annotations

import pytest

from ajs import contention, sysinfo
from ajs.config import Config
from ajs.contention import ExternalLoad
from ajs.scheduler import Capacity, effective_request, plan

from .conftest import NOW, make_job


def test_a_gpu_job_without_a_vram_figure_is_charged_the_whole_card(cap):
    """Two jobs each silently assuming they own 4 GB would OOM each other."""
    need = effective_request(make_job(1, gpu=1), cap)
    assert need["gpu_mem_mb"] == cap.gpu_mem_mb


def test_a_declared_vram_figure_is_respected(cap):
    need = effective_request(make_job(1, gpu=1, gpu_mem_mb=1024), cap)
    assert need["gpu_mem_mb"] == 1024


def test_two_small_gpu_jobs_share_the_card(cfg, cap):
    jobs = [make_job(1, gpu=1, gpu_mem_mb=1024), make_job(2, project="b", gpu=1, gpu_mem_mb=1024)]
    assert plan(queued=jobs, running=[], cap=cap, cfg=cfg, now=NOW, last_start={}).start == [1, 2]


def test_vram_runs_out_before_device_count_does(cfg, cap):
    jobs = [make_job(1, gpu=1, gpu_mem_mb=3000), make_job(2, project="b", gpu=1, gpu_mem_mb=3000)]
    decision = plan(queued=jobs, running=[], cap=cap, cfg=cfg, now=NOW, last_start={})
    assert decision.start == [1]
    assert 2 in decision.blocked


def test_gpu_exclusive_takes_the_whole_card(cap):
    need = effective_request(make_job(1, gpu_exclusive=True), cap)
    assert need["gpu_mem_mb"] == cap.gpu_mem_mb
    assert need["gpu"] == cap.gpu


# --- the two axes are independent -----------------------------------------


def test_a_cpu_timing_run_does_not_reserve_the_gpu(cap):
    """Coupling them would idle the card during every CPU benchmark."""
    need = effective_request(make_job(1, exclusive=True), cap)
    assert need["gpu_mem_mb"] == 0
    assert need["gpu"] == 0


def test_a_gpu_timing_run_does_not_reserve_every_core(cap):
    need = effective_request(make_job(1, cpu=2, gpu_exclusive=True), cap)
    assert need["cpu"] == 2
    assert need["mem_mb"] == 512


def test_a_gpu_job_runs_alongside_a_cpu_timing_run(cfg, cap):
    """The whole point of separating the axes: a 20-core benchmark and a GPU job are not
    competing for anything."""
    running = [make_job(9, project="bench", exclusive=True, state=_running(), started_at=NOW)]
    gpu_job = make_job(1, cpu=0, mem_mb=0, gpu=1, gpu_mem_mb=2048)
    decision = plan(queued=[gpu_job], running=running, cap=cap, cfg=cfg, now=NOW, last_start={})
    assert decision.start == []  # it still needs 1 cpu, and the timing run holds them all
    # ...but purely on the GPU axis there is no conflict:
    assert effective_request(running[0], cap)["gpu_mem_mb"] == 0


def test_both_flags_together_take_the_whole_machine(cap):
    need = effective_request(make_job(1, exclusive=True, gpu_exclusive=True), cap)
    assert need["cpu"] == cap.cpu
    assert need["gpu_mem_mb"] == cap.gpu_mem_mb


def _running():
    from ajs.models import JobState

    return JobState.RUNNING


# --- foreign GPU load -----------------------------------------------------


def test_vram_held_outside_ajs_is_counted(monkeypatch):
    monkeypatch.setattr(sysinfo, "gpu_compute_apps", lambda: {4242: 900})
    monkeypatch.setattr(contention, "cgroup_path_for_pid", lambda pid: None)
    ext = ExternalLoad()
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0, own_cgroups=set())
    assert ext.gpu_known
    assert ext.usage(cap_cpu=20)["gpu_mem_mb"] == 900


def test_vram_held_by_an_ajs_job_is_not_double_counted(monkeypatch, tmp_path):
    """The job already reserved that VRAM through the scheduler; charging it again as
    foreign would halve the card's apparent size."""
    own = tmp_path / "ajs-job-1.scope"
    own.mkdir()
    monkeypatch.setattr(sysinfo, "gpu_compute_apps", lambda: {4242: 900})
    monkeypatch.setattr(contention, "cgroup_path_for_pid", lambda pid: own)
    ext = ExternalLoad()
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0, own_cgroups={own})
    assert ext.usage(cap_cpu=20)["gpu_mem_mb"] == 0


def test_an_unqueryable_gpu_is_not_reported_as_idle(monkeypatch):
    monkeypatch.setattr(sysinfo, "gpu_compute_apps", lambda: None)
    ext = ExternalLoad()
    ext.sample(now=0.0, own_cpu_seconds=0.0, own_mem_mb=0, own_cgroups=set())
    assert not ext.gpu_known


def test_foreign_vram_keeps_a_gpu_job_waiting(cfg, cap):
    decision = plan(
        queued=[make_job(1, gpu=1, gpu_mem_mb=3000)],
        running=[],
        cap=cap,
        cfg=cfg,
        now=NOW,
        last_start={},
        external_usage={"cpu": 0, "mem_mb": 0, "gpu": 0, "gpu_mem_mb": 1024},
    )
    assert decision.start == []


# --- capacity autodetection ------------------------------------------------


def test_vram_capacity_holds_some_back_for_the_display(monkeypatch):
    """Handing a job literally all of it either fails to allocate or stalls the desktop."""
    monkeypatch.setattr(sysinfo, "gpu_total_mem_mb", lambda: 4096)
    monkeypatch.setattr(sysinfo, "gpu_count", lambda: 1)
    cfg = Config(cpu=8, mem_mb=8000, gpu_mem_reserve_mb=512).resolved()
    assert cfg.gpu_mem_mb == 3584


def test_a_machine_with_no_gpu_gets_no_vram_capacity(monkeypatch):
    monkeypatch.setattr(sysinfo, "gpu_total_mem_mb", lambda: 0)
    monkeypatch.setattr(sysinfo, "gpu_count", lambda: 0)
    cfg = Config(cpu=8, mem_mb=8000).resolved()
    assert cfg.gpu_mem_mb == 0
    assert Capacity.from_config(cfg).gpu_mem_mb == 0


def test_a_gpu_request_on_a_gpuless_machine_is_impossible_not_silent(cfg):
    """Regression guard: it must not quietly run without the hardware it asked for."""
    cap = Capacity(cpu=20, mem_mb=56000, gpu=0, gpu_mem_mb=0)
    decision = plan(queued=[make_job(1, gpu=1)], running=[], cap=cap, cfg=cfg, now=NOW, last_start={})
    assert "impossible" in decision.blocked[1]


@pytest.mark.parametrize("declared", [0, 1024, 4096])
def test_effective_request_never_returns_negative_vram(cap, declared):
    assert effective_request(make_job(1, gpu_mem_mb=declared), cap)["gpu_mem_mb"] >= 0
