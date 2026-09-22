"""The foreign-CPU measurement that decides whether a timing run is trustworthy."""

import os

from ajs import contention
from ajs.contention import ContentionMonitor


class TestCounters:
    def test_system_busy_is_monotonic(self):
        first = contention.system_busy_seconds()
        assert first is not None
        for _ in range(200_000):
            pass
        second = contention.system_busy_seconds()
        assert second >= first

    def test_cgroup_resolves_for_this_process(self):
        path = contention.cgroup_path_for_pid(os.getpid())
        assert path is not None
        assert contention.cgroup_cpu_seconds(path) is not None

    def test_missing_pid_resolves_to_none(self):
        assert contention.cgroup_path_for_pid(999_999_99) is None


class TestMonitor:
    def test_quiet_run_is_not_flagged(self, monkeypatch):
        """Job used 10 CPU-seconds, system used 10: nothing foreign happened."""
        monitor = ContentionMonitor(threshold_cores=2.0)
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 1000.0)
        monkeypatch.setattr(contention, "cgroup_path_for_pid", lambda pid: "/fake")
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 100.0)
        monitor.start(pid=1, now=0.0)

        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 1010.0)
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 110.0)
        report = monitor.finish(now=10.0)

        assert report.foreign_cores == 0.0
        assert not report.contended
        assert report.trustworthy

    def test_foreign_load_is_flagged(self, monkeypatch):
        """System burned 50 CPU-s over 10s but the job only accounts for 10 of them."""
        monitor = ContentionMonitor(threshold_cores=2.0)
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 1000.0)
        monkeypatch.setattr(contention, "cgroup_path_for_pid", lambda pid: "/fake")
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 100.0)
        monitor.start(pid=1, now=0.0)

        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 1050.0)
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 110.0)
        report = monitor.finish(now=10.0)

        assert report.foreign_cores == 4.0  # 40 foreign CPU-s / 10s
        assert report.contended
        assert not report.trustworthy
        assert "4.00 cores" in report.note

    def test_a_busy_job_alone_is_not_contention(self, monkeypatch):
        """The failure mode of the naive load-average check.

        A job saturating 16 cores drives load sky-high, but it is the only thing running,
        so the measurement is clean. Load average would wrongly condemn it.
        """
        monitor = ContentionMonitor(threshold_cores=2.0)
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 0.0)
        monkeypatch.setattr(contention, "cgroup_path_for_pid", lambda pid: "/fake")
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 0.0)
        monitor.start(pid=1, now=0.0)

        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 160.0)
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 160.0)
        report = monitor.finish(now=10.0)

        assert report.foreign_cores == 0.0
        assert not report.contended

    def test_threshold_is_respected(self, monkeypatch):
        monitor = ContentionMonitor(threshold_cores=5.0)
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 0.0)
        monkeypatch.setattr(contention, "cgroup_path_for_pid", lambda pid: "/fake")
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 0.0)
        monitor.start(pid=1, now=0.0)

        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 40.0)
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 0.0)
        report = monitor.finish(now=10.0)

        assert report.foreign_cores == 4.0
        assert not report.contended  # under the 5-core threshold

    def test_unmeasurable_job_reports_unknown_rather_than_guessing(self, monkeypatch):
        monitor = ContentionMonitor(threshold_cores=2.0)
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 1000.0)
        monkeypatch.setattr(contention, "cgroup_path_for_pid", lambda pid: None)
        monitor.start(pid=1, now=0.0)

        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 1100.0)
        report = monitor.finish(now=10.0)

        assert report.foreign_cores is None
        assert not report.contended
        assert "unavailable" in report.note

    def test_counter_going_backwards_does_not_produce_negative_load(self, monkeypatch):
        monitor = ContentionMonitor(threshold_cores=2.0)
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 1000.0)
        monkeypatch.setattr(contention, "cgroup_path_for_pid", lambda pid: "/fake")
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 100.0)
        monitor.start(pid=1, now=0.0)

        # Job cgroup reports more CPU than the system did; clamp rather than go negative.
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 1001.0)
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 200.0)
        report = monitor.finish(now=10.0)

        assert report.foreign_cpu_seconds >= 0.0
        assert not report.contended


class TestVerdictScope:
    def _monitor(self, monkeypatch, assess, elapsed, foreign):
        monitor = ContentionMonitor(threshold_cores=2.0, assess=assess)
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 0.0)
        monkeypatch.setattr(contention, "cgroup_path_for_pid", lambda pid: "/fake")
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 0.0)
        monitor.start(pid=1, now=0.0)
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: foreign)
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 0.0)
        return monitor.finish(now=elapsed)

    def test_non_timing_run_gets_no_verdict(self, monkeypatch):
        """Competing load is expected for an ordinary job; flagging it would be noise."""
        report = self._monitor(monkeypatch, assess=False, elapsed=10.0, foreign=100.0)
        assert report.foreign_cores == 10.0  # still measured
        assert not report.contended  # but no verdict delivered

    def test_timing_run_does_get_a_verdict(self, monkeypatch):
        report = self._monitor(monkeypatch, assess=True, elapsed=10.0, foreign=100.0)
        assert report.contended

    def test_run_too_short_to_assess(self, monkeypatch):
        """A 40 ms run divides jittery 10 ms ticks by a tiny elapsed: meaningless."""
        report = self._monitor(monkeypatch, assess=True, elapsed=0.04, foreign=0.6)
        assert not report.contended
        assert "too short" in report.note


class TestVanishingCgroup:
    """systemd removes a scope's cgroup the instant its last process exits, which is also
    when the daemon finds out. The verdict must survive that, or every job on a systemd
    host reports 'accounting unavailable'."""

    def test_falls_back_to_the_last_polled_sample(self, monkeypatch):
        monitor = ContentionMonitor(threshold_cores=2.0)
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 1000.0)
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 100.0)
        monkeypatch.setattr(contention, "cgroup_mem_bytes", lambda p: 0)
        monitor.start(pid=None, now=0.0, cgroup=contention.Path("/fake"))

        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: 109.0)
        monitor.poll(now=9.0)

        # Job exits; cgroup gone; system did 10 CPU-s, job accounted for 9 of them.
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: None)
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 1010.0)
        report = monitor.finish(now=10.0)

        assert report.foreign_cpu_seconds == 1.0
        assert not report.contended
        assert "last sampled 1.0s before exit" in report.note

    def test_without_any_sample_the_verdict_is_unknown(self, monkeypatch):
        monitor = ContentionMonitor(threshold_cores=2.0)
        monkeypatch.setattr(contention, "system_busy_seconds", lambda: 1000.0)
        monkeypatch.setattr(contention, "cgroup_cpu_seconds", lambda p: None)
        monitor.start(pid=None, now=0.0, cgroup=contention.Path("/fake"))
        report = monitor.finish(now=10.0)
        assert report.foreign_cores is None
        assert "unknown" in report.note
