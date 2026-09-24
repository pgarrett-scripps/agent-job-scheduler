"""Docker containers started by a job count as the job's work."""

import os
import subprocess
import time

from ajs import containers
from ajs.contention import ContentionMonitor

MB = 1024 * 1024


def _cgroup(path, *, cpu_s, anon_mb):
    path.mkdir(exist_ok=True)
    (path / "cpu.stat").write_text(f"usage_usec {int(cpu_s * 1_000_000)}\n")
    (path / "memory.current").write_text(str(anon_mb * MB))
    (path / "memory.stat").write_text(f"anon {anon_mb * MB}\nfile 0\n")


def test_shim_labels_containers_with_the_job(tmp_path):
    fake = tmp_path / "real-docker"
    fake.write_text('#!/bin/sh\necho "$@"\n')
    fake.chmod(0o755)
    bin_dir = containers.install_shim(tmp_path / "bin", real=str(fake))
    assert bin_dir is not None
    shim = str(bin_dir / "docker")
    env = {**os.environ, "AJS_JOB_ID": "42"}

    def run(*args):
        return subprocess.run([shim, *args], env=env, capture_output=True, text=True, check=True).stdout.strip()

    assert run("run", "--rm", "img") == "run --label ajs.job=42 --rm img"
    assert run("container", "create", "img") == "container create --label ajs.job=42 img"
    assert run("ps", "-q") == "ps -q"
    env.pop("AJS_JOB_ID")
    assert run("run", "img") == "run img"


def test_parse_ps_keeps_only_well_formed_lines():
    out = "abc123 7\nbroken\ndef456 notanid\n"
    assert containers.parse_ps(out) == [("abc123", 7)]


def test_attached_container_counts_as_the_job(tmp_path):
    job, box = tmp_path / "job", tmp_path / "box"
    _cgroup(job, cpu_s=10, anon_mb=100)
    _cgroup(box, cpu_s=50, anon_mb=900)
    monitor = ContentionMonitor(2.0)
    monitor.start(None, time.time(), cgroup=job)
    monitor.attach(box)
    _cgroup(box, cpu_s=80, anon_mb=900)
    monitor.poll(time.time())
    assert monitor.current_cpu_seconds() == 10 + 30  # only the container's work since attaching
    assert monitor.mem_held_mb == 1000
    assert monitor.mem_peak_mb == 1000
    assert monitor.extra_cgroups == {box}


def test_cpu_total_does_not_drop_when_the_container_exits(tmp_path):
    job, box = tmp_path / "job", tmp_path / "box"
    _cgroup(job, cpu_s=10, anon_mb=100)
    _cgroup(box, cpu_s=0, anon_mb=500)
    monitor = ContentionMonitor(2.0)
    monitor.start(None, time.time(), cgroup=job)
    monitor.attach(box)
    _cgroup(box, cpu_s=20, anon_mb=500)
    assert monitor.current_cpu_seconds() == 30
    for f in box.iterdir():
        f.unlink()
    box.rmdir()
    assert monitor.current_cpu_seconds() == 30
    assert monitor.current_mem_bytes() == 100 * MB
