"""`ajs top` rendering, driven by a canned status payload so no daemon is needed."""

from rich.console import Console

from ajs.top import bar, fmt_mem, fmt_secs, render

NOW = 1_000_000.0


def _payload():
    return {
        "capacity": {"cpu": 20, "mem_mb": 56000, "gpu": 1, "gpu_mem_mb": 3584},
        "used": {"cpu": 8, "mem_mb": 8192, "gpu": 0, "gpu_mem_mb": 0},
        "external": {"cpu": 3, "mem_mb": 4096, "gpu": 0, "gpu_mem_mb": 300},
        "free": {"cpu": 9, "mem_mb": 43712, "gpu": 1, "gpu_mem_mb": 3284},
        "running": [
            {
                "id": 41,
                "project": "koth_rust",
                "class": "batch",
                "cmd_str": "cargo test --release",
                "cpu": 8,
                "mem_mb": 8192,
                "gpu": 0,
                "gpu_mem_mb": 0,
                "exclusive": False,
                "gpu_exclusive": False,
                "max_runtime_s": 1800,
                "started_at": NOW - 72,
                "runtime_s": 72.0,
                "cpu_now": 11.2,
                "mem_now_mb": 3100,
            }
        ],
        "queued": [
            {
                "id": 42,
                "project": "blitz",
                "class": "batch",
                "cmd_str": "./bench.sh",
                "cpu": 1,
                "mem_mb": 512,
                "gpu": 0,
                "gpu_mem_mb": 0,
                "exclusive": True,
                "gpu_exclusive": False,
                "locks": [],
                "max_runtime_s": 600,
                "submitted_at": NOW - 58,
                "blocked_reason": "waiting for resources; reserved to start by 1728s from now",
            }
        ],
        "reservation": {"job_id": 42, "start_at": NOW + 1728, "in_seconds": 1728.0, "external": False},
        "paused": False,
        "draining": False,
        "free_disk_mb": 169920,
        "disk_floor_mb": 5120,
        "load": (4.38, 4.0, 3.9),
        "active_leases": 0,
    }


def _text(payload) -> str:
    console = Console(record=True, width=140, force_terminal=False, color_system=None)
    console.print(render(payload, now=NOW))
    return console.export_text()


def test_frame_shows_capacity_running_and_queue():
    out = _text(_payload())
    assert "8 ajs + 3 outside of 20" in out
    assert "8.0G ajs + 4.0G outside of 54.7G" in out
    assert "300M outside" in out  # vram row appears because the machine has a card
    assert "11.2/8" in out  # actual over declared, and over budget
    assert "3.0G/8.0G" in out
    assert "1m12s/30m00s" in out
    assert "cargo test --release" in out
    assert "waiting for resources; reserved" in out
    assert "starts in ~28m48s" in out


def test_over_budget_cpu_is_highlighted():
    console = Console(record=True, width=140, force_terminal=True, color_system="standard")
    console.print(render(_payload(), now=NOW))
    styled = console.export_text(styles=True)
    assert "11.2/8" in styled
    assert "\x1b[31m" in styled  # red


def test_idle_machine_renders_without_error():
    payload = _payload()
    payload.update(running=[], queued=[], reservation=None, used={"cpu": 0, "mem_mb": 0, "gpu": 0, "gpu_mem_mb": 0})
    out = _text(payload)
    assert "nothing running" in out
    assert "queue empty" in out


def test_status_from_an_older_daemon_lacks_live_usage():
    """`cpu_now` arrives only from a daemon running the current code; show a dash."""
    payload = _payload()
    del payload["running"][0]["cpu_now"]
    del payload["running"][0]["mem_now_mb"]
    out = _text(payload)
    assert "-/8" in out


def test_formatters():
    assert fmt_secs(5) == "5s"
    assert fmt_secs(65) == "1m05s"
    assert fmt_secs(3720) == "1h02m"
    assert fmt_mem(512) == "512M"
    assert fmt_mem(8192) == "8.0G"
    assert bar([(10, "green")], 20, width=10).plain == "█████░░░░░"
    assert bar([(30, "green")], 20, width=10).plain == "██████████"  # never overflows


def test_measured_breakdown_names_outside_processes():
    payload = _payload()
    payload["measured"] = {
        "cpu_ajs": 4.0,
        "cpu_outside": 5.5,
        "cpu_allowance": 2.0,
        "iowait_pct": 11.0,
        "mem_total_mb": 64000,
        "mem_used_mb": 30000,
        "mem_ajs_mb": 12000,
        "mem_outside_mb": 18000,
        "mem_reserve_mb": 6144,
        "quiet": False,
        "noise": "iowait 11%",
        "top_cpu": [{"pid": 42, "name": "cc1", "cwd": "~/Repos/tacular-omics", "cores": 3.0, "rss_mb": 200}],
        "top_mem": [{"pid": 42, "name": "cc1", "cwd": "~/Repos/tacular-omics", "cores": 3.0, "rss_mb": 200}],
    }
    payload["running"][0]["interference"] = ["12:21:00 cc1"]
    out = _text(payload)
    assert "4.0 ajs + 5.5 outside" in out
    assert "not quiet" in out and "iowait 11%" in out
    assert "cc1" in out and "~/Repos/tacular-omics" in out
    assert out.count("42") >= 1
    assert "!1" in out
