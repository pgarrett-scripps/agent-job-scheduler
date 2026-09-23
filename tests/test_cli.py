def test_inbox_lines_name_the_problem_and_skip_stalls_already_reported():
    from ajs.cli import inbox_lines

    box = {
        "finished": [
            {"id": 3, "title": "sweep", "state": "failed", "exit_code": 2, "log_path": "/l/3"},
            {"id": 4, "title": "bench", "state": "done", "exit_code": 0, "contended": 1, "contention_note": "x"},
        ],
        "events": [{"job_id": 5, "action": "hold", "actor": "claude:abcd-ef", "detail": "", "reason": "later"}],
        "stalled": [{"id": 6, "title": "t", "log_quiet_s": 1200, "log_path": "/l/6"}],
    }
    lines = inbox_lines(box, already_stalled={6})
    assert lines[0] == "job 3 sweep: failed (exit 2) - log /l/3"
    assert "CONTENDED" in lines[1]
    assert lines[2] == "job 5: hold by claude:abcd: later"
    assert len(lines) == 3
