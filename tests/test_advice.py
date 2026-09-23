from ajs.advice import submission_warnings


def test_ordinary_job_gets_no_warnings():
    assert submission_warnings(exclusive=False, gpu_exclusive=False, max_runtime_s=1800, title="t") == []


def test_exclusive_warns():
    (w,) = submission_warnings(exclusive=True, gpu_exclusive=False, max_runtime_s=600, title="t")
    assert w.startswith("exclusive")


def test_gpu_exclusive_warns():
    (w,) = submission_warnings(exclusive=False, gpu_exclusive=True, max_runtime_s=600, title="t")
    assert w.startswith("gpu_exclusive")


def test_long_runtime_suggests_chunks():
    (w,) = submission_warnings(exclusive=False, gpu_exclusive=False, max_runtime_s=4 * 3600, title="t")
    assert "240 min" in w and "chunks" in w


def test_one_hour_is_not_long():
    assert submission_warnings(exclusive=False, gpu_exclusive=False, max_runtime_s=3600, title="t") == []


def test_missing_title_warns():
    (w,) = submission_warnings(exclusive=False, gpu_exclusive=False, max_runtime_s=600)
    assert w.startswith("no title")
