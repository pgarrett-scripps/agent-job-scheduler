"""Submit-time advice: warn, never refuse, when a request will hold up other jobs.

Two patterns bottleneck the shared machine. An `exclusive` job waits for every other job
to drain and then runs alone, so using it for throughput stalls everyone. A job with a
ceiling of hours holds its slots the whole time, so nothing can start between its steps.
"""

from __future__ import annotations

LONG_RUNTIME_S = 3600


def submission_warnings(*, exclusive: bool, gpu_exclusive: bool, max_runtime_s: int) -> list[str]:
    warnings = []
    if exclusive or gpu_exclusive:
        flag = "exclusive" if exclusive else "gpu_exclusive"
        warnings.append(
            f"{flag} waits for every other job to finish, then runs alone. Use it only for "
            "timing you will report. For throughput, drop it and request the cores you use."
        )
    if max_runtime_s > LONG_RUNTIME_S:
        warnings.append(
            f"max_runtime is {max_runtime_s // 60} min. A long job holds its slots the whole "
            "time. If it can be split (per file, sample or parameter set), submit resumable "
            "chunks of about 30 min each so other jobs can start in between."
        )
    return warnings
