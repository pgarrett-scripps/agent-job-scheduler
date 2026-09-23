"""Submit-time advice: warn, never refuse, when a request will hold up other jobs.

A job without a title shows up in the queue as a bare command line, which tells whoever
manages the queue nothing about what it is for.

Two patterns bottleneck the shared machine. An `exclusive` job waits for every other job
to drain and then runs alone, so using it for throughput stalls everyone. A job with a
ceiling of hours holds its slots the whole time, so nothing can start between its steps.
"""

from __future__ import annotations

LONG_RUNTIME_S = 3600


def submission_warnings(
    *, exclusive: bool, gpu_exclusive: bool, max_runtime_s: int, title: str | None = None
) -> list[str]:
    warnings = []
    if not title:
        warnings.append(
            "no title. Pass a short title and a one-line description (what the job is for, "
            "what it unblocks) so whoever manages the queue can tell what it is."
        )
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
