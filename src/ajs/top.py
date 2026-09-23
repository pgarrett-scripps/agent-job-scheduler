"""`ajs top`: a live view of the machine, the running jobs, and the queue.

`ajs status` is a snapshot for scripts and agents. This is for a person watching the
box: capacity bars that show ajs's share against what is running outside it, declared
versus actual usage per job, and -- the part that is otherwise invisible -- what each
queued job is waiting for.

Rendering is a pure function of the status payload so it can be tested without a daemon.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from .models import display_name

BAR_WIDTH = 30


def fmt_secs(seconds: float | None) -> str:
    """12s / 3m05s / 1h02m -- compact enough for a table cell."""
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def fmt_mem(mb: float | None) -> str:
    if mb is None:
        return "-"
    return f"{mb / 1024:.1f}G" if mb >= 1024 else f"{int(mb)}M"


def bar(segments: list[tuple[float, str]], total: float, width: int = BAR_WIDTH) -> Text:
    """Stacked bar. ``segments`` are (amount, style) drawn left to right; the remainder
    is drawn dim as free."""
    text = Text()
    filled = 0
    for amount, style in segments:
        cells = int(round(width * amount / total)) if total > 0 else 0
        cells = max(0, min(cells, width - filled))
        text.append("█" * cells, style=style)
        filled += cells
    text.append("░" * (width - filled), style="dim")
    return text


def _capacity_rows(data: dict[str, Any]) -> Table:
    cap, used, ext = data["capacity"], data["used"], data.get("external") or {}
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_column()

    def row(label: str, key: str, fmt) -> None:
        total = cap.get(key, 0)
        if key in ("gpu", "gpu_mem_mb") and not total:
            return
        own = used.get(key, 0)
        outside = ext.get(key, 0)
        legend = f"{fmt(own)} ajs"
        if outside:
            legend += f" + {fmt(outside)} outside"
        legend += f" of {fmt(total)}"
        table.add_row(label, bar([(own, "green"), (outside, "yellow")], total), legend)

    row("cpu", "cpu", lambda v: f"{v:g}")
    row("mem", "mem_mb", fmt_mem)
    row("vram", "gpu_mem_mb", fmt_mem)
    return table


def _measured_rows(data: dict[str, Any]) -> RenderableType | None:
    """Real usage, ajs jobs against everything else, plus who the outside load is.

    The capacity bars above show *reservations*; these show what the kernel says is
    actually busy, which is what a timing run or an OOM cares about.
    """
    m = data.get("measured")
    if not m or m.get("cpu_outside") is None:
        return None
    cap = data["capacity"]
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_column()
    out_cpu, allow = m["cpu_outside"], m["cpu_allowance"]
    table.add_row(
        "cpu real",
        bar([(m["cpu_ajs"], "green"), (out_cpu, "yellow" if out_cpu > allow + 1 else "dim yellow")], cap["cpu"]),
        f"{m['cpu_ajs']:.1f} ajs + {out_cpu:.1f} outside (allowance {allow:g})   iowait {m['iowait_pct']:.0f}%",
    )
    if m.get("mem_used_mb") is not None:
        out_mem = m["mem_outside_mb"]
        style = "yellow" if out_mem > m["mem_reserve_mb"] else "dim yellow"
        table.add_row(
            "mem real",
            bar([(m["mem_ajs_mb"], "green"), (out_mem, style)], m["mem_total_mb"]),
            f"{fmt_mem(m['mem_ajs_mb'])} ajs + {fmt_mem(out_mem)} outside of {fmt_mem(m['mem_total_mb'])}",
        )
    if m.get("quiet"):
        table.add_row("timing", Text("quiet", style="green"), "")
    else:
        table.add_row("timing", Text("not quiet", style="yellow"), m.get("noise") or "")

    procs: dict[int, dict[str, Any]] = {}
    for row in (m.get("top_cpu") or []) + (m.get("top_mem") or []):
        procs[row["pid"]] = row
    if not procs:
        return table
    outside = Table(title="outside ajs", title_justify="left", header_style="bold", expand=True)
    outside.add_column("pid", justify="right", no_wrap=True)
    outside.add_column("process", no_wrap=True)
    outside.add_column("cpu", justify="right", no_wrap=True)
    outside.add_column("mem", justify="right", no_wrap=True)
    outside.add_column("where", overflow="ellipsis", ratio=1)
    for row in sorted(procs.values(), key=lambda r: (r["cores"], r["rss_mb"]), reverse=True):
        outside.add_row(
            str(row["pid"]), row["name"], f"{row['cores']:.1f}", fmt_mem(row["rss_mb"]), row.get("cwd") or "-"
        )
    return Group(table, outside)


def _header(data: dict[str, Any]) -> Text:
    text = Text()
    load = data.get("load") or (0.0, 0.0, 0.0)
    text.append(f"load {load[0]:.2f}", style="bold")
    disk_bad = data["free_disk_mb"] < data["disk_floor_mb"]
    text.append(f"   disk {fmt_mem(data['free_disk_mb'])} free", style="red" if disk_bad else "")
    if data.get("active_leases"):
        text.append(f"   {data['active_leases']} lease(s)")
    if data.get("paused"):
        text.append("   PAUSED", style="bold yellow")
    if data.get("draining"):
        text.append("   DRAINING", style="bold yellow")
    return text


def _running_table(jobs: list[dict[str, Any]], now: float) -> RenderableType:
    if not jobs:
        return Text("nothing running", style="dim")
    table = Table(title=f"running ({len(jobs)})", title_justify="left", header_style="bold", expand=True)
    table.add_column("id", justify="right", no_wrap=True)
    table.add_column("project", no_wrap=True)
    table.add_column("class", no_wrap=True)
    table.add_column("cpu now/decl", justify="right", no_wrap=True)
    table.add_column("mem now/decl", justify="right", no_wrap=True)
    table.add_column("elapsed/max", justify="right", no_wrap=True)
    table.add_column("job", overflow="ellipsis", ratio=1)
    for job in jobs:
        tags = ""
        if job.get("exclusive"):
            tags += " [magenta]X[/magenta]"
        if job.get("gpu_exclusive"):
            tags += " [magenta]GX[/magenta]"
        elif job.get("gpu_mem_mb"):
            tags += f" [cyan]{fmt_mem(job['gpu_mem_mb'])} vram[/cyan]"
        if job.get("interference"):
            # A timing run something outside ajs has disturbed; `ajs job N` says what.
            tags += f" [yellow]!{len(job['interference'])}[/yellow]"
        cpu_now = job.get("cpu_now")
        cpu_cell = f"{cpu_now:.1f}/{job['cpu']}" if cpu_now is not None else f"-/{job['cpu']}"
        if cpu_now is not None and cpu_now > job["cpu"] + 0.5:
            cpu_cell = f"[red]{cpu_cell}[/red]"
        mem_cell = f"{fmt_mem(job.get('mem_now_mb'))}/{fmt_mem(job['mem_mb'])}"
        elapsed = job.get("runtime_s")
        if elapsed is None and job.get("started_at"):
            elapsed = now - job["started_at"]
        table.add_row(
            f"{job['id']}{tags}",
            job["project"],
            job.get("class", "batch"),
            cpu_cell,
            mem_cell,
            f"{fmt_secs(elapsed)}/{fmt_secs(job['max_runtime_s'])}",
            display_name(job),
        )
    return table


def _queued_table(jobs: list[dict[str, Any]], now: float) -> RenderableType:
    if not jobs:
        return Text("queue empty", style="dim")
    table = Table(title=f"queued ({len(jobs)})", title_justify="left", header_style="bold", expand=True)
    table.add_column("id", justify="right", no_wrap=True)
    table.add_column("project", no_wrap=True)
    table.add_column("class", no_wrap=True)
    table.add_column("needs", no_wrap=True)
    table.add_column("waited", justify="right", no_wrap=True)
    table.add_column("waiting on", overflow="fold", ratio=2)
    table.add_column("job", overflow="ellipsis", ratio=1)
    for job in jobs:
        needs = f"{job['cpu']}cpu {fmt_mem(job['mem_mb'])}"
        if job.get("gpu_mem_mb"):
            needs += f" {fmt_mem(job['gpu_mem_mb'])}vram"
        elif job.get("gpu"):
            needs += " gpu"
        if job.get("exclusive"):
            needs += " [magenta]X[/magenta]"
        if job.get("gpu_exclusive"):
            needs += " [magenta]GX[/magenta]"
        if job.get("locks"):
            needs += f" lock:{','.join(job['locks'])}"
        table.add_row(
            str(job["id"]),
            job["project"],
            job.get("class", "batch"),
            needs,
            fmt_secs(now - job["submitted_at"]),
            job.get("blocked_reason") or "-",
            display_name(job),
        )
    return table


def _reservation_line(data: dict[str, Any]) -> Text | None:
    res = data.get("reservation")
    if not res:
        return None
    if res.get("external"):
        return Text(f"reservation: job {res['job_id']} waiting on load outside ajs (no ETA)", style="cyan")
    return Text(f"reservation: job {res['job_id']} starts in ~{fmt_secs(max(0, res['in_seconds']))}", style="cyan")


def render(data: dict[str, Any], now: float | None = None) -> RenderableType:
    """One frame of `ajs top` from a `status` payload."""
    now = time.time() if now is None else now
    parts: list[RenderableType] = [
        _header(data),
        _capacity_rows(data),
        *([measured] if (measured := _measured_rows(data)) is not None else []),
        Text(""),
        _running_table(data.get("running") or [], now),
        Text(""),
        _queued_table(data.get("queued") or [], now),
    ]
    line = _reservation_line(data)
    if line is not None:
        parts.append(line)
    parts.append(Text(time.strftime("%H:%M:%S", time.localtime(now)) + "   ctrl-c to quit", style="dim"))
    return Group(*parts)
