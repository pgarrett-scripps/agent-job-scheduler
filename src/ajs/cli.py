"""Command line interface.

The universal front door: works from a shell, from Codex, from Claude Code's Bash tool,
from cron, from a git hook. The MCP server exists to give agents structured tools, but
anything that can run a command can use the scheduler through this.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from . import protocol
from .advice import submission_warnings
from .client import Client
from .config import Config, config_path, socket_path, state_dir

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Schedule jobs so several agents can share one machine without colliding.",
)
daemon_app = typer.Typer(no_args_is_help=True, help="Control the scheduler daemon.")
app.add_typer(daemon_app, name="daemon")

console = Console()
err_console = Console(stderr=True)


def detect_project(cwd: Path | None = None) -> str:
    """Name the submitting project, so fair-share and `ajs ps` mean something.

    Uses the git repository root when there is one, since that is the unit agents
    actually work in; otherwise the directory name.
    """
    cwd = cwd or Path.cwd()
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip()).name
    except (OSError, subprocess.SubprocessError):
        pass
    return cwd.name


def parse_duration(value: str) -> int:
    """Accept 30s / 10m / 2h / 90 (bare seconds)."""
    value = value.strip().lower()
    if not value:
        raise ValueError("empty duration")
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if value[-1] in units:
        return int(float(value[:-1]) * units[value[-1]])
    return int(float(value))


def parse_mem(value: str) -> int:
    """Accept 512 / 512M / 8G, returning MB."""
    value = value.strip().upper()
    if not value:
        raise ValueError("empty size")
    units = {"M": 1, "G": 1024}
    if value[-1] in units:
        return int(float(value[:-1]) * units[value[-1]])
    if value[-1].isalpha():
        raise ValueError(f"unknown size unit in {value!r}; use M or G")
    return int(float(value))


#: Environment the submitter's shell has that the daemon's does not. The daemon runs as
#: a systemd user service with a minimal environment, so without this a job cannot find
#: cargo, uv, nvm-managed node or an activated virtualenv, and agents end up wrapping
#: every command in `bash -c "export PATH=...; ..."`. Deliberately an allowlist rather
#: than the whole environment: job records are persisted to the database, and API keys
#: do not belong there.
FORWARDED_ENV = (
    "PATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONPATH",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "GOPATH",
    "NVM_DIR",
    "JAVA_HOME",
    "LANG",
    "LC_ALL",
)


def forwarded_env(source: Mapping[str, str] | None = None, extra: list[str] | None = None) -> dict[str, str]:
    """Build the environment a job inherits from its submitter.

    ``extra`` entries are ``KEY=VALUE`` or bare ``KEY`` (copied from ``source``).
    """
    src: Mapping[str, str] = os.environ if source is None else source
    env = {k: src[k] for k in FORWARDED_ENV if k in src}
    for item in extra or []:
        key, sep, value = item.partition("=")
        if not key:
            raise ValueError(f"bad --env entry: {item!r}")
        if sep:
            env[key] = value
        elif key in src:
            env[key] = src[key]
    return env


def _client() -> Client:
    return Client()


def _fail(message: str) -> None:
    err_console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(1)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def submit(
    ctx: typer.Context,
    cpu: Annotated[int, typer.Option("--cpu", "-c", help="CPU slots required.")] = 1,
    mem: Annotated[str, typer.Option("--mem", "-m", help="Memory, e.g. 8G or 512M.")] = "512M",
    gpu: Annotated[int, typer.Option("--gpu", help="GPUs required.")] = 0,
    gpu_mem: Annotated[
        str,
        typer.Option("--gpu-mem", help="VRAM required, e.g. 2G. Defaults to the whole card when --gpu is set."),
    ] = "0M",
    gpu_exclusive: Annotated[
        bool,
        typer.Option("--gpu-exclusive", "-X", help="GPU timing run: take the whole card. Independent of -x."),
    ] = False,
    disk: Annotated[str, typer.Option("--disk", help="Disk this job will write, e.g. 20G.")] = "0M",
    exclusive: Annotated[
        bool,
        typer.Option("--exclusive", "-x", help="Timing run: take the whole machine, alone and settled."),
    ] = False,
    lock: Annotated[list[str] | None, typer.Option("--lock", "-l", help="Named exclusive lock(s).")] = None,
    max_runtime: Annotated[
        str, typer.Option("--max-runtime", "-t", help="Hard ceiling, e.g. 30m. Required for backfill.")
    ] = "1h",
    job_class: Annotated[str, typer.Option("--class", help="interactive | batch | background")] = "batch",
    project: Annotated[str | None, typer.Option("--project", "-p", help="Defaults to the git repo name.")] = None,
    session: Annotated[str, typer.Option("--session", help="Opaque submitter id.")] = "",
    env: Annotated[
        list[str] | None,
        typer.Option("--env", "-e", help="Extra environment for the job: KEY=VALUE, or KEY to copy from your shell."),
    ] = None,
    wait: Annotated[bool, typer.Option("--wait", "-w", help="Block until the job finishes.")] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON instead of prose.")] = False,
) -> None:
    """Queue a command. Everything after `--` is the command to run.

    Example: ajs submit --cpu 8 --mem 8G -- cargo test --release
    """
    cmd = list(ctx.args)
    if not cmd:
        _fail("no command given. Put it after `--`, e.g. ajs submit --cpu 4 -- pytest tests")

    try:
        job_env = forwarded_env(extra=env)
        mem_mb, gpu_mem_mb, disk_mb = parse_mem(mem), parse_mem(gpu_mem), parse_mem(disk)
        max_runtime_s = parse_duration(max_runtime)
    except ValueError as exc:
        _fail(str(exc))
        return

    client = _client()
    try:
        job = client.submit(
            project=project or detect_project(),
            session_id=session or os.environ.get("AJS_SESSION", ""),
            cmd=cmd,
            cwd=str(Path.cwd()),
            env=job_env,
            cpu=cpu,
            mem_mb=mem_mb,
            gpu=gpu,
            gpu_mem_mb=gpu_mem_mb,
            disk_mb=disk_mb,
            exclusive=exclusive,
            gpu_exclusive=gpu_exclusive,
            locks=list(lock or []),
            max_runtime_s=max_runtime_s,
            job_class=job_class,
        )
    except protocol.SchedulerError as exc:
        _fail(str(exc))
        return

    for warning in submission_warnings(exclusive=exclusive, gpu_exclusive=gpu_exclusive, max_runtime_s=max_runtime_s):
        err_console.print(f"[yellow]note:[/yellow] {warning}")

    if not wait:
        if json_out:
            console.print_json(data=job)
        else:
            console.print(f"[green]queued[/green] job [bold]{job['id']}[/bold]: {job['cmd_str']}")
            console.print(f"  follow with: [dim]ajs wait {job['id']}[/dim]")
        return

    result = _wait_loop(client, int(job["id"]))
    _report_finished(result, json_out)


def _wait_loop(client: Client, job_id: int, poll: float = 60.0) -> dict[str, Any]:
    """Long-poll until terminal.

    The daemon holds each request open, so this makes roughly one call per minute rather
    than spinning -- which matters when the caller is an agent paying per token.
    """
    while True:
        result = client.wait(job_id, timeout=poll)
        if not result.get("timed_out"):
            return result


def _report_finished(job: dict[str, Any], json_out: bool) -> None:
    if json_out:
        console.print_json(data=job)
        raise typer.Exit(0 if job.get("exit_code") == 0 else 1)

    state = job.get("state")
    colour = {"done": "green", "failed": "red", "timeout": "yellow", "cancelled": "yellow"}.get(str(state), "white")
    runtime = job.get("runtime_s") or 0.0
    console.print(f"[{colour}]{state}[/{colour}] job {job['id']} in {runtime:.1f}s (exit={job.get('exit_code')})")
    if job.get("exclusive"):
        note = job.get("contention_note")
        if job.get("contended"):
            console.print(f"[yellow]  CONTENDED:[/yellow] {note}")
            console.print("[yellow]  this timing is not trustworthy; rerun it[/yellow]")
        elif note:
            console.print(f"[green]  clean:[/green] {note}")
    raise typer.Exit(0 if job.get("exit_code") == 0 else 1)


@app.command()
def wait(
    job_id: Annotated[int, typer.Argument(help="Job id.")],
    timeout: Annotated[str, typer.Option("--timeout", "-t", help="Give up after, e.g. 10m.")] = "0",
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Block until a job reaches a terminal state."""
    client = _client()
    limit = parse_duration(timeout)
    deadline = time.time() + limit if limit else None
    try:
        while True:
            result = client.wait(job_id, timeout=min(60.0, limit) if limit else 60.0)
            if not result.get("timed_out"):
                _report_finished(result, json_out)
                return
            if deadline and time.time() >= deadline:
                reason = result.get("blocked_reason")
                console.print(f"[yellow]still {result['state']}[/yellow]" + (f": {reason}" if reason else ""))
                raise typer.Exit(2)
    except protocol.SchedulerError as exc:
        _fail(str(exc))


@app.command()
def cancel(
    job_id: Annotated[int, typer.Argument()],
    reason: Annotated[str, typer.Option("--reason", "-r")] = "cancelled by user",
) -> None:
    """Cancel a queued or running job."""
    try:
        if _client().cancel(job_id, reason):
            console.print(f"[yellow]cancelled[/yellow] job {job_id}")
        else:
            _fail(f"job {job_id} is not cancellable (already finished, or does not exist)")
    except protocol.SchedulerError as exc:
        _fail(str(exc))


@app.command()
def logs(
    job_id: Annotated[int, typer.Argument()],
    lines: Annotated[int, typer.Option("--lines", "-n")] = 50,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Tail until the job exits.")] = False,
) -> None:
    """Show a job's captured output."""
    client = _client()
    try:
        if not follow:
            console.print(client.logs(job_id, lines))
            return
        seen = ""
        while True:
            job = client.job(job_id)
            text = client.logs(job_id, 10_000)
            if text != seen:
                sys.stdout.write(text[len(seen) :])
                sys.stdout.flush()
                seen = text
            if job["state"] in {"done", "failed", "cancelled", "timeout"}:
                return
            time.sleep(1.0)
    except protocol.SchedulerError as exc:
        _fail(str(exc))
    except KeyboardInterrupt:
        return


@app.command(name="status")
def status_cmd(json_out: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Show capacity, running jobs, and why queued jobs are waiting."""
    try:
        data = _client().status()
    except protocol.SchedulerError as exc:
        _fail(str(exc))
        return

    if json_out:
        console.print_json(data=data)
        return

    cap, used = data["capacity"], data["used"]
    flags = []
    if data["paused"]:
        flags.append("[yellow]PAUSED[/yellow]")
    if data["draining"]:
        flags.append("[yellow]DRAINING[/yellow]")
    header = "  ".join(flags)
    console.print(
        f"[bold]cpu[/bold] {used['cpu']}/{cap['cpu']}   "
        f"[bold]mem[/bold] {used['mem_mb']}/{cap['mem_mb']} MB   "
        f"[bold]gpu[/bold] {used['gpu']}/{cap['gpu']} "
        f"({used.get('gpu_mem_mb', 0)}/{cap.get('gpu_mem_mb', 0)} MB)   "
        f"[bold]load[/bold] {data['load'][0]:.2f}   {header}"
    )
    ext = data.get("external") or {}
    if ext.get("gpu_mem_mb"):
        console.print(f"[yellow]outside ajs[/yellow] {ext['gpu_mem_mb']} MB VRAM held by non-ajs processes")
    if ext.get("cpu") or ext.get("mem_mb"):
        # Shown separately from `used` so it is obvious these cores are not ajs's doing
        # and will not be freed by cancelling a job.
        console.print(
            f"[dim]outside ajs {ext.get('cpu', 0)} cpu, {ext.get('mem_mb', 0)} MB "
            f"beyond the desktop allowance; timing runs ignore this[/dim]"
        )
    disk_colour = "red" if data["free_disk_mb"] < data["disk_floor_mb"] else "dim"
    console.print(
        f"[{disk_colour}]disk {data['free_disk_mb']} MB free (floor {data['disk_floor_mb']} MB)[/{disk_colour}]"
    )
    if data.get("active_leases"):
        console.print(f"[dim]{data['active_leases']} active lease(s)[/dim]")

    if data["running"]:
        table = Table(title="running", title_justify="left", header_style="bold")
        table.add_column("id", justify="right")
        table.add_column("project")
        table.add_column("cpu", justify="right")
        table.add_column("mem", justify="right")
        table.add_column("elapsed", justify="right")
        table.add_column("command", overflow="fold")
        for job in data["running"]:
            tag = " [magenta]X[/magenta]" if job["exclusive"] else ""
            table.add_row(
                str(job["id"]) + tag,
                job["project"],
                str(job["cpu"]),
                f"{job['mem_mb']}M",
                f"{job['runtime_s'] or 0:.0f}s",
                job["cmd_str"],
            )
        console.print(table)

    if data["queued"]:
        table = Table(title="queued", title_justify="left", header_style="bold")
        table.add_column("id", justify="right")
        table.add_column("project")
        table.add_column("needs")
        table.add_column("waiting on", overflow="fold")
        for job in data["queued"]:
            needs = f"{job['cpu']}cpu/{job['mem_mb']}M"
            if job.get("gpu_mem_mb"):
                needs += f"/{job['gpu_mem_mb']}M vram"
            if job["exclusive"]:
                needs += "/exclusive"
            if job.get("gpu_exclusive"):
                needs += "/gpu-exclusive"
            table.add_row(
                str(job["id"]),
                job["project"],
                needs,
                job.get("blocked_reason") or "-",
            )
        console.print(table)

    res = data.get("reservation")
    if res:
        if res.get("external"):
            # Announcing "~0s" here would be a promise the scheduler cannot keep: what
            # blocks the job is load ajs does not control and cannot time.
            console.print(f"[cyan]reservation:[/cyan] job {res['job_id']} waiting on load outside ajs (no ETA)")
        else:
            console.print(f"[cyan]reservation:[/cyan] job {res['job_id']} starts in ~{max(0, res['in_seconds']):.0f}s")

    if not data["running"] and not data["queued"]:
        console.print("[dim]idle[/dim]")


@app.command(name="top")
def top_cmd(
    interval: Annotated[float, typer.Option("--interval", "-i", help="Seconds between refreshes.")] = 1.0,
    once: Annotated[bool, typer.Option("--once", help="Print one frame and exit.")] = False,
) -> None:
    """Live view: capacity bars, running jobs with actual vs declared usage, and the queue."""
    from rich.live import Live
    from rich.text import Text

    from .top import render

    client = _client()
    try:
        data = client.status()
    except protocol.SchedulerError as exc:
        _fail(str(exc))
        return
    if once:
        console.print(render(data))
        return

    try:
        with Live(render(data), console=console, screen=True, refresh_per_second=4) as live:
            while True:
                time.sleep(max(0.2, interval))
                try:
                    data = client.status()
                except protocol.SchedulerError as exc:
                    live.update(Text(f"error: {exc}\n(retrying)", style="red"))
                    continue
                live.update(render(data))
    except KeyboardInterrupt:
        return


@app.command(name="ps")
def ps_cmd(
    project: Annotated[str | None, typer.Option("--project", "-p")] = None,
    all_jobs: Annotated[bool, typer.Option("--all", "-a", help="Include finished jobs.")] = False,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List jobs, most recent first."""
    states = None if all_jobs else ["queued", "running"]
    try:
        jobs = _client().jobs(project=project, states=states, limit=limit)
    except protocol.SchedulerError as exc:
        _fail(str(exc))
        return

    if json_out:
        console.print_json(data=jobs)
        return
    if not jobs:
        console.print("[dim]no jobs[/dim]")
        return

    table = Table(header_style="bold")
    table.add_column("id", justify="right")
    table.add_column("state")
    table.add_column("project")
    table.add_column("time", justify="right")
    table.add_column("command", overflow="fold")
    colours = {"done": "green", "failed": "red", "running": "cyan", "queued": "yellow"}
    for job in jobs:
        state = job["state"]
        mark = " [yellow]!contended[/yellow]" if job.get("contended") else ""
        table.add_row(
            str(job["id"]),
            f"[{colours.get(state, 'white')}]{state}[/{colours.get(state, 'white')}]{mark}",
            job["project"],
            f"{job['runtime_s'] or 0:.0f}s",
            job["cmd_str"],
        )
    console.print(table)


@app.command()
def pause() -> None:
    """Stop starting new jobs. Running jobs continue."""
    _client().call("pause")
    console.print("[yellow]paused[/yellow] - no new jobs will start")


@app.command()
def resume() -> None:
    """Resume scheduling."""
    _client().call("resume")
    console.print("[green]resumed[/green]")


@app.command()
def drain() -> None:
    """Let running jobs finish, start nothing new."""
    _client().call("drain")
    console.print("[yellow]draining[/yellow] - running jobs will finish, none will start")


@daemon_app.command("start")
def daemon_start(
    foreground: Annotated[bool, typer.Option("--foreground", "-f")] = False,
) -> None:
    """Start the scheduler daemon."""
    if Client().is_running():
        console.print("[dim]daemon already running[/dim]")
        return
    if foreground:
        from .daemon import main as daemon_main

        raise typer.Exit(daemon_main(["--foreground"]))
    exe = Path(sys.argv[0]).parent / "ajsd"
    argv = [str(exe)] if exe.exists() else [sys.executable, "-m", "ajs.daemon"]
    subprocess.Popen(
        argv,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        time.sleep(0.1)
        if Client().is_running():
            console.print(f"[green]daemon started[/green] on {socket_path()}")
            return
    _fail(f"daemon did not come up; check {state_dir() / 'ajsd.log'}")


@daemon_app.command("stop")
def daemon_stop() -> None:
    """Stop the daemon. Running jobs are terminated."""
    client = Client()
    if not client.is_running():
        console.print("[dim]daemon is not running[/dim]")
        return
    try:
        client.call("shutdown")
    except protocol.SchedulerError:
        pass  # it may well close the connection before replying
    for _ in range(50):
        time.sleep(0.1)
        if not client.is_running():
            console.print("[yellow]daemon stopped[/yellow]")
            return
    _fail("daemon did not stop")


@daemon_app.command("status")
def daemon_status() -> None:
    """Is the daemon up?"""
    if Client().is_running():
        console.print(f"[green]running[/green] on {socket_path()}")
    else:
        console.print(f"[red]not running[/red] (no socket at {socket_path()})")
        raise typer.Exit(1)


@app.command(name="config")
def config_cmd(
    show_path: Annotated[bool, typer.Option("--path", help="Print the config file path.")] = False,
    init: Annotated[bool, typer.Option("--init", help="Write defaults to the config file.")] = False,
) -> None:
    """Show or initialise configuration."""
    if show_path:
        console.print(str(config_path()))
        return
    if init:
        cfg = Config.load()
        cfg.save()
        console.print(f"[green]wrote[/green] {config_path()}")
        return
    try:
        console.print_json(data=_client().call("config"))
    except protocol.SchedulerError:
        from dataclasses import asdict

        console.print_json(data=asdict(Config.load()))


@app.command(name="mcp-config")
def mcp_config(
    codex: Annotated[bool, typer.Option("--codex", help="Emit Codex TOML instead of Claude Code JSON.")] = False,
) -> None:
    """Print the MCP registration snippet for Claude Code or Codex."""
    exe = Path(sys.argv[0]).parent / "ajs-mcp"
    command = str(exe) if exe.exists() else "ajs-mcp"
    if codex:
        console.print("# add to ~/.codex/config.toml\n")
        console.print(f'[mcp_servers.ajs]\ncommand = "{command}"\nargs = []')
    else:
        console.print("# register with: claude mcp add ajs -- " + command)
        console.print("# or add to .mcp.json:\n")
        console.print(json.dumps({"mcpServers": {"ajs": {"command": command, "args": []}}}, indent=2))


service_app = typer.Typer(no_args_is_help=True, help="Manage the systemd user service.")
app.add_typer(service_app, name="service")

UNIT_TEMPLATE = """[Unit]
Description=Agent job scheduler
Documentation=https://github.com/pgarrett-scripps/agent_job_scheduler

[Service]
Type=simple
ExecStart={exec_start}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def _unit_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "systemd" / "user" / "ajsd.service"


@service_app.command("install")
def service_install(
    enable: Annotated[bool, typer.Option("--enable/--no-enable", help="Start now and on login.")] = True,
) -> None:
    """Write and enable a systemd user unit so the daemon survives logout."""
    exe = Path(sys.argv[0]).parent / "ajsd"
    exec_start = str(exe) if exe.exists() else f"{sys.executable} -m ajs.daemon"
    path = _unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(UNIT_TEMPLATE.format(exec_start=exec_start))
    console.print(f"[green]wrote[/green] {path}")

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    if enable:
        subprocess.run(["systemctl", "--user", "enable", "--now", "ajsd.service"], check=False)
        console.print("[green]enabled[/green] ajsd.service")
        console.print("[dim]tip: run `loginctl enable-linger $USER` to keep it up without a session[/dim]")


@service_app.command("uninstall")
def service_uninstall() -> None:
    """Disable and remove the systemd user unit."""
    subprocess.run(["systemctl", "--user", "disable", "--now", "ajsd.service"], check=False)
    path = _unit_path()
    if path.exists():
        path.unlink()
        console.print(f"[yellow]removed[/yellow] {path}")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)


@service_app.command("status")
def service_status() -> None:
    """Show the systemd unit's status."""
    subprocess.run(["systemctl", "--user", "status", "ajsd.service", "--no-pager"], check=False)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
