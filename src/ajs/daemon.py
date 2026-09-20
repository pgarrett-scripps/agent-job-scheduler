"""The singleton daemon: socket server plus scheduler loop.

Exactly one of these runs per machine. Everything else -- CLI, MCP servers, shell
scripts -- is a stateless client. That singleton property is the whole point: it is the
only way a decision about "is the machine busy" can be authoritative.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import protocol
from .config import Config, ensure_dirs, runtime_dir, socket_path, state_dir
from .engine import Engine
from .models import JobState, ResourceRequest

log = logging.getLogger("ajs.daemon")


class Server:
    """Dispatches wire requests onto the engine."""

    def __init__(self, engine: Engine, on_shutdown: Callable[[], None] | None = None) -> None:
        self.engine = engine
        self._on_shutdown = on_shutdown

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            try:
                msg = protocol.decode(line)
            except json.JSONDecodeError as exc:
                writer.write(protocol.encode(protocol.err(f"malformed request: {exc}")))
                await writer.drain()
                return

            method = msg.get("method", "")
            params = msg.get("params", {}) or {}
            handler = getattr(self, f"do_{method}", None)
            if handler is None:
                response = protocol.err(f"unknown method: {method}")
            else:
                try:
                    result = handler(**params)
                    if asyncio.iscoroutine(result):
                        result = await result
                    response = protocol.ok(result)
                except TypeError as exc:
                    response = protocol.err(f"bad parameters for {method}: {exc}")
                except Exception as exc:  # pragma: no cover - surface to the caller
                    log.exception("error handling %s", method)
                    response = protocol.err(f"{type(exc).__name__}: {exc}")

            writer.write(protocol.encode(response))
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    # --- methods ----------------------------------------------------------

    def do_ping(self) -> str:
        return "pong"

    def do_shutdown(self) -> bool:
        """Ask the daemon to exit.

        Done over the socket rather than by matching process names: the only process that
        can answer here is the one actually holding the socket, so there is no way to kill
        the wrong thing.
        """
        if self._on_shutdown is not None:
            self._on_shutdown()
        return True

    def do_submit(self, **params: Any) -> dict[str, Any]:
        job = self.engine.submit(**params)
        return job.to_dict()

    def do_status(self) -> dict[str, Any]:
        return self.engine.status()

    def do_job(self, job_id: int) -> dict[str, Any]:
        job = self.engine.store.get_job(int(job_id))
        if job is None:
            raise KeyError(f"no such job: {job_id}")
        data = job.to_dict()
        data["blocked_reason"] = self.engine.blocked_reason(job.id)
        return data

    def do_jobs(
        self,
        project: str | None = None,
        states: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        parsed = [JobState(s) for s in states] if states else None
        return [j.to_dict() for j in self.engine.store.list_jobs(project=project, states=parsed, limit=limit)]

    async def do_wait(self, job_id: int, timeout: float = 60.0) -> dict[str, Any] | None:
        job = await self.engine.wait(int(job_id), float(timeout))
        if job is None:
            raise KeyError(f"no such job: {job_id}")
        data = job.to_dict()
        data["blocked_reason"] = self.engine.blocked_reason(job.id)
        data["timed_out"] = not job.state.is_terminal
        return data

    async def do_cancel(self, job_id: int, reason: str = "cancelled by user") -> bool:
        return await self.engine.cancel(int(job_id), reason)

    def do_logs(self, job_id: int, lines: int = 50) -> str:
        return self.engine.log_tail(int(job_id), int(lines))

    def do_pause(self) -> bool:
        self.engine.paused = True
        return True

    def do_resume(self) -> bool:
        self.engine.paused = False
        self.engine.draining = False
        self.engine.wake()
        return True

    def do_drain(self) -> bool:
        self.engine.draining = True
        return True

    def do_acquire_lease(
        self,
        project: str,
        session_id: str = "",
        cpu: int = 1,
        mem_mb: int = 512,
        gpu: int = 0,
        gpu_mem_mb: int = 0,
        reason: str = "",
    ) -> str | None:
        return self.engine.acquire_lease(
            project,
            session_id,
            ResourceRequest(cpu=cpu, mem_mb=mem_mb, gpu=gpu, gpu_mem_mb=gpu_mem_mb),
            reason,
        )

    def do_heartbeat_lease(self, lease_id: str) -> bool:
        return self.engine.store.heartbeat_lease(lease_id)

    def do_release_lease(self, lease_id: str) -> bool:
        released = self.engine.store.release_lease(lease_id)
        self.engine.wake()
        return released

    def do_config(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self.engine.cfg)


async def serve(cfg: Config) -> None:
    ensure_dirs()
    path = socket_path()

    if path.exists():
        # A live daemon already owns this socket; refuse rather than steal it, since two
        # schedulers would each hand out the machine's full capacity.
        from .client import Client

        if Client(path).is_running():
            raise SystemExit(f"a scheduler daemon is already running on {path}")
        log.warning("removing stale socket %s", path)
        path.unlink()

    engine = Engine(cfg)
    loop = asyncio.get_running_loop()
    stop: asyncio.Future[None] = loop.create_future()

    def request_stop() -> None:
        if not stop.done():
            loop.call_soon(lambda: stop.done() or stop.set_result(None))

    server = Server(engine, on_shutdown=request_stop)

    unix_server = await asyncio.start_unix_server(server.handle, path=str(path))
    os.chmod(path, 0o600)
    log.info("listening on %s", path)
    log.info(
        "capacity: cpu=%s mem=%sMB gpu=%s vram=%sMB | disk floor %sMB | settle %ss",
        cfg.cpu,
        cfg.mem_mb,
        cfg.gpu,
        cfg.gpu_mem_mb,
        cfg.disk_floor_mb,
        cfg.settle_seconds,
    )

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, request_stop)

    scheduler_task = asyncio.create_task(engine.run())
    try:
        await stop
    finally:
        log.info("shutting down")
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task
        unix_server.close()
        await unix_server.wait_closed()
        await engine.shutdown()
        with contextlib.suppress(OSError):
            path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ajsd", description="agent job scheduler daemon")
    parser.add_argument("--foreground", action="store_true", help="log to stderr (default under systemd)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if not args.foreground:
        handlers.append(logging.FileHandler(state_dir() / "ajsd.log"))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
    )

    cfg = Config.load()
    if not Path(str(runtime_dir())).exists():  # pragma: no cover
        ensure_dirs()

    try:
        asyncio.run(serve(cfg))
    except KeyboardInterrupt:  # pragma: no cover
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
