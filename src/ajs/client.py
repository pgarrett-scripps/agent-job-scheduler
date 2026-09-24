"""Thin, stateless client for the daemon.

Both the CLI and the MCP server sit on top of this. Neither holds scheduler state: every
Claude Code window and every Codex session spawns its own MCP server process, so any
state kept here would fragment into N schedulers that each believe they own the machine.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from . import protocol
from .config import socket_path


class Client:
    """Synchronous request/response against the daemon socket."""

    def __init__(self, path: Path | str | None = None, timeout: float = 30.0) -> None:
        self.path = Path(path) if path else socket_path()
        self.timeout = timeout

    def is_running(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(2.0)
                sock.connect(str(self.path))
            return True
        except OSError:
            return False

    def call(self, method: str, *, _socket_timeout: float | None = None, **params: Any) -> Any:
        """Send one request and return its result, raising SchedulerError on failure.

        ``_socket_timeout`` is the transport deadline and is never sent to the daemon;
        it is underscored to keep it from colliding with a ``timeout`` parameter that
        some methods (notably ``wait``) legitimately take.
        """
        if not self.path.exists():
            raise protocol.SchedulerError(
                f"scheduler daemon is not running (no socket at {self.path}).\nStart it with: ajs daemon start"
            )
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(_socket_timeout if _socket_timeout is not None else self.timeout)
                sock.connect(str(self.path))
                sock.sendall(protocol.encode(protocol.request(method, **params)))
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
        except OSError as exc:
            raise protocol.SchedulerError(f"could not talk to daemon: {exc}") from exc

        if not buf.strip():
            raise protocol.SchedulerError("daemon closed the connection without responding")

        resp = protocol.decode(buf)
        if not resp.get("ok"):
            raise protocol.SchedulerError(str(resp.get("error", "unknown error")))
        return resp.get("result")

    # --- convenience wrappers --------------------------------------------

    def submit(self, **params: Any) -> dict[str, Any]:
        try:
            return self.call("submit", **params)
        except protocol.SchedulerError as exc:
            described = {k: params[k] for k in ("title", "description", "meta") if k in params}
            if not described or "unexpected keyword argument" not in str(exc):
                raise
        # The daemon predates title/description/meta: fold them into its `note` field so a
        # client upgrade never has to wait for a daemon restart.
        folded = {k: v for k, v in params.items() if k not in described}
        meta = " ".join(f"{k}={v}" for k, v in (described.get("meta") or {}).items())
        folded["note"] = " | ".join(
            str(p) for p in (described.get("title"), described.get("description") or params.get("note"), meta) if p
        )
        return self.call("submit", **folded)

    def status(self) -> dict[str, Any]:
        return self.call("status")

    def job(self, job_id: int) -> dict[str, Any]:
        return self.call("job", job_id=job_id)

    def wait(self, job_id: int, timeout: float = 60.0) -> dict[str, Any]:
        # Allow the socket a margin beyond the server-side long-poll deadline, or the
        # client would time out first and look like a daemon failure.
        return self.call("wait", _socket_timeout=timeout + 15.0, job_id=job_id, timeout=timeout)

    def cancel(self, job_id: int, reason: str = "cancelled by user") -> bool:
        return bool(self.call("cancel", job_id=job_id, reason=reason))

    def hold(self, job_id: int, *, actor: str, reason: str, until: float | None = None) -> dict[str, Any]:
        # `until` only when set, so this still talks to a daemon that predates it.
        extra = {"until": until} if until is not None else {}
        return self.call("hold", job_id=job_id, actor=actor, reason=reason, **extra)

    def release(self, job_id: int, *, actor: str, reason: str = "") -> dict[str, Any]:
        return self.call("release", job_id=job_id, actor=actor, reason=reason)

    def set_priority(self, job_id: int, job_class: str, *, actor: str, reason: str) -> dict[str, Any]:
        return self.call("set_priority", job_id=job_id, job_class=job_class, actor=actor, reason=reason)

    def set_runtime(self, job_id: int, max_runtime_s: int, *, actor: str, reason: str) -> dict[str, Any]:
        return self.call("set_runtime", job_id=job_id, max_runtime_s=max_runtime_s, actor=actor, reason=reason)

    def events(self, job_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return list(self.call("events", job_id=job_id, limit=limit))

    def jobs(self, **params: Any) -> list[dict[str, Any]]:
        return list(self.call("jobs", **params))

    def inbox(self, session_id: str, since: float, stall_s: float = 900.0) -> dict[str, Any]:
        return dict(self.call("inbox", session_id=session_id, since=since, stall_s=stall_s))

    def logs(self, job_id: int, lines: int = 50) -> str:
        return str(self.call("logs", job_id=job_id, lines=lines))
