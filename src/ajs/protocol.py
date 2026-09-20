"""Newline-delimited JSON over a Unix socket.

Deliberately boring: one JSON object per line, request/response. Small enough that the
CLI, the MCP server, or a shell one-liner with `socat` can all speak it.
"""

from __future__ import annotations

import json
from typing import Any


def encode(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def decode(raw: bytes) -> dict[str, Any]:
    return json.loads(raw.decode())


def request(method: str, **params: Any) -> dict[str, Any]:
    return {"method": method, "params": params}


def ok(result: Any) -> dict[str, Any]:
    return {"ok": True, "result": result}


def err(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


class SchedulerError(RuntimeError):
    """Raised client-side when the daemon returns an error response."""
