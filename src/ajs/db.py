"""SQLite persistence.

The daemon is the sole writer, so WAL mode plus a single connection is enough; there is
no lock contention to manage. Readers (`sqlite3 jobs.db` for ad-hoc benchmark analysis)
can attach concurrently without blocking the scheduler.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

from .models import Job, JobClass, JobState, ResourceRequest

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project         TEXT NOT NULL,
    session_id      TEXT NOT NULL DEFAULT '',
    cmd             TEXT NOT NULL,
    cwd             TEXT NOT NULL,
    env             TEXT NOT NULL DEFAULT '{}',
    resources       TEXT NOT NULL,
    max_runtime_s   INTEGER NOT NULL,
    job_class       TEXT NOT NULL,
    state           TEXT NOT NULL,
    submitted_at    REAL NOT NULL,
    started_at      REAL,
    finished_at     REAL,
    exit_code       INTEGER,
    pid             INTEGER,
    unit            TEXT,
    log_path        TEXT,
    load_before     REAL,
    load_after      REAL,
    contended       INTEGER NOT NULL DEFAULT 0,
    contention_note TEXT,
    reserved_until  REAL,
    cancel_reason   TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project);
CREATE INDEX IF NOT EXISTS idx_jobs_submitted ON jobs(submitted_at);

-- Escape hatch for work that must run inside the agent's own process. Heartbeat-based
-- so a dead agent cannot wedge the queue holding resources forever.
CREATE TABLE IF NOT EXISTS leases (
    id           TEXT PRIMARY KEY,
    project      TEXT NOT NULL,
    session_id   TEXT NOT NULL DEFAULT '',
    resources    TEXT NOT NULL,
    reason       TEXT,
    acquired_at  REAL NOT NULL,
    heartbeat_at REAL NOT NULL,
    released_at  REAL
);

CREATE INDEX IF NOT EXISTS idx_leases_active ON leases(released_at);

-- Audit trail for queue management: who held, released or re-prioritised what, and why.
CREATE TABLE IF NOT EXISTS job_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  INTEGER NOT NULL,
    at      REAL NOT NULL,
    actor   TEXT NOT NULL DEFAULT '',
    action  TEXT NOT NULL,
    detail  TEXT,
    reason  TEXT
);

CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id);

-- Round-robin bookkeeping for fair-share between projects.
CREATE TABLE IF NOT EXISTS project_stats (
    project        TEXT PRIMARY KEY,
    last_started_at REAL NOT NULL DEFAULT 0
);
"""


_ADDED_COLUMNS = (
    ("held", "INTEGER NOT NULL DEFAULT 0"),
    ("note", "TEXT"),  # the job's description; named before `title` existed
    ("title", "TEXT"),
    ("meta", "TEXT NOT NULL DEFAULT '{}'"),
)


class Store:
    """Owns the job database."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the first schema. CREATE TABLE IF NOT EXISTS does
        not touch an existing table, so a live database needs them added in place."""
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(jobs)")}
        for name, ddl in _ADDED_COLUMNS:
            if name not in have:
                self.conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {ddl}")

    def close(self) -> None:
        self.conn.close()

    # --- jobs -------------------------------------------------------------

    def add_job(
        self,
        *,
        project: str,
        session_id: str,
        cmd: list[str],
        cwd: str,
        env: dict[str, str],
        resources: ResourceRequest,
        max_runtime_s: int,
        job_class: JobClass,
        title: str | None = None,
        description: str | None = None,
        meta: dict[str, str] | None = None,
        held: bool = False,
    ) -> Job:
        now = time.time()
        cur = self.conn.execute(
            """INSERT INTO jobs
               (project, session_id, cmd, cwd, env, resources, max_runtime_s, job_class, state, submitted_at,
                title, note, meta, held)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                project,
                session_id,
                json.dumps(cmd),
                cwd,
                json.dumps(env),
                resources.to_json(),
                max_runtime_s,
                str(job_class),
                str(JobState.QUEUED),
                now,
                title,
                description,
                json.dumps(meta or {}),
                int(held),
            ),
        )
        job_id = int(cur.lastrowid or 0)
        got = self.get_job(job_id)
        assert got is not None
        return got

    def get_job(self, job_id: int) -> Job | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def update_job(self, job_id: int, **fields: object) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def jobs_in_state(self, *states: JobState) -> list[Job]:
        marks = ",".join("?" * len(states))
        rows = self.conn.execute(
            f"SELECT * FROM jobs WHERE state IN ({marks}) ORDER BY submitted_at ASC",
            tuple(str(s) for s in states),
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    def list_jobs(
        self,
        *,
        project: str | None = None,
        states: Iterable[JobState] | None = None,
        limit: int = 50,
    ) -> list[Job]:
        clauses: list[str] = []
        args: list[object] = []
        if project:
            clauses.append("project=?")
            args.append(project)
        states = list(states or [])
        if states:
            clauses.append(f"state IN ({','.join('?' * len(states))})")
            args.extend(str(s) for s in states)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        args.append(limit)
        rows = self.conn.execute(f"SELECT * FROM jobs {where} ORDER BY id DESC LIMIT ?", tuple(args)).fetchall()
        return [_row_to_job(r) for r in rows]

    # --- queue-management audit trail ---------------------------------------

    def add_event(self, job_id: int, *, actor: str, action: str, detail: str = "", reason: str = "") -> None:
        self.conn.execute(
            "INSERT INTO job_events(job_id, at, actor, action, detail, reason) VALUES(?,?,?,?,?,?)",
            (job_id, time.time(), actor, action, detail, reason),
        )

    def events(self, *, job_id: int | None = None, limit: int = 50) -> list[dict[str, object]]:
        where, args = ("WHERE job_id=?", (job_id, limit)) if job_id is not None else ("", (limit,))
        rows = self.conn.execute(f"SELECT * FROM job_events {where} ORDER BY id DESC LIMIT ?", args).fetchall()
        return [dict(r) for r in rows]

    def last_event(self, job_id: int, action: str) -> dict[str, object] | None:
        row = self.conn.execute(
            "SELECT * FROM job_events WHERE job_id=? AND action=? ORDER BY id DESC LIMIT 1", (job_id, action)
        ).fetchone()
        return dict(row) if row else None

    # --- fair share -------------------------------------------------------

    def note_project_start(self, project: str, when: float) -> None:
        self.conn.execute(
            """INSERT INTO project_stats(project, last_started_at) VALUES(?,?)
               ON CONFLICT(project) DO UPDATE SET last_started_at=excluded.last_started_at""",
            (project, when),
        )

    def project_last_start(self) -> dict[str, float]:
        rows = self.conn.execute("SELECT project, last_started_at FROM project_stats").fetchall()
        return {r["project"]: r["last_started_at"] for r in rows}

    # --- leases -----------------------------------------------------------

    def add_lease(self, lease_id: str, project: str, session_id: str, resources: ResourceRequest, reason: str) -> None:
        now = time.time()
        self.conn.execute(
            """INSERT INTO leases(id, project, session_id, resources, reason, acquired_at, heartbeat_at)
               VALUES(?,?,?,?,?,?,?)""",
            (lease_id, project, session_id, resources.to_json(), reason, now, now),
        )

    def active_leases(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM leases WHERE released_at IS NULL").fetchall()

    def heartbeat_lease(self, lease_id: str) -> bool:
        cur = self.conn.execute(
            "UPDATE leases SET heartbeat_at=? WHERE id=? AND released_at IS NULL",
            (time.time(), lease_id),
        )
        return cur.rowcount > 0

    def release_lease(self, lease_id: str) -> bool:
        cur = self.conn.execute(
            "UPDATE leases SET released_at=? WHERE id=? AND released_at IS NULL",
            (time.time(), lease_id),
        )
        return cur.rowcount > 0

    def expire_leases(self, older_than: float) -> list[str]:
        rows = self.conn.execute(
            "SELECT id FROM leases WHERE released_at IS NULL AND heartbeat_at < ?",
            (older_than,),
        ).fetchall()
        ids = [r["id"] for r in rows]
        for lease_id in ids:
            self.release_lease(lease_id)
        return ids


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        project=row["project"],
        session_id=row["session_id"],
        cmd=json.loads(row["cmd"]),
        cwd=row["cwd"],
        env=json.loads(row["env"]),
        resources=ResourceRequest.from_json(row["resources"]),
        max_runtime_s=row["max_runtime_s"],
        job_class=JobClass(row["job_class"]),
        state=JobState(row["state"]),
        submitted_at=row["submitted_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        exit_code=row["exit_code"],
        pid=row["pid"],
        unit=row["unit"],
        log_path=row["log_path"],
        load_before=row["load_before"],
        load_after=row["load_after"],
        contended=bool(row["contended"]),
        contention_note=row["contention_note"],
        reserved_until=row["reserved_until"],
        cancel_reason=row["cancel_reason"],
        held=bool(row["held"]),
        title=row["title"],
        description=row["note"],
        meta=json.loads(row["meta"] or "{}"),
    )
