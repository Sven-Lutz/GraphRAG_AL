# crawler/core/seed_jobs.py
from __future__ import annotations

import os
import socket
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_CRAWL_DB_PATH = Path("crawler/data/db/crawl.sqlite")

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"

def ensure_seed_job_events_schema(con: sqlite3.Connection) -> None:
    """Sollte nur EINMAL beim App-Start aufgerufen werden!"""
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS seed_job_events (
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          ts              TEXT NOT NULL,
          municipality_id TEXT NOT NULL,
          worker_id       TEXT,
          event           TEXT NOT NULL,  -- claim|done|failed|reclaim
          details         TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_seed_job_events_muni_ts
        ON seed_job_events(municipality_id, ts);

        CREATE INDEX IF NOT EXISTS idx_seed_job_events_event_ts
        ON seed_job_events(event, ts);
        """
    )

def _log_event(
    con: sqlite3.Connection,
    municipality_id: str,
    worker_id: Optional[str],
    event: str,
    details: Optional[str] = None,
) -> None:
    try:
        con.execute(
            """
            INSERT INTO seed_job_events(ts, municipality_id, worker_id, event, details)
            VALUES (?,?,?,?,?)
            """,
            (_utc_now_iso(), municipality_id, worker_id, event, details),
        )
    except Exception:
        pass

@dataclass(frozen=True, slots=True)
class SeedJob:
    municipality_id: str
    seed_url: str

def claim_next_job(
    con: sqlite3.Connection,
    worker_id: str,
    stale_after_seconds: int = 15 * 60,
) -> Optional[SeedJob]:
    """Atomically claim one pending job (or reclaim stale claimed job)."""
    
    now = _utc_now_iso()
    stale_cutoff = time.time() - float(stale_after_seconds)
    stale_cutoff_iso = datetime.fromtimestamp(stale_cutoff, tz=timezone.utc).isoformat()

    row = con.execute(
        """
        UPDATE seed_jobs
        SET status='claimed',
            claimed_by=?,
            claimed_at=?,
            heartbeat_at=?,
            attempt_count=attempt_count+1,
            last_error=NULL
        WHERE municipality_id = (
            SELECT municipality_id
            FROM seed_jobs
            WHERE
                status='pending'
                OR (
                    status='claimed'
                    AND (heartbeat_at IS NULL OR heartbeat_at < ?)
                )
            ORDER BY
                CASE status WHEN 'pending' THEN 0 ELSE 1 END,
                COALESCE(claimed_at, '') ASC,
                municipality_id ASC
            LIMIT 1
        )
        RETURNING municipality_id, seed_url
        """,
        (worker_id, now, now, stale_cutoff_iso),
    ).fetchone()

    if row is None:
        return None

    muni_id, seed_url = str(row[0]), str(row[1])
    _log_event(con, muni_id, worker_id, "claim", f"seed_url={seed_url}")
    return SeedJob(muni_id, seed_url)

def heartbeat_job(con: sqlite3.Connection, municipality_id: str, worker_id: str) -> None:
    con.execute(
        """
        UPDATE seed_jobs
        SET heartbeat_at=?
        WHERE municipality_id=? AND status='claimed' AND claimed_by=?
        """,
        (_utc_now_iso(), municipality_id, worker_id),
    )

def mark_done(con: sqlite3.Connection, municipality_id: str, worker_id: str) -> None:
    cur = con.execute(
        """
        UPDATE seed_jobs
        SET status='done', done_at=?, heartbeat_at=?, last_error=NULL
        WHERE municipality_id=? AND claimed_by=? AND status='claimed'
        """,
        (_utc_now_iso(), _utc_now_iso(), municipality_id, worker_id),
    )
    if cur.rowcount:
        _log_event(con, municipality_id, worker_id, "done", None)

def mark_failed(con: sqlite3.Connection, municipality_id: str, worker_id: str, error: str) -> None:
    cur = con.execute(
        """
        UPDATE seed_jobs
        SET status='failed', done_at=?, heartbeat_at=?, last_error=?
        WHERE municipality_id=? AND claimed_by=? AND status='claimed'
        """,
        (_utc_now_iso(), _utc_now_iso(), (error or "")[:2000], municipality_id, worker_id),
    )
    if cur.rowcount:
        _log_event(con, municipality_id, worker_id, "failed", (error or "")[:500])