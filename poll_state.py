"""When did each pipeline stage last poll StreetEasy?

A thin domain layer over the db ``meta`` key/value store so the scraper and
backfill can stamp a "last run" time and print a friendly "last polled 6h ago"
at startup — without each script re-implementing the bookkeeping. Stages are
free-form strings; we use ``"scrape"`` and ``"backfill"``.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional

import db as dbm


def _key(stage: str) -> str:
    return f"last_run:{stage}"


def record(conn: sqlite3.Connection, stage: str) -> str:
    """Stamp ``stage`` as having just run. Returns the ISO-8601 timestamp."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dbm.set_meta(conn, _key(stage), ts)
    return ts


def last(conn: sqlite3.Connection, stage: str) -> Optional[str]:
    """The ISO-8601 timestamp of ``stage``'s last run, or None if never."""
    return dbm.get_meta(conn, _key(stage))


def ago(iso: Optional[str], now: Optional[datetime] = None) -> str:
    """Human-friendly elapsed string: 'never', 'just now', '6h ago', '3d ago'.

    ``now`` is injectable for testing; defaults to the current UTC time.
    """
    if not iso:
        return "never"
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    now = now or datetime.now(timezone.utc)
    secs = (now - then).total_seconds()
    if secs < 90:
        return "just now"
    if secs < 90 * 60:
        return f"{round(secs / 60)}m ago"
    if secs < 36 * 3600:
        return f"{round(secs / 3600)}h ago"
    return f"{round(secs / 86400)}d ago"
