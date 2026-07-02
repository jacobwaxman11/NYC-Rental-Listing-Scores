"""Connection + schema management, plus the shared ``_utcnow`` timestamp helper
every writer stamps rows with.

The DDL lives in ``schema.sql`` at the repo root (one level above this package)
for real SQL highlighting.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from config import DEFAULT_DB_PATH

# schema.sql sits at the repo root; this module lives in db/, so go up two levels.
SCHEMA = (Path(__file__).resolve().parent.parent / "schema.sql").read_text()


def connect(path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def open_db(path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    init_schema(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
