"""The ``listing_reactions`` table — the user's hearts / swipes from the web UI."""

from __future__ import annotations

import sqlite3
from typing import Optional

from db.connection import _utcnow


def set_reaction(
    conn: sqlite3.Connection, listing_id: str, reaction: Optional[str]
) -> None:
    """Set or clear a user's reaction to a listing.

    ``reaction`` is 'liked' or 'passed'; pass None (or '' / 'none') to clear it.
    """
    if reaction in (None, "", "none"):
        conn.execute("DELETE FROM listing_reactions WHERE listing_id=?", (listing_id,))
        return
    conn.execute(
        """
        INSERT INTO listing_reactions (listing_id, reaction, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(listing_id) DO UPDATE SET
            reaction=excluded.reaction, updated_at=excluded.updated_at
        """,
        (listing_id, reaction, _utcnow()),
    )


def get_reactions(conn: sqlite3.Connection) -> dict[str, str]:
    """Return {listing_id: reaction} for every listing the user has reacted to."""
    return {
        r["listing_id"]: r["reaction"]
        for r in conn.execute("SELECT listing_id, reaction FROM listing_reactions")
    }
