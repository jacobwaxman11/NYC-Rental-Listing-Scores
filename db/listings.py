"""The ``listings`` table — one row per StreetEasy listing, keyed by
``listing_id`` (``building_slug`` + unit). Core upserts plus the "what still
needs work" queries the backfill and scorer drive off.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from db.connection import _utcnow

LISTING_COLS = (
    "listing_id", "url", "name", "street", "neighborhood", "zip", "lat", "lng",
    "beds", "baths", "sqft", "rent", "lease_months", "furnished", "building_type",
    "building_slug", "unit", "description", "available_from",
    "floor_plan_url", "local_floor_plan_path",
)


def upsert_listing(conn: sqlite3.Connection, listing: dict) -> None:
    """Insert or update a listing's core columns. Does not touch amenities or
    images — those have their own helpers."""
    values = [listing.get(c) for c in LISTING_COLS]
    placeholders = ",".join("?" * len(LISTING_COLS))
    cols = ",".join(LISTING_COLS)

    set_clause = ",".join(f"{c}=excluded.{c}" for c in LISTING_COLS if c != "listing_id")

    conn.execute(
        f"""
        INSERT INTO listings ({cols}, scraped_at)
        VALUES ({placeholders}, ?)
        ON CONFLICT(listing_id) DO UPDATE SET {set_clause}
        """,
        values + [_utcnow()],
    )


def update_listing_fields(
    conn: sqlite3.Connection, listing_id: str, fields: dict
) -> None:
    """Update specific columns on an existing listing (used by backfill)."""
    if not fields:
        return
    cols = ",".join(f"{k}=?" for k in fields)
    conn.execute(
        f"UPDATE listings SET {cols} WHERE listing_id=?",
        list(fields.values()) + [listing_id],
    )


def mark_detail_fetched(conn: sqlite3.Connection, listing_id: str) -> None:
    conn.execute(
        "UPDATE listings SET detail_fetched_at=? WHERE listing_id=?",
        (_utcnow(), listing_id),
    )


def get_listing(conn: sqlite3.Connection, listing_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM listings WHERE listing_id=?", (listing_id,)
    ).fetchone()
    return dict(row) if row else None


def get_listing_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT listing_id FROM listings")}


def listings_missing_amenities(conn: sqlite3.Connection) -> list[dict]:
    """Listings whose amenity table is empty — i.e. detail page hasn't been
    backfilled yet."""
    rows = conn.execute(
        """
        SELECT l.*
        FROM listings l
        LEFT JOIN listing_amenities a ON l.listing_id = a.listing_id
        WHERE a.amenity IS NULL
        ORDER BY l.listing_id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def listings_never_fetched(conn: sqlite3.Connection) -> list[dict]:
    """Listings whose detail page has never been fetched — i.e.
    ``detail_fetched_at`` is still NULL.

    Unlike :func:`listings_missing_amenities`, this never re-selects a listing
    that was already attempted, even if the fetch turned up no amenities (some
    listings genuinely have none). Use this to backfill only the truly-missing
    data without re-hitting StreetEasy for listings we've already processed.
    """
    rows = conn.execute(
        """
        SELECT l.*
        FROM listings l
        WHERE l.detail_fetched_at IS NULL
        ORDER BY l.listing_id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def listings_missing_scores(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT l.*
        FROM listings l
        LEFT JOIN listing_scores s ON l.listing_id = s.listing_id
        WHERE s.listing_id IS NULL
        ORDER BY l.listing_id
        """
    ).fetchall()
    return [dict(r) for r in rows]
