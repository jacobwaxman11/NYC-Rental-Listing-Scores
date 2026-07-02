"""The ``listing_amenities`` child table — normalized StreetEasy amenity tags
(pool, washer_dryer, gym, dishwasher, …), distinct from vision-derived photo
tags (see :mod:`db.image_scores`)."""

from __future__ import annotations

import sqlite3
from typing import Iterable


def set_amenities(
    conn: sqlite3.Connection, listing_id: str, amenities: Iterable[str]
) -> None:
    conn.execute("DELETE FROM listing_amenities WHERE listing_id=?", (listing_id,))
    rows = [(listing_id, a) for a in amenities if a]
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO listing_amenities (listing_id, amenity) VALUES (?, ?)",
            rows,
        )


def get_amenities(conn: sqlite3.Connection, listing_id: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT amenity FROM listing_amenities WHERE listing_id=? ORDER BY amenity",
            (listing_id,),
        )
    ]


def get_all_listing_amenities(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Return {listing_id: [amenity, ...]} for every listing with amenities.

    The bulk counterpart to :func:`get_amenities` — used to attach the
    StreetEasy-scraped amenities to the web UI's suggestion rows for amenity
    chips and search, alongside (and kept distinct from) the vision-derived
    photo tags.
    """
    rows = conn.execute(
        "SELECT listing_id, amenity FROM listing_amenities ORDER BY listing_id, amenity"
    ).fetchall()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["listing_id"], []).append(r["amenity"])
    return out
