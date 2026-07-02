"""The ``listing_images`` child table — per-listing image references, positional
and ordered. Image *scores* (keyed by URL, shared across listings) live in
:mod:`db.image_scores`."""

from __future__ import annotations

import sqlite3
from typing import Optional


def set_listing_images(
    conn: sqlite3.Connection,
    listing_id: str,
    image_urls: list[str],
    local_image_paths: Optional[list[str]] = None,
) -> None:
    """Replace the entire image list for a listing."""
    conn.execute("DELETE FROM listing_images WHERE listing_id=?", (listing_id,))
    paths = local_image_paths or []
    rows = [
        (
            listing_id,
            i,
            url,
            paths[i] if i < len(paths) else None,
        )
        for i, url in enumerate(image_urls)
    ]
    if rows:
        conn.executemany(
            "INSERT INTO listing_images (listing_id, position, image_url, local_image_path) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )


def get_listing_images(conn: sqlite3.Connection, listing_id: str) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT position, image_url, local_image_path "
            "FROM listing_images WHERE listing_id=? ORDER BY position",
            (listing_id,),
        )
    ]
