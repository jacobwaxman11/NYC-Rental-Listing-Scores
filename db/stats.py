"""A DB-level summary of row counts across the core tables — handy for CLI
tools reporting scrape/backfill/score progress."""

from __future__ import annotations

import sqlite3


def stats(conn: sqlite3.Connection) -> dict:
    def cnt(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    return {
        "listings": cnt("SELECT COUNT(*) FROM listings"),
        "listings_with_amenities": cnt(
            "SELECT COUNT(DISTINCT listing_id) FROM listing_amenities"
        ),
        "listings_with_scores": cnt("SELECT COUNT(*) FROM listing_scores"),
        "image_refs": cnt("SELECT COUNT(*) FROM listing_images"),
        "unique_images": cnt("SELECT COUNT(DISTINCT image_url) FROM listing_images"),
        "scored_images": cnt("SELECT COUNT(*) FROM image_scores"),
    }
