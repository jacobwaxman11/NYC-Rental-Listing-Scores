"""The ``listing_scores`` table — aggregated per-listing scores (the output of
``score_listings.aggregate_scores``), rebuildable from ``image_scores``."""

from __future__ import annotations

import json
import sqlite3

from db.connection import _utcnow

LISTING_SCORE_COLS = (
    "avg_natural_light", "avg_finish_quality", "avg_space_feeling", "avg_condition",
    "max_view_quality", "pct_bright_rooms", "has_good_view",
    "common_avg_finish_quality", "common_avg_condition", "common_avg_appeal",
    "photos_total", "photos_scored", "photos_apartment", "photos_common",
    "photos_irrelevant",
)


def upsert_listing_scores(
    conn: sqlite3.Connection, listing_id: str, summary: dict
) -> None:
    """Persist aggregated per-listing scores (output of aggregate_scores)."""
    apt = summary.get("apartment") or {}
    com = summary.get("common_space") or {}

    values = (
        listing_id,
        apt.get("avg_natural_light"),
        apt.get("avg_finish_quality"),
        apt.get("avg_space_feeling"),
        apt.get("avg_condition"),
        apt.get("max_view_quality"),
        apt.get("pct_bright_rooms"),
        1 if apt.get("has_good_view") else 0,
        com.get("avg_finish_quality"),
        com.get("avg_condition"),
        com.get("avg_appeal"),
        summary.get("photos_total"),
        summary.get("photos_scored"),
        summary.get("photos_apartment"),
        summary.get("photos_common"),
        summary.get("photos_irrelevant"),
        json.dumps(apt.get("room_type_counts") or {}),
        json.dumps(com.get("space_type_counts") or {}),
        _utcnow(),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO listing_scores (
            listing_id,
            avg_natural_light, avg_finish_quality, avg_space_feeling, avg_condition,
            max_view_quality, pct_bright_rooms, has_good_view,
            common_avg_finish_quality, common_avg_condition, common_avg_appeal,
            photos_total, photos_scored, photos_apartment, photos_common,
            photos_irrelevant,
            room_type_counts_json, space_type_counts_json,
            aggregated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
