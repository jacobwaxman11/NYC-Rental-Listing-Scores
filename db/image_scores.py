"""The ``image_scores`` table (Gemini vision scores keyed by ``image_url``, one
per unique image) and its companion ``image_tags``.

Scores are stored flat; :func:`image_score_to_dict` re-nests a row into the
shape the aggregator expects. :func:`get_all_listing_tags` rolls image-level
tags up to listings via ``listing_images``.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable

from db.connection import _utcnow


def get_existing_image_scores(
    conn: sqlite3.Connection, image_urls: Iterable[str]
) -> dict[str, dict]:
    """Return {image_url: row_dict} for any URLs we've already scored."""
    urls = list(image_urls)
    if not urls:
        return {}
    placeholders = ",".join("?" * len(urls))
    rows = conn.execute(
        f"SELECT * FROM image_scores WHERE image_url IN ({placeholders})", urls
    ).fetchall()
    return {r["image_url"]: dict(r) for r in rows}


def upsert_image_score(
    conn: sqlite3.Connection,
    image_url: str,
    score: dict,
    model: str,
) -> None:
    """Persist a Gemini score for one image. ``score`` is the full nested dict
    returned by the model (with image_category, apartment_scores,
    common_space_scores, irrelevant_reason)."""
    apt = score.get("apartment_scores") or {}
    com = score.get("common_space_scores") or {}

    # The "shared" fields (finish_quality, condition) come from whichever sub-
    # object is populated. apt and com are mutually exclusive per the schema.
    finish = apt.get("finish_quality") if apt else com.get("finish_quality")
    cond = apt.get("condition") if apt else com.get("condition")

    conn.execute(
        """
        INSERT OR REPLACE INTO image_scores (
            image_url, image_category,
            room_type, natural_light, space_feeling, view_quality,
            space_type, appeal,
            finish_quality, condition_score,
            irrelevant_reason,
            model, scored_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_url,
            score.get("image_category"),
            apt.get("room_type"),
            apt.get("natural_light"),
            apt.get("space_feeling"),
            apt.get("view_quality"),
            com.get("space_type"),
            com.get("appeal"),
            finish,
            cond,
            score.get("irrelevant_reason"),
            model,
            _utcnow(),
        ),
    )

    # Replace the image's descriptive tags (controlled-vocabulary keywords).
    conn.execute("DELETE FROM image_tags WHERE image_url=?", (image_url,))
    tags = score.get("tags") or []
    if tags:
        conn.executemany(
            "INSERT OR IGNORE INTO image_tags (image_url, tag) VALUES (?, ?)",
            [(image_url, t) for t in tags],
        )


def get_all_listing_tags(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Roll image tags up to the listing level via listing_images.

    Returns {listing_id: [tag, ...]} ordered by how many of the listing's photos
    carry each tag (most frequent first). A tag on a shared image counts once
    per listing that references it.
    """
    rows = conn.execute(
        """
        SELECT li.listing_id AS listing_id, it.tag AS tag, COUNT(*) AS n
        FROM listing_images li
        JOIN image_tags it ON li.image_url = it.image_url
        GROUP BY li.listing_id, it.tag
        """
    ).fetchall()

    acc: dict[str, list[tuple[str, int]]] = {}
    for r in rows:
        acc.setdefault(r["listing_id"], []).append((r["tag"], r["n"]))
    return {
        lid: [t for t, _ in sorted(pairs, key=lambda p: (-p[1], p[0]))]
        for lid, pairs in acc.items()
    }


def image_score_to_dict(row: dict) -> dict:
    """Convert a flat image_scores row back into the nested shape the
    aggregator expects."""
    cat = row.get("image_category")
    out: dict[str, Any] = {
        "image_category": cat,
        "apartment_scores": None,
        "common_space_scores": None,
        "irrelevant_reason": row.get("irrelevant_reason"),
    }
    if cat == "apartment":
        out["apartment_scores"] = {
            "room_type": row.get("room_type"),
            "natural_light": row.get("natural_light"),
            "finish_quality": row.get("finish_quality"),
            "space_feeling": row.get("space_feeling"),
            "view_quality": row.get("view_quality"),
            "condition": row.get("condition_score"),
        }
    elif cat == "common_space":
        out["common_space_scores"] = {
            "space_type": row.get("space_type"),
            "finish_quality": row.get("finish_quality"),
            "condition": row.get("condition_score"),
            "appeal": row.get("appeal"),
        }
    return out
