"""The ``listing_embeddings`` table — description/tag vectors used for semantic
search and taste matching. Vectors are stored as JSON text."""

from __future__ import annotations

import json
import sqlite3

from db.connection import _utcnow


def upsert_embedding(
    conn: sqlite3.Connection,
    listing_id: str,
    model: str,
    vector: list[float],
    text_hash: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO listing_embeddings
            (listing_id, model, dim, text_hash, vector, embedded_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (listing_id, model, len(vector), text_hash, json.dumps(vector), _utcnow()),
    )


def get_embedding_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    """{listing_id: text_hash} for already-embedded listings (incremental skip)."""
    return {
        r["listing_id"]: r["text_hash"]
        for r in conn.execute("SELECT listing_id, text_hash FROM listing_embeddings")
    }


def get_all_embeddings(conn: sqlite3.Connection) -> dict[str, list[float]]:
    """{listing_id: vector} for every embedded listing."""
    return {
        r["listing_id"]: json.loads(r["vector"])
        for r in conn.execute("SELECT listing_id, vector FROM listing_embeddings")
    }
