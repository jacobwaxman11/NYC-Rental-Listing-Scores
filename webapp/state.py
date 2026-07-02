"""Process-wide state shared across the web layer.

``_STATE`` is a single mutable dict populated at startup (and on ``?refresh=1``)
and read by every route/helper. Modules bind to it by importing this one object,
so mutations made here are visible everywhere — never reassign ``_STATE`` itself,
only mutate its keys. ``_SEARCH_CACHE`` memoizes AI query plans.
"""

from __future__ import annotations

import numpy as np

import db as dbm

# Populated at startup (and on ?refresh=1).
# "llm" holds a TextLLM or None (AI search disabled). "embedder" embeds queries
# (None if sentence-transformers isn't installed). "embeddings" maps listing_id
# -> np.ndarray loaded from the DB. "_SEARCH_CACHE" memoizes query plans.
_STATE: dict = {
    "suggestions": [], "meta": {}, "db_path": dbm.DEFAULT_DB_PATH,
    "llm": None, "ai": {"enabled": False, "provider": None, "model": None},
    "embedder": None, "embeddings": {},
}
_SEARCH_CACHE: dict = {}

# Price-slider bounds (dollars). Sitting at an end means "no bound that way";
# PRICE_MAX is rendered as "10k+".
PRICE_MIN, PRICE_MAX, PRICE_STEP = 2000, 10000, 250


def load_embeddings(db_path: str) -> dict:
    """Load stored listing vectors into memory as float32 arrays for fast cosine
    ranking (vectors are already L2-normalized at embed time)."""
    with dbm.open_db(db_path) as conn:
        raw = dbm.get_all_embeddings(conn)
    return {lid: np.asarray(vec, dtype=np.float32) for lid, vec in raw.items()}