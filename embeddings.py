"""Local text embeddings for semantic listing search.

Uses sentence-transformers (runs offline, no API cost) to embed a compact
per-listing profile string built from the StreetEasy description plus the
photo-derived tags and basic facts. Embeddings are computed once by
``embed_listings.py`` and stored in the ``listing_embeddings`` table; the web
UI loads them for "based on what I've liked" matching and free-text vibe search.

The heavy ``sentence_transformers`` import is lazy, so importing this module is
cheap and the rest of the app works even when the package isn't installed.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np


DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, small and fast


def profile_text(listing: dict, tags: list[str]) -> str:
    """Build the text we embed for a listing: location + size + photo tags +
    the broker description. Order puts structured facts first, prose last."""
    parts: list[str] = []

    nb = listing.get("neighborhood")
    if nb:
        parts.append(f"Neighborhood: {nb}.")

    beds, baths, sqft = listing.get("beds"), listing.get("baths"), listing.get("sqft")
    bb: list[str] = []
    if beds is not None:
        bb.append(f"{int(beds)} bed" if float(beds) == int(beds) else f"{beds} bed")
    if baths is not None:
        bb.append(f"{baths} bath")
    if sqft:
        bb.append(f"{sqft} sqft")
    if bb:
        parts.append(", ".join(bb) + ".")

    if tags:
        parts.append("Features: " + ", ".join(t.replace("_", " ") for t in tags) + ".")

    desc = (listing.get("description") or "").strip()
    if desc:
        parts.append(desc)

    return " ".join(parts).strip()


def text_hash(model_name: str, text: str) -> str:
    """Stable hash of (model, text) so re-embedding is skipped when nothing
    relevant changed (and forced when the model changes)."""
    return hashlib.sha1(f"{model_name}|{text}".encode("utf-8")).hexdigest()


class Embedder:
    """Lazy wrapper around a SentenceTransformer model. Vectors are L2-normalized
    so cosine similarity is a plain dot product."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def try_build_embedder(model_name: str = DEFAULT_MODEL) -> Optional[Embedder]:
    """Return an Embedder, or None if sentence-transformers isn't installed
    (query-time semantic search is then disabled; stored vectors still work)."""
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return None
    return Embedder(model_name)
