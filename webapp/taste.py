"""Taste matching and "more like this" — pure local vector/heuristic math, no
model calls.

:func:`match_likes` ranks unreviewed listings by similarity to the centroid of
everything you've liked. :func:`similar_to` ranks by similarity to ONE reference
listing (the "✨ More like this" button), using description embeddings when
available and :func:`_similarity_heuristic` as the structured-field fallback.
"""

from __future__ import annotations

import numpy as np

import geo
from webapp.state import _STATE


def match_likes(limit: int = 40):
    """Rank unreviewed listings by cosine similarity to the centroid of the
    user's liked listings' embeddings. Pure local vector math — no model call."""
    emb = _STATE["embeddings"]
    sugg = _STATE["suggestions"]
    if not emb:
        return [], "⚠ No embeddings yet — run: python embed_listings.py"
    liked = [r for r in sugg if r["reaction"] == "liked" and r["listing_id"] in emb]
    if not liked:
        return [], "⚠ Like a few listings first (with embeddings) to match your taste"

    centroid = np.mean([emb[r["listing_id"]] for r in liked], axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm:
        centroid = centroid / norm

    cands = [r for r in sugg if not r["reaction"] and r["listing_id"] in emb]
    cands.sort(key=lambda r: float(np.dot(centroid, emb[r["listing_id"]])), reverse=True)
    n = len(liked)
    return cands[:limit], f"🧭 Matched to your {n} liked listing{'s' if n != 1 else ''}"


def _similarity_heuristic(ref: dict, r: dict) -> float:
    """A 0-ish..~8 similarity score between two listings using only structured
    fields — the fallback for "more like this" when description embeddings
    haven't been built yet. Higher = more alike."""
    s = 0.0
    if ref.get("neighborhood") not in (None, "—") and r.get("neighborhood") == ref.get("neighborhood"):
        s += 3.0
    if ref.get("beds") is not None and r.get("beds") is not None:
        s += max(0.0, 1.0 - abs(ref["beds"] - r["beds"]))          # exact beds → +1
    if ref.get("baths") is not None and r.get("baths") is not None:
        s += max(0.0, 0.5 - 0.5 * abs(ref["baths"] - r["baths"]))
    if ref.get("rent") and r.get("rent"):
        diff = abs(ref["rent"] - r["rent"]) / ref["rent"]
        s += max(0.0, 1.5 * (1 - diff / 0.5))                      # full at equal, 0 at +50%
    if ref.get("sqft") and r.get("sqft"):
        diff = abs(ref["sqft"] - r["sqft"]) / ref["sqft"]
        s += max(0.0, 1.0 * (1 - diff / 0.6))
    rt, tt = set(ref.get("tags") or []), set(r.get("tags") or [])
    if rt and tt:
        s += 2.0 * len(rt & tt) / len(rt | tt)                     # weighted Jaccard on tags
    return s


def similar_to(listing_id: str, limit: int = 40):
    """Rank listings by similarity to ONE reference listing — the engine behind
    the "✨ More like this" button. Uses description embeddings when available,
    otherwise :func:`_similarity_heuristic`. Pure local compute, no model call."""
    sugg = _STATE["suggestions"]
    ref = next((r for r in sugg if r["listing_id"] == listing_id), None)
    if ref is None:
        return [], "⚠ That listing isn't in the current set", {}

    emb = _STATE["embeddings"]
    cands = [r for r in sugg
             if r["listing_id"] != listing_id and r["reaction"] != "passed"]

    if emb and listing_id in emb:
        rv = emb[listing_id]
        cands = [r for r in cands if r["listing_id"] in emb]
        cands.sort(key=lambda r: float(np.dot(rv, emb[r["listing_id"]])), reverse=True)
    else:
        cands.sort(key=lambda r: _similarity_heuristic(ref, r), reverse=True)

    pool = cands[:limit]   # the "like this" set, chosen by feature/style similarity

    # Liking an apartment usually means caring about its area too — so we keep the
    # selection style-driven but ORDER it by vicinity to the reference: similar in
    # feel, nearest first. Listings without coordinates sink to the bottom.
    rlat, rlng = ref.get("lat"), ref.get("lng")
    distances = {}
    if rlat is not None and rlng is not None:
        for r in pool:
            distances[r["listing_id"]] = geo.haversine_mi(rlat, rlng, r.get("lat"), r.get("lng"))
        pool.sort(key=lambda r: (distances[r["listing_id"]] is None,
                                 distances[r["listing_id"]] or 0.0))
        banner = f"✨ Like {ref['name']} — similar style, nearest first"
    else:
        banner = f"✨ Like {ref['name']} — similar style"
    return pool, banner, distances
