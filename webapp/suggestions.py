"""Build the ranked "deals" set: predict each listing's market rent and enrich
it with the display fields the UI needs.

The core is a Ridge regression on ``log_rent`` with **out-of-fold** predictions
grouped by building (``GroupKFold`` on ``building_slug``) — the same leakage
guard the notebook uses, so a listing's predicted rent never comes from a model
that trained on its own building. Listings are ranked by how far their actual
rent sits below the prediction; the biggest discounts sort to the top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import db as dbm
from features import collapse_amenity_tiers, collapse_image_scores, prepare_features


def _predict_market_rent(df: pd.DataFrame, groups: list[str]) -> np.ndarray:
    """Return predicted rent (dollars) for every row in ``df``.

    Uses out-of-fold predictions grouped by building when there are enough
    buildings to split on; otherwise falls back to an in-sample fit (flagged
    in the caller). ``df`` must contain ``log_rent``; ``listing_id`` is dropped.
    """
    y = df["log_rent"].to_numpy()
    X = df.drop(columns=["listing_id", "log_rent"])

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    n_groups = len(set(groups))

    if n_groups >= 2 and len(df) >= 5:
        n_splits = min(5, n_groups)
        pred_log = cross_val_predict(
            model, X, y, cv=GroupKFold(n_splits=n_splits), groups=groups
        )
    else:
        # Too little data to cross-validate honestly — fit and predict in-sample.
        model.fit(X, y)
        pred_log = model.predict(X)

    return np.exp(pred_log)


def _building_groups(conn, listing_ids: list[str]) -> list[str]:
    """Map each listing_id to its building_slug (fallback: the id itself)."""
    rows = conn.execute(
        "SELECT listing_id, building_slug FROM listings"
    ).fetchall()
    slug = {r["listing_id"]: (r["building_slug"] or r["listing_id"]) for r in rows}
    return [slug.get(i, i) for i in listing_ids]


def _choose_image_position(conn, listing_id: str, images=None):
    """Pick a representative photo: the brightest apartment shot, else the
    first non-irrelevant photo, else the first photo. Returns a position or None.

    ``images`` may be passed in to reuse an already-fetched image list (the
    suggestions loop does this to avoid a second query per listing)."""
    if images is None:
        images = dbm.get_listing_images(conn, listing_id)
    if not images:
        return None
    scores = dbm.get_existing_image_scores(conn, [im["image_url"] for im in images])

    best_pos, best_light = None, -1
    for im in images:
        s = scores.get(im["image_url"])
        if s and s.get("image_category") == "apartment":
            light = s.get("natural_light") or 0
            if light > best_light:
                best_pos, best_light = im["position"], light
    if best_pos is not None:
        return best_pos

    for im in images:
        s = scores.get(im["image_url"])
        if not s or s.get("image_category") != "irrelevant":
            return im["position"]
    return images[0]["position"]


def _apt_quality(scores_row: dict | None):
    """Average of the four apartment quality aggregates, 1-10, or None."""
    if not scores_row:
        return None
    vals = [
        scores_row.get(c)
        for c in ("avg_natural_light", "avg_finish_quality",
                  "avg_space_feeling", "avg_condition")
        if scores_row.get(c) is not None
    ]
    return round(sum(vals) / len(vals), 1) if vals else None


def build_suggestions(db_path: str) -> dict:
    """Compute predicted rent + discount for every scored listing and enrich
    with display fields. Returns {"suggestions": [...], "meta": {...}}."""
    df = prepare_features(db_path=db_path, require_scores=True, verbose=False)
    df = collapse_image_scores(df)
    df = collapse_amenity_tiers(df)

    if df.empty:
        return {"suggestions": [], "meta": {"total": 0, "scored": 0,
                                            "underpriced": 0, "neighborhoods": []}}

    ids = df["listing_id"].tolist()

    with dbm.open_db(db_path) as conn:
        groups = _building_groups(conn, ids)
        pred_rent = _predict_market_rent(df, groups)
        actual_rent = np.exp(df["log_rent"].to_numpy())
        tags_by_listing = dbm.get_all_listing_tags(conn)

        rows = []
        for i, listing_id in enumerate(ids):
            listing = dbm.get_listing(conn, listing_id) or {}
            scores_row = conn.execute(
                "SELECT * FROM listing_scores WHERE listing_id=?", (listing_id,)
            ).fetchone()
            scores_row = dict(scores_row) if scores_row else None

            # Fetch images once and reuse: pick a representative photo to show
            # first, and hand the full ordered position list to the UI so the
            # Tinder card can page left/right through every photo.
            images = dbm.get_listing_images(conn, listing_id)
            img_positions = [im["position"] for im in images]

            actual = float(actual_rent[i])
            predicted = float(pred_rent[i])
            pct_diff = (actual - predicted) / predicted if predicted else 0.0

            rows.append({
                "listing_id": listing_id,
                "building_slug": listing.get("building_slug") or "",
                "name": listing.get("name") or listing.get("street") or listing_id,
                "neighborhood": listing.get("neighborhood") or "—",
                "lat": listing.get("lat"),
                "lng": listing.get("lng"),
                "beds": listing.get("beds"),
                "baths": listing.get("baths"),
                "sqft": listing.get("sqft"),
                "url": listing.get("url"),
                "available_from": listing.get("available_from"),  # ISO YYYY-MM-DD or None
                "rent": round(actual),
                "predicted": round(predicted),
                "pct_diff": pct_diff,
                "delta": round(predicted - actual),  # positive = below market
                "apt_quality": _apt_quality(scores_row),
                "photos": (scores_row or {}).get("photos_total"),
                "img_pos": _choose_image_position(conn, listing_id, images),
                "img_positions": img_positions,
                "tags": tags_by_listing.get(listing_id, []),
                "reaction": None,  # overlaid per-request from the DB
            })

    n_groups = len(set(groups))
    meta = {
        "total": len(rows),
        "scored": len(rows),
        "underpriced": sum(1 for r in rows if r["pct_diff"] < 0),
        "cv": n_groups >= 2 and len(rows) >= 5,
        "neighborhoods": sorted({r["neighborhood"] for r in rows if r["neighborhood"] != "—"}),
        "tags": sorted({t for r in rows for t in (r.get("tags") or [])}),
    }
    return {"suggestions": rows, "meta": meta}
