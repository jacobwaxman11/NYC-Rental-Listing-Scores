"""Quick web UI that serves the underpriced listings the model suggests.

This is the interactive front-end for the pipeline. It:

  1. Builds the engineered feature frame (``features.prepare_features``).
  2. Fits a Ridge regression on ``log_rent`` and predicts each listing's
     "market" rent with **out-of-fold** predictions grouped by building
     (``GroupKFold`` on ``building_slug``) — the same leakage guard the
     notebook uses, so a listing's predicted rent never comes from a model
     that trained on its own building.
  3. Ranks listings by how far their actual rent sits below the prediction
     and renders them as photo cards. The biggest discounts ("deals") sort
     to the top.

You can ❤ listings (persisted to the ``listing_reactions`` table) and triage
them in "Tinder mode" — swipe right to like, left to pass.

Run it (after scraping + scoring have populated rentals.db):

    pip install -r requirements.txt
    python web.py                       # http://127.0.0.1:5000
    python web.py --db rentals.db --port 8000

The suggestion set is computed once at startup and cached. Hit
``/?refresh=1`` to recompute after re-running the scorer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask, abort, jsonify, redirect, render_template, request, send_file,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import db as dbm
import embeddings as emb_mod
import geo
import llm as llm_mod
from features import collapse_amenity_tiers, collapse_image_scores, prepare_features


app = Flask(__name__)

# Populated at startup (and on ?refresh=1).
# "llm" holds a TextLLM or None (AI search disabled). "embedder" embeds queries
# (None if sentence-transformers isn't installed). "embeddings" maps listing_id
# -> np.ndarray loaded from the DB. "_search_cache" memoizes query plans.
_STATE: dict = {
    "suggestions": [], "meta": {}, "db_path": dbm.DEFAULT_DB_PATH,
    "llm": None, "ai": {"enabled": False, "provider": None, "model": None},
    "embedder": None, "embeddings": {},
}
_SEARCH_CACHE: dict = {}


def _load_embeddings(db_path: str) -> dict:
    """Load stored listing vectors into memory as float32 arrays for fast cosine
    ranking (vectors are already L2-normalized at embed time)."""
    with dbm.open_db(db_path) as conn:
        raw = dbm.get_all_embeddings(conn)
    return {lid: np.asarray(vec, dtype=np.float32) for lid, vec in raw.items()}


# ── Pipeline job runner (scrape / backfill / score from the UI) ───────────────
#
# At most one stage runs at a time. We spawn the existing CLI scripts as
# subprocesses (the same Python interpreter that's serving this app, so the
# venv is inherited), capture stdout line by line into a buffer, and let the
# browser poll /api/run/log for incremental progress. Params are passed as
# discrete argv elements (never through a shell), so UI input can't inject
# commands.

import subprocess
import sys
import threading

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))

_JOB: dict = {
    "stage": None,
    "status": "idle",     # idle | running | done | failed | stopped
    "lines": [],          # captured stdout/stderr lines
    "returncode": None,
    "proc": None,
}
_JOB_LOCK = threading.Lock()

_STAGES = {
    "scrape": "scrape_listings.py",
    "backfill": "backfill_details.py",
    "score": "score_listings.py",
}


def _build_argv(stage: str, params: dict) -> list[str]:
    """Translate posted params into a safe argv list for the stage's script.

    Only known flags are forwarded, and each value is cast (ints/floats) or
    passed verbatim as its own argv element — no shell interpolation.
    """
    argv = [_STAGES[stage]]

    def add(flag: str, key: str, cast=str) -> None:
        v = params.get(key)
        if v is None or v == "":
            return
        try:
            argv.extend([flag, str(cast(v))])
        except (TypeError, ValueError):
            pass  # ignore an unparseable value rather than crash the launch

    if stage == "scrape":
        add("--areas", "areas")
        add("--price-min", "price_min", int)
        add("--price-max", "price_max", int)
        add("--max-listings", "max_listings", int)
        add("--max-pages", "max_pages", int)
    elif stage == "backfill":
        add("--impersonate", "impersonate")
        add("--min-delay", "min_delay", float)
        add("--max-delay", "max_delay", float)
        add("--max-listings", "max_listings", int)
    elif stage == "score":
        add("--provider", "provider")
        add("--model", "model")
        add("--max-listings", "max_listings", int)
    return argv


def _job_reader(proc: "subprocess.Popen") -> None:
    """Stream a running job's output into the shared buffer until it exits."""
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        with _JOB_LOCK:
            _JOB["lines"].append(line.rstrip("\n"))
    proc.stdout.close()
    rc = proc.wait()
    with _JOB_LOCK:
        _JOB["returncode"] = rc
        if _JOB["status"] == "running":
            _JOB["status"] = "done" if rc == 0 else "failed"
        _JOB["lines"].append(f"── process exited with code {rc} ──")


def start_job(stage: str, params: dict) -> tuple[bool, str]:
    """Spawn a stage. Returns (started, message)."""
    with _JOB_LOCK:
        if _JOB["status"] == "running":
            return False, "A job is already running — wait for it to finish or stop it."
        argv = _build_argv(stage, params)
        _JOB.update(stage=stage, status="running", lines=[], returncode=None, proc=None)
        _JOB["lines"].append("$ python " + " ".join(argv))

    proc = subprocess.Popen(
        [sys.executable, "-u", *argv],
        cwd=_REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with _JOB_LOCK:
        _JOB["proc"] = proc
    threading.Thread(target=_job_reader, args=(proc,), daemon=True).start()
    return True, "started"


# ── Model + suggestion construction ──────────────────────────────────────────


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
                "name": listing.get("name") or listing.get("street") or listing_id,
                "neighborhood": listing.get("neighborhood") or "—",
                "lat": listing.get("lat"),
                "lng": listing.get("lng"),
                "beds": listing.get("beds"),
                "baths": listing.get("baths"),
                "sqft": listing.get("sqft"),
                "url": listing.get("url"),
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
    }
    return {"suggestions": rows, "meta": meta}


# ── Natural-language search (cost-smart: one small call → local execution) ────


SEARCH_SYSTEM = """You translate a renter's natural-language request into a STRICT
JSON query plan that a program executes over a fixed set of NYC rental listings.
You never see the full listings — only summary facets, an optional reference
listing, and a profile of what the user has liked. Output ONLY the JSON object.

Numeric fields per listing: beds, baths, sqft, rent (monthly USD),
apt_quality (photo quality 1-10), pct_diff (actual rent vs model-predicted rent;
NEGATIVE means below market / underpriced / a better deal).

Each listing also has descriptive "tags" derived from its photos (e.g.
hardwood_floors, exposed_brick, duplex, high_ceilings, private_balcony,
windowed_kitchen). The tags PRESENT in this dataset are listed in
context.facets.tags — only use tags from that list.

Output schema (use null for anything not implied):
{
  "explanation": "one short sentence on how you read the request",
  "filters": {
    "beds_min": number|null, "beds_max": number|null,
    "baths_min": number|null,
    "sqft_min": number|null, "sqft_max": number|null,
    "rent_min": number|null, "rent_max": number|null,
    "neighborhoods": [string]|null,
    "exclude_neighborhoods": [string]|null,
    "min_quality": number|null,
    "tags_any": [string]|null,        // listing must have at least one of these tags
    "tags_all": [string]|null,        // listing must have all of these tags
    "only_underpriced": boolean,
    "exclude_reviewed": boolean
  },
  "rank_by": [ {"field": "sqft"|"rent"|"pct_diff"|"apt_quality"|"beds",
                "direction": "asc"|"desc", "weight": number} ],
  "semantic_query": string|null,
  "limit": number
}

Rules:
- semantic_query: a short natural-language phrase capturing the descriptive
  "vibe"/qualities the user wants that are NOT covered by the structured fields
  or tags (charm, character, light-and-airy, cozy, prewar elegance, quiet,
  modern minimalist, etc.). It is matched against listing descriptions by
  embedding similarity to re-rank results. Use null for purely structural asks.
- Map descriptive wants to tags from context.facets.tags: "exposed brick" ->
  tags_all ["exposed_brick"]; "bright/lots of light" -> ["bright"] or
  ["lots_of_windows"]; "duplex"/"stairs" -> ["duplex"]/["internal_stairs"];
  "renovated kitchen" -> ["renovated_kitchen"]. Use tags_all when the user clearly
  requires a feature, tags_any when listing alternatives. Ignore tags not in the list.
- "like this" with a reference -> bias toward the reference's tags via tags_any.
- Prefer ranking over hard cutoffs unless the user is explicit ("at least 2 beds",
  "under $5000"). Keep filters loose so good matches aren't excluded.
- "bigger"/"more space" -> rank sqft desc (optionally sqft_min near the reference).
  "cheaper" -> rank rent asc. "nicer"/"better" -> rank apt_quality desc.
  "better deal"/"underpriced" -> only_underpriced true and/or rank pct_diff asc.
- "like this" with a reference -> prefer the same neighborhood and similar
  beds/baths, then apply the modifier.
- "based on what I liked"/"similar to my likes" -> use the liked profile to set
  sensible ranges/areas and set exclude_reviewed=true.
- "haven't reviewed"/"new"/"not seen yet" -> exclude_reviewed=true.
- Weights in [0,1]; multiple rank_by entries combine. limit default 24, max 60."""

_RANK_FIELDS = {"sqft", "rent", "pct_diff", "apt_quality", "beds"}


def _facets(sugg: list[dict]) -> dict:
    def rng(key):
        vals = [r[key] for r in sugg if r.get(key) is not None]
        return {"min": min(vals), "max": max(vals)} if vals else None

    return {
        "count": len(sugg),
        "beds": rng("beds"),
        "sqft": rng("sqft"),
        "rent": rng("rent"),
        "neighborhoods": sorted({r["neighborhood"] for r in sugg
                                 if r["neighborhood"] != "—"}),
        "tags": sorted({t for r in sugg for t in (r.get("tags") or [])}),
    }


def _build_context(sugg: list[dict], ref_id: str) -> dict:
    ctx = {"facets": _facets(sugg)}

    if ref_id:
        m = next((r for r in sugg if r["listing_id"] == ref_id), None)
        if m:
            ctx["reference"] = {k: m[k] for k in
                ("name", "neighborhood", "beds", "baths", "sqft", "rent",
                 "apt_quality", "pct_diff")}
            ctx["reference"]["tags"] = m.get("tags") or []

    liked = [r for r in sugg if r["reaction"] == "liked"]
    if liked:
        def avg(key):
            vals = [r[key] for r in liked if r.get(key) is not None]
            return round(sum(vals) / len(vals), 1) if vals else None

        ctx["liked_profile"] = {
            "count": len(liked),
            "avg_beds": avg("beds"), "avg_sqft": avg("sqft"),
            "avg_rent": avg("rent"), "avg_quality": avg("apt_quality"),
            "top_neighborhoods": [n for n, _ in
                Counter(r["neighborhood"] for r in liked).most_common(5)],
            "top_tags": [t for t, _ in
                Counter(t for r in liked for t in (r.get("tags") or [])).most_common(10)],
            "examples": [{k: r[k] for k in ("name", "neighborhood", "beds", "sqft", "rent")}
                         for r in liked[:5]],
        }
    return ctx


def _apply_plan(plan: dict, sugg: list[dict], query_vec=None) -> list[dict]:
    """Execute a query plan locally over the listings — no model calls here.

    ``query_vec`` (optional) is an embedding of the plan's semantic_query; when
    present (and listing embeddings are loaded) it adds a description-similarity
    term to the ranking."""
    rows = list(sugg)
    f = plan.get("filters") or {}

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    if f.get("exclude_reviewed"):
        rows = [r for r in rows if not r["reaction"]]
    if f.get("only_underpriced"):
        rows = [r for r in rows if r["pct_diff"] < 0]

    bounds = {
        "beds_min": num(f.get("beds_min")), "beds_max": num(f.get("beds_max")),
        "baths_min": num(f.get("baths_min")),
        "sqft_min": num(f.get("sqft_min")), "sqft_max": num(f.get("sqft_max")),
        "rent_min": num(f.get("rent_min")), "rent_max": num(f.get("rent_max")),
        "min_quality": num(f.get("min_quality")),
    }
    nhs = f.get("neighborhoods") or None
    ex_nhs = set(f.get("exclude_neighborhoods") or [])
    tags_any = set(f.get("tags_any") or [])
    tags_all = set(f.get("tags_all") or [])

    def keep(r):
        beds, baths, sqft, rent = r["beds"] or 0, r["baths"] or 0, r["sqft"], r["rent"]
        q = r["apt_quality"]
        if bounds["beds_min"] is not None and beds < bounds["beds_min"]: return False
        if bounds["beds_max"] is not None and beds > bounds["beds_max"]: return False
        if bounds["baths_min"] is not None and baths < bounds["baths_min"]: return False
        if bounds["sqft_min"] is not None and (sqft or 0) < bounds["sqft_min"]: return False
        if bounds["sqft_max"] is not None and sqft is not None and sqft > bounds["sqft_max"]: return False
        if bounds["rent_min"] is not None and rent < bounds["rent_min"]: return False
        if bounds["rent_max"] is not None and rent > bounds["rent_max"]: return False
        if bounds["min_quality"] is not None and (q or 0) < bounds["min_quality"]: return False
        if nhs and r["neighborhood"] not in nhs: return False
        if r["neighborhood"] in ex_nhs: return False
        if tags_any or tags_all:
            rtags = set(r.get("tags") or [])
            if tags_any and not (rtags & tags_any): return False
            if tags_all and not (tags_all <= rtags): return False
        return True

    rows = [r for r in rows if keep(r)]

    rank = [e for e in (plan.get("rank_by") or []) if e.get("field") in _RANK_FIELDS]
    emb = _STATE["embeddings"]
    use_sem = query_vec is not None and bool(emb)

    if (rank or use_sem) and rows:
        ranges = {}
        for e in rank:
            fld = e["field"]
            vals = [r[fld] for r in rows if r.get(fld) is not None]
            ranges[fld] = (min(vals), max(vals)) if vals else (0.0, 0.0)

        sims, slo, shi = {}, 0.0, 0.0
        if use_sem:
            for r in rows:
                v = emb.get(r["listing_id"])
                sims[r["listing_id"]] = float(np.dot(query_vec, v)) if v is not None else 0.0
            svals = list(sims.values())
            slo, shi = (min(svals), max(svals)) if svals else (0.0, 0.0)

        def score(r):
            s = 0.0
            for e in rank:
                fld = e["field"]
                lo, hi = ranges[fld]
                w = num(e.get("weight")) or 1.0
                v = r.get(fld)
                nrm = 0.5 if (v is None or hi == lo) else (v - lo) / (hi - lo)
                if e.get("direction") == "asc":
                    nrm = 1.0 - nrm
                s += w * nrm
            if use_sem:
                sv = sims[r["listing_id"]]
                s += 0.5 if shi == slo else (sv - slo) / (shi - slo)
            return s

        rows.sort(key=score, reverse=True)

    limit = max(1, min(int(num(plan.get("limit")) or 24), 60))
    return rows[:limit]


def ai_search(query: str, ref_id: str):
    """Return (plan, rows). One cached model call builds the plan; we execute it
    locally. Cache key includes the liked set so 'based on my likes' refreshes
    when your likes change."""
    sugg = _STATE["suggestions"]
    if not sugg:
        return {"explanation": "No listings to search."}, []

    liked_ids = tuple(sorted(r["listing_id"] for r in sugg if r["reaction"] == "liked"))
    key = (query.strip().lower(), ref_id, liked_ids)

    if key in _SEARCH_CACHE:
        plan = _SEARCH_CACHE[key]
    else:
        ctx = _build_context(sugg, ref_id)
        user = json.dumps({"request": query, "context": ctx}, default=str)
        raw = _STATE["llm"].complete(SEARCH_SYSTEM, user, max_tokens=700)
        plan = llm_mod.loads_lenient(raw)
        _SEARCH_CACHE[key] = plan

    query_vec = None
    sq = (plan.get("semantic_query") or "").strip()
    if sq and _STATE["embedder"] is not None and _STATE["embeddings"]:
        try:
            query_vec = _STATE["embedder"].encode([sq])[0]
        except Exception:
            query_vec = None

    return plan, _apply_plan(plan, sugg, query_vec=query_vec)


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


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    if request.args.get("refresh") or not _STATE["suggestions"]:
        _STATE.update(build_suggestions(_STATE["db_path"]))
        _STATE["embeddings"] = _load_embeddings(_STATE["db_path"])

    # Overlay current reactions from the DB so hearts reflect persisted state.
    with dbm.open_db(_STATE["db_path"]) as conn:
        reactions = dbm.get_reactions(conn)
    for r in _STATE["suggestions"]:
        r["reaction"] = reactions.get(r["listing_id"])

    base = list(_STATE["suggestions"])
    meta = _STATE["meta"]

    show = request.args.get("show", "under")           # under | all | liked | passed
    sort = request.args.get("sort", "shuffle")         # shuffle | deal | rent_asc | rent_desc | quality
    nh = request.args.get("nh", "")
    nh_set = {a for a in nh.split(",") if a}   # multi-area filter (comma-separated)
    try:
        min_beds = int(float(request.args.get("beds", "") or 0))
    except ValueError:
        min_beds = 0

    q = request.args.get("q", "").strip()
    ref = request.args.get("ref", "").strip()
    tag = request.args.get("tag", "").strip()
    match = request.args.get("match", "").strip()
    similar = request.args.get("similar", "").strip()
    ai = _STATE["ai"]
    ref_name = ""
    if ref:
        m = next((r for r in base if r["listing_id"] == ref), None)
        ref_name = m["name"] if m else ""

    ai_banner = None
    searching = False
    distances = {}   # listing_id -> miles from reference, for "more like this"

    if q and ai["enabled"]:
        # AI search supersedes the dropdown filters; the plan controls the set.
        searching = True
        try:
            plan, rows = ai_search(q, ref)
            ai_banner = "🔎 " + (plan.get("explanation") or "Results")
        except Exception as e:  # model/parse failure — surface it, show nothing
            rows = []
            ai_banner = f"⚠ AI search failed: {e}"
    elif similar:
        # "More like this" — style-similar to one reference, ordered by vicinity.
        searching = True
        rows, ai_banner, distances = similar_to(similar)
    elif match == "likes":
        # Taste matching — pure local vector math, no model call.
        searching = True
        rows, ai_banner = match_likes()
    else:
        # Standard browse path. Passed listings are hidden except in their view.
        rows = base
        if show == "passed":
            rows = [r for r in rows if r["reaction"] == "passed"]
        elif show == "liked":
            rows = [r for r in rows if r["reaction"] == "liked"]
        elif show == "under":
            rows = [r for r in rows if r["pct_diff"] < 0 and r["reaction"] != "passed"]
        else:  # all
            rows = [r for r in rows if r["reaction"] != "passed"]
        if nh_set:
            rows = [r for r in rows if r["neighborhood"] in nh_set]
        if min_beds:
            rows = [r for r in rows if (r["beds"] or 0) >= min_beds]
        if tag:
            rows = [r for r in rows if tag in (r.get("tags") or [])]

        if sort == "shuffle":
            # Reshuffled every load so same-building units (which often share a
            # hero image) don't cluster and look like duplicates.
            random.shuffle(rows)
        elif sort == "deal":
            rows.sort(key=lambda r: r["pct_diff"])
        elif sort == "rent_asc":
            rows.sort(key=lambda r: r["rent"])
        elif sort == "rent_desc":
            rows.sort(key=lambda r: -r["rent"])
        elif sort == "quality":
            rows.sort(key=lambda r: -(r["apt_quality"] or 0))

    liked_total = sum(1 for r in _STATE["suggestions"] if r["reaction"] == "liked")
    passed_total = sum(1 for r in _STATE["suggestions"] if r["reaction"] == "passed")

    html = render_template(
        "index.html", rows=rows, meta=meta,
        liked_total=liked_total, passed_total=passed_total,
        show=show, sort=sort, nh=nh, nh_list=sorted(nh_set), min_beds=str(min_beds),
        q=q, ref=ref, ref_name=ref_name, searching=searching, active_tag=tag,
        ai_enabled=ai["enabled"], ai_provider=ai["provider"],
        ai_model=ai["model"], ai_banner=ai_banner, distances=distances,
        emb_ready=bool(_STATE["embeddings"]),
    )
    # Don't let the browser cache the grid — otherwise a reload serves the old
    # HTML and the Shuffle sort looks "stuck" on the same order.
    return html, 200, {"Cache-Control": "no-store"}


@app.route("/listing/<listing_id>")
def listing_detail(listing_id: str):
    """Full consumer-style detail page: gallery, description, amenities, map,
    model price vs. asking, and a 'more like this' jump-off."""
    if not _STATE["suggestions"]:
        _STATE.update(build_suggestions(_STATE["db_path"]))
        _STATE["embeddings"] = _load_embeddings(_STATE["db_path"])

    # The model-derived fields (predicted rent, % vs model, photo quality, tags)
    # live on the in-memory suggestion row; the rest comes straight from the DB.
    srow = next((r for r in _STATE["suggestions"] if r["listing_id"] == listing_id), None)
    with dbm.open_db(_STATE["db_path"]) as conn:
        listing = dbm.get_listing(conn, listing_id)
        if not listing:
            abort(404)
        amenities = dbm.get_amenities(conn, listing_id)
        images = dbm.get_listing_images(conn, listing_id)
        reaction = dbm.get_reactions(conn).get(listing_id)

    positions = [im["position"] for im in images]
    tags = srow.get("tags", []) if srow else []
    return render_template(
        "listing.html",
        l=listing, s=srow, amenities=amenities, positions=positions,
        tags=tags, reaction=reaction,
    )


@app.route("/api/react", methods=["POST"])
def api_react():
    data = request.get_json(silent=True) or {}
    listing_id = data.get("listing_id")
    reaction = data.get("reaction")
    if not listing_id:
        return jsonify({"ok": False, "error": "missing listing_id"}), 400
    if reaction not in ("liked", "passed", "none", "", None):
        return jsonify({"ok": False, "error": "bad reaction"}), 400

    norm = None if reaction in ("none", "", None) else reaction
    with dbm.open_db(_STATE["db_path"]) as conn:
        dbm.set_reaction(conn, listing_id, norm)

    for r in _STATE["suggestions"]:
        if r["listing_id"] == listing_id:
            r["reaction"] = norm
            break

    return jsonify({"ok": True, "reaction": norm})


@app.route("/img/<listing_id>/<int:pos>")
def image(listing_id: str, pos: int):
    """Serve a listing photo from its local path, falling back to the CDN URL."""
    with dbm.open_db(_STATE["db_path"]) as conn:
        images = dbm.get_listing_images(conn, listing_id)
    for im in images:
        if im["position"] == pos:
            path = im["local_image_path"]
            if path and os.path.exists(path):
                return send_file(os.path.abspath(path))
            if im["image_url"]:
                return redirect(im["image_url"])
    abort(404)


# ── Pipeline control routes ───────────────────────────────────────────────────


@app.route("/run")
def run_panel():
    """Control-panel page: set params, launch a stage, watch live progress."""
    return render_template(
        "run.html", ai=_STATE["ai"], db_path=_STATE["db_path"], areas=AREA_OPTIONS
    )


@app.route("/api/run/<stage>", methods=["POST"])
def api_run(stage: str):
    if stage not in _STAGES:
        return jsonify({"ok": False, "error": f"unknown stage {stage!r}"}), 400
    params = request.get_json(silent=True) or {}
    started, msg = start_job(stage, params)
    if not started:
        return jsonify({"ok": False, "error": msg}), 409
    return jsonify({"ok": True, "stage": stage})


@app.route("/api/run/log")
def api_run_log():
    """Return job status plus any log lines after ``cursor`` (incremental poll)."""
    try:
        cursor = int(request.args.get("cursor", 0))
    except ValueError:
        cursor = 0
    with _JOB_LOCK:
        lines = _JOB["lines"]
        return jsonify({
            "stage": _JOB["stage"],
            "status": _JOB["status"],
            "returncode": _JOB["returncode"],
            "cursor": len(lines),
            "lines": lines[max(0, cursor):],
        })


@app.route("/api/run/stop", methods=["POST"])
def api_run_stop():
    with _JOB_LOCK:
        proc = _JOB.get("proc")
        if proc and _JOB["status"] == "running":
            proc.terminate()
            _JOB["status"] = "stopped"
            _JOB["lines"].append("── stopped by user ──")
            return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "no running job"}), 409


# ── Area options for the Pipeline UI ──────────────────────────────────────────
#
# StreetEasy area IDs -> neighborhood names for the area picker. Chelsea,
# Gramercy Park, Flatiron, Hudson Yards, and Upper East Side are confirmed by
# scrape_listings.py's docstring; the rest are the working set used in this
# repo. The numeric ID is shown alongside each name in the UI so it stays
# verifiable.
AREA_OPTIONS = [
    ("104", "Financial District"),
    ("105", "Tribeca"),
    ("106", "Stuyvesant Town/PCV"),
    ("113", "Chelsea"),
    ("115", "Gramercy Park"),
    ("116", "Greenwich Village"),
    ("117", "East Village"),
    ("130", "Turtle Bay"),
    ("131", "Kips Bay"),
    ("133", "Sutton Place"),
    ("140", "Upper East Side"),
    ("146", "Hudson Yards"),
    ("152", "West Village"),
    ("157", "Murray Hill"),
    ("158", "Flatiron"),
    ("162", "Nolita"),
]


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Web UI for suggested (underpriced) listings.")
    p.add_argument("--db", default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini",
                   help="Model provider for AI search (needs the matching API key)")
    p.add_argument("--model", default=None,
                   help="Model for AI search (default: per-provider — gemini-2.5-flash "
                        "or claude-opus-4-8; claude-haiku-4-5 is cheaper)")
    p.add_argument("--embed-model", default=emb_mod.DEFAULT_MODEL,
                   help="sentence-transformers model for query embedding (semantic search)")
    p.add_argument("--host", default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=5000, help="Bind port")
    p.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    args = p.parse_args()

    load_dotenv()
    _STATE["db_path"] = args.db

    model = args.model or llm_mod.DEFAULT_MODELS.get(args.provider)
    engine = llm_mod.build_llm(args.provider, model)
    _STATE["llm"] = engine
    _STATE["ai"] = {"enabled": engine is not None, "provider": args.provider, "model": model}
    _STATE["embedder"] = emb_mod.try_build_embedder(args.embed_model)
    _STATE["embeddings"] = _load_embeddings(args.db)

    print(f"Building suggestions from {args.db} …")
    _STATE.update(build_suggestions(args.db))
    m = _STATE["meta"]
    print(f"  {m.get('underpriced', 0)} underpriced of {m.get('total', 0)} scored listings")
    if engine:
        print(f"  AI search: enabled ({args.provider} / {model})")
    else:
        key = "GOOGLE_API_KEY" if args.provider == "gemini" else "ANTHROPIC_API_KEY"
        print(f"  AI search: disabled — set {key} (in .env) to enable")
    n_emb = len(_STATE["embeddings"])
    if n_emb:
        sem = "on" if _STATE["embedder"] is not None else "stored-only (install sentence-transformers)"
        print(f"  Semantic: {n_emb} embeddings loaded — Match-my-likes on, free-text {sem}")
    else:
        print("  Semantic: no embeddings — run `python embed_listings.py` to enable taste matching")
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
