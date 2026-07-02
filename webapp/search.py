"""Natural-language search — cost-smart: one small model call builds a query
plan, then we execute it locally over the in-memory suggestion set.

The model never sees the full listings, only summary facets (:func:`_facets`),
an optional reference listing, and a profile of the user's likes
(:func:`_build_context`). :func:`_apply_plan` runs the returned plan — filters,
multi-field ranking, optional distance and description-embedding terms — with no
further model calls. Plans are cached per (query, reference, liked-set).
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

import geo
import llm as llm_mod
from webapp.search_prompt import SEARCH_SYSTEM
from webapp.state import _SEARCH_CACHE, _STATE

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


def _apply_plan(plan: dict, sugg: list[dict], query_vec=None, ref_point=None) -> list[dict]:
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

    raw_rank = plan.get("rank_by") or []
    rank = [e for e in raw_rank if e.get("field") in _RANK_FIELDS]
    # "distance" ranks by proximity to the reference (only valid with a ref_point).
    dist_rank = [e for e in raw_rank if e.get("field") == "distance"] if ref_point else []
    emb = _STATE["embeddings"]
    use_sem = query_vec is not None and bool(emb)

    if (rank or dist_rank or use_sem) and rows:
        ranges = {}
        for e in rank:
            fld = e["field"]
            vals = [r[fld] for r in rows if r.get(fld) is not None]
            ranges[fld] = (min(vals), max(vals)) if vals else (0.0, 0.0)

        dists, dlo, dhi = {}, 0.0, 0.0
        if dist_rank:
            for r in rows:
                dists[r["listing_id"]] = geo.haversine_mi(
                    ref_point[0], ref_point[1], r.get("lat"), r.get("lng"))
            dvals = [d for d in dists.values() if d is not None]
            dlo, dhi = (min(dvals), max(dvals)) if dvals else (0.0, 0.0)

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
            for e in dist_rank:
                w = num(e.get("weight")) or 1.0
                d = dists.get(r["listing_id"])
                nrm = 0.5 if (d is None or dhi == dlo) else (d - dlo) / (dhi - dlo)
                if e.get("direction") != "desc":   # default: nearer is better
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

    ref_point = None
    if ref_id:
        ref = next((r for r in sugg if r["listing_id"] == ref_id), None)
        if ref and ref.get("lat") is not None and ref.get("lng") is not None:
            ref_point = (ref["lat"], ref["lng"])

    return plan, _apply_plan(plan, sugg, query_vec=query_vec, ref_point=ref_point)
