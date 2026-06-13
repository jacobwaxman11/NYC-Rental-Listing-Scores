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
from collections import Counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask, abort, jsonify, redirect, render_template_string, request, send_file,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import db as dbm
import llm as llm_mod
from features import collapse_amenity_tiers, collapse_image_scores, prepare_features


app = Flask(__name__)

# Populated at startup (and on ?refresh=1) by build_suggestions().
# "llm" holds a TextLLM or None (AI search disabled). "_search_cache" memoizes
# query plans so repeat questions cost nothing.
_STATE: dict = {
    "suggestions": [], "meta": {}, "db_path": dbm.DEFAULT_DB_PATH,
    "llm": None, "ai": {"enabled": False, "provider": None, "model": None},
}
_SEARCH_CACHE: dict = {}


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


def _choose_image_position(conn, listing_id: str):
    """Pick a representative photo: the brightest apartment shot, else the
    first non-irrelevant photo, else the first photo. Returns a position or None."""
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

            actual = float(actual_rent[i])
            predicted = float(pred_rent[i])
            pct_diff = (actual - predicted) / predicted if predicted else 0.0

            rows.append({
                "listing_id": listing_id,
                "name": listing.get("name") or listing.get("street") or listing_id,
                "neighborhood": listing.get("neighborhood") or "—",
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
                "img_pos": _choose_image_position(conn, listing_id),
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
  "limit": number
}

Rules:
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


def _apply_plan(plan: dict, sugg: list[dict]) -> list[dict]:
    """Execute a query plan locally over the listings — no model calls here."""
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
    if rank and rows:
        ranges = {}
        for e in rank:
            fld = e["field"]
            vals = [r[fld] for r in rows if r.get(fld) is not None]
            ranges[fld] = (min(vals), max(vals)) if vals else (0.0, 0.0)

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

    return plan, _apply_plan(plan, sugg)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    if request.args.get("refresh") or not _STATE["suggestions"]:
        _STATE.update(build_suggestions(_STATE["db_path"]))

    # Overlay current reactions from the DB so hearts reflect persisted state.
    with dbm.open_db(_STATE["db_path"]) as conn:
        reactions = dbm.get_reactions(conn)
    for r in _STATE["suggestions"]:
        r["reaction"] = reactions.get(r["listing_id"])

    base = list(_STATE["suggestions"])
    meta = _STATE["meta"]

    show = request.args.get("show", "under")           # under | all | liked | passed
    sort = request.args.get("sort", "deal")            # deal | rent_asc | rent_desc | quality
    nh = request.args.get("nh", "")
    try:
        min_beds = int(float(request.args.get("beds", "") or 0))
    except ValueError:
        min_beds = 0

    q = request.args.get("q", "").strip()
    ref = request.args.get("ref", "").strip()
    tag = request.args.get("tag", "").strip()
    ai = _STATE["ai"]
    ref_name = ""
    if ref:
        m = next((r for r in base if r["listing_id"] == ref), None)
        ref_name = m["name"] if m else ""

    ai_banner = None
    searching = False

    if q and ai["enabled"]:
        # AI search supersedes the dropdown filters; the plan controls the set.
        searching = True
        try:
            plan, rows = ai_search(q, ref)
            ai_banner = "🔎 " + (plan.get("explanation") or "Results")
        except Exception as e:  # model/parse failure — surface it, show nothing
            rows = []
            ai_banner = f"⚠ AI search failed: {e}"
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
        if nh:
            rows = [r for r in rows if r["neighborhood"] == nh]
        if min_beds:
            rows = [r for r in rows if (r["beds"] or 0) >= min_beds]
        if tag:
            rows = [r for r in rows if tag in (r.get("tags") or [])]

        if sort == "deal":
            rows.sort(key=lambda r: r["pct_diff"])
        elif sort == "rent_asc":
            rows.sort(key=lambda r: r["rent"])
        elif sort == "rent_desc":
            rows.sort(key=lambda r: -r["rent"])
        elif sort == "quality":
            rows.sort(key=lambda r: -(r["apt_quality"] or 0))

    liked_total = sum(1 for r in _STATE["suggestions"] if r["reaction"] == "liked")
    passed_total = sum(1 for r in _STATE["suggestions"] if r["reaction"] == "passed")

    return render_template_string(
        TEMPLATE, rows=rows, meta=meta,
        liked_total=liked_total, passed_total=passed_total,
        show=show, sort=sort, nh=nh, min_beds=str(min_beds),
        q=q, ref=ref, ref_name=ref_name, searching=searching, active_tag=tag,
        ai_enabled=ai["enabled"], ai_provider=ai["provider"],
        ai_model=ai["model"], ai_banner=ai_banner,
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
    return render_template_string(
        RUN_TEMPLATE, ai=_STATE["ai"], db_path=_STATE["db_path"]
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


# ── Template ─────────────────────────────────────────────────────────────────


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NYC Rental Deals</title>
<style>
  :root { --bg:#0f1115; --card:#181b22; --line:#262b35; --txt:#e6e8ec;
          --muted:#9aa3b2; --good:#34d399; --bad:#f87171; --accent:#60a5fa;
          --heart:#fb7185; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { padding:20px 28px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  h1 { font-size:20px; margin:0; font-weight:650; }
  .sub { color:var(--muted); font-size:13px; }
  .tools { margin-left:auto; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  a.refresh { color:var(--accent); text-decoration:none; font-size:13px; align-self:center; }

  /* custom dropdown */
  .cs { position:relative; user-select:none; }
  .cs-btn { background:var(--card); color:var(--txt); border:1px solid var(--line);
            border-radius:9px; padding:8px 12px; font-size:13px; cursor:pointer;
            display:flex; align-items:center; gap:8px; min-width:130px;
            justify-content:space-between; transition:border-color .15s; }
  .cs-btn:hover { border-color:#3a4252; }
  .cs.open .cs-btn { border-color:var(--accent); }
  .cs-btn i { font-style:normal; color:var(--muted); font-size:11px; transition:transform .15s; }
  .cs.open .cs-btn i { transform:rotate(180deg); }
  .cs-menu { position:absolute; top:calc(100% + 6px); left:0; right:0; z-index:30;
             background:var(--card); border:1px solid var(--line); border-radius:10px;
             padding:5px; margin:0; list-style:none; box-shadow:0 12px 30px rgba(0,0,0,.45);
             max-height:300px; overflow:auto; display:none; }
  .cs.open .cs-menu { display:block; }
  .cs-menu li { padding:8px 10px; border-radius:7px; cursor:pointer; font-size:13px; }
  .cs-menu li:hover { background:#222732; }
  .cs-menu li.sel { color:var(--accent); }
  .cs-menu li.sel::after { content:"✓"; float:right; }

  .tinder-open { background:linear-gradient(135deg,#fb7185,#f59e0b); color:#fff;
                 border:none; border-radius:9px; padding:9px 14px; font-size:13px;
                 font-weight:650; cursor:pointer; }
  .tinder-open:disabled { opacity:.4; cursor:default; }

  /* AI search bar */
  .searchbar { padding:14px 28px; border-bottom:1px solid var(--line);
               display:flex; flex-direction:column; gap:8px; }
  .searchbar form { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .searchbar input[type=text] { flex:1; min-width:260px; background:var(--card);
    color:var(--txt); border:1px solid var(--line); border-radius:10px;
    padding:11px 14px; font-size:14px; }
  .searchbar input[type=text]:focus { outline:none; border-color:var(--accent); }
  .searchbar input[type=text]:disabled { opacity:.6; }
  .ask { background:linear-gradient(135deg,#60a5fa,#34d399); color:#06281f; border:none;
         border-radius:10px; padding:11px 18px; font-weight:700; font-size:14px; cursor:pointer; }
  .ask:disabled { opacity:.4; cursor:default; }
  .chip { display:inline-flex; align-items:center; gap:6px; background:#1f2734;
          border:1px solid var(--line); border-radius:999px; padding:5px 6px 5px 12px;
          font-size:13px; color:var(--muted); }
  .chip.hidden { display:none; }
  .chip b { color:var(--txt); font-weight:600; max-width:160px; overflow:hidden;
            text-overflow:ellipsis; white-space:nowrap; }
  .chip button { background:none; border:none; color:var(--muted); cursor:pointer; font-size:13px; }
  .ai-banner { color:var(--txt); font-size:13px; }
  .ai-banner a { color:var(--accent); text-decoration:none; }
  .ai-note { color:var(--muted); font-size:12px; }
  .foot-actions { display:flex; gap:12px; align-items:center; }
  .similar { background:none; border:none; color:var(--accent); cursor:pointer;
             font-size:13px; padding:0; }
  .tags { display:flex; flex-wrap:wrap; gap:5px; }
  .tag { font-size:11px; background:#1f2734; border:1px solid var(--line);
         color:var(--muted); border-radius:6px; padding:2px 7px; text-decoration:none; }
  .tag:hover { color:var(--txt); border-color:#3a4252; }
  .tag.active { color:var(--accent); border-color:var(--accent); }

  main { padding:24px 28px; display:grid; gap:18px;
         grid-template-columns:repeat(auto-fill,minmax(290px,1fr)); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px;
          overflow:hidden; display:flex; flex-direction:column; position:relative; }
  .photo { aspect-ratio:4/3; background:#0b0d11 center/cover no-repeat; position:relative; }
  .heart { position:absolute; top:10px; right:10px; width:36px; height:36px;
           border:none; border-radius:50%; background:rgba(15,17,21,.55);
           color:#fff; font-size:18px; cursor:pointer; backdrop-filter:blur(4px);
           display:flex; align-items:center; justify-content:center; transition:transform .12s; }
  .heart:hover { transform:scale(1.12); }
  .heart.on { color:var(--heart); }
  .undo { position:absolute; top:10px; left:10px; width:36px; height:36px;
          border:none; border-radius:50%; background:rgba(15,17,21,.55);
          color:#fff; font-size:16px; cursor:pointer; backdrop-filter:blur(4px);
          display:flex; align-items:center; justify-content:center; transition:transform .12s; }
  .undo:hover { transform:scale(1.12); }
  .body { padding:14px 16px 16px; display:flex; flex-direction:column; gap:8px; }
  .addr { font-weight:600; line-height:1.3; }
  .meta { color:var(--muted); font-size:13px; }
  .prices { display:flex; align-items:baseline; gap:10px; margin-top:2px; }
  .rent { font-size:22px; font-weight:700; }
  .pred { color:var(--muted); font-size:13px; }
  .badge { align-self:flex-start; padding:3px 9px; border-radius:999px;
           font-size:12px; font-weight:650; }
  .badge.good { background:rgba(52,211,153,.14); color:var(--good); }
  .badge.bad  { background:rgba(248,113,113,.14); color:var(--bad); }
  .foot { display:flex; justify-content:space-between; align-items:center;
          margin-top:4px; font-size:13px; color:var(--muted); }
  .q { color:var(--txt); }
  .empty { padding:60px 28px; color:var(--muted); max-width:620px; }
  code { background:var(--card); padding:2px 6px; border-radius:6px; }
  a.view { color:var(--accent); text-decoration:none; }

  /* Tinder mode */
  .tinder { position:fixed; inset:0; background:rgba(8,9,12,.92); z-index:100;
            display:flex; flex-direction:column; align-items:center; padding:18px; }
  .tinder.hidden { display:none; }
  .t-head { width:100%; max-width:420px; display:flex; justify-content:space-between;
            align-items:center; color:var(--muted); font-size:14px; }
  .t-head button { background:none; border:none; color:var(--txt); font-size:22px; cursor:pointer; }
  .t-stage { position:relative; width:min(420px,92vw); height:min(560px,68vh);
             margin:14px 0; }
  .t-card { position:absolute; inset:0; background:var(--card); border:1px solid var(--line);
            border-radius:18px; overflow:hidden; display:flex; flex-direction:column;
            box-shadow:0 20px 50px rgba(0,0,0,.5); touch-action:none; cursor:grab; }
  .t-card.behind { transform:scale(.95) translateY(14px); filter:brightness(.7); }
  .t-photo { flex:1; background:#0b0d11 center/cover no-repeat; }
  .t-info { padding:14px 18px 18px; display:flex; flex-direction:column; gap:7px; }
  .t-badge { align-self:flex-start; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:650; }
  .t-badge.good { background:rgba(52,211,153,.16); color:var(--good); }
  .t-badge.bad  { background:rgba(248,113,113,.16); color:var(--bad); }
  .t-addr { font-weight:650; font-size:17px; }
  .t-meta { color:var(--muted); font-size:13px; }
  .t-tags { display:flex; flex-wrap:wrap; gap:5px; margin-top:2px; }
  .t-tag { font-size:11px; background:#1f2734; border:1px solid var(--line);
           color:var(--muted); border-radius:6px; padding:2px 7px; }
  .t-price { font-size:20px; font-weight:700; }
  .t-price span { color:var(--muted); font-size:13px; font-weight:400; }
  .stamp { position:absolute; top:24px; padding:6px 14px; border:4px solid; border-radius:10px;
           font-weight:800; font-size:26px; letter-spacing:2px; opacity:0; transform:rotate(-15deg); }
  .stamp.like { left:20px; color:var(--good); border-color:var(--good); }
  .stamp.pass { right:20px; color:var(--bad); border-color:var(--bad); transform:rotate(15deg); }
  .t-actions { display:flex; gap:22px; margin-top:6px; }
  .t-btn { width:64px; height:64px; border-radius:50%; border:1px solid var(--line);
           background:var(--card); font-size:24px; cursor:pointer; transition:transform .12s; }
  .t-btn:hover { transform:scale(1.08); }
  .t-btn.pass { color:var(--bad); }
  .t-btn.like { color:var(--good); }
  .t-undo { width:48px; height:48px; font-size:18px; color:var(--muted); align-self:center; }
  .t-undo:disabled { opacity:.35; cursor:default; transform:none; }
  .t-hint { color:var(--muted); font-size:12px; margin-top:10px; }
  .t-toast { position:fixed; bottom:26px; left:50%; transform:translateX(-50%);
             background:var(--card); border:1px solid var(--line); border-radius:999px;
             padding:9px 10px 9px 16px; display:flex; align-items:center; gap:12px;
             font-size:13px; box-shadow:0 10px 30px rgba(0,0,0,.5); z-index:110;
             transition:opacity .2s, transform .2s; }
  .t-toast.hidden { opacity:0; pointer-events:none; transform:translate(-50%, 8px); }
  .t-toast button { background:#222732; border:1px solid var(--line); color:var(--accent);
                    border-radius:999px; padding:5px 12px; font-size:13px; cursor:pointer; }
  .t-done { color:var(--muted); text-align:center; margin-top:40px; }
  .t-done button { background:none; border:none; color:var(--accent); cursor:pointer; font-size:15px; }
</style>
</head>
<body>

{% macro csel(param, current, options) %}
<div class="cs" data-param="{{ param }}">
  <button type="button" class="cs-btn">
    <span>{{ options[current] if current in options else (options.values()|list)[0] }}</span><i>▾</i>
  </button>
  <ul class="cs-menu">
    {% for val, label in options.items() %}
      <li data-value="{{ val }}" class="{{ 'sel' if val == current }}">{{ label }}</li>
    {% endfor %}
  </ul>
</div>
{% endmacro %}

<header>
  <h1>NYC Rental Deals</h1>
  <span class="sub">
    {{ meta.underpriced }} below-model of {{ meta.total }} scored ·
    {{ liked_total }} ❤ · {{ passed_total }} ✕ ·
    {{ "out-of-fold (grouped by building)" if meta.cv else "in-sample fit" }}
  </span>
  <div class="tools">
    {{ csel('show', show, {'under':'Underpriced only','all':'All listings','liked':'❤ Liked','passed':'✕ Passed'}) }}
    {{ csel('sort', sort, {'deal':'Biggest discount','rent_asc':'Rent: low → high','rent_desc':'Rent: high → low','quality':'Photo quality'}) }}
    {{ csel('beds', min_beds, {'0':'Any beds','1':'1+ bed','2':'2+ bed','3':'3+ bed'}) }}

    <div class="cs" data-param="nh">
      <button type="button" class="cs-btn"><span>{{ nh if nh else 'All areas' }}</span><i>▾</i></button>
      <ul class="cs-menu">
        <li data-value="" class="{{ 'sel' if not nh }}">All areas</li>
        {% for n in meta.neighborhoods %}
          <li data-value="{{ n }}" class="{{ 'sel' if nh == n }}">{{ n }}</li>
        {% endfor %}
      </ul>
    </div>

    <button class="tinder-open" id="tinder-open" {{ 'disabled' if not rows }}>🔥 Tinder mode</button>
    <a class="refresh" href="/run">⚙ Pipeline</a>
    <a class="refresh" href="/?refresh=1">↻ recompute</a>
  </div>
</header>

<div class="searchbar">
  <form method="get">
    <input type="text" name="q" id="q" autocomplete="off" value="{{ q }}"
           placeholder="Ask AI: “like this but bigger” · “based on what I liked, find ones I haven't reviewed”"
           {{ 'disabled' if not ai_enabled }}>
    <input type="hidden" name="ref" id="ref" value="{{ ref }}">
    <span id="ref-chip" class="chip {{ '' if ref else 'hidden' }}">
      like: <b id="ref-name">{{ ref_name }}</b>
      <button type="button" id="ref-clear" title="Clear reference">✕</button>
    </span>
    <button type="submit" class="ask" {{ 'disabled' if not ai_enabled }}>Ask AI</button>
  </form>
  {% if ai_banner %}
    <div class="ai-banner">{{ ai_banner }} · {{ rows|length }} result{{ '' if rows|length == 1 else 's' }} · <a href="/">clear</a></div>
  {% endif %}
  {% if active_tag and not searching %}
    <div class="ai-banner">🏷 {{ active_tag.replace('_', ' ') }} · {{ rows|length }} result{{ '' if rows|length == 1 else 's' }} · <a href="/">clear</a></div>
  {% endif %}
  {% if not ai_enabled %}
    <div class="ai-note">AI search is off — start the server with <code>--provider gemini</code>
      (or <code>anthropic</code>) and set the matching API key in <code>.env</code> to enable.</div>
  {% endif %}
</div>

{% if not rows %}
  <div class="empty">
    {% if meta.total == 0 %}
      <p>No scored listings yet. Run the pipeline first:</p>
      <p><code>python scrape_listings.py</code> →
         <code>python backfill_details.py</code> →
         <code>python score_listings.py</code></p>
      <p>Then reload this page.</p>
    {% else %}
      <p>No listings match these filters. Try “All listings” or clearing the area/beds filter.</p>
    {% endif %}
  </div>
{% else %}
  <main>
  {% for r in rows %}
    <div class="card">
      <div class="photo" {% if r.img_pos is not none %}style="background-image:url('/img/{{ r.listing_id }}/{{ r.img_pos }}')"{% endif %}>
        <button class="heart {{ 'on' if r.reaction == 'liked' }}" data-id="{{ r.listing_id }}" title="Save">♥</button>
        {% if show == 'passed' %}<button class="undo" data-id="{{ r.listing_id }}" title="Restore">↩</button>{% endif %}
      </div>
      <div class="body">
        {% if r.pct_diff < 0 %}
          <span class="badge good">{{ '%.0f'|format(-r.pct_diff*100) }}% below model · save ${{ '{:,}'.format(r.delta) }}/mo</span>
        {% else %}
          <span class="badge bad">{{ '%.0f'|format(r.pct_diff*100) }}% above model</span>
        {% endif %}
        <div class="addr">{{ r.name }}</div>
        <div class="meta">
          {{ r.neighborhood }} ·
          {{ (r.beds|int) if r.beds is not none else '?' }} bd /
          {{ r.baths if r.baths is not none else '?' }} ba{% if r.sqft %} · {{ '{:,}'.format(r.sqft) }} ft²{% endif %}
        </div>
        {% if r.tags %}
          <div class="tags">
            {% for t in r.tags[:6] %}<a class="tag {{ 'active' if t == active_tag }}" href="/?tag={{ t }}">{{ t.replace('_', ' ') }}</a>{% endfor %}
          </div>
        {% endif %}
        <div class="prices">
          <span class="rent">${{ '{:,}'.format(r.rent) }}</span>
          <span class="pred">model ${{ '{:,}'.format(r.predicted) }}/mo</span>
        </div>
        <div class="foot">
          <span>photo quality: <span class="q">{{ r.apt_quality if r.apt_quality is not none else '—' }}</span>/10
            {% if r.photos %}· {{ r.photos }} photos{% endif %}</span>
          <span class="foot-actions">
            {% if ai_enabled %}<button class="similar" data-id="{{ r.listing_id }}" data-name="{{ r.name }}" title="Find similar with AI">✨ similar</button>{% endif %}
            {% if r.url %}<a class="view" href="{{ r.url }}" target="_blank" rel="noopener">view ↗</a>{% endif %}
          </span>
        </div>
      </div>
    </div>
  {% endfor %}
  </main>
{% endif %}

<!-- Tinder mode -->
<div id="tinder" class="tinder hidden">
  <div class="t-head">
    <span id="t-count"></span>
    <button id="t-close" title="Close">✕</button>
  </div>
  <div id="t-stage" class="t-stage"></div>
  <div class="t-actions">
    <button class="t-btn t-undo" id="t-undo" title="Undo (z)" disabled>↩</button>
    <button class="t-btn pass" id="t-pass" title="Pass (←)">✕</button>
    <button class="t-btn like" id="t-like" title="Like (→)">♥</button>
  </div>
  <div class="t-hint">drag the card, use the buttons, or press ← / → · z to undo</div>
  <div id="t-done" class="t-done hidden">
    <p>That's everything in this view.</p>
    <button id="t-back">← back to grid</button>
  </div>
  <div id="t-toast" class="t-toast hidden">
    <span id="t-toast-msg"></span>
    <button id="t-toast-undo">Undo</button>
  </div>
</div>

<script>
const DECK = {{ rows|tojson }};

// ── Navigation helper (preserves other query params) ──
function go(param, value) {
  const u = new URL(window.location);
  u.searchParams.set(param, value);
  u.searchParams.delete('refresh');
  window.location = u;
}

// ── Custom dropdowns ──
document.querySelectorAll('.cs').forEach(cs => {
  const btn = cs.querySelector('.cs-btn');
  btn.addEventListener('click', e => {
    e.stopPropagation();
    document.querySelectorAll('.cs.open').forEach(o => { if (o !== cs) o.classList.remove('open'); });
    cs.classList.toggle('open');
  });
  cs.querySelectorAll('.cs-menu li').forEach(li => {
    li.addEventListener('click', () => go(cs.dataset.param, li.dataset.value));
  });
});
document.addEventListener('click', () => document.querySelectorAll('.cs.open').forEach(o => o.classList.remove('open')));

// ── Reactions (hearts + swipes) ──
async function react(id, reaction) {
  const res = await fetch('/api/react', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({listing_id: id, reaction}),
  });
  return res.json();
}
function paintHeart(id, liked) {
  const h = document.querySelector('.heart[data-id="' + CSS.escape(id) + '"]');
  if (h) h.classList.toggle('on', liked);
}
document.querySelectorAll('.heart').forEach(h => {
  h.addEventListener('click', async e => {
    e.preventDefault(); e.stopPropagation();
    const liked = h.classList.contains('on');
    const out = await react(h.dataset.id, liked ? 'none' : 'liked');
    h.classList.toggle('on', out.reaction === 'liked');
  });
});
// Undo (restore) a passed listing — clears the reaction and drops the card.
document.querySelectorAll('.undo').forEach(b => {
  b.addEventListener('click', async e => {
    e.preventDefault(); e.stopPropagation();
    await react(b.dataset.id, 'none');
    b.closest('.card').remove();
  });
});

// ── AI "✨ similar": set this listing as the reference, focus the search box ──
const refInput = document.getElementById('ref');
const refChip = document.getElementById('ref-chip');
document.querySelectorAll('.similar').forEach(b => {
  b.addEventListener('click', e => {
    e.preventDefault(); e.stopPropagation();
    refInput.value = b.dataset.id;
    document.getElementById('ref-name').textContent = b.dataset.name;
    refChip.classList.remove('hidden');
    const q = document.getElementById('q');
    q.focus();
    if (!q.value.trim()) q.value = 'like this but ';
    q.setSelectionRange(q.value.length, q.value.length);
  });
});
const refClear = document.getElementById('ref-clear');
if (refClear) refClear.addEventListener('click', () => {
  refInput.value = '';
  refChip.classList.add('hidden');
});

// ── Tinder mode ──
const tinder = document.getElementById('tinder');
const stage  = document.getElementById('t-stage');
const counter = document.getElementById('t-count');
const doneEl = document.getElementById('t-done');
const undoBtn = document.getElementById('t-undo');
const toast = document.getElementById('t-toast');
const toastMsg = document.getElementById('t-toast-msg');
let ti = 0;
let history = [];   // {idx, id, prev} per decision, for undo
let toastTimer = null;

function cardMarkup(r) {
  const bg = r.img_pos !== null ? "background-image:url('/img/" + r.listing_id + "/" + r.img_pos + "')" : "";
  const badge = r.pct_diff < 0
    ? '<span class="t-badge good">' + Math.round(-r.pct_diff*100) + '% below model · save $' + r.delta.toLocaleString() + '/mo</span>'
    : '<span class="t-badge bad">' + Math.round(r.pct_diff*100) + '% above model</span>';
  const sqft = r.sqft ? ' · ' + r.sqft.toLocaleString() + ' ft²' : '';
  const beds = (r.beds === null || r.beds === undefined) ? '?' : Math.round(r.beds);
  const baths = (r.baths === null || r.baths === undefined) ? '?' : r.baths;
  const tags = (r.tags && r.tags.length)
    ? '<div class="t-tags">' + r.tags.slice(0, 6).map(
        t => '<span class="t-tag">' + t.replace(/_/g, ' ') + '</span>').join('') + '</div>'
    : '';
  return '<div class="t-photo" style="' + bg + '"></div>' +
    '<span class="stamp like">LIKE</span><span class="stamp pass">NOPE</span>' +
    '<div class="t-info">' + badge +
      '<div class="t-addr">' + r.name + '</div>' +
      '<div class="t-meta">' + r.neighborhood + ' · ' + beds + ' bd / ' + baths + ' ba' + sqft + '</div>' + tags +
      '<div class="t-price">$' + r.rent.toLocaleString() + ' <span>model $' + r.predicted.toLocaleString() + '/mo</span></div>' +
    '</div>';
}

function renderDeck() {
  stage.innerHTML = '';
  counter.textContent = ti < DECK.length ? (ti + 1) + ' / ' + DECK.length : DECK.length + ' / ' + DECK.length;
  if (ti >= DECK.length) { doneEl.classList.remove('hidden'); return; }
  doneEl.classList.add('hidden');

  // Card behind (depth), if present.
  if (ti + 1 < DECK.length) {
    const behind = document.createElement('div');
    behind.className = 't-card behind';
    behind.innerHTML = cardMarkup(DECK[ti + 1]);
    stage.appendChild(behind);
  }
  // Top card.
  const top = document.createElement('div');
  top.className = 't-card';
  top.innerHTML = cardMarkup(DECK[ti]);
  stage.appendChild(top);
  attachDrag(top);
}

function decide(reaction) {
  const r = DECK[ti];
  history.push({idx: ti, id: r.listing_id, prev: r.reaction || null});
  r.reaction = reaction;
  react(r.listing_id, reaction);                 // persist (fire and forget)
  paintHeart(r.listing_id, reaction === 'liked'); // reflect in grid behind
  ti += 1;
  showToast(reaction);
  updateUndo();
  renderDeck();
}

function undoLast() {
  const last = history.pop();
  if (!last) return;
  ti = last.idx;
  DECK[ti].reaction = last.prev;
  react(last.id, last.prev || 'none');           // persist the restored state
  paintHeart(last.id, last.prev === 'liked');
  hideToast();
  updateUndo();
  renderDeck();
}

function updateUndo() { undoBtn.disabled = history.length === 0; }

function showToast(reaction) {
  toastMsg.textContent = reaction === 'liked' ? 'Liked ❤' : 'Passed ✕';
  toast.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(hideToast, 2600);
}
function hideToast() { clearTimeout(toastTimer); toast.classList.add('hidden'); }

function flyTop(dir) {                            // dir: 1 = like/right, -1 = pass/left
  const top = stage.querySelector('.t-card:not(.behind)');
  if (!top) return;
  top.style.transition = 'transform .3s, opacity .3s';
  top.style.transform = 'translate(' + (dir * 700) + 'px,0) rotate(' + (dir * 28) + 'deg)';
  top.style.opacity = '0';
  setTimeout(() => decide(dir > 0 ? 'liked' : 'passed'), 230);
}

function attachDrag(el) {
  let sx = 0, sy = 0, dx = 0, dragging = false;
  const like = el.querySelector('.stamp.like');
  const pass = el.querySelector('.stamp.pass');
  el.addEventListener('pointerdown', e => {
    dragging = true; sx = e.clientX; sy = e.clientY; dx = 0;
    el.setPointerCapture(e.pointerId); el.style.transition = 'none';
  });
  el.addEventListener('pointermove', e => {
    if (!dragging) return;
    dx = e.clientX - sx; const dy = e.clientY - sy;
    el.style.transform = 'translate(' + dx + 'px,' + dy + 'px) rotate(' + (dx / 18) + 'deg)';
    if (like) like.style.opacity = Math.max(0, Math.min(1, dx / 110));
    if (pass) pass.style.opacity = Math.max(0, Math.min(1, -dx / 110));
  });
  function end() {
    if (!dragging) return;
    dragging = false;
    if (dx > 110) { flyTop(1); }
    else if (dx < -110) { flyTop(-1); }
    else {
      el.style.transition = 'transform .25s';
      el.style.transform = '';
      if (like) like.style.opacity = 0;
      if (pass) pass.style.opacity = 0;
    }
  }
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
}

function openTinder() {
  ti = 0; history = []; hideToast(); updateUndo();
  tinder.classList.remove('hidden'); renderDeck();
}
function closeTinder() { hideToast(); tinder.classList.add('hidden'); }

const openBtn = document.getElementById('tinder-open');
if (openBtn) openBtn.addEventListener('click', openTinder);
document.getElementById('t-close').addEventListener('click', closeTinder);
document.getElementById('t-back').addEventListener('click', closeTinder);
document.getElementById('t-like').addEventListener('click', () => flyTop(1));
document.getElementById('t-pass').addEventListener('click', () => flyTop(-1));
undoBtn.addEventListener('click', undoLast);
document.getElementById('t-toast-undo').addEventListener('click', undoLast);
document.addEventListener('keydown', e => {
  if (tinder.classList.contains('hidden')) return;
  if (e.key === 'ArrowRight') flyTop(1);
  else if (e.key === 'ArrowLeft') flyTop(-1);
  else if (e.key === 'z' || e.key === 'Backspace') { e.preventDefault(); undoLast(); }
  else if (e.key === 'Escape') closeTinder();
});
</script>
</body>
</html>
"""


# ── Pipeline control-panel template ───────────────────────────────────────────


RUN_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pipeline · NYC Rental Deals</title>
<style>
  :root { --bg:#0f1115; --card:#181b22; --line:#262b35; --txt:#e6e8ec;
          --muted:#9aa3b2; --good:#34d399; --bad:#f87171; --accent:#60a5fa; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { padding:18px 28px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:16px; }
  h1 { font-size:19px; margin:0; font-weight:650; }
  a.back { color:var(--accent); text-decoration:none; font-size:13px; }
  .wrap { display:grid; grid-template-columns:minmax(330px,440px) 1fr;
          min-height:calc(100vh - 59px); }
  .panel { padding:22px 26px; border-right:1px solid var(--line); overflow:auto; }
  .stage { background:var(--card); border:1px solid var(--line); border-radius:12px;
           padding:16px 18px; margin-bottom:18px; }
  .stage h2 { font-size:15px; margin:0 0 4px; }
  .stage p { color:var(--muted); font-size:12px; margin:0 0 8px; }
  label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
  input, select { width:100%; background:#0f1218; color:var(--txt);
    border:1px solid var(--line); border-radius:8px; padding:8px 10px; font-size:13px; }
  input:focus, select:focus { outline:none; border-color:var(--accent); }
  .row { display:flex; gap:10px; } .row > div { flex:1; }
  button.run { margin-top:14px; width:100%; border:none; border-radius:9px;
    padding:10px; font-weight:700; font-size:14px; cursor:pointer;
    background:linear-gradient(135deg,#60a5fa,#34d399); color:#06281f; }
  button.run:disabled { opacity:.4; cursor:default; }
  .console { display:flex; flex-direction:column; min-width:0; }
  .console-head { padding:14px 22px; border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:12px; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--muted); }
  .dot.running { background:var(--accent); animation:pulse 1s infinite; }
  .dot.done { background:var(--good); }
  .dot.failed, .dot.stopped { background:var(--bad); }
  @keyframes pulse { 50% { opacity:.3; } }
  .status { font-size:13px; color:var(--muted); text-transform:capitalize; }
  button.stop { margin-left:auto; background:#2a1417; color:var(--bad);
    border:1px solid #5b2027; border-radius:8px; padding:6px 12px; font-size:12px; cursor:pointer; }
  button.stop:disabled { opacity:.4; cursor:default; }
  pre.log { flex:1; margin:0; padding:16px 22px; overflow:auto; white-space:pre-wrap;
    word-break:break-word; color:#cdd3dd;
    font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }
  .hint { color:var(--muted); font-size:12px; padding:0 22px 16px; }
</style>
</head>
<body>
<header>
  <h1>⚙ Pipeline control</h1>
  <a class="back" href="/">← back to deals</a>
  <span class="status" style="margin-left:auto">db: {{ db_path }}</span>
</header>
<div class="wrap">
  <div class="panel">

    <div class="stage">
      <h2>1 · Scrape listings</h2>
      <p>Search StreetEasy and download photos. Re-runs only fetch new listings.</p>
      <label>Area IDs (comma-separated)</label>
      <input id="s_areas" value="152,116,113,162,117,104,115,158,146">
      <div class="row">
        <div><label>Min rent ($/mo)</label><input id="s_price_min" type="number" value="4000"></div>
        <div><label>Max rent ($/mo)</label><input id="s_price_max" type="number" value="7500"></div>
      </div>
      <div class="row">
        <div><label>Max new listings</label><input id="s_max_listings" type="number" value="250"></div>
        <div><label>Max pages</label><input id="s_max_pages" type="number" value="100"></div>
      </div>
      <button class="run" data-stage="scrape">Start scraping</button>
    </div>

    <div class="stage">
      <h2>2 · Backfill details</h2>
      <p>Amenities, availability, floor plans. Rotates browser fingerprints on a 403.</p>
      <label>Browser fingerprint(s)</label>
      <input id="b_impersonate" value="safari180,safari170,chrome131,firefox144">
      <div class="row">
        <div><label>Min delay (s)</label><input id="b_min_delay" type="number" value="10"></div>
        <div><label>Max delay (s)</label><input id="b_max_delay" type="number" value="25"></div>
      </div>
      <label>Max listings (blank = all)</label>
      <input id="b_max_listings" type="number" placeholder="all">
      <button class="run" data-stage="backfill">Start backfill</button>
    </div>

    <div class="stage">
      <h2>3 · Score photos</h2>
      <p>Vision model scores each photo. Needs the provider's API key in <code>.env</code>.</p>
      <div class="row">
        <div><label>Provider</label>
          <select id="c_provider">
            <option value="anthropic">anthropic</option>
            <option value="gemini">gemini</option>
          </select></div>
        <div><label>Max listings</label><input id="c_max_listings" type="number" value="50"></div>
      </div>
      <label>Model</label>
      <input id="c_model" value="claude-haiku-4-5">
      <button class="run" data-stage="score">Start scoring</button>
    </div>

  </div>

  <div class="console">
    <div class="console-head">
      <span id="dot" class="dot"></span>
      <span id="status" class="status">idle</span>
      <button id="stop" class="stop" disabled>Stop</button>
    </div>
    <pre id="log" class="log"></pre>
    <div class="hint">Output streams live. A job keeps running if you navigate away —
      come back here to watch it. Only one stage runs at a time.</div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const logEl = $('log'), dot = $('dot'), statusEl = $('status'), stopBtn = $('stop');
const runBtns = document.querySelectorAll('button.run');
let cursor = 0, polling = false;

const PARAMS = {
  scrape: () => ({ areas:$('s_areas').value, price_min:$('s_price_min').value,
    price_max:$('s_price_max').value, max_listings:$('s_max_listings').value,
    max_pages:$('s_max_pages').value }),
  backfill: () => ({ impersonate:$('b_impersonate').value, min_delay:$('b_min_delay').value,
    max_delay:$('b_max_delay').value, max_listings:$('b_max_listings').value }),
  score: () => ({ provider:$('c_provider').value, model:$('c_model').value,
    max_listings:$('c_max_listings').value }),
};

// Keep the model field sensible when the provider changes.
$('c_provider').addEventListener('change', e => {
  $('c_model').value = e.target.value === 'anthropic' ? 'claude-haiku-4-5' : 'gemini-2.5-flash';
});

function setStatus(s) {
  dot.className = 'dot ' + (s || '');
  statusEl.textContent = s || 'idle';
  const running = s === 'running';
  runBtns.forEach(b => b.disabled = running);
  stopBtn.disabled = !running;
}

async function poll() {
  let d;
  try { d = await (await fetch('/api/run/log?cursor=' + cursor)).json(); }
  catch (e) { setTimeout(poll, 1200); return; }
  if (d.lines && d.lines.length) {
    logEl.textContent += (logEl.textContent ? '\\n' : '') + d.lines.join('\\n');
    logEl.scrollTop = logEl.scrollHeight;
    cursor = d.cursor;
  }
  setStatus(d.status);
  if (d.status === 'running') { setTimeout(poll, 800); }
  else { polling = false; }
}
function ensurePolling() { if (!polling) { polling = true; poll(); } }

runBtns.forEach(b => b.addEventListener('click', async () => {
  const stage = b.dataset.stage;
  logEl.textContent = ''; cursor = 0;
  const d = await (await fetch('/api/run/' + stage, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(PARAMS[stage]())
  })).json();
  if (!d.ok) { logEl.textContent = '⚠ ' + d.error; return; }
  setStatus('running'); ensurePolling();
}));

stopBtn.addEventListener('click', () => fetch('/api/run/stop', { method:'POST' }));

// On load, re-attach to any job already in flight.
(async () => {
  const d = await (await fetch('/api/run/log?cursor=0')).json();
  if (d.lines && d.lines.length) { logEl.textContent = d.lines.join('\\n'); cursor = d.cursor; }
  setStatus(d.status);
  if (d.status === 'running') ensurePolling();
})();
</script>
</body>
</html>
"""


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Web UI for suggested (underpriced) listings.")
    p.add_argument("--db", default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini",
                   help="Model provider for AI search (needs the matching API key)")
    p.add_argument("--model", default=None,
                   help="Model for AI search (default: per-provider — gemini-2.5-flash "
                        "or claude-opus-4-8; claude-haiku-4-5 is cheaper)")
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

    print(f"Building suggestions from {args.db} …")
    _STATE.update(build_suggestions(args.db))
    m = _STATE["meta"]
    print(f"  {m.get('underpriced', 0)} underpriced of {m.get('total', 0)} scored listings")
    if engine:
        print(f"  AI search: enabled ({args.provider} / {model})")
    else:
        key = "GOOGLE_API_KEY" if args.provider == "gemini" else "ANTHROPIC_API_KEY"
        print(f"  AI search: disabled — set {key} (in .env) to enable")
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
