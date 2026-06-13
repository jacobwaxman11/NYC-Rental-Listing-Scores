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
import os

import numpy as np
import pandas as pd
from flask import (
    Flask, abort, jsonify, redirect, render_template_string, request, send_file,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import db as dbm
from features import collapse_amenity_tiers, collapse_image_scores, prepare_features


app = Flask(__name__)

# Populated at startup (and on ?refresh=1) by build_suggestions().
_STATE: dict = {"suggestions": [], "meta": {}, "db_path": dbm.DEFAULT_DB_PATH}


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

    rows = list(_STATE["suggestions"])
    meta = _STATE["meta"]

    show = request.args.get("show", "under")           # under | all | liked
    sort = request.args.get("sort", "deal")            # deal | rent_asc | rent_desc | quality
    nh = request.args.get("nh", "")
    try:
        min_beds = int(float(request.args.get("beds", "") or 0))
    except ValueError:
        min_beds = 0

    # Passed listings are hidden from the browsing views; the "passed" view
    # surfaces them on their own so you can restore (undo) them.
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
    <a class="refresh" href="/?refresh=1">↻ recompute</a>
  </div>
</header>

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
        <div class="prices">
          <span class="rent">${{ '{:,}'.format(r.rent) }}</span>
          <span class="pred">model ${{ '{:,}'.format(r.predicted) }}/mo</span>
        </div>
        <div class="foot">
          <span>photo quality: <span class="q">{{ r.apt_quality if r.apt_quality is not none else '—' }}</span>/10
            {% if r.photos %}· {{ r.photos }} photos{% endif %}</span>
          {% if r.url %}<a class="view" href="{{ r.url }}" target="_blank" rel="noopener">view ↗</a>{% endif %}
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
  return '<div class="t-photo" style="' + bg + '"></div>' +
    '<span class="stamp like">LIKE</span><span class="stamp pass">NOPE</span>' +
    '<div class="t-info">' + badge +
      '<div class="t-addr">' + r.name + '</div>' +
      '<div class="t-meta">' + r.neighborhood + ' · ' + beds + ' bd / ' + baths + ' ba' + sqft + '</div>' +
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


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Web UI for suggested (underpriced) listings.")
    p.add_argument("--db", default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument("--host", default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=5000, help="Bind port")
    p.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    args = p.parse_args()

    _STATE["db_path"] = args.db
    print(f"Building suggestions from {args.db} …")
    _STATE.update(build_suggestions(args.db))
    m = _STATE["meta"]
    print(f"  {m.get('underpriced', 0)} underpriced of {m.get('total', 0)} scored listings")
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
