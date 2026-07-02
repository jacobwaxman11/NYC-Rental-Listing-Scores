"""Listing-facing routes: the deals grid, a listing detail page, the react
endpoint, and the photo proxy.

The grid (:func:`index`) overlays fresh reactions from the DB, then dispatches to
one of four views — AI search, "more like this", taste matching, or the standard
browse path (:mod:`webapp.browse`) — and finally collapses same-building units
into one lead card before rendering.
"""

from __future__ import annotations

import os

from flask import (
    Blueprint, abort, jsonify, redirect, render_template, request, send_file,
)

import db as dbm
import grouping
from webapp.browse import apply_browse, parse_browse_args
from webapp.search import ai_search
from webapp.state import (
    PRICE_MAX, PRICE_MIN, PRICE_STEP, _STATE, load_embeddings,
)
from webapp.suggestions import build_suggestions
from webapp.taste import match_likes, similar_to

bp = Blueprint("listings", __name__)


def _ensure_loaded(refresh: bool = False) -> None:
    """Populate the cached suggestion set + embeddings if empty (or on refresh)."""
    if refresh or not _STATE["suggestions"]:
        _STATE.update(build_suggestions(_STATE["db_path"]))
        _STATE["embeddings"] = load_embeddings(_STATE["db_path"])


@bp.route("/")
def index():
    _ensure_loaded(refresh=bool(request.args.get("refresh")))

    # Overlay current reactions from the DB so hearts reflect persisted state.
    with dbm.open_db(_STATE["db_path"]) as conn:
        reactions = dbm.get_reactions(conn)
    for r in _STATE["suggestions"]:
        r["reaction"] = reactions.get(r["listing_id"])

    base = list(_STATE["suggestions"])
    meta = _STATE["meta"]
    spec = parse_browse_args(request.args)

    q = request.args.get("q", "").strip()
    ref = request.args.get("ref", "").strip()
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
        # Standard browse path.
        rows = apply_browse(base, spec)

    # Collapse same-building units into one lead card so a big building doesn't
    # flood the grid with near-identical photos. The lead is the most relevant
    # unit (rows are already sorted); the rest hang off it in an expandable
    # panel. ?group=0 flattens back to one card per unit.
    group_on = request.args.get("group", "1") != "0"
    groups = grouping.group_by_building(rows)
    n_units = len(rows)
    n_buildings = len(groups)
    group_others: dict = {}
    if group_on:
        rows = [g["lead"] for g in groups]
        group_others = {g["lead"]["listing_id"]: g["others"]
                        for g in groups if g["count"] > 1}

    liked_total = sum(1 for r in _STATE["suggestions"] if r["reaction"] == "liked")
    passed_total = sum(1 for r in _STATE["suggestions"] if r["reaction"] == "passed")

    html = render_template(
        "index.html", rows=rows, meta=meta,
        liked_total=liked_total, passed_total=passed_total,
        show=spec["show"], sort=spec["sort"], nh=spec["nh"],
        nh_list=sorted(spec["nh_set"]), min_beds=str(spec["min_beds"]),
        pmin=spec["pmin"], pmax=spec["pmax"],
        price_min=PRICE_MIN, price_max=PRICE_MAX, price_step=PRICE_STEP,
        avail_before=spec["avail_before"],
        tag_list=sorted(spec["tag_set"]), tag_mode=spec["tag_mode"],
        near_lat=spec["near_lat"], near_lng=spec["near_lng"],
        near_label=spec["near_label"], radius_mi=spec["radius_mi"],
        near_on=spec["near_on"], bbox=spec["bbox"],
        q=q, ref=ref, ref_name=ref_name, searching=searching,
        active_tag=spec["active_tag"],
        ai_enabled=ai["enabled"], ai_provider=ai["provider"],
        ai_model=ai["model"], ai_banner=ai_banner, distances=distances,
        emb_ready=bool(_STATE["embeddings"]),
        group_on=group_on, group_others=group_others,
        n_units=n_units, n_buildings=n_buildings,
    )
    # Don't let the browser cache the grid — otherwise a reload serves the old
    # HTML and the Shuffle sort looks "stuck" on the same order.
    return html, 200, {"Cache-Control": "no-store"}


@bp.route("/listing/<listing_id>")
def listing_detail(listing_id: str):
    """Full consumer-style detail page: gallery, description, amenities, map,
    model price vs. asking, and a 'more like this' jump-off."""
    _ensure_loaded()

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

        # Other units in the same building — small cards linking to their pages,
        # enriched with the model's deal % where the unit has been scored.
        siblings = []
        building = listing.get("building_slug")
        if building:
            sugg_by_id = {r["listing_id"]: r for r in _STATE["suggestions"]}
            sib_rows = conn.execute(
                "SELECT listing_id, name, unit, rent, beds, baths "
                "FROM listings WHERE building_slug=? AND listing_id<>? "
                "ORDER BY rent IS NULL, rent",
                (building, listing_id),
            ).fetchall()
            for row in sib_rows:
                sib = dict(row)
                sr = sugg_by_id.get(sib["listing_id"])
                if sr and sr.get("img_pos") is not None:
                    sib["img_pos"] = sr["img_pos"]
                else:
                    sib_imgs = dbm.get_listing_images(conn, sib["listing_id"])
                    sib["img_pos"] = sib_imgs[0]["position"] if sib_imgs else None
                sib["pct_diff"] = sr["pct_diff"] if sr else None
                siblings.append(sib)

    positions = [im["position"] for im in images]
    tags = srow.get("tags", []) if srow else []
    return render_template(
        "listing.html",
        l=listing, s=srow, amenities=amenities, positions=positions,
        tags=tags, reaction=reaction, siblings=siblings,
    )


@bp.route("/api/react", methods=["POST"])
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


@bp.route("/img/<listing_id>/<int:pos>")
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
