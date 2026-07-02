"""The standard (non-AI) browse path: parse the grid's query-string filters and
apply them to the suggestion rows.

:func:`parse_browse_args` turns ``request.args`` into a flat ``spec`` dict — the
same dict feeds both the filtering here and the template's control state (so the
sliders/chips render their current values). :func:`apply_browse` filters and
sorts a base row list per that spec. The AI-search / similar / match-likes paths
bypass this module entirely.
"""

from __future__ import annotations

import random

import geo
from webapp.state import PRICE_MAX, PRICE_MIN


def parse_browse_args(args) -> dict:
    """Normalize ``request.args`` into a browse ``spec`` dict."""
    show = args.get("show", "under")           # under | all | liked | passed | unreviewed
    sort = args.get("sort", "shuffle")         # shuffle | deal | rent_asc | rent_desc | quality

    nh = args.get("nh", "")
    nh_set = {a for a in nh.split(",") if a}    # multi-area filter (comma-separated)

    try:
        min_beds = int(float(args.get("beds", "") or 0))
    except ValueError:
        min_beds = 0

    # Price range filter. The slider runs PRICE_MIN..PRICE_MAX; sitting at an end
    # means "no bound that way" (PRICE_MAX is shown as "10k+").
    def price_arg(name: str, default: int) -> int:
        try:
            return max(PRICE_MIN, min(PRICE_MAX, int(float(args.get(name) or default))))
        except (TypeError, ValueError):
            return default
    pmin = price_arg("pmin", PRICE_MIN)
    pmax = price_arg("pmax", PRICE_MAX)
    if pmin > pmax:
        pmin, pmax = pmax, pmin

    # "Available by" cutoff — keep listings whose move-in date is on/before this
    # ISO YYYY-MM-DD date. Empty = no availability filter.
    avail_before = args.get("avail_before", "").strip()

    # Tag filter — multi-select with an any/all mode. `tag` (legacy single-tag
    # chip link) folds into the same set.
    tag = args.get("tag", "").strip()
    tag_set = {t for t in args.get("tags", "").split(",") if t}
    if tag:
        tag_set.add(tag)
    tag_mode = args.get("tagmode", "any").lower()
    if tag_mode not in ("any", "all"):
        tag_mode = "any"

    # Location filters. `near_*` = radius from a geocoded point; `bbox` = a box
    # drawn on the map ("south,west,north,east").
    def float_arg(name: str):
        try:
            return float(args.get(name))
        except (TypeError, ValueError):
            return None
    near_lat = float_arg("near_lat")
    near_lng = float_arg("near_lng")
    near_label = args.get("near_label", "").strip()
    radius_mi = float_arg("radius_mi") or 1.0
    near_on = near_lat is not None and near_lng is not None

    bbox = args.get("bbox", "").strip()
    bbox_vals = None
    if bbox:
        try:
            s, w, n, e = (float(x) for x in bbox.split(","))
            bbox_vals = (s, w, n, e)
        except (ValueError, TypeError):
            bbox_vals = None

    return {
        "show": show, "sort": sort,
        "nh": nh, "nh_set": nh_set,
        "min_beds": min_beds,
        "pmin": pmin, "pmax": pmax,
        "avail_before": avail_before,
        "active_tag": tag, "tag_set": tag_set, "tag_mode": tag_mode,
        "near_lat": near_lat, "near_lng": near_lng, "near_label": near_label,
        "radius_mi": radius_mi, "near_on": near_on,
        "bbox": bbox, "bbox_vals": bbox_vals,
    }


def apply_browse(base: list[dict], spec: dict) -> list[dict]:
    """Filter + sort ``base`` rows per a parsed ``spec`` (standard browse path).

    Passed listings are hidden except in their own view."""
    show = spec["show"]
    rows = base
    if show == "passed":
        rows = [r for r in rows if r["reaction"] == "passed"]
    elif show == "liked":
        rows = [r for r in rows if r["reaction"] == "liked"]
    elif show == "unreviewed":
        rows = [r for r in rows if not r["reaction"]]
    elif show == "under":
        rows = [r for r in rows if r["pct_diff"] < 0 and r["reaction"] != "passed"]
    else:  # all
        rows = [r for r in rows if r["reaction"] != "passed"]

    if spec["nh_set"]:
        rows = [r for r in rows if r["neighborhood"] in spec["nh_set"]]
    if spec["min_beds"]:
        rows = [r for r in rows if (r["beds"] or 0) >= spec["min_beds"]]
    if spec["pmin"] > PRICE_MIN:
        rows = [r for r in rows if (r["rent"] or 0) >= spec["pmin"]]
    if spec["pmax"] < PRICE_MAX:
        rows = [r for r in rows if (r["rent"] or 0) <= spec["pmax"]]

    tag_set = spec["tag_set"]
    if tag_set:
        if spec["tag_mode"] == "all":
            rows = [r for r in rows if tag_set <= set(r.get("tags") or [])]
        else:
            rows = [r for r in rows if tag_set & set(r.get("tags") or [])]

    if spec["near_on"]:
        near_lat, near_lng, radius_mi = spec["near_lat"], spec["near_lng"], spec["radius_mi"]
        rows = [
            r for r in rows
            if (d := geo.haversine_mi(r.get("lat"), r.get("lng"), near_lat, near_lng)) is not None
            and d <= radius_mi
        ]
    if spec["bbox_vals"]:
        s, w, n, e = spec["bbox_vals"]
        rows = [
            r for r in rows
            if r.get("lat") is not None and r.get("lng") is not None
            and s <= r["lat"] <= n and w <= r["lng"] <= e
        ]
    if spec["avail_before"]:
        # ISO dates sort lexically. Keep listings available on/before the cutoff;
        # those with no known date are excluded while the filter is on.
        cutoff = spec["avail_before"]
        rows = [r for r in rows
                if r.get("available_from") and r["available_from"] <= cutoff]

    sort = spec["sort"]
    if sort == "shuffle":
        # Reshuffled every load so same-building units (which often share a hero
        # image) don't cluster and look like duplicates.
        random.shuffle(rows)
    elif sort == "deal":
        rows.sort(key=lambda r: r["pct_diff"])
    elif sort == "rent_asc":
        rows.sort(key=lambda r: r["rent"])
    elif sort == "rent_desc":
        rows.sort(key=lambda r: -r["rent"])
    elif sort == "quality":
        rows.sort(key=lambda r: -(r["apt_quality"] or 0))

    return rows
