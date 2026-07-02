"""Roll per-image vision scores up into per-listing features, plus the console
logging for the scoring run.

Kept separate from the scoring orchestration so the aggregation logic (the part
worth unit-testing) has no I/O or model dependencies.
"""

from __future__ import annotations

import os
from typing import Optional


def aggregate_scores(image_scores: list[Optional[dict]]) -> dict:
    """Roll up per-image scores into per-listing features.

    Separates apartment photos from common-space photos so averages over
    apartment features are not polluted by lobby/gym/exterior shots, and
    counts irrelevant photos separately so we can flag low-quality listings.
    """
    scored = [s for s in image_scores if s]

    apartment = [s["apartment_scores"] for s in scored
                 if s.get("image_category") == "apartment" and s.get("apartment_scores")]
    common = [s["common_space_scores"] for s in scored
              if s.get("image_category") == "common_space" and s.get("common_space_scores")]
    irrelevant = [s for s in scored if s.get("image_category") == "irrelevant"]

    def _avg(items: list[dict], key: str) -> Optional[float]:
        vals = [it[key] for it in items if it.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def _max(items: list[dict], key: str):
        vals = [it[key] for it in items if it.get(key) is not None]
        return max(vals) if vals else None

    # ── Apartment metrics ────────────────────────────────────────────────────
    room_types = [a.get("room_type") for a in apartment if a.get("room_type")]
    room_type_counts = {r: room_types.count(r) for r in set(room_types)}

    apartment_summary = {
        "avg_natural_light":  _avg(apartment, "natural_light"),
        "avg_finish_quality": _avg(apartment, "finish_quality"),
        "avg_space_feeling":  _avg(apartment, "space_feeling"),
        "avg_condition":      _avg(apartment, "condition"),
        "max_view_quality":   _max(apartment, "view_quality"),
        "pct_bright_rooms":   round(
            sum(1 for a in apartment if a.get("natural_light", 0) >= 7) / len(apartment), 2
        ) if apartment else None,
        "has_good_view":      (_max(apartment, "view_quality") or 0) >= 7,
        "room_type_counts":   room_type_counts,
    } if apartment else {}

    # ── Common-space metrics ─────────────────────────────────────────────────
    # NOTE: we deliberately do NOT roll a "detected_amenities" boolean list out
    # of the photos — StreetEasy's `amenities` (populated by backfill_details.py)
    # is the canonical source. Including a Gemini-inferred copy would create
    # autocorrelated features for downstream modeling. We keep `space_type_counts`
    # because the *quantity* of photos per amenity carries a different signal
    # (how prominently the listing showcases each space) than a flat existence flag.
    space_types = [c.get("space_type") for c in common if c.get("space_type")]
    space_type_counts = {s: space_types.count(s) for s in set(space_types)}

    common_summary = {
        "avg_finish_quality": _avg(common, "finish_quality"),
        "avg_condition":      _avg(common, "condition"),
        "avg_appeal":         _avg(common, "appeal"),
        "space_type_counts":  space_type_counts,
    } if common else {}

    return {
        "apartment":         apartment_summary,
        "common_space":      common_summary,
        "photos_total":      len(image_scores),
        "photos_scored":     len(scored),
        "photos_apartment":  len(apartment),
        "photos_common":     len(common),
        "photos_irrelevant": len(irrelevant),
    }


def log_score(idx: int, total: int, path: str, scores: Optional[dict]) -> None:
    """One-line-per-image progress log during a scoring run."""
    prefix = f"    [{idx}/{total}] {os.path.basename(path)}"
    if not scores:
        print(f"{prefix}  ✗ no score")
        return
    cat = scores.get("image_category")
    if cat == "apartment":
        a = scores.get("apartment_scores") or {}
        print(
            f"{prefix}  ✓ apartment/{a.get('room_type')} "
            f"light={a.get('natural_light')} "
            f"finish={a.get('finish_quality')} "
            f"condition={a.get('condition')}"
        )
    elif cat == "common_space":
        c = scores.get("common_space_scores") or {}
        print(
            f"{prefix}  ✓ common/{c.get('space_type')} "
            f"finish={c.get('finish_quality')} "
            f"appeal={c.get('appeal')}"
        )
    elif cat == "irrelevant":
        print(f"{prefix}  ⊘ irrelevant: {scores.get('irrelevant_reason')}")
    else:
        print(f"{prefix}  ? unknown category: {cat}")


def print_summary(s: dict) -> None:
    """Per-listing aggregate summary printed after a listing is scored."""
    if not s:
        return
    print("  Summary:")
    print(
        f"    photos: {s.get('photos_apartment', 0)} apt / "
        f"{s.get('photos_common', 0)} common / "
        f"{s.get('photos_irrelevant', 0)} irrelevant "
        f"(total {s.get('photos_total', 0)})"
    )
    apt = s.get("apartment") or {}
    if apt:
        print(
            f"    apt:    light={apt.get('avg_natural_light')} "
            f"finish={apt.get('avg_finish_quality')} "
            f"space={apt.get('avg_space_feeling')} "
            f"cond={apt.get('avg_condition')} "
            f"view_max={apt.get('max_view_quality')} "
            f"bright%={apt.get('pct_bright_rooms')}"
        )
        print(f"    rooms:  {apt.get('room_type_counts')}")
    common = s.get("common_space") or {}
    if common:
        print(
            f"    common: finish={common.get('avg_finish_quality')} "
            f"cond={common.get('avg_condition')} "
            f"appeal={common.get('avg_appeal')}"
        )
        print(f"    spaces: {common.get('space_type_counts')}")
