"""Optional post-hoc feature transforms + the amenity-tier vocabulary and
column-ordering helper.

``prepare_features`` returns a base frame with individual amenity one-hots and
all per-image score columns intact; these transforms compress those groups and
are applied by the CLI (and by the web app before modelling):

    collapse_amenity_tiers(df)   — replaces amen_* one-hots with 3 tier counts
    collapse_image_scores(df)    — replaces correlated image columns with
                                   apt_quality / common_quality composites
"""

from __future__ import annotations

import pandas as pd

# Raw image-score columns loaded from listing_scores. These are collapsed into
# two composites (apt_quality, common_quality) by ``collapse_image_scores``.
_APT_QUALITY_COLS = (
    "avg_natural_light", "avg_finish_quality", "avg_space_feeling",
    "avg_condition",
)
_COMMON_QUALITY_COLS = (
    "common_avg_finish_quality", "common_avg_condition", "common_avg_appeal",
)

# Union used for NaN-detection (deciding whether a listing has been scored).
_ALL_RAW_IMAGE_COLS = _APT_QUALITY_COLS + _COMMON_QUALITY_COLS

# ── Amenity tiers ─────────────────────────────────────────────────────────────
# Tier A: premium amenities that meaningfully drive rent up.
# Tier B: standard/expected amenities in managed NYC buildings.
# Tier C: low-signal tags (view descriptors, marketing fluff, near-universal).
# Amenities not listed in any tier are silently ignored.

AMENITY_TIER_A = frozenset({
    "pool", "hot_tub", "gym", "roof_deck", "private_roof_deck",
    "washer_dryer", "central_ac", "parking", "garage",
    "patio", "balcony", "terrace", "private_outdoor_space",
    "concierge", "media_room", "childrens_playroom",
    "hardwood_floors", "fireplace", "loft",
    "doorman", "valet_service", "valet",
})

AMENITY_TIER_B = frozenset({
    "elevator", "dishwasher", "laundry", "package_room",
    "storage_space", "bike_room", "shared_outdoor_space", "garden",
    "courtyard", "cold_storage", "locker_cage", "deck",
})

AMENITY_TIER_C = frozenset({
    "fios_available", "virtual", "city", "skyline", "water",
    "park", "view", "decorative", "part_time", "wheelchair_access",
    "live_in_super", "full_time", "assigned", "furnished",
})


def collapse_image_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Replace correlated per-dimension image columns with two composites.

    ``apt_quality``    = mean(avg_natural_light, avg_finish_quality,
                              avg_space_feeling, avg_condition)
    ``common_quality`` = mean(common_avg_finish_quality,
                              common_avg_condition, common_avg_appeal)
    """
    df = df.copy()
    apt_present = [c for c in _APT_QUALITY_COLS if c in df.columns]
    com_present = [c for c in _COMMON_QUALITY_COLS if c in df.columns]

    if apt_present:
        df["apt_quality"] = df[apt_present].mean(axis=1)
        df = df.drop(columns=apt_present)
    if com_present:
        df["common_quality"] = df[com_present].mean(axis=1)
        df = df.drop(columns=com_present)

    return _reorder_columns(df)


def collapse_amenity_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Replace individual ``amen_*`` one-hot columns with three tier counts.

    Each listing gets ``amen_tier_a``, ``amen_tier_b``, ``amen_tier_c``
    equal to the number of its amenities falling into that tier.
    """
    df = df.copy()
    amen_cols = [c for c in df.columns if c.startswith("amen_")]
    if not amen_cols:
        return df

    a_cols = [c for c in amen_cols if c.removeprefix("amen_") in AMENITY_TIER_A]
    b_cols = [c for c in amen_cols if c.removeprefix("amen_") in AMENITY_TIER_B]
    c_cols = [c for c in amen_cols if c.removeprefix("amen_") in AMENITY_TIER_C]

    df["amen_tier_a"] = df[a_cols].sum(axis=1).astype(int) if a_cols else 0
    df["amen_tier_b"] = df[b_cols].sum(axis=1).astype(int) if b_cols else 0
    df["amen_tier_c"] = df[c_cols].sum(axis=1).astype(int) if c_cols else 0

    df = df.drop(columns=amen_cols)
    return _reorder_columns(df)


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Stable, inspection-friendly column ordering: id, target, then numeric
    listing features, then one-hot groups, then amenity columns last."""
    front = [c for c in ("listing_id", "log_rent") if c in df.columns]

    amen = sorted(c for c in df.columns if c.startswith("amen_"))
    nh   = sorted(c for c in df.columns if c.startswith("nh_"))
    bt   = sorted(c for c in df.columns if c.startswith("bt_"))

    rest = [c for c in df.columns if c not in front + amen + nh + bt]
    return df[front + sorted(rest) + bt + nh + amen]
