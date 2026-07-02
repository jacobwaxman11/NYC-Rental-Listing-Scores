"""Build the training frame from rentals.db.

Joins ``listings`` + ``listing_scores`` + ``listing_amenities`` into one row
per listing and applies the engineering decisions documented inline. The
output is a fully-numeric DataFrame ready to split into X / y for any
sklearn estimator. No scaling is applied — leave that for a Pipeline so it
can be fit inside CV.

``prepare_features()`` returns a *base* frame with individual amenity one-hot
columns and all per-image score columns intact. The optional compression
transforms — ``collapse_amenity_tiers`` and ``collapse_image_scores`` — live in
:mod:`features_transforms` and are re-exported here for convenience.

    # notebook: df = prepare_features(); y = df["log_rent"]; then optionally
    #           df2 = collapse_amenity_tiers(collapse_image_scores(df))
    # CLI (applies all transforms):  python features.py --output features.csv
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_DB_PATH
from features_transforms import (  # noqa: F401  (collapse_* re-exported)
    _ALL_RAW_IMAGE_COLS, collapse_amenity_tiers, collapse_image_scores,
    _reorder_columns,
)


# ── Loaders ──────────────────────────────────────────────────────────────────


def _load_listings(conn: sqlite3.Connection) -> pd.DataFrame:
    """Listing-level columns + listing_scores fields. Image-score columns are
    NaN for listings the scorer hasn't reached yet.
    """
    return pd.read_sql(
        """
        SELECT
            l.listing_id,
            l.rent,
            l.beds, l.baths, l.sqft,
            l.neighborhood,
            l.building_type,
            (l.floor_plan_url IS NOT NULL) AS has_floor_plan,
            ls.avg_natural_light,
            ls.avg_finish_quality,
            ls.avg_space_feeling,
            ls.avg_condition,
            ls.common_avg_finish_quality,
            ls.common_avg_condition,
            ls.common_avg_appeal
        FROM listings l
        LEFT JOIN listing_scores ls ON l.listing_id = ls.listing_id
        """,
        conn,
    )


def _load_amenities_one_hot(
    conn: sqlite3.Connection, min_count: int = 10
) -> pd.DataFrame:
    """Pivot listing_amenities into a one-hot matrix (one column per amenity).

    Amenities present in fewer than ``min_count`` listings are dropped to
    avoid fitting noise on the long tail of rare tags.
    """
    raw = pd.read_sql(
        "SELECT listing_id, amenity FROM listing_amenities",
        conn,
    )
    if raw.empty:
        return pd.DataFrame(columns=["listing_id"])

    counts = raw["amenity"].value_counts()
    keep = counts[counts >= min_count].index
    raw = raw[raw["amenity"].isin(keep)]

    one_hot = (
        pd.crosstab(raw["listing_id"], raw["amenity"])
        .clip(upper=1)
        .add_prefix("amen_")
        .astype(int)
        .reset_index()
    )
    return one_hot


# ── Engineering ──────────────────────────────────────────────────────────────


def engineer_features(
    df: pd.DataFrame,
    require_scores: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply target transform, sqft imputation, categorical encoding, and
    NaN handling. Returns a frame with individual amenity one-hots and raw
    image-score columns preserved. Use ``collapse_amenity_tiers`` and
    ``collapse_image_scores`` afterwards for optional compression.

    Args:
        df: raw frame from the SQL join (listings ⋈ scores ⋈ amenities).
        require_scores: if True (default), drop listings whose images haven't
            been scored yet. If False, mean-impute the score columns and add
            a ``has_image_scores`` flag.
        verbose: print row-count deltas for diagnostics.
    """
    df = df.copy()

    # ── Target ───────────────────────────────────────────────────────────────
    df["log_rent"] = np.log(df["rent"])

    # ── Drop or flag unscored rows ───────────────────────────────────────────
    if require_scores:
        before = len(df)
        df = df.dropna(subset=["avg_natural_light"]).reset_index(drop=True)
        if verbose:
            print(f"  Dropped {before - len(df)} listings without image scores "
                  f"({len(df)} remaining)")
    else:
        df["has_image_scores"] = df["avg_natural_light"].notna().astype(int)

    # ── Mean-impute remaining NaN in image-score columns ─────────────────────
    for c in _ALL_RAW_IMAGE_COLS:
        if c in df.columns and df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mean())

    # ── Sqft imputation (median by beds) + missingness flag ──────────────────
    df["has_sqft"] = df["sqft"].notna().astype(int)
    bed_medians = df.groupby("beds")["sqft"].transform("median")
    df["sqft"] = df["sqft"].fillna(bed_medians)
    df["sqft"] = df["sqft"].fillna(df["sqft"].median())

    # ── Bool/int normalization ───────────────────────────────────────────────
    for col in ("has_floor_plan",):   # note the comma: a 1-tuple, not a string
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # ── One-hot encode categoricals ──────────────────────────────────────────
    df = pd.get_dummies(
        df,
        columns=["neighborhood", "building_type"],
        prefix=["nh", "bt"],
        drop_first=True,
        dtype=int,
    )

    # ── Drop the raw target column (log_rent is the canonical y) ─────────────
    df = df.drop(columns=["rent"])

    return _reorder_columns(df)


# ── Public entry point ───────────────────────────────────────────────────────


def prepare_features(
    db_path: str = DEFAULT_DB_PATH,
    amenity_min_count: int = 10,
    require_scores: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load + engineer in one call. Returns a DataFrame with individual
    amenity one-hot columns and raw image-score columns. Apply
    ``collapse_amenity_tiers`` / ``collapse_image_scores`` afterwards to
    compress these feature groups."""
    conn = sqlite3.connect(db_path)
    try:
        listings = _load_listings(conn)
        amen = _load_amenities_one_hot(conn, amenity_min_count)
    finally:
        conn.close()

    df = listings.merge(amen, on="listing_id", how="left")

    amen_cols = [c for c in df.columns if c.startswith("amen_")]
    if amen_cols:
        df[amen_cols] = df[amen_cols].fillna(0).astype(int)

    return engineer_features(df, require_scores=require_scores, verbose=verbose)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _summary(df: pd.DataFrame) -> None:
    n_amen = sum(1 for c in df.columns if c.startswith("amen_"))
    n_nh   = sum(1 for c in df.columns if c.startswith("nh_"))
    n_bt   = sum(1 for c in df.columns if c.startswith("bt_"))
    n_features = len(df.columns) - 2  # excl. listing_id + log_rent

    print(f"\nTraining frame: {len(df)} rows × {len(df.columns)} columns "
          f"({n_features} features)")
    print(f"  target log_rent: mean={df['log_rent'].mean():.3f}, "
          f"std={df['log_rent'].std():.3f}")
    print(f"  amenity columns:                     {n_amen}")
    print(f"  neighborhood (one-hot, drop_first):  {n_nh}")
    print(f"  building_type (one-hot, drop_first): {n_bt}")
    print(f"  numeric listing features:            "
          f"{n_features - n_amen - n_nh - n_bt}")


def _save(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Saved to {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument("--output", default=None,
                   help="Optional path to dump the frame (.csv or .parquet)")
    p.add_argument("--include-unscored", action="store_true",
                   help="Keep listings missing image scores (mean-impute + flag)")
    p.add_argument("--raw", action="store_true",
                   help="Skip amenity-tier and image-composite transforms")
    args = p.parse_args()

    df = prepare_features(
        db_path=args.db,
        require_scores=not args.include_unscored,
    )

    if not args.raw:
        df = collapse_image_scores(df)
        df = collapse_amenity_tiers(df)

    _summary(df)

    if args.output:
        _save(df, Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
