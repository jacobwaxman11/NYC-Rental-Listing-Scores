"""Tests for the feature-engineering frame (features.py)."""

import numpy as np
import pandas as pd

import features


def _raw_frame():
    # Mirrors the columns _load_listings() produces from the SQL join.
    return pd.DataFrame({
        "listing_id": ["a", "b", "c"],
        "rent": [4000, 5000, 6000],
        "beds": [1.0, 2.0, 1.0],
        "baths": [1.0, 1.0, 1.0],
        "sqft": [700, None, 900],
        "neighborhood": ["Chelsea", "LES", "Chelsea"],
        "building_type": ["RENTAL", "CONDO", "RENTAL"],
        "has_floor_plan": [1, 0, 1],
        "avg_natural_light": [7.0, None, 6.0],         # b is unscored
        "avg_finish_quality": [7.0, None, 6.0],
        "avg_space_feeling": [6.0, None, 6.0],
        "avg_condition": [8.0, None, 7.0],
        "common_avg_finish_quality": [8.0, None, None],
        "common_avg_condition": [7.0, None, None],
        "common_avg_appeal": [9.0, None, None],
    })


def test_require_scores_drops_unscored_and_logs_target():
    out = features.engineer_features(_raw_frame(), require_scores=True, verbose=False)
    assert set(out["listing_id"]) == {"a", "c"}        # b dropped (no image scores)
    assert "log_rent" in out.columns and "rent" not in out.columns
    a_log = out.set_index("listing_id").loc["a", "log_rent"]
    assert abs(a_log - np.log(4000)) < 1e-9


def test_include_unscored_keeps_all_with_flag():
    out = features.engineer_features(_raw_frame(), require_scores=False, verbose=False)
    assert len(out) == 3
    assert "has_image_scores" in out.columns
    flags = out.set_index("listing_id")["has_image_scores"].to_dict()
    assert flags == {"a": 1, "b": 0, "c": 1}
    # sqft was imputed (no NaN remaining)
    assert out["sqft"].notna().all()
    # categoricals one-hot encoded
    assert any(c.startswith("nh_") for c in out.columns)


def test_collapse_image_scores_composites():
    out = features.engineer_features(_raw_frame(), require_scores=True, verbose=False)
    collapsed = features.collapse_image_scores(out)
    assert "apt_quality" in collapsed.columns
    assert "common_quality" in collapsed.columns
    assert "avg_natural_light" not in collapsed.columns   # folded into apt_quality
