"""Tests for 'more like this' — style selection ordered by vicinity (web.similar_to)."""

import numpy as np
import pytest

import web


def _listings():
    # a is the reference; b/c/d are all stylistically similar (embeddings near a),
    # at increasing distances: b closest, then d, then c.
    return [
        {"listing_id": "a", "name": "Ref", "neighborhood": "Chelsea", "beds": 1.0, "baths": 1.0,
         "sqft": 700, "rent": 4000, "pct_diff": -0.1, "apt_quality": 7.0, "reaction": None,
         "tags": ["bright"], "lat": 40.7400, "lng": -73.9900},
        {"listing_id": "b", "name": "B", "neighborhood": "Chelsea", "beds": 1.0, "baths": 1.0,
         "sqft": 720, "rent": 4100, "pct_diff": -0.05, "apt_quality": 7.0, "reaction": None,
         "tags": ["bright"], "lat": 40.7405, "lng": -73.9905},   # ~0.04 mi
        {"listing_id": "c", "name": "C", "neighborhood": "Harlem", "beds": 1.0, "baths": 1.0,
         "sqft": 700, "rent": 4000, "pct_diff": 0.0, "apt_quality": 7.0, "reaction": None,
         "tags": ["bright"], "lat": 40.7600, "lng": -73.9900},   # ~1.4 mi
        {"listing_id": "d", "name": "D", "neighborhood": "Chelsea", "beds": 1.0, "baths": 1.0,
         "sqft": 700, "rent": 4000, "pct_diff": -0.2, "apt_quality": 7.0, "reaction": None,
         "tags": ["bright"], "lat": 40.7450, "lng": -73.9900},   # ~0.35 mi
    ]


@pytest.fixture(autouse=True)
def _state():
    saved = {k: web._STATE[k] for k in ("suggestions", "embeddings")}
    web._STATE["suggestions"] = _listings()
    web._STATE["embeddings"] = {
        "a": np.asarray([1.0, 0.0], dtype=np.float32),
        "b": np.asarray([0.9, 0.1], dtype=np.float32),
        "c": np.asarray([0.8, 0.2], dtype=np.float32),
        "d": np.asarray([0.85, 0.15], dtype=np.float32),
    }
    yield
    web._STATE.update(saved)


def test_similar_orders_by_vicinity():
    rows, banner, distances = web.similar_to("a")
    assert [r["listing_id"] for r in rows] == ["b", "d", "c"]   # nearest first
    assert "nearest first" in banner
    # distances are miles, ascending in result order
    ds = [distances[r["listing_id"]] for r in rows]
    assert ds == sorted(ds)
    assert distances["b"] < distances["c"]


def test_similar_excludes_self_and_passed():
    web._STATE["suggestions"][1]["reaction"] = "passed"   # b passed
    rows, _, _ = web.similar_to("a")
    ids = [r["listing_id"] for r in rows]
    assert "a" not in ids and "b" not in ids


def test_similar_without_coords_falls_back_to_style_order():
    for r in web._STATE["suggestions"]:
        r["lat"] = r["lng"] = None
    rows, banner, distances = web.similar_to("a")
    assert distances == {}
    assert "nearest first" not in banner
    # still style-ranked (embeddings): closest vector to a is b, then d, then c
    assert [r["listing_id"] for r in rows] == ["b", "d", "c"]


def test_similar_unknown_reference():
    rows, banner, distances = web.similar_to("zzz")
    assert rows == [] and distances == {} and "isn't in the current set" in banner


def test_apply_plan_distance_rank():
    sugg = web._STATE["suggestions"]   # a/b/c/d carry coords (b nearest a, then d, then c)
    plan = {"filters": {}, "rank_by": [{"field": "distance", "direction": "asc", "weight": 1}]}
    out = web._apply_plan(plan, sugg, ref_point=(40.7400, -73.9900))
    assert [r["listing_id"] for r in out] == ["a", "b", "d", "c"]   # a is the ref point (0 mi)
    # Without a reference point, a distance rank is silently ignored (no crash).
    assert len(web._apply_plan(plan, sugg)) == len(sugg)
