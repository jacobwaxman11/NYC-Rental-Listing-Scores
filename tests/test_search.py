"""Tests for the AI-search plan executor and taste matching (web.py).

These exercise the local, deterministic parts — no model calls. Importing web
pulls in Flask/sklearn (present in the venv); main() is never invoked.
"""

import numpy as np
import pytest

import web


def _listings():
    return [
        {"listing_id": "a", "name": "A", "neighborhood": "Chelsea", "beds": 1.0, "baths": 1.0,
         "sqft": 700, "rent": 4000, "pct_diff": -0.10, "apt_quality": 7.0, "reaction": "liked",
         "tags": ["exposed_brick", "bright", "hardwood_floors"]},
        {"listing_id": "b", "name": "B", "neighborhood": "LES", "beds": 1.0, "baths": 1.0,
         "sqft": 720, "rent": 4200, "pct_diff": -0.05, "apt_quality": 7.0, "reaction": None,
         "tags": ["duplex", "hardwood_floors"]},
        {"listing_id": "c", "name": "C", "neighborhood": "SoHo", "beds": 2.0, "baths": 1.0,
         "sqft": 1100, "rent": 6000, "pct_diff": 0.0, "apt_quality": 8.0, "reaction": None,
         "tags": ["carpet"]},
        {"listing_id": "d", "name": "D", "neighborhood": "FiDi", "beds": 1.0, "baths": 1.0,
         "sqft": 710, "rent": 4100, "pct_diff": -0.20, "apt_quality": 6.0, "reaction": "passed",
         "tags": ["hardwood_floors"]},
    ]


@pytest.fixture(autouse=True)
def _reset_state():
    """Isolate web._STATE mutations between tests."""
    saved = {k: web._STATE[k] for k in ("suggestions", "embeddings")}
    web._STATE["suggestions"] = _listings()
    web._STATE["embeddings"] = {}
    yield
    web._STATE.update(saved)


def _ids(rows):
    return [r["listing_id"] for r in rows]


def test_facets():
    f = web._facets(_listings())
    assert f["count"] == 4
    assert f["beds"] == {"min": 1.0, "max": 2.0}
    assert f["rent"] == {"min": 4000, "max": 6000}
    assert set(f["neighborhoods"]) == {"Chelsea", "LES", "SoHo", "FiDi"}
    assert "hardwood_floors" in f["tags"] and "duplex" in f["tags"]


def test_build_context_reference_and_liked_profile():
    ctx = web._build_context(_listings(), "a")
    assert ctx["reference"]["name"] == "A"
    assert ctx["reference"]["tags"] == ["exposed_brick", "bright", "hardwood_floors"]
    assert ctx["liked_profile"]["count"] == 1
    assert "exposed_brick" in ctx["liked_profile"]["top_tags"]


def test_apply_plan_tag_filters():
    sugg = _listings()
    assert _ids(web._apply_plan({"filters": {"tags_all": ["exposed_brick"]}, "rank_by": []}, sugg)) == ["a"]
    assert set(_ids(web._apply_plan(
        {"filters": {"tags_any": ["exposed_brick", "duplex"]}, "rank_by": []}, sugg))) == {"a", "b"}


def test_apply_plan_numeric_filters_and_review_exclusion():
    sugg = _listings()
    # only_underpriced + exclude_reviewed → only 'b' (a/d reviewed, c not underpriced)
    out = web._apply_plan({"filters": {"only_underpriced": True, "exclude_reviewed": True},
                           "rank_by": []}, sugg)
    assert _ids(out) == ["b"]
    # beds_min 2 → only 'c'
    assert _ids(web._apply_plan({"filters": {"beds_min": 2}, "rank_by": []}, sugg)) == ["c"]
    # rent_max 4150 keeps a, d (b is 4200, c is 6000)
    assert set(_ids(web._apply_plan({"filters": {"rent_max": 4150}, "rank_by": []}, sugg))) == {"a", "d"}


def test_apply_plan_ranking():
    sugg = _listings()
    by_sqft = web._apply_plan({"filters": {}, "rank_by": [{"field": "sqft", "direction": "desc", "weight": 1}]}, sugg)
    assert _ids(by_sqft)[0] == "c"            # 1100 sqft is largest
    by_rent = web._apply_plan({"filters": {}, "rank_by": [{"field": "rent", "direction": "asc", "weight": 1}]}, sugg)
    assert _ids(by_rent)[0] == "a"            # cheapest


def test_apply_plan_limit():
    out = web._apply_plan({"filters": {}, "rank_by": [], "limit": 2}, _listings())
    assert len(out) == 2


def test_apply_plan_semantic_blend():
    web._STATE["embeddings"] = {
        "a": np.asarray([1.0, 0.0], dtype=np.float32),
        "b": np.asarray([0.9, 0.1], dtype=np.float32),
        "c": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    # query points at 'c' direction → c ranks first among the unreviewed set
    out = web._apply_plan({"filters": {"exclude_reviewed": True}, "rank_by": []},
                          _listings(), query_vec=np.asarray([0.0, 1.0], dtype=np.float32))
    assert _ids(out)[0] == "c"


def test_match_likes_centroid():
    web._STATE["embeddings"] = {
        "a": np.asarray([1.0, 0.0], dtype=np.float32),
        "b": np.asarray([0.9, 0.1], dtype=np.float32),
        "c": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    rows, banner = web.match_likes()
    assert _ids(rows) == ["b", "c"]          # b closer to liked 'a'; a/d excluded (reviewed)
    assert "1 liked listing" in banner


def test_match_likes_guards():
    web._STATE["embeddings"] = {}
    assert web.match_likes()[0] == []
    assert "No embeddings" in web.match_likes()[1]

    web._STATE["embeddings"] = {"b": np.asarray([1.0, 0.0], dtype=np.float32)}
    for r in web._STATE["suggestions"]:
        r["reaction"] = None                  # no likes
    rows, banner = web.match_likes()
    assert rows == [] and "Like a few" in banner
