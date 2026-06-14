"""Tests for the scoring pipeline's aggregation (score_listings.aggregate_scores).

The provider scorers + their JSON/tag helpers moved to scorers.py — see
test_scorers.py. This file covers the per-listing score rollup.
"""

import score_listings as sl


def test_aggregate_scores_rollup():
    scores = [
        {"image_category": "apartment", "common_space_scores": None, "irrelevant_reason": None,
         "apartment_scores": {"room_type": "living_room", "natural_light": 8,
                              "finish_quality": 7, "space_feeling": 6, "view_quality": 9, "condition": 8}},
        {"image_category": "apartment", "common_space_scores": None, "irrelevant_reason": None,
         "apartment_scores": {"room_type": "bedroom", "natural_light": 4,
                              "finish_quality": 5, "space_feeling": 5, "view_quality": 3, "condition": 6}},
        {"image_category": "common_space", "apartment_scores": None, "irrelevant_reason": None,
         "common_space_scores": {"space_type": "gym", "finish_quality": 8, "condition": 7, "appeal": 9}},
        {"image_category": "irrelevant", "apartment_scores": None,
         "common_space_scores": None, "irrelevant_reason": "neighborhood map"},
        None,  # a failed image
    ]
    s = sl.aggregate_scores(scores)

    assert s["photos_total"] == 5
    assert s["photos_scored"] == 4
    assert s["photos_apartment"] == 2
    assert s["photos_common"] == 1
    assert s["photos_irrelevant"] == 1

    apt = s["apartment"]
    assert apt["avg_natural_light"] == 6.0          # (8 + 4) / 2
    assert apt["max_view_quality"] == 9
    assert apt["has_good_view"] is True             # max view >= 7
    assert apt["pct_bright_rooms"] == 0.5           # one of two rooms light >= 7
    assert apt["room_type_counts"] == {"living_room": 1, "bedroom": 1}

    assert s["common_space"]["avg_appeal"] == 9.0


def test_aggregate_scores_empty():
    s = sl.aggregate_scores([])
    assert s["photos_total"] == 0
    assert s["apartment"] == {}
    assert s["common_space"] == {}
