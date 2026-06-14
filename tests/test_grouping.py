"""Tests for grouping.py — collapsing listings into per-building groups."""

import grouping


def _r(lid, slug=None):
    row = {"listing_id": lid}
    if slug is not None:
        row["building_slug"] = slug
    return row


def test_building_key_from_slug_or_listing_id():
    assert grouping.building_key({"building_slug": "stonehenge-gardens"}) == "stonehenge-gardens"
    # derived from "{building_slug}_{unit}"
    assert grouping.building_key({"listing_id": "stonehenge-gardens_006j"}) == "stonehenge-gardens"
    assert grouping.building_key({"listing_id": "loftco"}) == "loftco"   # no unit suffix


def test_groups_preserve_order_lead_and_count():
    rows = [
        _r("bldgA_1"), _r("bldgB_1"), _r("bldgA_2"), _r("bldgA_3"), _r("bldgC_1"),
    ]
    groups = grouping.group_by_building(rows)
    # building order = first appearance: A, B, C
    assert [g["building"] for g in groups] == ["bldgA", "bldgB", "bldgC"]
    a = groups[0]
    assert a["lead"]["listing_id"] == "bldgA_1"          # most relevant = first seen
    assert [o["listing_id"] for o in a["others"]] == ["bldgA_2", "bldgA_3"]
    assert a["count"] == 3
    assert groups[1]["count"] == 1 and groups[1]["others"] == []


def test_explicit_slug_takes_precedence():
    rows = [_r("x_1", slug="shared"), _r("y_2", slug="shared")]
    groups = grouping.group_by_building(rows)
    assert len(groups) == 1 and groups[0]["count"] == 2


def test_empty():
    assert grouping.group_by_building([]) == []
