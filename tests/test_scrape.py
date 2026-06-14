"""Tests for scrape_listings URL building (delta sort + pagination)."""

import scrape_listings as s


def test_build_url_defaults_to_newest_first():
    url = s.build_url(3000, 7000, "west-village")
    assert "/for-rent/west-village/price:3000-7000" in url
    assert "sort_by=listed_desc" in url           # delta polling: newest first
    assert "page=" not in url                      # page 1 has no page param


def test_build_url_paginates_and_keeps_sort():
    url = s.build_url(3000, 7000, "west-village", page=3)
    assert "sort_by=listed_desc" in url and "page=3" in url


def test_build_url_numeric_area():
    url = s.build_url(2000, 5000, "115", page=2)
    assert "|area:115" in url and "sort_by=listed_desc" in url and "page=2" in url


def test_build_url_sort_can_be_disabled():
    url = s.build_url(3000, 7000, "soho", sort="")
    assert "sort_by" not in url
