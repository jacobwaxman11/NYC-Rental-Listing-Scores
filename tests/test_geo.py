"""Tests for geo.py haversine helpers."""

import geo


def test_zero_distance():
    assert geo.haversine_km(40.74, -73.99, 40.74, -73.99) == 0.0


def test_one_degree_latitude_is_about_111km():
    d = geo.haversine_km(0.0, 0.0, 1.0, 0.0)
    assert abs(d - 111.19) < 0.5


def test_symmetric():
    a = geo.haversine_km(40.74, -73.99, 40.71, -74.01)
    b = geo.haversine_km(40.71, -74.01, 40.74, -73.99)
    assert abs(a - b) < 1e-9


def test_missing_coords_returns_none():
    assert geo.haversine_km(None, -73.99, 40.71, -74.01) is None
    assert geo.haversine_km(40.74, -73.99, 40.71, None) is None
    assert geo.haversine_mi(None, None, None, None) is None


def test_miles_conversion():
    km = geo.haversine_km(40.74, -73.99, 40.71, -74.01)
    mi = geo.haversine_mi(40.74, -73.99, 40.71, -74.01)
    assert abs(mi - km * 0.621371) < 1e-6
