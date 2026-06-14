"""Geographic helpers — great-circle distance between lat/lng points."""

from __future__ import annotations

import math

_EARTH_KM = 6371.0088
MI_PER_KM = 0.621371


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in kilometres between two lat/lng points.

    Returns None if any coordinate is missing (so callers can sink
    unlocatable listings to the bottom of a vicinity sort)."""
    if lat1 is None or lng1 is None or lat2 is None or lng2 is None:
        return None
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def haversine_mi(lat1, lng1, lat2, lng2):
    """Great-circle distance in miles, or None if any coordinate is missing."""
    km = haversine_km(lat1, lng1, lat2, lng2)
    return None if km is None else km * MI_PER_KM
