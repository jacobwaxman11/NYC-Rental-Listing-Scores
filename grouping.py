"""Collapse a sorted list of listing rows into per-building groups.

A big NYC building can have many units in the results, all sharing the same hero
photo — which reads like duplicates. Grouping shows one lead card per building
(the most relevant unit, since the input is already sorted) with the rest of
that building's units attached, plus a count.

Pure and dependency-free so it can be unit-tested without the web app. The web
layer calls ``group_by_building(rows)`` after filtering/sorting and renders one
card per group.
"""

from __future__ import annotations

from typing import Callable, Optional


def building_key(row: dict) -> str:
    """The building a row belongs to: ``building_slug`` if present, else derived
    from the listing_id (which is ``"{building_slug}_{unit}"``)."""
    slug = row.get("building_slug")
    if slug:
        return slug
    lid = row.get("listing_id", "")
    return lid.rsplit("_", 1)[0] if "_" in lid else lid


def group_by_building(
    rows: list[dict], key: Optional[Callable[[dict], str]] = None
) -> list[dict]:
    """Group ``rows`` by building, preserving order.

    Input order is assumed to be relevance-sorted, so the first unit seen for a
    building becomes its ``lead`` (most relevant) and the building's position is
    fixed by where its lead first appears (best building first). Returns a list
    of dicts::

        {"building": str, "lead": row, "others": [row, ...], "count": int}
    """
    key = key or building_key
    groups: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        k = key(r)
        g = groups.get(k)
        if g is None:
            groups[k] = {"building": k, "lead": r, "others": [], "count": 1}
            order.append(k)
        else:
            g["others"].append(r)
            g["count"] += 1
    return [groups[k] for k in order]
