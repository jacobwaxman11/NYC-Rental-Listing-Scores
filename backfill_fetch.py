"""Detail-page parsing + fetching for the backfill stage.

Extracts amenities/description/availability/floor-plan from a StreetEasy detail
page, and wraps the fetch in fingerprint-rotation retry logic (PerimeterX
sometimes flags one browser's TLS signature but not another's). The shared
request headers + file downloader live in :mod:`scrape_http`.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

from bs4 import BeautifulSoup

from scrape_http import HEADERS, _download_one


def find_floor_plan(soup: BeautifulSoup) -> Optional[str]:
    """Best-effort extraction of a floor plan image URL from a detail page.

    StreetEasy's HTML changes over time. We look for common signals:
      - <img> with alt/class containing "floor plan" / "floorplan"
      - <a> with class/data attribute pointing to a floor plan
      - Any zillowstatic image whose alt text mentions floor plan
    Returns None if nothing looks like a floor plan.
    """
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").lower()
        cls = " ".join(img.get("class") or []).lower()
        if "floor plan" in alt or "floorplan" in cls or "floor-plan" in cls:
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if src:
                return src

    for a in soup.find_all("a"):
        cls = " ".join(a.get("class") or []).lower()
        data_cat = (a.get("data-gtm-category") or "").lower()
        if "floor-plan" in cls or "floorplan" in cls or "floorplan" in data_cat:
            href = a.get("href")
            if href and href.startswith("http"):
                return href

    return None


def parse_detail(graph: list, soup: BeautifulSoup) -> dict:
    """Extract amenities/description/available_from/floor_plan from a detail page."""
    result: dict = {}
    for item in graph:
        t = item.get("@type")
        if t == "Apartment" and "amenityFeature" in item:
            result["amenities"] = [
                f["name"]
                for f in item["amenityFeature"]
                if f.get("value") is True
            ]
        if t == "ItemPage":
            about = item.get("about", {}) or {}
            result["description"] = about.get("description")
            offers = about.get("offers") or {}
            result["available_from"] = offers.get("validFrom")

    floor_plan_url = find_floor_plan(soup)
    if floor_plan_url:
        result["floor_plan_url"] = floor_plan_url
    return result


def fetch_detail(url: str, session, timeout: int = 20) -> tuple[Optional[dict], int]:
    """Fetch and parse a detail page. Returns (detail_dict_or_None, status_code).

    A non-200 response yields (None, status). We don't raise on 4xx so the
    caller can decide what to do (e.g. terminate on 403). ``session`` is a
    ``curl_cffi`` session (mimics a real browser's TLS fingerprint)."""
    try:
        response = session.get(url, headers=HEADERS, timeout=timeout)
    except Exception as e:
        print(f"    ✗ request failed: {e}")
        return None, -1

    status = response.status_code
    if status != 200:
        return None, status

    soup = BeautifulSoup(response.text, "html.parser")
    tag = soup.find("script", {"type": "application/ld+json"})
    graph: list = []
    if tag:
        try:
            graph = json.loads(tag.string).get("@graph", [])
        except json.JSONDecodeError:
            graph = []

    return parse_detail(graph, soup), status


def fetch_detail_with_rotation(
    url: str, get_session, fingerprints: list[str], fp_idx: int,
    timeout: int, min_delay: float, max_delay: float,
) -> tuple[Optional[dict], Optional[int], int]:
    """Fetch a detail page, rotating browser fingerprints on a 403.

    Tries the current fingerprint; on a 403 advances to the next and retries the
    same listing. Returns ``(detail, status, fp_idx)`` where ``fp_idx`` is the
    (possibly advanced) index — the caller reuses it for follow-up requests to
    the same listing and carries it into the next listing so a working
    fingerprint sticks. A 403 in ``status`` means EVERY fingerprint was blocked.
    """
    detail, status = None, None
    for attempt in range(len(fingerprints)):
        fp = fingerprints[fp_idx]
        detail, status = fetch_detail(url, get_session(fp), timeout=timeout)
        if status != 403:
            break
        print(f"  ⚠ 403 with impersonate={fp!r} — rotating fingerprint...")
        fp_idx = (fp_idx + 1) % len(fingerprints)
        if attempt < len(fingerprints) - 1:
            time.sleep(random.uniform(min_delay, max_delay))
    return detail, status, fp_idx


def maybe_download_floor_plan(listing: dict, image_dir: str, session) -> Optional[str]:
    """Download the listing's floor plan if we have a URL but no local file."""
    url = listing.get("floor_plan_url")
    if not url or listing.get("local_floor_plan_path"):
        return listing.get("local_floor_plan_path")

    building = listing.get("building_slug") or listing["listing_id"]
    unit = listing.get("unit") or listing["listing_id"]
    unit_dir = os.path.join(image_dir, building, unit)
    os.makedirs(unit_dir, exist_ok=True)

    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
    dest = os.path.join(unit_dir, f"floor_plan{ext}")

    if _download_one(url, dest, session):
        return dest
    return None


def merge_detail(listing: dict, detail: dict) -> dict:
    """Compute the subset of detail fields that should be written to the DB.

    We only fill in fields that are currently missing, so re-running the
    backfill after a partial success can't clobber good data.
    """
    updates: dict = {}
    for key in ("description", "available_from", "floor_plan_url"):
        if not listing.get(key) and detail.get(key):
            updates[key] = detail[key]
    return updates
