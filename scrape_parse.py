"""StreetEasy search-page parsing: build a search URL, parse a schema.org
Apartment object into a listing dict, and fetch+parse one search-results page.
"""

from __future__ import annotations

import json

import httpx
from bs4 import BeautifulSoup

from scrape_http import HEADERS  # noqa: F401  (re-exported for callers/tests)


def build_url(price_min: int, price_max: int, area: str, page: int = 1,
              sort: str = "listed_desc") -> str:
    """Build a search URL for ONE StreetEasy area.

    ``area`` is either a neighborhood slug — the reliable, human-readable form,
    e.g. ``west-village``, ``williamsburg``, ``hoboken`` (grab it from a
    ``streeteasy.com/for-rent/<slug>`` URL) — or a legacy numeric area ID. Slugs
    use ``/for-rent/<slug>/…``; numeric IDs use ``/for-rent/nyc/…|area:<id>``.

    ``sort`` defaults to ``listed_desc`` (newest first) so delta polling can
    early-stop once it reaches already-known listings.
    """
    area = str(area).strip()
    if area.isdigit():
        path = f"/for-rent/nyc/price:{price_min}-{price_max}|area:{area}"
    else:
        path = f"/for-rent/{area}/price:{price_min}-{price_max}"
    params = []
    if sort:
        params.append(f"sort_by={sort}")
    if page > 1:
        params.append(f"page={page}")
    if params:
        path += "?" + "&".join(params)
    return f"https://streeteasy.com{path}"


def parse_listing(apt: dict) -> dict:
    """Extract fields from a schema.org Apartment object on the search page."""
    props = {p["name"]: p["value"] for p in apt.get("additionalProperty", [])}

    rent_str = props.get("Monthly Rent", "")
    rent = (
        int(rent_str.replace("$", "").replace(",", "").replace("/mo", ""))
        if rent_str
        else None
    )

    sqft = apt.get("floorSize", {}).get("value") if apt.get("floorSize") else None
    lease = apt.get("leaseLength", {}).get("value") if apt.get("leaseLength") else None
    address = apt.get("address", {}) or {}
    image_urls = [img["url"] for img in apt.get("image", []) if img.get("url")]

    # Split the URL path into building slug + unit.
    # e.g. ".../building/stonehenge-gardens/006j" -> building="stonehenge-gardens", unit="006j"
    id_parts = [p for p in apt.get("@id", "").split("/") if p]
    building_slug = id_parts[-2] if len(id_parts) >= 2 else ""
    unit = id_parts[-1] if id_parts else ""
    listing_id = f"{building_slug}_{unit}" if building_slug else unit

    return {
        "listing_id": listing_id,
        "building_slug": building_slug,
        "unit": unit,
        "url": apt.get("url"),
        "name": apt.get("name"),
        "street": address.get("streetAddress"),
        "neighborhood": address.get("addressLocality"),
        "zip": address.get("postalCode"),
        "lat": apt.get("geo", {}).get("latitude") if apt.get("geo") else None,
        "lng": apt.get("geo", {}).get("longitude") if apt.get("geo") else None,
        "beds": apt.get("numberOfBedrooms"),
        "baths": apt.get("numberOfBathroomsTotal"),
        "sqft": sqft,
        "rent": rent,
        "lease_months": lease,
        "furnished": props.get("Furnished"),
        "building_type": props.get("Building Type"),
        # Image URLs are kept on the dict during scraping so download_images
        # can consume them; persist_listing then writes them to listing_images.
        "image_urls": image_urls,
    }


def fetch_search_page(url: str, client: httpx.Client) -> list[dict]:
    print(f"  → GET {url}")
    try:
        response = client.get(url, timeout=15)
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return []

    print(f"  ← {response.status_code} ({len(response.text):,} chars)")
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    tag = soup.find("script", {"type": "application/ld+json"})
    if not tag:
        print("  ✗ No ld+json tag found")
        return []

    try:
        graph = json.loads(tag.string).get("@graph", [])
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse ld+json: {e}")
        return []

    listings = [parse_listing(it) for it in graph if it.get("@type") == "Apartment"]
    print(f"  ✓ Parsed {len(listings)} listings from page")
    return listings
