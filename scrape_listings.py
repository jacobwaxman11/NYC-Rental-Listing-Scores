"""Scrape StreetEasy rental listings (search pages) and download images.

This is stage 1 of the pipeline. It walks search-result pages, parses the
schema.org Apartment objects, downloads each listing's photos, and writes the
results to ``rentals.db``. Detail-page fields (amenities, description,
available_from, floor plan) are filled in afterwards by ``backfill_details.py``.

Re-runs are additive and idempotent: existing listing_ids are skipped, and only
newly-discovered listings get processed.

Example:
    python scrape_listings.py \\
        --price-min 4000 --price-max 6000 \\
        --areas west-village,soho,williamsburg \\
        --max-listings 100

Areas are StreetEasy neighborhood slugs — grab one from a
``streeteasy.com/for-rent/<slug>`` URL. Examples: west-village, soho, nolita,
tribeca, les (Lower East Side), williamsburg, greenpoint, dumbo, hoboken,
jersey-city. Legacy numeric area IDs (e.g. 115) still work too.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import time
from typing import Optional

import httpx
from bs4 import BeautifulSoup

import db as dbm


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://streeteasy.com/",
}


# ── URL + parsing helpers ────────────────────────────────────────────────────


def build_url(price_min: int, price_max: int, area: str, page: int = 1) -> str:
    """Build a search URL for ONE StreetEasy area.

    ``area`` is either a neighborhood slug — the reliable, human-readable form,
    e.g. ``west-village``, ``williamsburg``, ``hoboken`` (grab it from a
    ``streeteasy.com/for-rent/<slug>`` URL) — or a legacy numeric area ID. Slugs
    use ``/for-rent/<slug>/…``; numeric IDs use ``/for-rent/nyc/…|area:<id>``.
    """
    area = str(area).strip()
    if area.isdigit():
        path = f"/for-rent/nyc/price:{price_min}-{price_max}|area:{area}"
    else:
        path = f"/for-rent/{area}/price:{price_min}-{price_max}"
    if page > 1:
        path += f"?page={page}"
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


# ── Fetchers ─────────────────────────────────────────────────────────────────


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


def _download_one(url: str, dest_path: str, client) -> bool:
    """Download a single URL to dest_path. Returns True on success or if it already exists.

    Generic enough that ``backfill_details.py`` reuses it for floor-plan downloads
    (passing in its own curl_cffi session in place of an httpx.Client)."""
    if os.path.exists(dest_path):
        return True
    try:
        r = client.get(url, timeout=15)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"    ✗ download failed ({url}): {e}")
        return False


def download_images(listing: dict, base_dir: str, client: httpx.Client) -> list[Optional[str]]:
    """Download all listing images into images/<building>/<unit>/.

    Filenames use the Zillow CDN hash (last path segment of the URL).
    Returns a list of local paths in the same order as listing['image_urls'];
    failed downloads produce a None placeholder so positional alignment is
    preserved when we write to listing_images.
    """
    building = listing.get("building_slug") or listing["listing_id"]
    unit = listing.get("unit") or listing["listing_id"]
    unit_dir = os.path.join(base_dir, building, unit)
    os.makedirs(unit_dir, exist_ok=True)

    local_paths: list[Optional[str]] = []
    for img_url in listing.get("image_urls", []):
        filename = img_url.split("/")[-1]  # e.g. "b0c21bbf...-p_e.webp"
        filepath = os.path.join(unit_dir, filename)

        if _download_one(img_url, filepath, client):
            local_paths.append(filepath)
        else:
            local_paths.append(None)  # keep positional alignment

    return local_paths


# ── Persist ──────────────────────────────────────────────────────────────────


def persist_listing(conn: sqlite3.Connection, listing: dict) -> None:
    """Write a listing's core columns + image refs to the DB.

    Detail-page fields (amenities, description, available_from, floor plan)
    are intentionally not touched here — they belong to backfill_details.py.
    """
    dbm.upsert_listing(conn, listing)

    if listing.get("image_urls"):
        dbm.set_listing_images(
            conn,
            listing["listing_id"],
            listing["image_urls"],
            listing.get("local_image_paths"),
        )

    conn.commit()


# ── Main ─────────────────────────────────────────────────────────────────────


def scrape(
    db_path: str,
    price_min: int,
    price_max: int,
    areas: list[str],
    max_listings: Optional[int],
    max_pages: int,
    image_dir: str,
    skip_images: bool,
    min_delay: float,
    max_delay: float,
) -> None:
    os.makedirs(image_dir, exist_ok=True)

    with dbm.open_db(db_path) as conn:
        existing_ids: set[str] = dbm.get_listing_ids(conn)
        if existing_ids:
            print(f"Resuming — {len(existing_ids)} listings already in {db_path}")
        else:
            print(f"Starting fresh (no existing entries in {db_path})")

        new_to_process: list[dict] = []

        with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
            # ── Stage 1: search pages, one StreetEasy area at a time ─────────
            # ``max_listings`` is a GLOBAL cap across all areas. For each area we
            # paginate until it runs out of pages (or we hit the cap). Listings
            # already in the DB — or already queued from another area — are
            # skipped, so overlapping areas don't double-count.
            for area in areas:
                if max_listings is not None and len(new_to_process) >= max_listings:
                    break
                print(f"\n══ Area: {area} ══════════════════════════════════════")

                for page in range(1, max_pages + 1):
                    if max_listings is not None and len(new_to_process) >= max_listings:
                        break

                    print(f"── {area} · page {page} ──")
                    url = build_url(price_min, price_max, area, page)
                    page_listings = fetch_search_page(url, client)

                    if not page_listings:
                        print("  (no listings — on to the next area)")
                        break

                    dup_count = 0
                    for listing in page_listings:
                        lid = listing.get("listing_id")
                        if lid and lid in existing_ids:
                            dup_count += 1
                            continue
                        new_to_process.append(listing)
                        existing_ids.add(lid)  # avoid cross-area / in-page dups too
                        if max_listings is not None and len(new_to_process) >= max_listings:
                            break

                    if dup_count:
                        print(f"  ↺ {dup_count} already seen, skipping")

                    delay = random.uniform(min_delay, max_delay)
                    print(f"  sleeping {delay:.1f}s...")
                    time.sleep(delay)

            print(f"\n── {len(new_to_process)} new listings to process ──")

            if not new_to_process:
                print("Nothing new to scrape. Exiting.")
                return

            # ── Stage 2: image downloads per NEW listing ─────────────────────
            for i, listing in enumerate(new_to_process, start=1):
                print(
                    f"\n── [{i}/{len(new_to_process)}] "
                    f"{listing.get('name') or listing.get('listing_id')} "
                    f"────────────────"
                )

                if not skip_images and listing.get("image_urls"):
                    print(f"  downloading {len(listing['image_urls'])} images...")
                    local_paths = download_images(listing, image_dir, client)
                    listing["local_image_paths"] = local_paths
                    n_ok = sum(1 for p in local_paths if p)
                    print(f"  ✓ {n_ok}/{len(local_paths)} images saved")

                # Incremental persist: a Ctrl-C between listings keeps progress.
                persist_listing(conn, listing)

        st = dbm.stats(conn)
        print(
            f"\n── Done. {st['listings']} listings in {db_path} "
            f"({len(new_to_process)} new this run) ──"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape StreetEasy rental listings and download images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", type=str, default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument("--price-min", type=int, default=3000, help="Minimum rent ($/mo)")
    p.add_argument("--price-max", type=int, default=7000, help="Maximum rent ($/mo)")
    p.add_argument(
        "--areas",
        type=str,
        default=(
            "financial-district,battery-park-city,fulton-seaport,civic-center,tribeca,"
            "soho,nolita,little-italy,chinatown,les,west-village,greenwich-village,"
            "east-village,noho,chelsea,west-chelsea,flatiron,nomad,gramercy-park,"
            "murray-hill,kips-bay,hudson-yards,hells-kitchen"
        ),
        help="Comma-separated StreetEasy neighborhood slugs (e.g. "
             "west-village,williamsburg,hoboken). Grab a slug from a "
             "streeteasy.com/for-rent/<slug> URL. Legacy numeric area IDs also work.",
    )
    p.add_argument(
        "--max-listings",
        type=int,
        default=250,
        help="Stop after this many NEW listings (useful for testing, e.g. 100)",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Hard cap on search-result pages to crawl",
    )
    p.add_argument("--image-dir", type=str, default="images", help="Directory for downloaded images")
    p.add_argument("--skip-images", action="store_true", help="Skip image downloads")
    p.add_argument("--min-delay", type=float, default=2.0, help="Min polite delay between requests (s)")
    p.add_argument("--max-delay", type=float, default=4.0, help="Max polite delay between requests (s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    areas = [a.strip() for a in args.areas.split(",") if a.strip()]

    scrape(
        db_path=args.db,
        price_min=args.price_min,
        price_max=args.price_max,
        areas=areas,
        max_listings=args.max_listings,
        max_pages=args.max_pages,
        image_dir=args.image_dir,
        skip_images=args.skip_images,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
    )


if __name__ == "__main__":
    main()
