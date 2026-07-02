"""Scrape StreetEasy rental listings (search pages) and download images.

This is stage 1 of the pipeline. It walks search-result pages, parses the
schema.org Apartment objects, downloads each listing's photos, and writes the
results to ``rentals.db``. Detail-page fields (amenities, description,
available_from, floor plan) are filled in afterwards by ``backfill_details.py``.

Re-runs are additive and idempotent: existing listing_ids are skipped, and only
newly-discovered listings get processed.

Parsing/URL helpers live in :mod:`scrape_parse`; the shared request headers and
file downloader in :mod:`scrape_http` (both re-exported here for callers/tests).

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
import os
import random
import sqlite3
import time
from typing import Optional

import httpx

import db as dbm
import poll_state
from scrape_http import HEADERS, _download_one, download_images
from scrape_parse import build_url, fetch_search_page, parse_listing

# Re-exported so ``import scrape_listings`` stays a one-stop namespace for the
# parsing/URL/download helpers (tests and backfill_details rely on this).
__all__ = [
    "HEADERS", "_download_one", "download_images",
    "build_url", "parse_listing", "fetch_search_page",
    "persist_listing", "scrape", "main",
]


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
    full: bool = False,
) -> None:
    os.makedirs(image_dir, exist_ok=True)

    with dbm.open_db(db_path) as conn:
        existing_ids: set[str] = dbm.get_listing_ids(conn)
        if existing_ids:
            print(f"Resuming — {len(existing_ids)} listings already in {db_path}")
        else:
            print(f"Starting fresh (no existing entries in {db_path})")
        print(f"Last scrape: {poll_state.ago(poll_state.last(conn, 'scrape'))}"
              + ("  ·  --full re-crawl" if full else "  ·  delta mode (newest-first, early-stop)"))

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

                    before = len(new_to_process)
                    for listing in page_listings:
                        lid = listing.get("listing_id")
                        if lid and lid in existing_ids:
                            continue
                        new_to_process.append(listing)
                        existing_ids.add(lid)  # avoid cross-area / in-page dups too
                        if max_listings is not None and len(new_to_process) >= max_listings:
                            break

                    new_this_page = len(new_to_process) - before
                    dup_count = len(page_listings) - new_this_page
                    if dup_count:
                        print(f"  ↺ {dup_count} already seen, skipping")

                    # ── Delta early-stop ─────────────────────────────────────
                    # Results are sorted newest-first, so a page with no new
                    # listings means we've reached already-known territory —
                    # everything beyond it is older and already in the DB. Stop
                    # this area (unless --full forces a complete re-crawl).
                    if not full and new_this_page == 0:
                        print("  ✓ no new listings here — delta stop "
                              "(--full to re-crawl everything)")
                        break

                    delay = random.uniform(min_delay, max_delay)
                    print(f"  sleeping {delay:.1f}s...")
                    time.sleep(delay)

            print(f"\n── {len(new_to_process)} new listings to process ──")
            poll_state.record(conn, "scrape")   # we polled, regardless of yield

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
    p.add_argument("--full", action="store_true",
                   help="Re-crawl every page instead of delta early-stopping at the "
                        "first page with no new listings (newest-first sort)")
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
        full=args.full,
    )


if __name__ == "__main__":
    main()
