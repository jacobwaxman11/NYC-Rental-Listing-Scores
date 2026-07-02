"""Backfill amenities/description/availability/floor plan from StreetEasy detail pages.

This is the second stage of the pipeline. The scraper (`scrape_listings.py`)
collects basic listing data + images from search pages and writes them to
``rentals.db``. This script then walks through listings whose detail-page
fields haven't been filled in yet, fetches each detail page, and updates the DB.

Detail-page parsing/fetching (incl. the fingerprint-rotation retry) lives in
:mod:`backfill_fetch`; this module is the orchestration + CLI.

Design notes:
  - Listings are NEVER deleted; we only fill fields not yet populated. A listing
    "needs backfill" when its `listing_amenities` table has no rows for it.
  - We use `curl_cffi` (mimics a real browser's TLS fingerprint) instead of
    `httpx` to get through PerimeterX. `--impersonate` takes one OR MORE
    fingerprints; on a 403 we rotate and retry, terminating only when EVERY one
    is blocked (an IP-level block — continuing risks a longer ban).
  - Default delays are generous (5–12s), and saves are incremental (one commit
    per listing) so a Ctrl-C or 403 mid-run keeps what we've backfilled.

Example:
    python backfill_details.py --max-listings 50 --min-delay 8 --max-delay 15
    python backfill_details.py --only-missing   # skip already-fetched listings
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Optional

try:
    from curl_cffi import requests as cffi_requests
except ImportError as e:
    raise SystemExit(
        "curl_cffi is required for this script — install it with:\n"
        "    pip install curl_cffi"
    ) from e

import db as dbm
import poll_state
from backfill_fetch import (
    fetch_detail_with_rotation, maybe_download_floor_plan, merge_detail,
)


def run(
    db_path: str,
    image_dir: str,
    max_listings: Optional[int],
    min_delay: float,
    max_delay: float,
    fingerprints: list[str],
    timeout: int,
    only_missing: bool,
) -> int:
    """Returns the exit code (0 = ok, 2 = stopped on 403)."""
    with dbm.open_db(db_path) as conn:
        if only_missing:
            # Only listings we've never fetched a detail page for. This skips
            # listings that were already attempted but came back with no
            # amenities, so re-runs don't keep re-hitting them.
            todo = dbm.listings_never_fetched(conn)
            mode = "never-fetched"
        else:
            todo = dbm.listings_missing_amenities(conn)
            mode = "missing-amenities"
        total = dbm.stats(conn)["listings"]
        print(
            f"DB: {db_path} — {total} listings total; "
            f"{len(todo)} need detail backfill (mode: {mode})."
        )
        print(f"Last backfill: {poll_state.ago(poll_state.last(conn, 'backfill'))}")
        poll_state.record(conn, "backfill")

        if max_listings is not None:
            todo = todo[:max_listings]
            print(f"Capped at --max-listings {max_listings}.")

        if not todo:
            print("Nothing to do. Exiting.")
            return 0

        # One session per fingerprint, created lazily and reused. ``fp_idx``
        # persists across listings, so once a fingerprint gets through we keep
        # using it instead of re-trying a blocked one on every listing.
        sessions: dict[str, "cffi_requests.Session"] = {}

        def get_session(fp: str) -> "cffi_requests.Session":
            if fp not in sessions:
                sessions[fp] = cffi_requests.Session(impersonate=fp)
            return sessions[fp]

        fp_idx = 0
        print(f"Using curl_cffi — fingerprint rotation order: {fingerprints}\n")

        processed = 0
        succeeded = 0

        for n, listing in enumerate(todo, start=1):
            url = listing.get("url")
            name = listing.get("name") or listing["listing_id"]
            print(f"── [{n}/{len(todo)}] {name} ──")

            if not url:
                print("  ✗ no url, skipping")
                continue

            delay = random.uniform(min_delay, max_delay)
            print(f"  sleeping {delay:.1f}s...")
            time.sleep(delay)

            print(f"  → GET {url}")
            detail, status, fp_idx = fetch_detail_with_rotation(
                url, get_session, fingerprints, fp_idx, timeout, min_delay, max_delay
            )
            processed += 1

            if status == 403:
                print(
                    f"\n!!! 403 from ALL fingerprints ({', '.join(fingerprints)}) "
                    "— likely an IP-level block. Terminating to avoid further bans. !!!\n"
                    f"Processed {processed - 1} successful, "
                    f"{n - processed} skipped, before block.\n"
                    "Re-run later (try a longer delay or a different IP/VPN)."
                )
                return 2

            if detail is None:
                print(f"  ✗ status={status}, no detail extracted")
                continue

            # Apply updates to the listings row (description / available_from /
            # floor_plan_url) and persist amenities into their dedicated table.
            updates = merge_detail(listing, detail)
            if updates:
                dbm.update_listing_fields(conn, listing["listing_id"], updates)

            if detail.get("amenities"):
                dbm.set_amenities(conn, listing["listing_id"], detail["amenities"])
                print(f"  ✓ {len(detail['amenities'])} amenities")
            if "description" in updates and updates["description"]:
                print(f"  ✓ description ({len(updates['description'])} chars)")
            if "available_from" in updates:
                print(f"  ✓ available_from: {updates['available_from']}")
            if "floor_plan_url" in updates:
                print("  ✓ floor_plan_url found")

            # Floor plan download (best-effort, doesn't count against 403 logic).
            floor_plan_url = updates.get("floor_plan_url") or listing.get("floor_plan_url")
            if floor_plan_url and not listing.get("local_floor_plan_path"):
                fp_payload = {**listing, "floor_plan_url": floor_plan_url}
                # Reuse the fingerprint that just succeeded for this listing.
                fp_path = maybe_download_floor_plan(
                    fp_payload, image_dir, get_session(fingerprints[fp_idx])
                )
                if fp_path:
                    dbm.update_listing_fields(
                        conn, listing["listing_id"], {"local_floor_plan_path": fp_path}
                    )
                    print(f"  ✓ floor plan saved: {fp_path}")

            dbm.mark_detail_fetched(conn, listing["listing_id"])
            conn.commit()
            succeeded += 1
            print()

        print(
            f"── Done. {succeeded}/{processed} backfills succeeded "
            f"({len(todo) - processed} not attempted) ──"
        )
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill amenities + description + floor plan into the SQLite DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", type=str, default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument(
        "--image-dir",
        type=str,
        default="images",
        help="Where to save floor-plan images",
    )
    p.add_argument(
        "--max-listings",
        type=int,
        default=None,
        help="Cap on listings to process per run (useful for testing)",
    )
    p.add_argument(
        "--min-delay",
        type=float,
        default=5.0,
        help="Min sleep before each detail fetch (seconds)",
    )
    p.add_argument(
        "--max-delay",
        type=float,
        default=12.0,
        help="Max sleep before each detail fetch (seconds)",
    )
    p.add_argument(
        "--impersonate",
        type=str,
        default="safari180,safari170,chrome131,firefox144",
        help="curl_cffi browser fingerprint(s), comma-separated. On a 403 the "
             "scraper rotates to the next and retries, routing around a "
             "fingerprint-specific block; terminates only when all are blocked.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Per-request timeout in seconds",
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="Only fetch listings whose detail page has never been fetched "
             "(detail_fetched_at IS NULL) — skips ones already attempted that "
             "came back with no amenities, so re-runs don't re-hit them. Without "
             "it, any listing with an empty amenities table is (re)fetched.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fingerprints = [fp.strip() for fp in args.impersonate.split(",") if fp.strip()]
    if not fingerprints:
        raise SystemExit("--impersonate must list at least one fingerprint")
    code = run(
        db_path=args.db,
        image_dir=args.image_dir,
        max_listings=args.max_listings,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        fingerprints=fingerprints,
        timeout=args.timeout,
        only_missing=args.only_missing,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
