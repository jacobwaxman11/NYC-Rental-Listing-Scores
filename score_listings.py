"""Score apartment photos with Gemini or Claude and persist results to SQLite.

Reads listings (and their image URLs/local paths) from ``rentals.db``, scores
each unseen image with the chosen vision model, and writes per-image scores +
per-listing aggregates back into the same DB.

Aggregation + console logging live in :mod:`score_aggregate`; the provider
scorers in :mod:`scorers`. Two providers produce the same score shape, so they
can be compared head to head:

    # Gemini (default)
    python score_listings.py --provider gemini --model gemini-2.5-flash

    # Claude
    python score_listings.py --provider anthropic --model claude-opus-4-8

Requires GOOGLE_API_KEY (Gemini) or ANTHROPIC_API_KEY (Claude) in the
environment, or in a .env file.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from typing import Optional

from dotenv import load_dotenv

import db as dbm
from score_aggregate import aggregate_scores, log_score, print_summary
from scorers import PROVIDER_DEFAULT_MODELS, Scorer, build_scorer

# ``aggregate_scores`` is re-exported (tests import it from here).
__all__ = ["aggregate_scores", "score_listing", "run", "main"]


def score_listing(
    conn: sqlite3.Connection,
    listing: dict,
    scorer: Scorer,
    sleep_s: float,
    batch_size: int,
    max_images: int,
) -> dict:
    """Score images for one listing. Persists each new image score immediately
    (so a crash mid-listing keeps work) and aggregates at the end into
    ``listing_scores``.

    Returns the aggregated summary dict (same shape as before)."""
    listing_id = listing["listing_id"]

    # Pull the listing's images from the DB.
    images = dbm.get_listing_images(conn, listing_id)
    urls = [im["image_url"] for im in images]
    paths = [im["local_image_path"] for im in images]

    if not images:
        print("  ✗ no images to score")
        summary = aggregate_scores([])
        dbm.upsert_listing_scores(conn, listing_id, summary)
        return summary

    if len(images) > max_images:
        print(f"  capping {len(images)} → {max_images} images")
        images = images[:max_images]
        urls = urls[:max_images]
        paths = paths[:max_images]

    # Cache lookup is now keyed by image_url against the image_scores table.
    cached = dbm.get_existing_image_scores(conn, urls)
    raw_scores: list[Optional[dict]] = [None] * len(images)
    miss_indices: list[int] = []

    for i, url in enumerate(urls):
        if url in cached:
            raw_scores[i] = dbm.image_score_to_dict(cached[url])
        elif paths[i] and os.path.exists(paths[i]):
            miss_indices.append(i)
        else:
            print(f"    ! image {i + 1}: no local file at {paths[i]!r}, skipping")

    cache_hits = sum(1 for s in raw_scores if s is not None)
    if cache_hits:
        print(f"  ↺ {cache_hits}/{len(images)} images already scored in DB")

    # Send misses in batches.
    for batch_start in range(0, len(miss_indices), batch_size):
        batch_idxs = miss_indices[batch_start:batch_start + batch_size]
        batch_paths = [paths[i] for i in batch_idxs]
        batch_urls = [urls[i] for i in batch_idxs]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(miss_indices) + batch_size - 1) // batch_size
        print(
            f"  → batch {batch_num}/{total_batches} "
            f"({len(batch_paths)} image{'s' if len(batch_paths) != 1 else ''})"
        )

        batch_scores = scorer.score_batch(batch_paths)

        for idx, path, url, scores in zip(batch_idxs, batch_paths, batch_urls, batch_scores):
            raw_scores[idx] = scores
            if scores:
                dbm.upsert_image_score(conn, url, scores, scorer.model_name)
            log_score(idx + 1, len(images), path, scores)

        # Commit progress after every batch so a crash doesn't lose the API
        # spend we just incurred.
        conn.commit()

        if sleep_s > 0 and batch_start + batch_size < len(miss_indices):
            time.sleep(sleep_s)

    summary = aggregate_scores(raw_scores)
    dbm.upsert_listing_scores(conn, listing_id, summary)
    conn.commit()
    return summary


def run(
    db_path: str,
    provider: str,
    model_name: str,
    max_listings: Optional[int],
    sleep_s: float,
    timeout_ms: int,
    rescore: bool,
    batch_size: int,
    max_images: int,
) -> None:
    load_dotenv()

    scorer = build_scorer(provider, model_name, timeout_ms)

    with dbm.open_db(db_path) as conn:
        # Pick the queue. ``rescore`` re-aggregates everything (image scores
        # already in image_scores are still re-used; we just rebuild
        # listing_scores). Default is "only listings without aggregates".
        if rescore:
            rows = conn.execute("SELECT * FROM listings ORDER BY listing_id").fetchall()
            queue = [dict(r) for r in rows]
        else:
            queue = dbm.listings_missing_scores(conn)

        if max_listings is not None:
            queue = queue[:max_listings]

        already = dbm.stats(conn)["scored_images"]
        print(
            f"DB: {db_path} — provider={provider} model={model_name} — "
            f"{already} images already scored, "
            f"{len(queue)} listing{'s' if len(queue) != 1 else ''} queued."
        )

        for i, listing in enumerate(queue, start=1):
            name = listing.get("name") or listing["listing_id"]
            print(f"\n── [{i}/{len(queue)}] {name} ──────────────────")

            summary = score_listing(
                conn, listing, scorer, sleep_s,
                batch_size=batch_size, max_images=max_images,
            )
            print_summary(summary)

        st = dbm.stats(conn)
        print(
            f"\n── Done. {st['listings_with_scores']}/{st['listings']} listings scored, "
            f"{st['scored_images']} unique images scored in {db_path} ──"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score apartment photos with Gemini or Claude.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", type=str, default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "anthropic"],
        default="gemini",
        help="Vision model provider",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. Defaults per provider: gemini-2.5-flash (gemini), "
             "claude-opus-4-8 (anthropic). Other examples: gemini-2.5-pro, "
             "claude-haiku-4-5 (cheaper/faster).",
    )
    p.add_argument(
        "--max-listings",
        type=int,
        default=8,
        help="Only score the first N listings in the queue (useful for testing)",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=4,
        help="Seconds to sleep between batch calls",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of images to send per API call",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Cap images scored per listing (takes the first N)",
    )
    p.add_argument(
        "--timeout-ms",
        type=int,
        default=60_000,
        help="Per-call HTTP timeout in milliseconds",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="Re-aggregate every listing's listing_scores even if already present "
             "(individual image_scores rows are still cached, so this is cheap)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model or PROVIDER_DEFAULT_MODELS[args.provider]
    run(
        db_path=args.db,
        provider=args.provider,
        model_name=model_name,
        max_listings=args.max_listings,
        sleep_s=args.sleep,
        timeout_ms=args.timeout_ms,
        rescore=args.rescore,
        batch_size=args.batch_size,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
