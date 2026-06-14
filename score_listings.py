"""Score apartment photos with Gemini or Claude and persist results to SQLite.

Reads listings (and their image URLs/local paths) from ``rentals.db``, scores
each unseen image with the chosen vision model, and writes per-image scores +
per-listing aggregates back into the same DB.

Two providers are supported and produce the same score shape, so they can be
compared head to head:

    # Gemini (default)
    python score_listings.py --provider gemini --model gemini-2.5-flash

    # Claude
    python score_listings.py --provider anthropic --model claude-opus-4-8

Requires GOOGLE_API_KEY (Gemini) or ANTHROPIC_API_KEY (Claude) in the
environment, or in a .env file.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import db as dbm
from prompts import SCORE_PROMPT, SYSTEM_PROMPT
from tags import VALID_TAGS

# Prompts and the tag vocabulary now live in prompts.py / tags.py.


# ── Scorers ──────────────────────────────────────────────────────────────────
#
# Each provider implements ``_call(paths) -> str`` (raw model text). The shared
# ``score_batch`` wrapper parses that text into one score dict per image and
# pads/truncates so the result always matches the number of images sent.


def _loads_lenient(raw: str):
    """Parse a JSON array out of a model response.

    Gemini is asked for application/json and returns a clean array; Claude
    usually returns a bare array too. If either wraps the array in markdown
    fences or a sentence of prose, fall back to the substring between the first
    ``[`` and the last ``]``."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _clean_tags(raw) -> list:
    """Keep only known vocabulary tags, de-duplicated and order-preserved."""
    if not isinstance(raw, list):
        return []
    return [t for t in dict.fromkeys(raw) if t in VALID_TAGS]


def _coerce_to_list(parsed, n: int) -> list:
    """Force a parsed response into exactly ``n`` elements."""
    if not isinstance(parsed, list):
        # Model occasionally returns a lone object for a batch-of-one despite
        # the instructions — tolerate it.
        parsed = [parsed]
    if len(parsed) != n:
        print(f"    ! expected {n} results, got {len(parsed)} — padding with nulls")
        parsed = list(parsed)[:n] + [None] * max(0, n - len(parsed))
    return parsed


class Scorer:
    """Base scorer. Subclasses set ``model_name`` and implement ``_call``."""

    model_name: str

    def _call(self, paths: list[str]) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def score_batch(self, paths: list[str]) -> list[Optional[dict]]:
        """Send up to N images in one call and return one score dict per image
        in the same order. On any failure, returns [None] * len(paths)."""
        if not paths:
            return []
        try:
            raw = self._call(paths)
            results = _coerce_to_list(_loads_lenient(raw), len(paths))
            for s in results:
                if isinstance(s, dict):
                    s["tags"] = _clean_tags(s.get("tags"))
            return results
        except Exception as e:
            print(f"    ✗ Batch API error: {e}")
            return [None] * len(paths)


class GeminiScorer(Scorer):
    def __init__(self, model_name: str, timeout_ms: int):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit(
                "GOOGLE_API_KEY not set. Add it to your .env file or export it."
            )
        # Imported lazily so the Anthropic-only path doesn't require google-genai.
        from google import genai
        from google.genai import types

        self._types = types
        self.model_name = model_name
        self._timeout_ms = timeout_ms
        self._client = genai.Client(api_key=api_key)

    def _call(self, paths: list[str]) -> str:
        types = self._types
        parts: list = [SCORE_PROMPT]
        for path in paths:
            data = Path(path).read_bytes()
            mime = mimetypes.guess_type(path)[0] or "image/webp"
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=self._timeout_ms),
            ),
        )
        return response.text


class ClaudeScorer(Scorer):
    # Anthropic accepts JPEG, PNG, GIF, and WebP. Listing photos are usually
    # WebP, which is what the scraper downloads.
    _ALLOWED_MEDIA = {"image/jpeg", "image/png", "image/gif", "image/webp"}

    def __init__(self, model_name: str, timeout_ms: int, max_tokens: int = 8192):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY not set. Add it to your .env file or export it."
            )
        # Imported lazily so the Gemini-only path doesn't require anthropic.
        import anthropic

        self.model_name = model_name
        self._max_tokens = max_tokens
        # Anthropic's SDK timeout is in seconds.
        self._client = anthropic.Anthropic(timeout=timeout_ms / 1000.0)

    def _call(self, paths: list[str]) -> str:
        blocks: list = [{"type": "text", "text": SCORE_PROMPT}]
        for path in paths:
            data = base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")
            mime = mimetypes.guess_type(path)[0] or "image/webp"
            if mime not in self._ALLOWED_MEDIA:
                mime = "image/jpeg"
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": data},
            })

        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": blocks}],
        )
        # The system prompt constrains output to JSON; concatenate any text
        # blocks and let _loads_lenient pull the array out.
        return "".join(b.text for b in response.content if b.type == "text")


PROVIDER_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-opus-4-8",
}


def build_scorer(provider: str, model_name: str, timeout_ms: int) -> Scorer:
    if provider == "gemini":
        return GeminiScorer(model_name, timeout_ms)
    if provider == "anthropic":
        return ClaudeScorer(model_name, timeout_ms)
    raise SystemExit(f"Unknown provider: {provider!r} (expected 'gemini' or 'anthropic')")


def aggregate_scores(image_scores: list[Optional[dict]]) -> dict:
    """Roll up per-image scores into per-listing features.

    Separates apartment photos from common-space photos so averages over
    apartment features are not polluted by lobby/gym/exterior shots, and
    counts irrelevant photos separately so we can flag low-quality listings.
    """
    scored = [s for s in image_scores if s]

    apartment = [s["apartment_scores"] for s in scored
                 if s.get("image_category") == "apartment" and s.get("apartment_scores")]
    common = [s["common_space_scores"] for s in scored
              if s.get("image_category") == "common_space" and s.get("common_space_scores")]
    irrelevant = [s for s in scored if s.get("image_category") == "irrelevant"]

    def _avg(items: list[dict], key: str) -> Optional[float]:
        vals = [it[key] for it in items if it.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def _max(items: list[dict], key: str):
        vals = [it[key] for it in items if it.get(key) is not None]
        return max(vals) if vals else None

    # ── Apartment metrics ────────────────────────────────────────────────────
    room_types = [a.get("room_type") for a in apartment if a.get("room_type")]
    room_type_counts = {r: room_types.count(r) for r in set(room_types)}

    apartment_summary = {
        "avg_natural_light":  _avg(apartment, "natural_light"),
        "avg_finish_quality": _avg(apartment, "finish_quality"),
        "avg_space_feeling":  _avg(apartment, "space_feeling"),
        "avg_condition":      _avg(apartment, "condition"),
        "max_view_quality":   _max(apartment, "view_quality"),
        "pct_bright_rooms":   round(
            sum(1 for a in apartment if a.get("natural_light", 0) >= 7) / len(apartment), 2
        ) if apartment else None,
        "has_good_view":      (_max(apartment, "view_quality") or 0) >= 7,
        "room_type_counts":   room_type_counts,
    } if apartment else {}

    # ── Common-space metrics ─────────────────────────────────────────────────
    # NOTE: we deliberately do NOT roll a "detected_amenities" boolean list out
    # of the photos — StreetEasy's `amenities` (populated by backfill_details.py)
    # is the canonical source. Including a Gemini-inferred copy would create
    # autocorrelated features for downstream modeling. We keep `space_type_counts`
    # because the *quantity* of photos per amenity carries a different signal
    # (how prominently the listing showcases each space) than a flat existence flag.
    space_types = [c.get("space_type") for c in common if c.get("space_type")]
    space_type_counts = {s: space_types.count(s) for s in set(space_types)}

    common_summary = {
        "avg_finish_quality": _avg(common, "finish_quality"),
        "avg_condition":      _avg(common, "condition"),
        "avg_appeal":         _avg(common, "appeal"),
        "space_type_counts":  space_type_counts,
    } if common else {}

    return {
        "apartment":         apartment_summary,
        "common_space":      common_summary,
        "photos_total":      len(image_scores),
        "photos_scored":     len(scored),
        "photos_apartment":  len(apartment),
        "photos_common":     len(common),
        "photos_irrelevant": len(irrelevant),
    }


def _log_score(idx: int, total: int, path: str, scores: Optional[dict]) -> None:
    prefix = f"    [{idx}/{total}] {os.path.basename(path)}"
    if not scores:
        print(f"{prefix}  ✗ no score")
        return
    cat = scores.get("image_category")
    if cat == "apartment":
        a = scores.get("apartment_scores") or {}
        print(
            f"{prefix}  ✓ apartment/{a.get('room_type')} "
            f"light={a.get('natural_light')} "
            f"finish={a.get('finish_quality')} "
            f"condition={a.get('condition')}"
        )
    elif cat == "common_space":
        c = scores.get("common_space_scores") or {}
        print(
            f"{prefix}  ✓ common/{c.get('space_type')} "
            f"finish={c.get('finish_quality')} "
            f"appeal={c.get('appeal')}"
        )
    elif cat == "irrelevant":
        print(f"{prefix}  ⊘ irrelevant: {scores.get('irrelevant_reason')}")
    else:
        print(f"{prefix}  ? unknown category: {cat}")


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
            _log_score(idx + 1, len(images), path, scores)

        # Commit progress after every batch so a crash doesn't lose the API
        # spend we just incurred.
        conn.commit()

        if sleep_s > 0 and batch_start + batch_size < len(miss_indices):
            time.sleep(sleep_s)

    summary = aggregate_scores(raw_scores)
    dbm.upsert_listing_scores(conn, listing_id, summary)
    conn.commit()
    return summary


# ── Main ─────────────────────────────────────────────────────────────────────


def _print_summary(s: dict) -> None:
    if not s:
        return
    print("  Summary:")
    print(
        f"    photos: {s.get('photos_apartment', 0)} apt / "
        f"{s.get('photos_common', 0)} common / "
        f"{s.get('photos_irrelevant', 0)} irrelevant "
        f"(total {s.get('photos_total', 0)})"
    )
    apt = s.get("apartment") or {}
    if apt:
        print(
            f"    apt:    light={apt.get('avg_natural_light')} "
            f"finish={apt.get('avg_finish_quality')} "
            f"space={apt.get('avg_space_feeling')} "
            f"cond={apt.get('avg_condition')} "
            f"view_max={apt.get('max_view_quality')} "
            f"bright%={apt.get('pct_bright_rooms')}"
        )
        print(f"    rooms:  {apt.get('room_type_counts')}")
    common = s.get("common_space") or {}
    if common:
        print(
            f"    common: finish={common.get('avg_finish_quality')} "
            f"cond={common.get('avg_condition')} "
            f"appeal={common.get('avg_appeal')}"
        )
        print(f"    spaces: {common.get('space_type_counts')}")


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
                conn,
                listing,
                scorer,
                sleep_s,
                batch_size=batch_size,
                max_images=max_images,
            )
            _print_summary(summary)

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
