"""Embed each listing's text profile for semantic search (optional pipeline step).

Builds a compact profile string per listing (neighborhood + size + photo tags +
broker description), embeds it locally with sentence-transformers, and stores the
vector in ``listing_embeddings``. Runs offline, no API cost.

Re-runs are incremental: a listing is skipped when its (model, profile-text) hash
is unchanged. Use ``--rebuild`` to force re-embedding everything (e.g. after the
description or tags changed, or to switch models).

Run after scoring (tags improve the profile), though description alone also works:

    python embed_listings.py
    python embed_listings.py --model all-mpnet-base-v2 --rebuild

Then start the web UI — "Match my likes" and free-text vibe search light up.
"""

from __future__ import annotations

import argparse

import db as dbm
from embeddings import DEFAULT_MODEL, Embedder, profile_text, text_hash


def run(db_path: str, model_name: str, max_listings: int | None, rebuild: bool) -> None:
    embedder = Embedder(model_name)

    with dbm.open_db(db_path) as conn:
        tags_by_listing = dbm.get_all_listing_tags(conn)
        existing = {} if rebuild else dbm.get_embedding_hashes(conn)

        todo: list[tuple[str, str, str]] = []  # (listing_id, text, hash)
        skipped_empty = 0
        for row in conn.execute("SELECT * FROM listings"):
            listing = dict(row)
            lid = listing["listing_id"]
            text = profile_text(listing, tags_by_listing.get(lid, []))
            if not text:
                skipped_empty += 1
                continue
            h = text_hash(model_name, text)
            if existing.get(lid) == h:
                continue
            todo.append((lid, text, h))

        if max_listings is not None:
            todo = todo[:max_listings]

        print(
            f"DB: {db_path} — model {model_name} — {len(todo)} listing"
            f"{'s' if len(todo) != 1 else ''} to embed"
            f"{f' ({skipped_empty} have no text)' if skipped_empty else ''}."
        )
        if not todo:
            print("Nothing to do (all up to date). Use --rebuild to force.")
            return

        batch = 64
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            vectors = embedder.encode([t for _, t, _ in chunk])
            for (lid, _, h), vec in zip(chunk, vectors):
                dbm.upsert_embedding(conn, lid, model_name, vec.tolist(), h)
            conn.commit()
            print(f"  embedded {min(i + batch, len(todo))}/{len(todo)}")

    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Embed listing text profiles for semantic search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    p.add_argument("--max-listings", type=int, default=None, help="Embed at most N (testing)")
    p.add_argument("--rebuild", action="store_true", help="Re-embed all listings, ignoring cache")
    args = p.parse_args()
    run(args.db, args.model, args.max_listings, args.rebuild)


if __name__ == "__main__":
    main()
