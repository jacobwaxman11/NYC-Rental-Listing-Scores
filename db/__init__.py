"""SQLite schema + helpers for the NYCRentalRankings dataset.

The database is the single source of truth: each script (scraper, backfill,
scorer, web UI) reads/writes through the helpers here instead of through JSON
files. This package re-exports every helper so ``import db as dbm`` stays a flat,
drop-in namespace — the table groups just live in their own modules now:

  - :mod:`db.connection`     connect / open_db / schema
  - :mod:`db.listings`       the ``listings`` table + "needs work" queries
  - :mod:`db.amenities`      the ``listing_amenities`` child table
  - :mod:`db.images`         the ``listing_images`` child table
  - :mod:`db.image_scores`   ``image_scores`` + ``image_tags`` (keyed by URL)
  - :mod:`db.listing_scores` aggregated per-listing scores
  - :mod:`db.reactions`      hearts / swipes from the web UI
  - :mod:`db.embeddings`     description/tag vectors
  - :mod:`db.meta`           generic key/value bookkeeping
  - :mod:`db.stats`          row-count summary

Design notes:
  - ``listing_id`` is the canonical key (building_slug + unit).
  - ``image_url`` is the canonical image key. The same URL can be referenced by
    multiple listings (shared building photo) but is scored only once.
  - listing_amenities + listing_images are "child" tables: ON DELETE CASCADE so
    ``DELETE FROM listings ...`` cleans up automatically.
  - image_scores and listing_scores are NOT cascaded — keeping them across
    listing deletions would make any future revival cheap.
"""

from __future__ import annotations

from config import DEFAULT_DB_PATH
from db.amenities import (
    get_all_listing_amenities, get_amenities, set_amenities,
)
from db.connection import SCHEMA, connect, init_schema, open_db
from db.embeddings import (
    get_all_embeddings, get_embedding_hashes, upsert_embedding,
)
from db.image_scores import (
    get_all_listing_tags, get_existing_image_scores, image_score_to_dict,
    upsert_image_score,
)
from db.images import get_listing_images, set_listing_images
from db.listing_scores import LISTING_SCORE_COLS, upsert_listing_scores
from db.listings import (
    LISTING_COLS, get_listing, get_listing_ids, listings_missing_amenities,
    listings_missing_scores, listings_never_fetched, mark_detail_fetched,
    update_listing_fields, upsert_listing,
)
from db.meta import get_meta, set_meta
from db.reactions import get_reactions, set_reaction
from db.stats import stats

__all__ = [
    "DEFAULT_DB_PATH",
    # connection
    "SCHEMA", "connect", "init_schema", "open_db",
    # listings
    "LISTING_COLS", "upsert_listing", "update_listing_fields",
    "mark_detail_fetched", "get_listing", "get_listing_ids",
    "listings_missing_amenities", "listings_never_fetched",
    "listings_missing_scores",
    # amenities
    "set_amenities", "get_amenities", "get_all_listing_amenities",
    # images
    "set_listing_images", "get_listing_images",
    # image scores
    "get_existing_image_scores", "upsert_image_score", "get_all_listing_tags",
    "image_score_to_dict",
    # listing scores
    "LISTING_SCORE_COLS", "upsert_listing_scores",
    # reactions
    "set_reaction", "get_reactions",
    # embeddings
    "upsert_embedding", "get_embedding_hashes", "get_all_embeddings",
    # meta
    "set_meta", "get_meta",
    # stats
    "stats",
]
