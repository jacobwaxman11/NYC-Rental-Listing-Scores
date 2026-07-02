"""Backward-compatible shim for the web UI, which now lives in the ``webapp``
package (see ``webapp/__init__.py``).

``python web.py`` still works, and ``import web`` still exposes the app plus the
search/taste helpers that the test-suite and older callers reach for. New code
should import from ``webapp`` directly, e.g.::

    python -m webapp --db rentals.db --port 8000
    from webapp.search import ai_search
"""

from __future__ import annotations

from webapp import app, create_app
from webapp.areas import AREA_OPTIONS
from webapp.browse import apply_browse, parse_browse_args
from webapp.cli import main
from webapp.search import (
    SEARCH_SYSTEM, _apply_plan, _build_context, _facets, ai_search,
)
from webapp.state import (
    PRICE_MAX, PRICE_MIN, PRICE_STEP, _SEARCH_CACHE, _STATE, load_embeddings,
)
from webapp.suggestions import build_suggestions
from webapp.taste import match_likes, similar_to

# Legacy alias — the loader was ``web._load_embeddings`` before the split.
_load_embeddings = load_embeddings

__all__ = [
    "app", "create_app", "main",
    "AREA_OPTIONS", "PRICE_MIN", "PRICE_MAX", "PRICE_STEP",
    "_STATE", "_SEARCH_CACHE", "load_embeddings", "_load_embeddings",
    "build_suggestions", "parse_browse_args", "apply_browse",
    "SEARCH_SYSTEM", "_facets", "_build_context", "_apply_plan", "ai_search",
    "match_likes", "similar_to",
]


if __name__ == "__main__":
    main()
