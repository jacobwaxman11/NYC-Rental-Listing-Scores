"""Flask app for the underpriced-listings UI.

This package is the interactive front-end for the pipeline. It:

  1. Builds the engineered feature frame (``features.prepare_features``).
  2. Fits a Ridge regression on ``log_rent`` and predicts each listing's
     "market" rent with out-of-fold predictions grouped by building
     (:mod:`webapp.suggestions`).
  3. Ranks listings by how far their actual rent sits below the prediction and
     renders them as photo cards — biggest discounts ("deals") first.

You can ❤ listings (persisted to ``listing_reactions``) and triage them in
"Tinder mode". Natural-language / taste search live in :mod:`webapp.search` and
:mod:`webapp.taste`; the scrape/backfill/score control panel in
:mod:`webapp.routes.pipeline`.

Run it (after scraping + scoring have populated rentals.db)::

    pip install -r requirements.txt
    python web.py                       # http://127.0.0.1:5000  (thin shim)
    python -m webapp --db rentals.db --port 8000

The suggestion set is computed once at startup and cached; hit ``/?refresh=1``
to recompute after re-running the scorer.
"""

from __future__ import annotations

import os

from flask import Flask

# Templates and static assets live at the repo root, one level above this
# package, so point Flask there explicitly (its default would look inside
# ``webapp/``).
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_app() -> Flask:
    """Build the Flask app and register the UI + pipeline blueprints."""
    app = Flask(
        __name__,
        template_folder=os.path.join(_REPO_DIR, "templates"),
        static_folder=os.path.join(_REPO_DIR, "static"),
    )
    from webapp.routes import listings_bp, pipeline_bp
    app.register_blueprint(listings_bp)
    app.register_blueprint(pipeline_bp)
    return app


# Module-level singleton — imported as ``webapp.app`` (and re-exported by the
# ``web.py`` shim as ``web.app`` for the test client).
app = create_app()
