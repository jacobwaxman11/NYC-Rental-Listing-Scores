"""Flask blueprints for the web UI, registered by :func:`webapp.create_app`."""

from webapp.routes.listings import bp as listings_bp
from webapp.routes.pipeline import bp as pipeline_bp

__all__ = ["listings_bp", "pipeline_bp"]
