"""Command-line entry point: parse args, wire up the model/embedder, build the
initial suggestion set, and serve. Invoked via ``python -m webapp`` (and by the
``web.py`` shim's ``main()``)."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

import db as dbm
import embeddings as emb_mod
import llm as llm_mod
from webapp import app
from webapp.state import _STATE, load_embeddings
from webapp.suggestions import build_suggestions


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Web UI for suggested (underpriced) listings.")
    p.add_argument("--db", default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini",
                   help="Model provider for AI search (needs the matching API key)")
    p.add_argument("--model", default=None,
                   help="Model for AI search (default: per-provider — gemini-2.5-flash "
                        "or claude-opus-4-8; claude-haiku-4-5 is cheaper)")
    p.add_argument("--embed-model", default=emb_mod.DEFAULT_MODEL,
                   help="sentence-transformers model for query embedding (semantic search)")
    p.add_argument("--host", default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=5000, help="Bind port")
    p.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    return p.parse_args()


def _print_startup(args: argparse.Namespace, engine, model: str | None) -> None:
    m = _STATE["meta"]
    print(f"  {m.get('underpriced', 0)} underpriced of {m.get('total', 0)} scored listings")
    if engine:
        print(f"  AI search: enabled ({args.provider} / {model})")
    else:
        key = "GOOGLE_API_KEY" if args.provider == "gemini" else "ANTHROPIC_API_KEY"
        print(f"  AI search: disabled — set {key} (in .env) to enable")
    n_emb = len(_STATE["embeddings"])
    if n_emb:
        sem = "on" if _STATE["embedder"] is not None else "stored-only (install sentence-transformers)"
        print(f"  Semantic: {n_emb} embeddings loaded — Match-my-likes on, free-text {sem}")
    else:
        print("  Semantic: no embeddings — run `python embed_listings.py` to enable taste matching")


def main() -> None:
    args = _parse_args()

    load_dotenv()
    _STATE["db_path"] = args.db

    model = args.model or llm_mod.DEFAULT_MODELS.get(args.provider)
    engine = llm_mod.build_llm(args.provider, model)
    _STATE["llm"] = engine
    _STATE["ai"] = {"enabled": engine is not None, "provider": args.provider, "model": model}
    _STATE["embedder"] = emb_mod.try_build_embedder(args.embed_model)
    _STATE["embeddings"] = load_embeddings(args.db)

    print(f"Building suggestions from {args.db} …")
    _STATE.update(build_suggestions(args.db))
    _print_startup(args, engine, model)

    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
