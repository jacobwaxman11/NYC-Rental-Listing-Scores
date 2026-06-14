"""Provider-agnostic vision scorers: turn a batch of photos into score dicts.

Two providers with the same output schema, so they can be compared head to head:
Gemini (``google-genai``) and Claude (``anthropic``). The SDKs are imported
lazily, so you only need the one you actually use. ``score_listings.py`` drives
these over the DB; this module is just "photos in, score dicts out".
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Optional

from prompts import SCORE_PROMPT, SYSTEM_PROMPT
from tags import VALID_TAGS


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
