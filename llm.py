"""Thin text-generation wrapper over the same providers the scorer uses.

The web UI's natural-language search uses this to turn a renter's request into a
small structured "query plan" with a SINGLE model call. We never send the full
listing set to the model — only compact facets + an optional reference listing +
a profile of liked listings — so cost is a few hundred tokens per distinct query
regardless of how many listings exist.

Providers mirror score_listings.py: 'gemini' (google-genai) and 'anthropic'
(anthropic SDK). SDKs are imported lazily so you only need the one you use.
"""

from __future__ import annotations

import json
import os
from typing import Optional


# Cheap, capable defaults per provider. Override with --model on the web UI.
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-opus-4-8",
}


def loads_lenient(raw: str):
    """Parse a JSON object out of a model response, tolerating code fences or a
    sentence of surrounding prose (falls back to the first ``{`` … last ``}``)."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


class TextLLM:
    """Base text-completion interface: ``complete(system, user) -> str``."""

    provider: str
    model: str

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:  # pragma: no cover
        raise NotImplementedError


class GeminiTextLLM(TextLLM):
    def __init__(self, model: str):
        from google import genai
        from google.genai import types

        self._types = types
        self.provider = "gemini"
        self.model = model
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        types = self._types
        response = self._client.models.generate_content(
            model=self.model,
            contents=[user],
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                max_output_tokens=max_tokens,
            ),
        )
        return response.text


class ClaudeTextLLM(TextLLM):
    def __init__(self, model: str):
        import anthropic

        self.provider = "anthropic"
        self.model = model
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


def build_llm(provider: str, model: Optional[str] = None) -> Optional[TextLLM]:
    """Construct a text LLM, or return None if the provider's API key is missing
    or its SDK isn't installed (the UI then disables AI search gracefully)."""
    model = model or DEFAULT_MODELS.get(provider)
    try:
        if provider == "gemini":
            if not os.environ.get("GOOGLE_API_KEY"):
                return None
            return GeminiTextLLM(model)
        if provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                return None
            return ClaudeTextLLM(model)
    except Exception:
        return None
    return None
