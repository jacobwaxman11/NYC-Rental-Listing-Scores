"""Tests for the provider-agnostic scorer helpers (scorers.py) — no API calls."""

import scorers as sc
import tags


def test_clean_tags_filters_and_dedupes():
    out = sc._clean_tags(["duplex", "made_up_tag", "bright", "bright"])
    assert out == ["duplex", "bright"]                 # unknown dropped, deduped, order kept
    assert sc._clean_tags(None) == []
    assert sc._clean_tags("not-a-list") == []
    assert "hardwood_floors" in tags.VALID_TAGS


def test_loads_lenient_handles_fences_and_prose():
    assert sc._loads_lenient('[{"a": 1}]') == [{"a": 1}]
    assert sc._loads_lenient('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert sc._loads_lenient('Results: [{"a": 1}, {"b": 2}] done') == [{"a": 1}, {"b": 2}]


def test_coerce_to_list_pads_and_wraps():
    assert sc._coerce_to_list([{"x": 1}], 3) == [{"x": 1}, None, None]
    assert sc._coerce_to_list({"x": 1}, 1) == [{"x": 1}]          # lone object → list
    assert sc._coerce_to_list([1, 2, 3], 2) == [1, 2]            # truncate


def test_provider_defaults():
    assert sc.PROVIDER_DEFAULT_MODELS["gemini"].startswith("gemini")
    assert sc.PROVIDER_DEFAULT_MODELS["anthropic"].startswith("claude")
