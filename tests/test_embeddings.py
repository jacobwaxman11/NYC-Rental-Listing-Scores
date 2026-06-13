"""Tests for embeddings.py text-profile + hashing helpers (no model needed)."""

import embeddings as emb


def test_profile_text_full():
    listing = {"neighborhood": "Chelsea", "beds": 1.0, "baths": 1.0, "sqft": 720,
               "description": "Sun-drenched corner unit with exposed brick."}
    text = emb.profile_text(listing, ["exposed_brick", "hardwood_floors"])
    assert "Neighborhood: Chelsea." in text
    assert "1 bed" in text and "720 sqft" in text
    assert "exposed brick, hardwood floors" in text   # tags humanized
    assert text.endswith("exposed brick.")            # description appended last


def test_profile_text_studio_and_empty():
    assert emb.profile_text({"beds": 0.0}, []) == "0 bed."
    assert emb.profile_text({}, []) == ""             # nothing to embed


def test_text_hash_stable_and_model_sensitive():
    assert emb.text_hash("m", "abc") == emb.text_hash("m", "abc")
    assert emb.text_hash("m", "abc") != emb.text_hash("m", "abd")
    assert emb.text_hash("m", "abc") != emb.text_hash("n", "abc")
