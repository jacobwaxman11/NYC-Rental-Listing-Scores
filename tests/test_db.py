"""Tests for the SQLite layer (db.py) — pure stdlib, no heavy deps."""

import db as dbm


def _seed(conn, ids=("L1", "L2")):
    for lid in ids:
        conn.execute("INSERT INTO listings (listing_id, rent) VALUES (?, ?)", (lid, 4000))


def test_reactions_set_get_update_clear(tmp_path):
    with dbm.open_db(str(tmp_path / "r.db")) as conn:
        _seed(conn)
        dbm.set_reaction(conn, "L1", "liked")
        dbm.set_reaction(conn, "L2", "passed")
        assert dbm.get_reactions(conn) == {"L1": "liked", "L2": "passed"}

        dbm.set_reaction(conn, "L1", "passed")        # update in place
        dbm.set_reaction(conn, "L2", None)            # clear
        assert dbm.get_reactions(conn) == {"L1": "passed"}

        for empty in ("", "none"):
            dbm.set_reaction(conn, "L1", empty)
        assert dbm.get_reactions(conn) == {}


def test_image_tags_rollup_and_replace(tmp_path):
    with dbm.open_db(str(tmp_path / "t.db")) as conn:
        _seed(conn)
        # u2 is shared by both listings.
        dbm.set_listing_images(conn, "L1", ["u1", "u2"], ["/x/u1", "/x/u2"])
        dbm.set_listing_images(conn, "L2", ["u2"], ["/x/u2"])
        dbm.upsert_image_score(conn, "u1", {"image_category": "apartment",
                                            "apartment_scores": {}, "tags": ["hardwood_floors", "bright"]}, "m")
        dbm.upsert_image_score(conn, "u2", {"image_category": "apartment",
                                            "apartment_scores": {}, "tags": ["hardwood_floors", "exposed_brick"]}, "m")

        tags = dbm.get_all_listing_tags(conn)
        # L1 has hardwood on both images → ranked first by count.
        assert tags["L1"][0] == "hardwood_floors"
        assert set(tags["L1"]) == {"hardwood_floors", "bright", "exposed_brick"}
        assert set(tags["L2"]) == {"hardwood_floors", "exposed_brick"}

        # Re-scoring an image replaces its tags.
        dbm.upsert_image_score(conn, "u1", {"image_category": "apartment",
                                            "apartment_scores": {}, "tags": ["duplex"]}, "m")
        tags = dbm.get_all_listing_tags(conn)
        assert "duplex" in tags["L1"] and "bright" not in tags["L1"]
        assert set(tags["L2"]) == {"hardwood_floors", "exposed_brick"}  # untouched


def test_get_all_listing_amenities(tmp_path):
    with dbm.open_db(str(tmp_path / "am.db")) as conn:
        _seed(conn)
        dbm.set_amenities(conn, "L1", ["pool", "washer_dryer", "gym"])
        dbm.set_amenities(conn, "L2", ["dishwasher"])
        allam = dbm.get_all_listing_amenities(conn)
        assert allam["L1"] == ["gym", "pool", "washer_dryer"]   # sorted
        assert allam["L2"] == ["dishwasher"]
        # set_amenities replaces; a listing with none isn't in the dict
        dbm.set_amenities(conn, "L1", [])
        assert "L1" not in dbm.get_all_listing_amenities(conn)


def test_embeddings_roundtrip_and_replace(tmp_path):
    with dbm.open_db(str(tmp_path / "e.db")) as conn:
        _seed(conn, ids=("L1",))
        dbm.upsert_embedding(conn, "L1", "m", [0.1, 0.2, 0.3], "h1")
        assert dbm.get_embedding_hashes(conn) == {"L1": "h1"}
        assert dbm.get_all_embeddings(conn) == {"L1": [0.1, 0.2, 0.3]}

        dbm.upsert_embedding(conn, "L1", "m", [0.4, 0.5], "h2")  # replace, new dim
        assert dbm.get_embedding_hashes(conn) == {"L1": "h2"}
        assert dbm.get_all_embeddings(conn) == {"L1": [0.4, 0.5]}


def test_image_score_to_dict_roundtrip(tmp_path):
    with dbm.open_db(str(tmp_path / "s.db")) as conn:
        _seed(conn, ids=("L1",))
        dbm.set_listing_images(conn, "L1", ["u1", "u2"], [None, None])
        dbm.upsert_image_score(conn, "u1", {
            "image_category": "apartment",
            "apartment_scores": {"room_type": "kitchen", "natural_light": 8,
                                 "finish_quality": 7, "space_feeling": 6,
                                 "view_quality": 9, "condition": 8},
            "common_space_scores": None, "irrelevant_reason": None, "tags": [],
        }, "m")
        dbm.upsert_image_score(conn, "u2", {
            "image_category": "common_space",
            "apartment_scores": None,
            "common_space_scores": {"space_type": "gym", "finish_quality": 8,
                                    "condition": 7, "appeal": 9},
            "irrelevant_reason": None, "tags": [],
        }, "m")

        rows = dbm.get_existing_image_scores(conn, ["u1", "u2"])
        apt = dbm.image_score_to_dict(rows["u1"])
        assert apt["image_category"] == "apartment"
        assert apt["apartment_scores"]["condition"] == 8       # condition_score → condition
        assert apt["common_space_scores"] is None

        com = dbm.image_score_to_dict(rows["u2"])
        assert com["common_space_scores"]["appeal"] == 9
        assert com["apartment_scores"] is None


def test_upsert_listing_scores_persists(tmp_path):
    with dbm.open_db(str(tmp_path / "ls.db")) as conn:
        _seed(conn, ids=("L1",))
        summary = {
            "apartment": {"avg_natural_light": 6.0, "has_good_view": True,
                          "room_type_counts": {"bedroom": 2}},
            "common_space": {"avg_appeal": 9.0, "space_type_counts": {"gym": 1}},
            "photos_total": 5, "photos_apartment": 2, "photos_common": 1,
            "photos_irrelevant": 1, "photos_scored": 4,
        }
        dbm.upsert_listing_scores(conn, "L1", summary)
        row = dict(conn.execute("SELECT * FROM listing_scores WHERE listing_id='L1'").fetchone())
        assert row["avg_natural_light"] == 6.0
        assert row["has_good_view"] == 1           # bool → 0/1
        assert row["photos_total"] == 5
