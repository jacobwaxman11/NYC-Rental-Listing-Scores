"""End-to-end render smoke tests via the Flask test client (catches template
and route regressions). Uses a temp DB and a pre-seeded in-memory suggestion set
so the heavy build_suggestions path is skipped."""

import pytest

import web


def _row(lid, name, nh, lat, lng):
    return {"listing_id": lid, "name": name, "neighborhood": nh, "beds": 1.0, "baths": 1.0,
            "sqft": 700, "url": None, "rent": 4000, "predicted": 4500, "pct_diff": -0.1,
            "delta": 500, "apt_quality": 7.0, "photos": 5, "img_pos": 0, "tags": ["bright"],
            "reaction": None, "lat": lat, "lng": lng}


@pytest.fixture(autouse=True)
def _app_state(tmp_path):
    saved = dict(web._STATE)
    web._STATE["db_path"] = str(tmp_path / "w.db")
    web._STATE["suggestions"] = [
        _row("a", "Ref Apt", "Chelsea", 40.7400, -73.9900),
        _row("b", "Near Apt", "Chelsea", 40.7405, -73.9905),
    ]
    web._STATE["meta"] = {"total": 2, "underpriced": 2, "cv": True,
                          "neighborhoods": ["Chelsea", "LES"]}
    web._STATE["embeddings"] = {}
    web._STATE["ai"] = {"enabled": False, "provider": None, "model": None}
    yield
    web._STATE.clear(); web._STATE.update(saved)


def test_index_renders_with_multi_area_filter():
    html = web.app.test_client().get("/").get_data(as_text=True)
    assert "StreetEasier" in html
    assert 'class="cs multi" data-name="nh"' in html          # multi-area filter
    assert "data-apply" in html                                # apply button


def test_area_filter_narrows_results():
    # Only LES selected → neither seeded listing (both Chelsea) shows.
    html = web.app.test_client().get("/?nh=LES&show=all").get_data(as_text=True)
    assert "Ref Apt" not in html and "Near Apt" not in html
    # Chelsea selected → both show.
    html = web.app.test_client().get("/?nh=Chelsea&show=all").get_data(as_text=True)
    assert "Ref Apt" in html and "Near Apt" in html


def test_similar_renders_distance_chip():
    html = web.app.test_client().get("/?similar=a").get_data(as_text=True)
    assert "similar style" in html          # banner
    assert "mi away" in html                # distance chip on the nearby card
