"""Tests for poll_state — last-run bookkeeping over the db meta store."""

from datetime import datetime, timedelta, timezone

import db as dbm
import poll_state


def test_record_and_last_roundtrip(tmp_path):
    with dbm.open_db(str(tmp_path / "p.db")) as conn:
        assert poll_state.last(conn, "scrape") is None      # never run
        ts = poll_state.record(conn, "scrape")
        assert poll_state.last(conn, "scrape") == ts
        # stages are independent
        assert poll_state.last(conn, "backfill") is None
        poll_state.record(conn, "backfill")
        assert poll_state.last(conn, "backfill") is not None


def test_meta_kv_store(tmp_path):
    with dbm.open_db(str(tmp_path / "m.db")) as conn:
        assert dbm.get_meta(conn, "x") is None
        assert dbm.get_meta(conn, "x", "fallback") == "fallback"
        dbm.set_meta(conn, "x", "1")
        assert dbm.get_meta(conn, "x") == "1"
        dbm.set_meta(conn, "x", "2")                          # upsert
        assert dbm.get_meta(conn, "x") == "2"


def test_ago_formatting():
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc)

    def t(**kw):
        return (now - timedelta(**kw)).isoformat(timespec="seconds")

    assert poll_state.ago(None) == "never"
    assert poll_state.ago(t(seconds=10), now=now) == "just now"
    assert poll_state.ago(t(minutes=20), now=now) == "20m ago"
    assert poll_state.ago(t(hours=6), now=now) == "6h ago"
    assert poll_state.ago(t(days=3), now=now) == "3d ago"
    assert poll_state.ago("not-a-timestamp") == "not-a-timestamp"
