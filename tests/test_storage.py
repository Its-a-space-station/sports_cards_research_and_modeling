import json
from datetime import date

import pytest

from cardprice import storage


@pytest.fixture
def raw_root(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)
    return tmp_path


def test_save_then_load_latest_roundtrip(raw_root):
    storage.save_raw("stats", "592450_hitting_2022", {"splits": [1, 2]}, on=date(2026, 9, 14))
    assert storage.load_latest("stats", "592450_hitting_2022") == {"splits": [1, 2]}


def test_save_is_idempotent_same_day(raw_root):
    p1 = storage.save_raw("stats", "k", {"a": 1}, on=date(2026, 9, 14))
    mtime = p1.stat().st_mtime_ns
    p2 = storage.save_raw("stats", "k", {"a": 1}, on=date(2026, 9, 14))
    assert p1 == p2 and p2.stat().st_mtime_ns == mtime  # not rewritten


def test_save_new_date_does_not_touch_old(raw_root):
    storage.save_raw("stats", "k", {"v": 1}, on=date(2026, 9, 13))
    storage.save_raw("stats", "k", {"v": 2}, on=date(2026, 9, 14))
    old = raw_root / "stats" / "k" / "2026-09-13.json"
    assert json.loads(old.read_text()) == {"v": 1}
    assert storage.load_latest("stats", "k") == {"v": 2}


def test_load_latest_missing_raises(raw_root):
    with pytest.raises(FileNotFoundError):
        storage.load_latest("stats", "nope")
