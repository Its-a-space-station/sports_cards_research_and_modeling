# tests/test_collect_class_stats.py
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from collect_class_stats import collect_class_stats

from cardprice import stats_api, storage


@pytest.fixture
def raw_root(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)
    return tmp_path


def _players():
    return pd.DataFrame(
        {"player_name": ["A", "B"], "mlb_id": [1, 2], "role": ["hitter", "pitcher"]}
    )


def _fake_split(date="2024-05-01"):
    return {"date": date, "stat": {"gamesPlayed": 1, "hits": 2, "atBats": 4}}


def test_snapshot_exists(tmp_path, raw_root):
    assert not storage.snapshot_exists("stats", "k")
    storage.save_raw("stats", "k", {"splits": []})
    assert storage.snapshot_exists("stats", "k")


def test_collect_uses_snapshots_without_fetching(raw_root, monkeypatch):
    storage.save_raw("stats", "1_hitting_2024", {"splits": [_fake_split()]})
    calls = []
    monkeypatch.setattr(
        stats_api, "fetch_game_log",
        lambda *a, **k: calls.append(a) or [_fake_split("2024-06-01")],
    )
    # collect_class_stats imports fetch_game_log lazily via stats_api module:
    monkeypatch.setattr("collect_class_stats.fetch_game_log", stats_api.fetch_game_log)
    out = str(raw_root / "out.parquet")
    df = collect_class_stats(_players().head(1), [2024], [], out, sleep_s=0)
    assert calls == []  # snapshot hit -> zero network
    assert len(df) == 1
    assert df.iloc[0]["level"] == "mlb"
    assert Path(out).exists()


def test_collect_fetches_missing_and_flushes(raw_root, monkeypatch):
    calls = []

    def fake_fetch(mlb_id, group, season, sport_id=None):
        calls.append((mlb_id, group, season, sport_id))
        return [_fake_split()]

    monkeypatch.setattr("collect_class_stats.fetch_game_log", fake_fetch)
    out = str(raw_root / "out.parquet")
    df = collect_class_stats(_players(), [2024], [2024], out, flush_every=1, sleep_s=0)
    # player 1 (hitter): 1 MLB call + 4 minor-level calls; player 2 (pitcher): same
    assert len(calls) == 10
    levels = set(df["level"])
    assert levels == {"mlb", "aaa", "aa", "a_plus", "a"}
    # second run: everything is now snapshotted -> zero fetches, same output
    calls.clear()
    df2 = collect_class_stats(_players(), [2024], [2024], out, flush_every=1, sleep_s=0)
    assert calls == []
    assert len(df2) == len(df)
