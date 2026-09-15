# tests/test_collect_stats.py
import pandas as pd
import pytest

from cardprice import collect_stats, stats_api, storage

SPLITS = [
    {"date": "2022-04-08", "stat": {"gamesPlayed": 1, "atBats": 4, "hits": 1}},
    {"date": "2022-04-10", "stat": {"gamesPlayed": 1, "atBats": 3, "hits": 2}},
]


@pytest.fixture
def players():
    return pd.DataFrame(
        [
            {"mlb_id": 1, "name": "Test Hitter", "role": "hitter"},
            {"mlb_id": 2, "name": "Test Pitcher", "role": "pitcher"},
        ]
    )


def test_collect_fetches_role_group_and_saves_raw(players, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)
    calls = []

    def fake_fetch(mlb_id, group, season):
        calls.append((mlb_id, group, season))
        return SPLITS

    monkeypatch.setattr(stats_api, "fetch_game_log", fake_fetch)
    monkeypatch.setattr(collect_stats, "fetch_game_log", fake_fetch)

    df = collect_stats.collect(players, [2022], sleep_s=0)
    assert (1, "hitting", 2022) in calls
    assert (2, "pitching", 2022) in calls
    assert len(calls) == 2  # role gates the group; no wasted calls
    assert len(df) == 4  # 2 players x 2 games
    assert storage.load_latest("stats", "1_hitting_2022") == {"splits": SPLITS}


def test_collect_handles_empty_log(players, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(collect_stats, "fetch_game_log", lambda *a: [])
    df = collect_stats.collect(players, [2022], sleep_s=0)
    assert len(df) == 0
    assert storage.load_latest("stats", "1_hitting_2022") == {"splits": []}
