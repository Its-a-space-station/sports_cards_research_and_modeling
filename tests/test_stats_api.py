# tests/test_stats_api.py
import pandas as pd
import pytest

from cardprice import stats_api

SAMPLE_RESPONSE = {
    "stats": [
        {
            "splits": [
                {
                    "date": "2022-04-10",
                    "stat": {"gamesPlayed": 1, "atBats": 3, "hits": 2, "homeRuns": 1},
                },
                {
                    "date": "2022-04-08",
                    "stat": {"gamesPlayed": 1, "atBats": 4, "hits": 1, "homeRuns": 0},
                },
            ]
        }
    ]
}


def test_fetch_game_log_builds_url_and_returns_splits(monkeypatch):
    calls = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return SAMPLE_RESPONSE

    def fake_get(url, params=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        return FakeResp()

    monkeypatch.setattr(stats_api.requests, "get", fake_get)
    splits = stats_api.fetch_game_log(592450, "hitting", 2022)
    assert calls["url"] == "https://statsapi.mlb.com/api/v1/people/592450/stats"
    assert calls["params"] == {"stats": "gameLog", "group": "hitting", "season": 2022}
    assert len(splits) == 2


def test_fetch_game_log_empty_when_no_stats(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"stats": []}

    monkeypatch.setattr(stats_api.requests, "get", lambda *a, **k: FakeResp())
    assert stats_api.fetch_game_log(592450, "pitching", 2022) == []


def test_game_log_to_frame_flattens_and_sorts():
    splits = SAMPLE_RESPONSE["stats"][0]["splits"]
    df = stats_api.game_log_to_frame(splits, 592450, "hitting", 2022)
    assert list(df.columns)[:4] == ["mlb_id", "group", "season", "date"]
    assert df["date"].tolist() == [pd.Timestamp("2022-04-08"), pd.Timestamp("2022-04-10")]
    assert df.loc[df["date"] == pd.Timestamp("2022-04-10"), "homeRuns"].iloc[0] == 1
    assert (df["mlb_id"] == 592450).all()


@pytest.mark.live
def test_fetch_game_log_live_judge_2022():
    splits = stats_api.fetch_game_log(592450, "hitting", 2022)
    assert len(splits) == 157  # Judge played 157 games in 2022
