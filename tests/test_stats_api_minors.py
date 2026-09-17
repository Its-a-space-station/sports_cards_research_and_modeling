# tests/test_stats_api_minors.py
import pytest

from cardprice import stats_api


def test_fetch_game_log_sport_id_param(monkeypatch):
    calls = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"stats": [{"splits": [{"date": "2011-07-01", "stat": {"gamesPlayed": 1}}]}]}

    def fake_get(url, params=None, timeout=None):
        calls["params"] = params
        return FakeResp()

    monkeypatch.setattr(stats_api.requests, "get", fake_get)
    out = stats_api.fetch_game_log(545361, "hitting", 2011, sport_id=12)
    assert calls["params"]["sportId"] == 12
    assert len(out) == 1
    # None = MLB, param absent entirely (backwards compatible)
    stats_api.fetch_game_log(545361, "hitting", 2016)
    assert "sportId" not in calls["params"]


def test_fetch_minor_league_logs_omits_empty_levels(monkeypatch):
    responses = {11: [], 12: [{"date": "2011-07-01", "stat": {}}], 13: [], 14: []}

    def fake_fetch(mlb_id, group, season, sport_id=None):
        return responses[sport_id]

    monkeypatch.setattr(stats_api, "fetch_game_log", fake_fetch)
    out = stats_api.fetch_minor_league_logs(545361, "hitting", 2011)
    assert list(out) == ["aa"]


@pytest.mark.live
def test_minors_live_trout_2011_aa():
    splits = stats_api.fetch_game_log(545361, "hitting", 2011, sport_id=12)
    assert len(splits) == 91  # Trout's 2011 Arkansas (AA) season


@pytest.mark.live
def test_minors_live_henderson_2022_aaa():
    splits = stats_api.fetch_game_log(683002, "hitting", 2022, sport_id=11)
    assert len(splits) == 65  # Henderson's 2022 Norfolk (AAA) stint
