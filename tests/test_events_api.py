from unittest.mock import MagicMock

import pandas as pd
import pytest

from cardprice import events


def test_fetch_playoff_events_parses(monkeypatch):
    payload = {
        "stats": [
            {
                "splits": [
                    {"date": "2024-10-01", "stat": {"gamesPlayed": 1}},
                    {"date": "2024-10-05", "stat": {"gamesPlayed": 1}},
                ]
            }
        ]
    }

    def fake_get(url, params=None, timeout=None):
        assert params.get("gameType") == "P"
        r = MagicMock()
        r.json.return_value = payload
        r.raise_for_status = lambda: None
        return r

    monkeypatch.setattr(events.requests, "get", fake_get)
    out = events.fetch_playoff_events([683002], [2024], None)
    assert len(out) == 1
    assert out.iloc[0]["event_type"] == "playoff_appearance"
    assert out.iloc[0]["event_date"] == pd.Timestamp("2024-10-01")
    assert "2" in out.iloc[0]["details"]


def test_fetch_playoff_events_empty_when_no_postseason(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        r = MagicMock()
        r.json.return_value = {"stats": []}
        r.raise_for_status = lambda: None
        return r

    monkeypatch.setattr(events.requests, "get", fake_get)
    assert len(events.fetch_playoff_events([683002], [2024], None)) == 0


def test_fetch_award_events_parses_flat_recipients(monkeypatch):
    awards_list = {
        "awards": [
            {"id": "ALROY", "name": "Jackie Robinson AL Rookie of the Year", "league": {"id": 103}},
            {"id": "NLROY", "name": "Jackie Robinson NL Rookie of the Year", "league": {"id": 104}},
            {"id": "ILROY", "name": "INT Rookie of the Year"},  # no AL/NL league: skip
            {"id": "ALCSMVP", "name": "ALCS MVP", "league": {"id": 103}},  # not league MVP
        ]
    }
    recipients = {
        "awards": [
            {
                "id": "ALROY",
                "name": "Jackie Robinson AL Rookie of the Year",
                "date": "2023-11-13",
                "season": "2023",
                "player": {"id": 683002, "nameFirstLast": "Gunnar Henderson"},
            },
            {"id": "ALROY", "season": "2023", "player": {"id": 999999}},  # not our player
        ]
    }

    def fake_get(url, params=None, timeout=None):
        r = MagicMock()
        if url.endswith("/awards"):
            r.json.return_value = awards_list
        else:
            assert "/awards/ALROY/recipients" in url or "/awards/NLROY/recipients" in url
            r.json.return_value = recipients if "ALROY" in url else {"awards": []}
        r.raise_for_status = lambda: None
        return r

    monkeypatch.setattr(events.requests, "get", fake_get)
    out = events.fetch_award_events([683002], [2023])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_type"] == "award_win"
    assert row["event_date"] == pd.Timestamp("2023-11-13")  # real announcement date
    assert "Rookie of the Year" in row["details"]
    assert "approximated" not in row["details"]


def test_fetch_award_events_skips_unannounced_season_404(monkeypatch):
    awards_list = {"awards": [{"id": "ALMVP", "name": "AL MVP", "league": {"id": 103}}]}

    def fake_get(url, params=None, timeout=None):
        r = MagicMock()
        if url.endswith("/awards"):
            r.json.return_value = awards_list
            r.status_code = 200
        else:
            r.status_code = 404  # 2026 awards not yet announced
            r.raise_for_status.side_effect = events.requests.HTTPError("404")
        return r

    monkeypatch.setattr(events.requests, "get", fake_get)
    assert len(events.fetch_award_events([683002], [2026])) == 0


@pytest.mark.live
def test_awards_and_playoffs_live():
    ev = events.fetch_playoff_events([683002], [2024], None)  # Orioles made the 2024 postseason
    assert len(ev) == 1
    awards = events.fetch_award_events([683002], [2023])  # Henderson won 2023 AL ROY
    assert (awards["event_type"] == "award_win").any()
