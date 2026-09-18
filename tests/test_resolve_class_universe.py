# tests/test_resolve_class_universe.py
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from resolve_class_universe import (
    players_from_checklists,
    resolve_base_cards,
    resolve_class_player_id,
)


def _checklists():
    return pd.DataFrame(
        [
            # Witt: Draft only (2019); Henderson: Draft (2019); Bellinger: Chrome (2015)
            {"year": 2019, "family": "draft", "name": "Bobby Witt Jr.", "number": "CDA-BW",
             "product_id": 1, "card_url": "u/draft-witt"},
            {"year": 2019, "family": "draft", "name": "Gunnar Henderson", "number": "CDA-GH",
             "product_id": 2, "card_url": "u/draft-gunnar"},
            {"year": 2015, "family": "chrome", "name": "Cody Bellinger", "number": "BCAP-CBE",
             "product_id": 3, "card_url": "u/chrome-bell"},
            # same player in both families, different years -> earliest year wins
            {"year": 2017, "family": "chrome", "name": "Test Player", "number": "CPA-TP",
             "product_id": 4, "card_url": "u/chrome-tp"},
            {"year": 2016, "family": "draft", "name": "Test Player", "number": "CDA-TP",
             "product_id": 5, "card_url": "u/draft-tp"},
        ]
    )


def test_players_from_checklists_unique_earliest_family():
    players = players_from_checklists(_checklists()).set_index("player_name")
    assert len(players) == 4
    assert players.loc["Cody Bellinger", "class_year"] == 2015
    assert players.loc["Cody Bellinger", "family"] == "chrome"
    assert players.loc["Bobby Witt Jr.", "class_year"] == 2019
    assert players.loc["Bobby Witt Jr.", "family"] == "draft"
    # earliest year across families: 2016 draft (not 2017 chrome)
    assert players.loc["Test Player", "class_year"] == 2016
    assert players.loc["Test Player", "family"] == "draft"
    assert players.loc["Test Player", "auto_card_url"] == "u/draft-tp"


def test_resolve_base_cards_earliest_year_name_exact():
    base = {
        2019: pd.DataFrame(
            [
                {"product_id": 9, "name": "Fernando Tatis Jr.", "number": "BCP-25",
                 "card_url": "u/tatis-base", "raw_title": "t"},
                {"product_id": 10, "name": "Gunnar Henderson", "number": "BDC-99",
                 "card_url": "u/gunnar-base-19", "raw_title": "t"},
            ]
        ),
        2020: pd.DataFrame(
            [
                {"product_id": 11, "name": "Gunnar Henderson", "number": "BCP-5",
                 "card_url": "u/gunnar-base-20", "raw_title": "t"},
            ]
        ),
    }
    players = players_from_checklists(_checklists())
    resolved, unresolved = resolve_base_cards(players, base)
    gunnar = resolved[resolved["player_name"] == "Gunnar Henderson"].iloc[0]
    assert gunnar["base_year"] == 2019  # earliest, not 2020
    assert gunnar["scp_url"] == "u/gunnar-base-19"
    # Witt (class 2019) and Bellinger (class 2015) have no name-matched base rows
    assert set(unresolved["player_name"]) == {"Bobby Witt Jr.", "Cody Bellinger", "Test Player"}


def test_resolve_base_cards_rejects_near_names():
    # "Bobby Witt Jr" must NOT match an anchor named "Bobby Witt" (suffix-safe, whole-word)
    base = {2019: pd.DataFrame(
        [{"product_id": 1, "name": "Bobby Witt", "number": "BDC-1",
          "card_url": "u/witt-sr", "raw_title": "t"}]
    )}
    players = pd.DataFrame(
        [{"player_name": "Bobby Witt Jr.", "class_year": 2019, "family": "draft",
          "auto_card_url": "u"}]
    )
    resolved, unresolved = resolve_base_cards(players, base)
    assert len(resolved) == 0 and len(unresolved) == 1


# --- resolve_class_player_id (offline; requests.get monkeypatched) ---
# Mocked JSON mirrors the 2026-09-18 live probe of
# GET {BASE}/people/{id}?hydrate=stats(group=[hitting,pitching],type=yearByYear):
# people[0].stats is a list of {"type","group","splits"} groups; statsapi omits
# groups with no splits entirely (Suwinski probe returned hitting only).


class _FakeResp:
    """Minimal requests.Response stand-in returning a canned JSON payload."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _split(season, team):
    return {"season": season, "sport": {"abbreviation": "MLB"},
            "team": {"name": team}, "stat": {"gamesPlayed": 10}}


def _year_group(group, splits):
    return {"type": {"displayName": "yearByYear"},
            "group": {"displayName": group}, "splits": splits}


def _person_payload(mlb_id, name, stats):
    person = {"id": mlb_id, "fullName": name}
    if stats is not None:
        person["stats"] = stats
    return {"people": [person]}


def _fake_get(search_people, hydrate_payloads):
    def fake_get(url, params=None, timeout=None):
        if url.endswith("/people/search"):
            return _FakeResp({"people": search_people})
        mlb_id = int(url.rsplit("/people/", 1)[1])
        return _FakeResp(hydrate_payloads[mlb_id])

    return fake_get


def test_resolve_class_player_id_exact_with_pro_stats(monkeypatch):
    # probe id 669261 (Jack Suwinski): single hitting yearByYear group, MLB splits
    hydrate = _person_payload(669261, "Jack Suwinski", [
        _year_group("hitting", [_split("2022", "Pittsburgh Pirates"),
                                _split("2023", "Pittsburgh Pirates")]),
    ])
    monkeypatch.setattr(
        requests, "get",
        _fake_get([{"id": 669261, "fullName": "Jack Suwinski"}], {669261: hydrate}),
    )
    assert resolve_class_player_id("Jack Suwinski", 2019, sleep_s=0.0) == (669261, "ok")


def test_resolve_class_player_id_exact_no_pro_stats(monkeypatch):
    # exact match but every stats group has empty splits
    hydrate = _person_payload(700001, "Test Prospect", [
        _year_group("hitting", []), _year_group("pitching", []),
    ])
    monkeypatch.setattr(
        requests, "get",
        _fake_get([{"id": 700001, "fullName": "Test Prospect"}], {700001: hydrate}),
    )
    assert resolve_class_player_id("Test Prospect", 2024, sleep_s=0.0) == (None, "no_pro_stats")


def test_resolve_class_player_id_two_exact_none_with_stats_ambiguous(monkeypatch):
    people = [{"id": 700002, "fullName": "Twin Guy"}, {"id": 700003, "fullName": "Twin Guy"}]
    hydrates = {
        700002: _person_payload(700002, "Twin Guy", [_year_group("hitting", [])]),
        700003: {"people": [{"id": 700003, "fullName": "Twin Guy"}]},  # stats key absent
    }
    monkeypatch.setattr(requests, "get", _fake_get(people, hydrates))
    mlb_id, status = resolve_class_player_id("Twin Guy", 2020, sleep_s=0.0)
    assert mlb_id is None
    assert status == ("ambiguous:candidate 700002 without pro stats; "
                      "candidate 700003 without pro stats")


def test_resolve_class_player_id_no_exact_match(monkeypatch):
    monkeypatch.setattr(
        requests, "get", _fake_get([{"id": 700004, "fullName": "Jack Suwinskii"}], {})
    )
    assert resolve_class_player_id("Jack Suwinski", 2019, sleep_s=0.0) == (None, "no_exact_match")
