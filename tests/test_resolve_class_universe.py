# tests/test_resolve_class_universe.py
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from resolve_class_universe import (
    _fetch_player_info_batched,
    _universe_row,
    audit_multi_validated,
    load_player_directory,
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
    assert resolve_class_player_id("Jack Suwinski", 2019, sleep_s=0.0,
                                   directory={}) == (669261, "ok")


def test_resolve_class_player_id_exact_no_pro_stats(monkeypatch):
    # exact match but every stats group has empty splits
    hydrate = _person_payload(700001, "Test Prospect", [
        _year_group("hitting", []), _year_group("pitching", []),
    ])
    monkeypatch.setattr(
        requests, "get",
        _fake_get([{"id": 700001, "fullName": "Test Prospect"}], {700001: hydrate}),
    )
    assert resolve_class_player_id("Test Prospect", 2024, sleep_s=0.0,
                                   directory={}) == (None, "no_pro_stats")


def test_resolve_class_player_id_two_exact_none_with_stats_ambiguous(monkeypatch):
    people = [{"id": 700002, "fullName": "Twin Guy"}, {"id": 700003, "fullName": "Twin Guy"}]
    hydrates = {
        700002: _person_payload(700002, "Twin Guy", [_year_group("hitting", [])]),
        700003: {"people": [{"id": 700003, "fullName": "Twin Guy"}]},  # stats key absent
    }
    monkeypatch.setattr(requests, "get", _fake_get(people, hydrates))
    mlb_id, status = resolve_class_player_id("Twin Guy", 2020, sleep_s=0.0, directory={})
    assert mlb_id is None
    assert status == ("ambiguous:candidate 700002 without pro stats; "
                      "candidate 700003 without pro stats")


def test_resolve_class_player_id_no_exact_match(monkeypatch):
    monkeypatch.setattr(
        requests, "get", _fake_get([{"id": 700004, "fullName": "Jack Suwinskii"}], {})
    )
    assert resolve_class_player_id("Jack Suwinski", 2019, sleep_s=0.0,
                                   directory={}) == (None, "no_exact_match")


# --- rounds 2-3 (2026-09-18, plan amendments f1f0f63 + edf7cfd): multi-sportId
# hydrate retry, MiLB directory fallback {11..17}, suffix rule, fetch_player_info
# batching. Directory mocks mirror the live probe of
# GET {BASE}/sports/14/players?season=2015:
# {"copyright": ..., "people": [{"id", "fullName", "firstName", "lastName"}]}.


def test_resolve_class_player_id_validated_via_multi_sportid_retry(monkeypatch):
    # Cam Gibson 622057 (controller probe: splits at 12-17, zero at 11 and at
    # no-sportId) — mock is empty at no-sportId/14/13/12, non-empty at 16;
    # iteration must short-circuit at 16 (never query 11/17/15)
    hydrate_calls = []

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/people/search"):
            return _FakeResp({"people": [{"id": 622057, "fullName": "Cam Gibson"}]})
        hydrate_calls.append(params["hydrate"])
        stats = []
        if "sportId=16" in params["hydrate"]:
            stats = [_year_group("hitting", [_split("2016", "GCL Tigers West")])]
        return _FakeResp(_person_payload(622057, "Cam Gibson", stats))

    monkeypatch.setattr(requests, "get", fake_get)
    mlb_id, status = resolve_class_player_id("Cam Gibson", 2015, sleep_s=0.0, directory={})
    assert (mlb_id, status) == (622057, "ok")
    assert hydrate_calls == [
        "stats(group=[hitting,pitching],type=yearByYear)",
        "stats(group=[hitting,pitching],type=yearByYear,sportId=14)",
        "stats(group=[hitting,pitching],type=yearByYear,sportId=13)",
        "stats(group=[hitting,pitching],type=yearByYear,sportId=12)",
        "stats(group=[hitting,pitching],type=yearByYear,sportId=16)",
    ]


def test_load_player_directory_extended_sports_with_empty_cells(monkeypatch):
    saved = []

    def fake_load_latest(dataset, key):
        raise FileNotFoundError(f"no snapshots for {dataset}/{key}")

    def fake_save_raw(dataset, key, payload):
        saved.append(key)

    def fake_get(url, params=None, timeout=None):
        # probe shape: {"copyright": ..., "people": [{"id", "fullName", ...}]}
        sport_id = int(url.rsplit("/sports/", 1)[1].split("/")[0])
        people = [] if sport_id == 15 else [{"id": 900000 + sport_id, "fullName": "Dir Player"}]
        return _FakeResp({"copyright": "c", "people": people})

    monkeypatch.setattr("resolve_class_universe.load_latest", fake_load_latest)
    monkeypatch.setattr("resolve_class_universe.save_raw", fake_save_raw)
    monkeypatch.setattr(requests, "get", fake_get)
    directory = load_player_directory(sleep_s=0.0)
    expected = {(sid, season) for sid in (11, 12, 13, 14, 15, 16, 17)
                for season in range(2014, 2027)}
    assert set(directory) == expected  # 91 cells
    assert len(saved) == 91
    assert directory[(16, 2023)] == [{"id": 900016, "fullName": "Dir Player"}]
    assert directory[(15, 2023)] == []  # empty sport-15 cells tolerated as gaps


def test_resolve_class_player_id_directory_exact_fold_hit(monkeypatch):
    # search indexes nobody; accent-folded exact match in a directory cell
    monkeypatch.setattr(requests, "get", _fake_get([], {}))
    directory = {(14, 2023): [{"id": 683349, "fullName": "Yainer Díaz"}]}
    mlb_id, status = resolve_class_player_id("Yainer Diaz", 2020, sleep_s=0.0,
                                             directory=directory)
    assert (mlb_id, status) == (683349, "ok:directory")


def test_resolve_class_player_id_directory_suffix_hit(monkeypatch):
    # SCP "Jazz Chisholm" vs directory "Jazz Chisholm Jr." -> suffix rule
    monkeypatch.setattr(requests, "get", _fake_get([], {}))
    directory = {(11, 2021): [{"id": 665862, "fullName": "Jazz Chisholm Jr."}]}
    mlb_id, status = resolve_class_player_id("Jazz Chisholm", 2021, sleep_s=0.0,
                                             directory=directory)
    assert (mlb_id, status) == (665862, "ok:directory")


def test_resolve_class_player_id_directory_nearest_season_wins(monkeypatch):
    # hits in two (level, season) cells -> nearest season to class_year wins
    # (distinct ids per cell so the chosen cell is observable)
    monkeypatch.setattr(requests, "get", _fake_get([], {}))
    directory = {
        (11, 2018): [{"id": 800001, "fullName": "Multi Year"}],
        (14, 2021): [{"id": 800002, "fullName": "Multi Year"}],
    }
    mlb_id, status = resolve_class_player_id("Multi Year", 2020, sleep_s=0.0,
                                             directory=directory)
    assert (mlb_id, status) == (800002, "ok:directory")  # 2021 nearer 2020 than 2018


def test_resolve_class_player_id_directory_miss_everywhere(monkeypatch):
    # a non-suffix extra token ("Smith") must NOT satisfy the suffix rule
    monkeypatch.setattr(requests, "get", _fake_get([], {}))
    directory = {(14, 2023): [{"id": 800003, "fullName": "Somebody Else"},
                              {"id": 800004, "fullName": "Jazz Chisholm Smith"}]}
    mlb_id, status = resolve_class_player_id("Jazz Chisholm", 2021, sleep_s=0.0,
                                             directory=directory)
    assert (mlb_id, status) == (None, "no_exact_match")


def test_fetch_player_info_batched_chunks_at_100(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        assert url.endswith("/people")
        ids = [int(i) for i in params["personIds"].split(",")]
        calls.append(len(ids))
        return _FakeResp({"people": [
            {"id": i, "fullName": f"Player {i}", "birthDate": "2000-01-01",
             "primaryPosition": {"abbreviation": "P"}} for i in ids
        ]})

    monkeypatch.setattr(requests, "get", fake_get)
    info = _fetch_player_info_batched(list(range(250)), sleep_s=0.0)
    assert calls == [100, 100, 50]
    assert len(info) == 250


# --- round 4 (2026-09-18, plan amendment 1d1142d): multi-validated =
# ambiguous, resolution_source provenance, bidirectional suffix, NICKNAME_MAP,
# audit_multi_validated.


def test_resolve_class_player_id_two_validated_ambiguous(monkeypatch):
    # two exact candidates BOTH pass validation -> never a silent API-order pick
    people = [{"id": 700010, "fullName": "Twin Star"}, {"id": 700011, "fullName": "Twin Star"}]
    hydrates = {
        700010: _person_payload(700010, "Twin Star",
                                [_year_group("hitting", [_split("2021", "A Team")])]),
        700011: _person_payload(700011, "Twin Star",
                                [_year_group("pitching", [_split("2022", "B Team")])]),
    }
    monkeypatch.setattr(requests, "get", _fake_get(people, hydrates))
    mlb_id, status = resolve_class_player_id("Twin Star", 2021, sleep_s=0.0, directory={})
    assert mlb_id is None
    assert status == "ambiguous:multiple validated (700010, 700011)"


def test_resolve_class_player_id_second_candidate_validates(monkeypatch):
    # first exact candidate fails validation, second validates -> second wins
    people = [{"id": 700012, "fullName": "Twin Star"}, {"id": 700013, "fullName": "Twin Star"}]
    hydrates = {
        700012: _person_payload(700012, "Twin Star", [_year_group("hitting", [])]),
        700013: _person_payload(700013, "Twin Star",
                                [_year_group("hitting", [_split("2021", "A Team")])]),
    }
    monkeypatch.setattr(requests, "get", _fake_get(people, hydrates))
    mlb_id, status = resolve_class_player_id("Twin Star", 2021, sleep_s=0.0, directory={})
    assert (mlb_id, status) == (700013, "ok")


def test_universe_row_resolution_source():
    p = SimpleNamespace(player_name="T Player", class_year=2020, family="chrome",
                        auto_card_url="u/t")
    row = _universe_row(p, 123, "ok")
    assert row["resolution_source"] == "search"
    assert row["player_name"] == "T Player" and row["mlb_id"] == 123
    assert _universe_row(p, 123, "ok:directory")["resolution_source"] == "directory"
    assert _universe_row(p, None, "no_exact_match")["resolution_source"] == ""


def test_resolve_class_player_id_directory_suffix_on_query(monkeypatch):
    # SCP "Jimmy Crooks III" vs directory "Jimmy Crooks" -> query-minus-suffix
    monkeypatch.setattr(requests, "get", _fake_get([], {}))
    directory = {(14, 2022): [{"id": 700030, "fullName": "Jimmy Crooks"}]}
    mlb_id, status = resolve_class_player_id("Jimmy Crooks III", 2022, sleep_s=0.0,
                                             directory=directory)
    assert (mlb_id, status) == (700030, "ok:directory")


def test_resolve_class_player_id_nickname_map_hit(monkeypatch):
    # SCP "Joe Wendle" searches empty; curated map retries "joseph wendle"
    # through the full search+validation path -> "ok"
    searched = []

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/people/search"):
            searched.append(params["names"])
            if params["names"] == "joseph wendle":
                return _FakeResp({"people": [{"id": 621563, "fullName": "Joseph Wendle"}]})
            return _FakeResp({"people": []})
        return _FakeResp(_person_payload(621563, "Joseph Wendle", [
            _year_group("hitting", [_split("2016", "Durham Bulls")]),
        ]))

    monkeypatch.setattr(requests, "get", fake_get)
    mlb_id, status = resolve_class_player_id("Joe Wendle", 2016, sleep_s=0.0, directory={})
    assert (mlb_id, status) == (621563, "ok")
    assert searched == ["Joe Wendle", "joseph wendle"]


def test_resolve_class_player_id_nickname_map_unmapped_unaffected(monkeypatch):
    searched = []

    def fake_get(url, params=None, timeout=None):
        assert url.endswith("/people/search")
        searched.append(params["names"])
        return _FakeResp({"people": []})

    monkeypatch.setattr(requests, "get", fake_get)
    mlb_id, status = resolve_class_player_id("Random Player", 2020, sleep_s=0.0, directory={})
    assert (mlb_id, status) == (None, "no_exact_match")
    assert searched == ["Random Player"]  # no extra mapped search


def test_audit_multi_validated(monkeypatch):
    people_by_name = {
        "Twin Star": [{"id": 700010, "fullName": "Twin Star"},
                      {"id": 700011, "fullName": "Twin Star"}],
        "Solo Guy": [{"id": 700020, "fullName": "Solo Guy"}],
    }
    with_splits = {700010, 700011, 700020}

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/people/search"):
            return _FakeResp({"people": people_by_name[params["names"]]})
        mlb_id = int(url.rsplit("/people/", 1)[1])
        stats = [_year_group("hitting", [_split("2021", "X")])] if mlb_id in with_splits else []
        return _FakeResp(_person_payload(mlb_id, "N", stats))

    monkeypatch.setattr(requests, "get", fake_get)
    players = pd.DataFrame([
        {"player_name": "Twin Star", "mlb_id": 700010},
        {"player_name": "Solo Guy", "mlb_id": 700020},
    ])
    audit = audit_multi_validated(players, sleep_s=0.0)
    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["player_name"] == "Twin Star"
    assert row["current_mlb_id"] == 700010
    assert row["validated_ids"] == "700010,700011"
