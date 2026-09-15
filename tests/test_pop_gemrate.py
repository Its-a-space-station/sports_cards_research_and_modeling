# tests/test_pop_gemrate.py
import json
from pathlib import Path

import pandas as pd
import pytest

from cardprice import pop_gemrate, storage

FIXTURES = Path(__file__).parent / "fixtures" / "gemrate"


@pytest.fixture
def search_payload():
    return json.loads((FIXTURES / "search_henderson.json").read_text())


@pytest.fixture
def details_payload():
    return json.loads((FIXTURES / "card_details_henderson.json").read_text())


def test_set_query_from_slug():
    assert pop_gemrate.set_query_from_slug("baseball-cards-2023-topps-chrome") == (
        "2023",
        "Topps Chrome",
    )
    assert pop_gemrate.set_query_from_slug("baseball-cards-2024-topps-chrome-update") == (
        "2024",
        "Topps Chrome Update",
    )


def test_card_number_from_scp_url():
    base = "https://www.sportscardspro.com/game"
    assert (
        pop_gemrate.card_number_from_scp_url(
            f"{base}/baseball-cards-2023-topps-chrome/gunnar-henderson-2"
        )
        == "2"
    )
    assert (
        pop_gemrate.card_number_from_scp_url(
            f"{base}/baseball-cards-2024-topps-chrome-update/paul-skenes-usc88"
        )
        == "USC88"
    )


def test_build_query():
    q = pop_gemrate.build_query(
        "Gunnar Henderson",
        "baseball-cards-2023-topps-chrome",
        "https://www.sportscardspro.com/game/baseball-cards-2023-topps-chrome/gunnar-henderson-2",
    )
    assert q == "Gunnar Henderson 2023 Topps Chrome #2"


def test_pick_search_result_base_not_parallel(search_payload):
    hit = pop_gemrate.pick_search_result(
        search_payload["results"],
        name="Gunnar Henderson",
        year="2023",
        set_name="Topps Chrome",
        card_number="2",
    )
    assert hit is not None
    assert hit["gemrate_id"] == "7fbc276ca4b0956fbfd703bcc2cb2b29b63b432a"
    assert hit["parsed_description"]["parallel"] == "Base"


def test_pick_search_result_requires_year_and_number(search_payload):
    assert (
        pop_gemrate.pick_search_result(
            search_payload["results"],
            name="Gunnar Henderson",
            year="2024",
            set_name="Topps Chrome",
            card_number="2",
        )
        is None
    )
    assert (
        pop_gemrate.pick_search_result(
            search_payload["results"],
            name="Gunnar Henderson",
            year="2023",
            set_name="Topps Chrome",
            card_number="95",
        )
        is None
    )


def test_pick_search_result_tolerates_name_punctuation():
    results = [
        {
            "gemrate_id": "abc123",
            "is_universal_match": True,
            "total_population": 100,
            "parsed_description": {
                "year": "2022",
                "set_name": "Topps Chrome",
                "name": "Bobby Witt Jr.",
                "card_number": "221",
                "parallel": "Base",
            },
        }
    ]
    hit = pop_gemrate.pick_search_result(
        results, name="Bobby Witt Jr", year="2022", set_name="Topps Chrome", card_number="221"
    )
    assert hit is not None and hit["gemrate_id"] == "abc123"


def _sp_result(gemrate_id, parallel, pop):
    return {
        "gemrate_id": gemrate_id,
        "is_universal_match": True,
        "total_population": pop,
        "parsed_description": {
            "year": "2022",
            "set_name": "Topps Chrome",
            "name": "Julio Rodriguez",
            "card_number": "222",
            "parallel": parallel,
        },
    }


def test_pick_search_result_sp_fallback_when_no_base():
    # 2022 Topps Chrome #221/#222 exist only as SP entries in GemRate's taxonomy
    results = [_sp_result("sp1", "SP", 553), _sp_result("par1", "SP-Red Refractor", 3)]
    hit = pop_gemrate.pick_search_result(
        results, name="Julio Rodriguez", year="2022", set_name="Topps Chrome", card_number="222"
    )
    assert hit is not None and hit["gemrate_id"] == "sp1"


def test_pick_search_result_prefers_base_over_sp():
    results = [_sp_result("sp1", "SP", 553), _sp_result("base1", "Base", 10)]
    hit = pop_gemrate.pick_search_result(
        results, name="Julio Rodriguez", year="2022", set_name="Topps Chrome", card_number="222"
    )
    assert hit is not None and hit["gemrate_id"] == "base1"


def test_pick_search_result_digits_fallback_for_usc_prefix():
    # SCP slug says 'usc178'; GemRate's true base entry is 'Base 178'. The
    # digits-normalized strict-set match must beat an exact-number match in a
    # lookalike refractor-line set ('Topps Chrome Update Refractors').
    true_base = {
        "gemrate_id": "base178",
        "is_universal_match": True,
        "total_population": 629,
        "parsed_description": {
            "year": "2025",
            "set_name": "Topps Chrome Update",
            "name": "Nick Kurtz",
            "card_number": "178",
            "parallel": "Base",
        },
    }
    refractor_line = {
        "gemrate_id": "refr5",
        "is_universal_match": False,
        "total_population": 5,
        "parsed_description": {
            "year": "2025",
            "set_name": "Topps Chrome Update Refractors",
            "name": "Nick Kurtz",
            "card_number": "USC178",
            "parallel": None,
        },
    }
    hit = pop_gemrate.pick_search_result(
        [refractor_line, true_base],
        name="Nick Kurtz",
        year="2025",
        set_name="Topps Chrome Update",
        card_number="USC178",
    )
    assert hit is not None and hit["gemrate_id"] == "base178"


def test_pick_search_result_exact_number_beats_digits_fallback():
    base88 = _sp_result("exact", "Base", 100)
    base88["parsed_description"]["set_name"] = "Topps Chrome Update"
    base88["parsed_description"]["name"] = "Paul Skenes"
    base88["parsed_description"]["card_number"] = "USC88"
    other88 = _sp_result("digits", "Base", 999)
    other88["parsed_description"]["set_name"] = "Topps Chrome Update"
    other88["parsed_description"]["name"] = "Paul Skenes"
    other88["parsed_description"]["card_number"] = "88"
    hit = pop_gemrate.pick_search_result(
        [other88, base88],
        name="Paul Skenes",
        year="2022",
        set_name="Topps Chrome Update",
        card_number="USC88",
    )
    assert hit is not None and hit["gemrate_id"] == "exact"


def test_search_all_pages_paginates_until_match_possible():
    class FakePage:
        def __init__(self, payloads):
            self.payloads = payloads
            self.requested = []

        def evaluate(self, _js, args):
            self.requested.append(args[1])
            return self.payloads.get(args[1])

    page = FakePage(
        {
            2: {"results": [{"id": "p2"}], "has_more": True},
            3: {"results": [{"id": "p3"}], "has_more": True},
        }
    )
    first = {"results": [{"id": "p1"}], "has_more": True}
    results = pop_gemrate._search_all_pages(page, "q", first, max_pages=3, sleep_s=0)
    assert [r["id"] for r in results] == ["p1", "p2", "p3"]
    assert page.requested == [2, 3]
    # stops early when has_more is false
    page2 = FakePage({2: {"results": [{"id": "p2"}], "has_more": False}})
    results = pop_gemrate._search_all_pages(page2, "q", first, max_pages=3, sleep_s=0)
    assert [r["id"] for r in results] == ["p1", "p2"]
    assert page2.requested == [2]


def test_parse_card_details_henderson(details_payload):
    pop = pop_gemrate.parse_card_details(details_payload)
    assert pop["psa_10_pop"] == 2070
    assert pop["psa_total_pop"] == 3129
    assert pop["total_pop"] == 4798
    assert pop["gem_rate"] == pytest.approx(0.6244, abs=1e-4)
    assert pop["description"] == "2023 Topps Chrome Gunnar Henderson Base 2"
    assert pop["data_last_updated"] == "2026-09-14"


def test_parse_card_details_without_psa_row():
    payload = {
        "total_population": 50,
        "total_gems_or_greater": 30,
        "description": "X",
        "gemrate_id": "deadbeef",
        "population_data": [{"grader": "sgc", "card_total_grades": 50, "grades": {"sgc_10": 30}}],
    }
    pop = pop_gemrate.parse_card_details(payload)
    assert pop["psa_10_pop"] is None
    assert pop["psa_total_pop"] is None
    assert pop["total_pop"] == 50
    assert pop["gem_rate"] == pytest.approx(0.6)


def test_collect_pops_maps_slugs_and_skips_unresolved(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)
    cards = pd.DataFrame(
        [
            {
                "player_name": "Gunnar Henderson",
                "rookie_year": 2023,
                "set_slug": "baseball-cards-2023-topps-chrome",
                "scp_url": "https://www.sportscardspro.com/game/baseball-cards-2023-topps-chrome/gunnar-henderson-2",
            },
            {
                "player_name": "Unresolved Player",
                "rookie_year": 2025,
                "set_slug": "baseball-cards-2025-topps-chrome",
                "scp_url": "",
            },
        ]
    )
    canned = {"psa_10_pop": 2070, "total_pop": 4798, "gem_rate": 0.6244}
    monkeypatch.setattr(pop_gemrate, "_new_page", lambda: (None, None, None))
    monkeypatch.setattr(pop_gemrate, "fetch_pop", lambda query, match=None, page=None: canned)
    df = pop_gemrate.collect_pops(cards, sleep_s=0, on=pd.Timestamp("2026-09-15").date())
    assert list(df.columns) == ["date", "card_slug", "psa_10_pop", "total_pop", "gem_rate"]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["card_slug"] == "baseball-cards-2023-topps-chrome/gunnar-henderson-2"
    assert row["psa_10_pop"] == 2070
    assert row["date"] == "2026-09-15"


def test_collect_pops_records_lookup_failures_as_nan(monkeypatch):
    cards = pd.DataFrame(
        [
            {
                "player_name": "Ghost Player",
                "rookie_year": 2025,
                "set_slug": "baseball-cards-2025-topps-chrome",
                "scp_url": "https://www.sportscardspro.com/game/baseball-cards-2025-topps-chrome/ghost-player-1",
            }
        ]
    )

    def not_found(query, match=None, page=None):
        raise pop_gemrate.PopLookupError(query)

    monkeypatch.setattr(pop_gemrate, "_new_page", lambda: (None, None, None))
    monkeypatch.setattr(pop_gemrate, "fetch_pop", not_found)
    df = pop_gemrate.collect_pops(cards, sleep_s=0, on=pd.Timestamp("2026-09-15").date())
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["psa_10_pop"])
    assert pd.isna(df.iloc[0]["total_pop"])


def _gr_result(gemrate_id, set_name, parallel, card_number="USC178"):
    return {
        "gemrate_id": gemrate_id,
        "is_universal_match": True,
        "total_population": 100,
        "parsed_description": {
            "year": "2025",
            "set_name": set_name,
            "name": "Nick Kurtz",
            "card_number": card_number,
            "parallel": parallel,
        },
    }


def test_pick_search_result_loose_set_rejects_product_line_tokens():
    # Only loose (token-subset) set matches are possible here; candidates whose
    # set adds a product-line token must be rejected, not silently taken as base.
    refractors = _gr_result("r1", "Topps Chrome Update Refractors", None)
    logofractor = _gr_result("l1", "Topps Chrome Logofractor Edition", None)
    cosmic = _gr_result("c1", "Topps Cosmic Chrome", None, card_number="178")
    for results in ([refractors], [logofractor], [cosmic]):
        assert (
            pop_gemrate.pick_search_result(
                results,
                name="Nick Kurtz",
                year="2025",
                set_name="Topps Chrome Update",
                card_number="USC178",
            )
            is None
        )


def test_pick_search_result_loose_set_accepts_benign_set_drift():
    # Legit loose match: same product line, set name merely drifts ('... Series').
    hit = pop_gemrate.pick_search_result(
        [_gr_result("ok1", "Topps Chrome Update Series", "Base")],
        name="Nick Kurtz",
        year="2025",
        set_name="Topps Chrome Update",
        card_number="USC178",
    )
    assert hit is not None and hit["gemrate_id"] == "ok1"


def test_parse_card_details_missing_psa10_key_stays_null():
    payload = {
        "total_population": 100,
        "total_gems_or_greater": 60,
        "population_data": [{"grader": "psa", "card_total_grades": 100, "grades": {"psa_9": 40}}],
    }
    pop = pop_gemrate.parse_card_details(payload)
    assert pop["psa_10_pop"] is None  # absent key is unknown, not zero
    assert pop["psa_total_pop"] == 100
    payload["population_data"][0]["grades"]["psa_10"] = 0
    assert pop_gemrate.parse_card_details(payload)["psa_10_pop"] == 0  # real zero kept


def test_main_failed_rerun_does_not_clobber_good_row(monkeypatch, tmp_path):
    cards_csv = tmp_path / "cards.csv"
    cards_csv.write_text("player_name,role,rookie_year,set_slug,mlb_id,scp_url\n")
    out_csv = tmp_path / "pop.csv"
    out_csv.write_text(
        "date,card_slug,psa_10_pop,total_pop,gem_rate\n"
        "2026-09-15,set/a-1,10,100,0.5\n"
        "2026-09-15,set/b-2,20,200,0.6\n"
    )
    new = pd.DataFrame(
        [
            {
                "date": "2026-09-15",
                "card_slug": "set/a-1",
                "psa_10_pop": pd.NA,
                "total_pop": pd.NA,
                "gem_rate": pd.NA,
            },
            {
                "date": "2026-09-15",
                "card_slug": "set/b-2",
                "psa_10_pop": 21,
                "total_pop": 201,
                "gem_rate": 0.61,
            },
        ]
    ).astype({"psa_10_pop": "Int64", "total_pop": "Int64"})
    monkeypatch.setattr(pop_gemrate, "collect_pops", lambda cards, sleep_s: new)
    monkeypatch.setattr(
        "sys.argv", ["pop_gemrate", "--cards", str(cards_csv), "--out", str(out_csv)]
    )
    pop_gemrate.main()
    df = pd.read_csv(out_csv)
    assert len(df) == 2  # no duplicate (date, card_slug) rows
    assert df[df["card_slug"] == "set/a-1"].iloc[0]["psa_10_pop"] == 10  # good row kept
    assert df[df["card_slug"] == "set/b-2"].iloc[0]["psa_10_pop"] == 21  # good rerun replaces


@pytest.mark.live
def test_fetch_pop_henderson_live():
    pop = pop_gemrate.fetch_pop(
        "Gunnar Henderson 2023 Topps Chrome #2",
        match={
            "name": "Gunnar Henderson",
            "year": "2023",
            "set_name": "Topps Chrome",
            "card_number": "2",
        },
    )
    assert pop["psa_10_pop"] > 0
    assert pop["total_pop"] >= pop["psa_10_pop"]
    assert 0 < pop["gem_rate"] < 1
