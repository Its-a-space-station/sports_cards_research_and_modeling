# tests/test_expectations.py
"""Marcel goldens + draft-prospects parser goldens (Task 6).

The Pipeline Top-100 parser goldens depend on Task 6's live fixture captures,
which were blocked by source rate-limiting (2026-09-18); they land here when
the parsers join `cardprice.expectations`. The draft-prospects goldens below
run against the trimmed real capture of statsapi `/draft/prospects/2023`
(amendment 2, unblocked).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from cardprice.expectations import marcel_projection, parse_draft_prospects

FIX = Path(__file__).parent / "fixtures" / "expectations"


def load_draft_2023() -> dict:
    return json.loads((FIX / "draft_prospects_2023_trimmed.json").read_text())


def test_parse_draft_prospects_golden_first_rows():
    # hand-verified against the live 2023 payload (captured 2026-09-18):
    # entry order is payload order, NOT rank order. The fixture's 25 entries
    # hold one verbatim duplicated person (Jonah Cox, id 813841, rank 161, at
    # positions 17-18) -> 24 rows after dedupe.
    df = parse_draft_prospects(load_draft_2023(), 2023)
    assert len(df) == 24
    first = df.iloc[0]
    assert first["player_name"] == "Christian Oppor"
    assert first["mlb_id"] == 803291
    assert first["rank"] == 225
    second = df.iloc[1]
    assert second["player_name"] == "Quinn Mathews"
    assert second["mlb_id"] == 687273
    assert second["rank"] == 86
    third = df.iloc[2]
    assert third["player_name"] == "Zach Levenson"
    assert third["mlb_id"] == 804241
    assert third["rank"] == 204
    assert (df["mlb_id"] == 813841).sum() == 1  # duplicate collapsed, first kept


def test_parse_draft_prospects_column_contract():
    df = parse_draft_prospects(load_draft_2023(), 2023)
    assert list(df.columns) == [
        "player_name",
        "mlb_id",
        "season",
        "source",
        "rank",
        "fv",
        "as_of",
    ]
    assert str(df["mlb_id"].dtype) == "Int64"
    assert str(df["season"].dtype) == "int64"
    assert str(df["rank"].dtype) == "Int64"
    assert str(df["fv"].dtype) == "Float64"
    assert pd.api.types.is_datetime64_ns_dtype(df["as_of"])
    assert (df["source"] == "mlb_draft").all()
    assert (df["season"] == 2023).all()
    assert (df["as_of"] == pd.Timestamp("2023-07-01")).all()
    assert df["fv"].isna().all()
    assert df["mlb_id"].notna().all()  # statsapi id is direct, no mapping needed
    # the fixture keeps 3 unranked entries -> rank=NA, never fabricated
    assert df["rank"].isna().sum() == 3
    assert df["rank"].dropna().astype(int).between(1, 250).all()


def test_parse_draft_prospects_skips_malformed_entries():
    payload = {
        "prospects": [
            {"rank": 1},  # no person block
            {"rank": 2, "person": {}},  # person without name or id
            {"rank": 3, "person": None},  # null person
            {"rank": 4, "person": {"id": 700001, "fullName": "Good Entry"}},
            {"person": {"id": 700002}},  # id but no name (kept: id joinable)
            {"person": {"fullName": "Name Only"}},  # name but no id (kept)
        ]
    }
    df = parse_draft_prospects(payload, 2024)
    assert len(df) == 3
    assert df.iloc[0]["mlb_id"] == 700001
    assert df.iloc[0]["player_name"] == "Good Entry"
    assert df.iloc[1]["mlb_id"] == 700002
    assert pd.isna(df.iloc[1]["player_name"])
    assert df.iloc[2]["player_name"] == "Name Only"
    assert pd.isna(df.iloc[2]["mlb_id"])
    assert df["rank"].isna().sum() == 2  # entries 5 and 6 carry no rank
    assert (df["as_of"] == pd.Timestamp("2024-07-01")).all()


def test_parse_draft_prospects_unranked_year_keeps_players():
    # 2015-2016 payloads omit the rank field entirely (live-verified; 2017+
    # carry ranks): players are recorded with rank=NA (coverage + mlb_id
    # join) rather than skipped as a gap
    payload = {"prospects": [{"person": {"id": 1, "fullName": "A"}},
                             {"person": {"id": 2, "fullName": "B"}}]}
    df = parse_draft_prospects(payload, 2015)
    assert len(df) == 2
    assert df["rank"].isna().all()
    assert (df["as_of"] == pd.Timestamp("2015-07-01")).all()


def test_parse_draft_prospects_dedupes_repeat_persons():
    # the payload repeats some persons verbatim (2023: 59 ids twice); first
    # occurrence kept, id-less rows never collapsed into each other
    payload = {
        "prospects": [
            {"rank": 10, "person": {"id": 700010, "fullName": "Dup Entry"}},
            {"rank": 10, "person": {"id": 700010, "fullName": "Dup Entry"}},
            {"person": {"fullName": "No Id One"}},
            {"person": {"fullName": "No Id Two"}},
        ]
    }
    df = parse_draft_prospects(payload, 2023)
    assert len(df) == 3
    assert (df["mlb_id"] == 700010).sum() == 1
    assert df["mlb_id"].isna().sum() == 2


def test_parse_draft_prospects_empty_payload():
    df = parse_draft_prospects({}, 2023)
    assert len(df) == 0
    assert str(df["mlb_id"].dtype) == "Int64"
    assert pd.api.types.is_datetime64_ns_dtype(df["as_of"])


def test_marcel_projection_golden():
    # num = 5*.8*500 + 4*.7*300 + 3*.6*100 = 3020; den = 5*500+4*300+3*100 = 4000
    # proj = (3020 + .72*1200) / (4000+1200) = 3884/5200
    got = marcel_projection([(0.8, 500.0), (0.7, 300.0), (0.6, 100.0)], league_mean=0.72)
    assert got == pytest.approx(3884.0 / 5200.0)


def test_marcel_projection_uses_at_most_three_seasons():
    a = marcel_projection([(0.8, 500.0), (0.7, 300.0), (0.6, 100.0), (0.9, 999.0)], 0.72)
    b = marcel_projection([(0.8, 500.0), (0.7, 300.0), (0.6, 100.0)], 0.72)
    assert a == b


def test_marcel_projection_no_seasons_is_none():
    assert marcel_projection([], 0.72) is None


def test_marcel_projection_zero_pt_season_ignored():
    assert marcel_projection([(0.8, 0.0)], 0.72) is None  # den == 0
