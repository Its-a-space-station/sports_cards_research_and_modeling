# tests/test_career.py
from datetime import date

import pandas as pd

from cardprice.career import (
    awards_to_date,
    career_stage,
    career_to_date,
    minors_pedigree,
    season_year,
)


def make_logs(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


BRYANT_2015 = {  # one game standing in for a season's worth of counting stats
    "mlb_id": 592178,
    "group": "hitting",
    "season": 2015,
    "level": "mlb",
    "gamesPlayed": 151,
    "atBats": 559,
    "hits": 154,
    "doubles": 31,
    "triples": 5,
    "homeRuns": 26,
    "rbi": 99,
    "baseOnBalls": 77,
    "strikeOuts": 199,
    "hitByPitch": 10,
    "sacFlies": 3,
    "stolenBases": 13,
}


def test_season_year_rolls_offseason():
    assert season_year(pd.Timestamp("2023-04-01")) == 2023
    assert season_year(pd.Timestamp("2023-01-01")) == 2023
    assert season_year(pd.Timestamp("2022-12-01")) == 2023


def test_career_to_date_sums_across_seasons():
    logs = make_logs(
        [
            {"date": "2015-04-17", **BRYANT_2015},
            {"date": "2016-04-03", **BRYANT_2015, "season": 2016, "homeRuns": 39},
            {"date": "2017-04-02", **BRYANT_2015, "season": 2017, "homeRuns": 29},
        ]
    )
    out = career_to_date(logs, 592178, date(2017, 12, 31))
    assert out["home_runs"] == 94
    assert out["games"] == 453  # 3 synthetic rows x 151
    # strictly-lagged: through 2016 only sees 2 seasons
    out16 = career_to_date(logs, 592178, date(2016, 12, 31))
    assert out16["home_runs"] == 65


def test_minors_pedigree_levels():
    logs = make_logs(
        [
            {
                "mlb_id": 1,
                "group": "hitting",
                "season": 2017,
                "date": "2017-06-01",
                "level": "a",
                "gamesPlayed": 23,
                "atBats": 80,
                "hits": 20,
                "doubles": 5,
                "triples": 0,
                "homeRuns": 3,
                "rbi": 10,
                "baseOnBalls": 10,
                "strikeOuts": 15,
                "hitByPitch": 1,
                "sacFlies": 1,
                "stolenBases": 2,
            },
            {
                "mlb_id": 1,
                "group": "hitting",
                "season": 2018,
                "date": "2018-05-01",
                "level": "aa",
                "gamesPlayed": 40,
                "atBats": 150,
                "hits": 45,
                "doubles": 10,
                "triples": 1,
                "homeRuns": 8,
                "rbi": 30,
                "baseOnBalls": 20,
                "strikeOuts": 30,
                "hitByPitch": 2,
                "sacFlies": 2,
                "stolenBases": 5,
            },
        ]
    )
    out = minors_pedigree(logs, 1, date(2018, 6, 1))
    assert out["max_level"] == "aa" and out["max_level_rank"] == 3
    assert out["minor_games"] == 63
    assert out["rate_at_max_level"] is not None  # OPS at AA
    empty = minors_pedigree(logs, 999, date(2018, 6, 1))
    assert empty["max_level"] is None and empty["max_level_rank"] == 0
    assert empty["rate_at_max_level"] is None and empty["minor_games"] == 0


def test_career_stage_henderson_shape():
    logs = make_logs(
        [
            {
                "mlb_id": 683002,
                "group": "hitting",
                "season": 2022,
                "date": "2022-08-31",
                "level": "mlb",
            },
            {
                "mlb_id": 683002,
                "group": "hitting",
                "season": 2023,
                "date": "2023-03-30",
                "level": "mlb",
            },
        ]
    )
    assert career_stage(logs, 683002, pd.Timestamp("2022-06-01")) == "prospect"
    assert career_stage(logs, 683002, pd.Timestamp("2022-09-01")) == "rookie_year"
    assert career_stage(logs, 683002, pd.Timestamp("2023-04-01")) == "sophomore"
    assert career_stage(logs, 683002, pd.Timestamp("2025-04-01")) == "established"
    assert (
        career_stage(logs, 683002, pd.Timestamp("2022-12-01")) == "sophomore"
    )  # offseason rolls forward


def test_awards_to_date_counts():
    events = pd.DataFrame(
        [
            {
                "mlb_id": 592450,
                "event_date": pd.Timestamp("2017-11-13"),
                "event_type": "award_win",
                "details": "AL ROY",
            },
            {
                "mlb_id": 592450,
                "event_date": pd.Timestamp("2022-11-16"),
                "event_type": "award_win",
                "details": "AL MVP",
            },
            {
                "mlb_id": 592450,
                "event_date": pd.Timestamp("2022-08-31"),
                "event_type": "debut",
                "details": "x",
            },
        ]
    )
    assert awards_to_date(events, 592450, date(2017, 12, 31)) == 1
    assert awards_to_date(events, 592450, date(2023, 1, 1)) == 2
    assert awards_to_date(events, 592450, date(2016, 1, 1)) == 0
