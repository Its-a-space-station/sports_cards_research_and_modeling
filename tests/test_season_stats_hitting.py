# tests/test_season_stats_hitting.py
from datetime import date

import pandas as pd

from cardprice.season_stats import hitting_to_date


def make_log():
    # Three games: Apr 8, Apr 10, Apr 12
    rows = [
        {
            "date": "2022-04-08",
            "gamesPlayed": 1,
            "atBats": 4,
            "hits": 1,
            "doubles": 0,
            "triples": 0,
            "homeRuns": 0,
            "rbi": 0,
            "baseOnBalls": 1,
            "strikeOuts": 2,
            "hitByPitch": 0,
            "sacFlies": 0,
            "stolenBases": 0,
        },
        {
            "date": "2022-04-10",
            "gamesPlayed": 1,
            "atBats": 3,
            "hits": 2,
            "doubles": 1,
            "triples": 0,
            "homeRuns": 1,
            "rbi": 3,
            "baseOnBalls": 0,
            "strikeOuts": 1,
            "hitByPitch": 1,
            "sacFlies": 0,
            "stolenBases": 1,
        },
        {
            "date": "2022-04-12",
            "gamesPlayed": 1,
            "atBats": 5,
            "hits": 3,
            "doubles": 0,
            "triples": 1,
            "homeRuns": 1,
            "rbi": 2,
            "baseOnBalls": 0,
            "strikeOuts": 0,
            "hitByPitch": 0,
            "sacFlies": 1,
            "stolenBases": 0,
        },
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_partial_season_excludes_later_games():
    out = hitting_to_date(make_log(), date(2022, 4, 10))
    assert out["games"] == 2
    assert out["at_bats"] == 7
    assert out["home_runs"] == 1
    assert out["rbi"] == 3
    assert out["stolen_bases"] == 1


def test_full_season_rate_stats():
    out = hitting_to_date(make_log(), date(2022, 4, 12))
    # totals: AB=12, H=6, 2B=1, 3B=1, HR=2, BB=1, HBP=1, SF=1
    assert out["avg"] == round(6 / 12, 3)  # .500
    assert out["obp"] == round(8 / 15, 3)  # (6+1+1)/(12+1+1+1) = .533
    assert out["slg"] == round(15 / 12, 3)  # TB = 6+1+2+6 = 15 -> 1.250
    assert out["ops"] == round(8 / 15 + 15 / 12, 3)  # 1.783


def test_empty_window_returns_zeroes_and_none_rates():
    out = hitting_to_date(make_log(), date(2022, 4, 1))
    assert out["games"] == 0
    assert out["avg"] is None
    assert out["ops"] is None
