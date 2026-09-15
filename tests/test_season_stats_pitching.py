# tests/test_season_stats_pitching.py
from datetime import date

import pandas as pd
import pytest

from cardprice.season_stats import innings_to_outs, pitching_to_date


@pytest.mark.parametrize(
    "ip,outs",
    [("6.0", 18), ("6.1", 19), ("6.2", 20), ("7", 21), (0.1, 1), ("0.2", 2)],
)
def test_innings_to_outs(ip, outs):
    assert innings_to_outs(ip) == outs


def make_log():
    rows = [
        {"date": "2018-04-02", "gamesPlayed": 1, "gamesStarted": 1, "wins": 1, "losses": 0,
         "inningsPitched": "7.0", "hits": 4, "earnedRuns": 1, "baseOnBalls": 1,
         "strikeOuts": 11, "battersFaced": 25},
        {"date": "2018-04-08", "gamesPlayed": 1, "gamesStarted": 1, "wins": 0, "losses": 1,
         "inningsPitched": "5.2", "hits": 6, "earnedRuns": 3, "baseOnBalls": 2,
         "strikeOuts": 7, "battersFaced": 23},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_pitching_full_window():
    out = pitching_to_date(make_log(), date(2018, 4, 30))
    assert out["games"] == 2
    assert out["wins"] == 1 and out["losses"] == 1
    assert out["outs"] == 21 + 17          # 7.0 IP + 5.2 IP = 38 outs
    assert out["innings_pitched"] == round(38 / 3, 3)
    assert out["era"] == round(9 * 4 / (38 / 3), 3)     # 2.842
    assert out["whip"] == round(13 / (38 / 3), 3)       # (2+4+6)/IP = 1.026
    assert out["k_bb_pct"] == round((18 - 3) / 48, 3)   # .313


def test_pitching_empty_window():
    out = pitching_to_date(make_log(), date(2018, 3, 1))
    assert out["games"] == 0
    assert out["era"] is None
