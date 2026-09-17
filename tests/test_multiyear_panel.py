# tests/test_multiyear_panel.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_multiyear_panel import build_panel

from cardprice.multiyear import hold_returns, series_month_ends, trailing_features


def tiny_universe():
    chart = pd.DataFrame(
        {
            "card_slug": "set/card",
            "grade": "ungraded",
            "date": pd.date_range("2021-01", periods=30, freq="MS"),
            "price": [10.0 * np.exp(0.01 * i) for i in range(30)],
            "mlb_id": 1,
            "player_name": "Test Player",
            "rookie_year": 2021,
            "card_type": "flagship",
        }
    )
    game_logs = pd.DataFrame(
        [
            {
                "mlb_id": 1,
                "group": "hitting",
                "season": 2021,
                "date": "2021-04-01",
                "level": "mlb",
                "gamesPlayed": 1,
                "atBats": 4,
                "hits": 1,
                "doubles": 0,
                "triples": 0,
                "homeRuns": 0,
                "rbi": 0,
                "baseOnBalls": 0,
                "strikeOuts": 1,
                "hitByPitch": 0,
                "sacFlies": 0,
                "stolenBases": 0,
            },
            {
                "mlb_id": 1,
                "group": "hitting",
                "season": 2021,
                "date": "2021-07-01",
                "level": "mlb",
                "gamesPlayed": 1,
                "atBats": 3,
                "hits": 2,
                "doubles": 1,
                "triples": 0,
                "homeRuns": 1,
                "rbi": 2,
                "baseOnBalls": 1,
                "strikeOuts": 0,
                "hitByPitch": 0,
                "sacFlies": 0,
                "stolenBases": 0,
            },
        ]
    )
    game_logs["date"] = pd.to_datetime(game_logs["date"])
    events = pd.DataFrame(
        [
            {
                "mlb_id": 1,
                "event_date": pd.Timestamp("2021-11-01"),
                "event_type": "award_win",
                "details": "fake award",
            }
        ]
    )
    info = pd.DataFrame(
        [
            {
                "mlb_id": 1,
                "name": "Test Player",
                "birth_date": pd.Timestamp("1998-01-01"),
                "position": "OF",
            }
        ]
    )
    cards = pd.DataFrame(
        [
            {
                "card_slug": "set/card",
                "grade": "ungraded",
                "player_name": "Test Player",
                "mlb_id": 1,
                "rookie_year": 2021,
                "card_type": "flagship",
            }
        ]
    )
    return chart, game_logs, events, info, cards


def test_panel_rows_and_lag():
    chart, game_logs, events, info, _cards = tiny_universe()
    me = series_month_ends(chart)
    panel = build_panel(me, hold_returns(me), trailing_features(me), game_logs, events, info)
    row = panel[panel["entry_month"] == pd.Timestamp("2021-05-01")].iloc[0]
    # lag discipline: entry 2021-05 sees games through 2021-04-30 -> the Apr 1 game counts
    assert row["career_games"] == 1
    assert row["career_stage"] == "rookie_year"
    assert row["awards_to_date"] == 0
    assert row["age"] == round(
        (pd.Timestamp("2021-05-01") - pd.Timestamp("1998-01-01")).days / 365.25, 2
    )
    # pre-debut entry: the Apr 1 game is NOT strictly before entry month 2021-04-01,
    # and lag (Mar 31) sees zero games -> prospect with an empty career line
    pre = panel[panel["entry_month"] == pd.Timestamp("2021-04-01")]
    assert len(pre) == 1  # entry exists: 2021-04 has a month-end price
    assert pre.iloc[0]["career_stage"] == "prospect"
    assert pre.iloc[0]["career_games"] == 0
    assert pre.iloc[0]["age_at_debut"] == pre.iloc[0]["age"]  # pre-debut: age-now semantics
    # outcomes present and consistent with Task 3
    assert "ret_12m" in panel.columns
    # market regime column is the universe median trailing return for that month
    assert "market_ret_3m" in panel.columns


def test_no_lookahead_truncation_probe():
    chart, game_logs, events, info, _cards = tiny_universe()
    me = series_month_ends(chart)
    full = build_panel(me, hold_returns(me), trailing_features(me), game_logs, events, info)
    cut = pd.Timestamp("2021-06-01")
    truncated_logs = game_logs[game_logs["date"] < cut]
    truncated_events = events[events["event_date"] < cut] if len(events) else events
    part = build_panel(
        me, hold_returns(me), trailing_features(me), truncated_logs, truncated_events, info
    )
    pred_cols = [
        "career_games",
        "career_stage",
        "awards_to_date",
        "age",
        "age_at_debut",
        "max_level_rank",
        "minor_games",
        "price_level",
        "ret_3m",
        "market_ret_3m",
    ]
    early_full = full[full["entry_month"] <= cut].sort_values("entry_month")
    early_part = part[part["entry_month"] <= cut].sort_values("entry_month")
    for c in pred_cols:
        a, b = early_full[c].to_numpy(), early_part[c].to_numpy()
        assert len(a) == len(b)
        assert all((x == y) or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b)), c
