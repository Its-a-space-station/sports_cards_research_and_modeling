# tests/test_build_class_panel.py
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_class_panel import build_panel, league_means, pace_line, season_line


def _hitting_log(date, h=2, bb=1, hbp=0, ab=4, sf=0, tb=3, pa=5, season=2024):
    return {"mlb_id": 1, "group": "hitting", "season": season, "date": pd.Timestamp(date),
            "level": "mlb", "hits": h, "baseOnBalls": bb, "hitByPitch": hbp, "atBats": ab,
            "sacFlies": sf, "totalBases": tb, "plateAppearances": pa}


def test_season_line_hitting_ops_from_components():
    logs = pd.DataFrame([_hitting_log("2024-05-01"), _hitting_log("2024-05-02")])
    rate, pt = season_line(logs, "hitting")
    # obp = (4+2+0)/(8+2+0+0) = .6 ; slg = 6/8 = .75 ; ops = 1.35 ; pt = 10
    assert rate == pytest.approx(0.6 + 0.75)
    assert pt == 10.0


def test_season_line_pitching_era():
    logs = pd.DataFrame(
        [{"mlb_id": 2, "group": "pitching", "season": 2024, "date": pd.Timestamp("2024-05-01"),
          "level": "mlb", "earnedRuns": 3, "outs": 18, "battersFaced": 25}]
    )
    rate, pt = season_line(logs, "pitching")
    assert rate == pytest.approx(9 * 3 / 6.0)  # 4.5
    assert pt == 25.0


def test_pace_line_strictly_before_entry():
    logs = pd.DataFrame([_hitting_log("2024-05-01"), _hitting_log("2024-06-15")])
    assert pace_line(logs, "hitting", 2024, pd.Timestamp("2024-06-01")) is not None
    late = pace_line(logs, "hitting", 2024, pd.Timestamp("2024-05-01"))
    assert late is None  # the 05-01 game is NOT before 05-01
    got = pace_line(logs, "hitting", 2024, pd.Timestamp("2024-06-15"))
    assert got[1] == 5.0  # only the 05-01 game counts


def test_league_means_pt_weighted():
    logs = pd.DataFrame(
        [_hitting_log("2024-05-01") | {"mlb_id": 1, "plateAppearances": 100, "hits": 25,
                                       "atBats": 90, "baseOnBalls": 10, "totalBases": 40},
         _hitting_log("2024-05-01") | {"mlb_id": 2, "plateAppearances": 10}]
    )
    out = league_means(logs)
    row = out[(out["season"] == 2024) & (out["group"] == "hitting")].iloc[0]
    # weighted by PA: player 1 dominates
    p1 = (25 + 10 + 0) / (90 + 10 + 0 + 0) + 40 / 90
    p2 = 0.6 + 0.75
    assert row["league_rate"] == pytest.approx((p1 * 100 + p2 * 10) / 110)


def _mini_panel_inputs():
    me = pd.DataFrame(
        {
            "card_slug": ["set/p-bdc-1"] * 2, "grade": ["ungraded"] * 2,
            "month": pd.to_datetime(["2023-06-01", "2023-07-01"]),
            "price": [10.0, 11.0], "mlb_id": [1, 1], "player_name": ["P One", "P One"],
            "rookie_year": [2019, 2019], "card_type": ["bowman_1st_base"] * 2,
        }
    )
    outcomes = pd.DataFrame(
        {
            "card_slug": ["set/p-bdc-1"] * 2, "grade": ["ungraded"] * 2,
            "entry_month": pd.to_datetime(["2023-06-01", "2023-07-01"]),
            "entry_price": [10.0, 11.0], "ret_12m": [0.1, 0.2],
        }
    )
    trailing = pd.DataFrame(
        {
            "card_slug": ["set/p-bdc-1"] * 2, "grade": ["ungraded"] * 2,
            "month": pd.to_datetime(["2023-06-01", "2023-07-01"]),
            "price_level": [2.3, 2.4], "ret_3m": [0.05, 0.06],
        }
    )
    logs = pd.DataFrame(
        [_hitting_log("2022-05-01", season=2022) | {"plateAppearances": 500, "hits": 130,
                                                    "atBats": 450, "baseOnBalls": 50,
                                                    "totalBases": 200},
         _hitting_log("2023-05-01", season=2023),
         # a 1-for-4 single: distinct from the 05-01 game so the June/July pace
         # lines differ (two identical default games would give equal surprises)
         _hitting_log("2023-06-10", season=2023, h=1, bb=0, ab=4, tb=1, pa=4)]
    )
    expectations = pd.DataFrame(
        {
            "player_name": ["P One"], "mlb_id": [1], "season": [2023],
            "source": ["pipeline"], "rank": [7], "fv": [pd.NA],
            "as_of": [pd.Timestamp("2023-04-01")],
        }
    )
    info = pd.DataFrame({"mlb_id": [1], "name": ["P One"],
                         "birth_date": [pd.Timestamp("2000-01-01")], "position": ["SS"]})
    events = pd.DataFrame({"mlb_id": [1], "event_date": [pd.Timestamp("2023-04-15")],
                           "event_type": ["debut"], "details": ["x"]})
    return me, outcomes, trailing, logs, expectations, info, events


def test_build_panel_surprise_rank_era_and_no_lookahead():
    me, outcomes, trailing, logs, expectations, info, events = _mini_panel_inputs()
    panel = build_panel(me, outcomes, trailing, logs, expectations, info, events)
    assert len(panel) == 2
    row = panel[panel["month"] == "2023-07-01"].iloc[0]
    assert row["prospect_rank"] == 7  # as_of 2023-04-01 < 2023-07
    assert not pd.isna(row["surprise_ops"])  # pace (2 May/June games) vs Marcel (2022)
    assert row["price_visible_breakout"]  # first pro season 2022 >= 2020
    # hand-computed Marcel golden: ops_2022 = 180/500 + 200/450; the 2023 league
    # mean is the pace of both 2023 games (4/9 + 4/8); 5-weighted, 1200 regression
    ops22 = 180 / 500 + 200 / 450
    lg23 = 4 / 9 + 4 / 8
    marcel = (5 * ops22 * 500 + lg23 * 1200) / (5 * 500 + 1200)
    assert row["marcel_rate"] == pytest.approx(marcel)
    assert row["pace_rate"] == pytest.approx(lg23)
    assert row["surprise_ops"] == pytest.approx(lg23 - marcel)
    june = panel[panel["month"] == "2023-06-01"].iloc[0]
    # pace at 2023-06-01 = only the 2023-05-01 game; surprise differs from July's
    assert june["surprise_ops"] == pytest.approx(1.35 - marcel)
    assert june["surprise_ops"] != row["surprise_ops"]
    # as_of guard: an expectation dated AFTER the entry month must not appear
    late = expectations.assign(as_of=[pd.Timestamp("2023-08-01")])
    panel2 = build_panel(me, outcomes, trailing, logs, late, info, events)
    assert panel2["prospect_rank"].isna().all()


def test_build_panel_empty_expectations_yields_na_ranks():
    # contract while expectations.parquet is uncollected (Task 6 rate-limited):
    # the panel still builds, every expectation column NaN, Marcel unaffected
    me, outcomes, trailing, logs, expectations, info, events = _mini_panel_inputs()
    panel = build_panel(me, outcomes, trailing, logs, expectations.iloc[0:0], info, events)
    assert len(panel) == 2
    assert panel["prospect_rank"].isna().all()
    assert panel["rank_change"].isna().all()
    assert panel["fg_draft_fv"].isna().all()
    assert panel["surprise_ops"].notna().all()


def test_build_panel_unique_card_grade_month():
    me, outcomes, trailing, logs, expectations, info, events = _mini_panel_inputs()
    panel = build_panel(me, outcomes, trailing, logs, expectations, info, events)
    assert not panel.duplicated(subset=["card_slug", "grade", "month"]).any()
