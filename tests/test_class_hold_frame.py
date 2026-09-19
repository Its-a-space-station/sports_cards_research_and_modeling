# tests/test_class_hold_frame.py
import numpy as np
import pandas as pd
import pytest

from cardprice.class_hold_frame import (
    CLASS_HOLD_FEATURES,
    CLASS_PITCHER_FEATURES,
    build_class_hold_frame,
)


def _panel_row(mlb_id=1, month="2023-06-01", position="SS", stage="prospect",
               ret12=0.1, grade="ungraded", ops=0.8, era=None, games=100, hr=10,
               kbb=None, whip=None, ip=None, surprise_ops=0.05, surprise_era=None,
               draft_rank=5.0, rookie_year=2020):
    return {
        "card_slug": f"set/p-{mlb_id}", "grade": grade, "month": pd.Timestamp(month),
        "entry_price": 10.0, "ret_12m": ret12, "ret_6m": 0.05, "ret_24m": 0.2,
        "ret_36m": 0.3, "price_level": 2.3, "ret_3m": 0.02, "market_ret_3m": 0.01,
        "mlb_id": mlb_id, "player_name": f"P{mlb_id}", "rookie_year": rookie_year,
        "card_type": "bowman_1st_base", "career_stage": stage,
        "awards_to_date": 0, "age": 22.0, "position": position,
        "max_level": "aa", "max_level_rank": 3, "rate_at_max_level": 0.75,
        "minor_games": 200, "games": games, "ops": ops, "home_runs": hr,
        "era": era, "k_bb_pct": kbb, "innings_pitched": ip, "whip": whip,
        "surprise_ops": surprise_ops, "surprise_era": surprise_era,
        "marcel_rate": 0.75, "pace_rate": 0.8, "draft_rank": draft_rank,
        "prospect_rank": pd.NA, "rank_change": pd.NA, "fg_draft_fv": pd.NA,
        "price_visible_breakout": True,
    }


def _events():
    return pd.DataFrame(
        {"mlb_id": [1, 3], "event_date": pd.to_datetime(["2022-04-15", "2023-06-20"]),
         "event_type": ["debut", "debut"], "details": ["x", "y"]}
    )


def _info():
    return pd.DataFrame(
        {"mlb_id": [1, 2, 3], "name": ["P1", "P2", "P3"],
         "birth_date": pd.to_datetime(["2000-01-01", "1999-06-01", "2001-03-01"]),
         "position": ["SS", "P", "CF"]}
    )


def test_feature_lists_exact():
    assert CLASS_HOLD_FEATURES == [
        "career_ops", "career_hr_rate", "log_career_games", "max_level_rank",
        "rate_at_max_level", "log_minor_games", "awards_to_date", "age",
        "age_at_debut", "price_level", "ret_3m", "market_ret_3m", "sophomore",
        "established", "prospect", "years_since_class", "surprise_ops", "draft_rank",
    ]
    assert CLASS_PITCHER_FEATURES == [
        "career_era", "career_k_bb_pct", "log_career_games", "max_level_rank",
        "rate_at_max_level", "log_minor_games", "awards_to_date", "age",
        "age_at_debut", "price_level", "ret_3m", "market_ret_3m", "sophomore",
        "established", "prospect", "years_since_class", "surprise_era_flipped",
        "draft_rank",
    ]


def test_adapter_columns_and_derived_values():
    panel = pd.DataFrame([_panel_row()])
    out = build_class_hold_frame(panel, _events(), _info())
    row = out.iloc[0]
    assert row["entry_month"] == pd.Timestamp("2023-06-01")
    assert row["entry_year"] == 2023
    assert row["career_ops"] == 0.8
    assert row["career_hr_rate"] == pytest.approx(10 / 100)
    assert row["log_career_games"] == pytest.approx(np.log1p(100))
    assert row["log_minor_games"] == pytest.approx(np.log1p(200))
    assert row["prospect"] == 1 and row["sophomore"] == 0 and row["established"] == 0
    assert row["years_since_class"] == 3  # 2023 - 2020
    assert row["age_at_debut"] == pytest.approx(
        (pd.Timestamp("2022-04-15") - pd.Timestamp("2000-01-01")).days / 365.25
    )
    assert row["draft_rank"] == 5.0 and row["surprise_ops"] == 0.05


def test_zero_games_hr_rate_and_missing_debut():
    panel = pd.DataFrame([_panel_row(mlb_id=2, games=0, hr=0)])
    out = build_class_hold_frame(panel, _events(), _info())
    assert out.iloc[0]["career_hr_rate"] == 0.0
    assert pd.isna(out.iloc[0]["age_at_debut"])  # no debut event for mlb_id 2


def test_no_imputation_nans_preserved():
    panel = pd.DataFrame([_panel_row(mlb_id=2, ops=None, draft_rank=None)])
    out = build_class_hold_frame(panel, _events(), _info())
    assert pd.isna(out.iloc[0]["career_ops"])
    assert pd.isna(out.iloc[0]["draft_rank"])


def test_group_filters_and_pitcher_sign_flip():
    panel = pd.DataFrame(
        [_panel_row(mlb_id=1), _panel_row(mlb_id=2, position="P", era=3.5, kbb=0.15,
                                          surprise_era=-0.4, surprise_ops=None)]
    )
    hitters = build_class_hold_frame(panel, _events(), _info(), group="hitter")
    pitchers = build_class_hold_frame(panel, _events(), _info(), group="pitcher")
    assert len(hitters) == 1 and len(pitchers) == 1
    prow = pitchers.iloc[0]
    assert prow["career_era"] == 3.5
    assert prow["career_k_bb_pct"] == 0.15
    assert prow["surprise_era_flipped"] == pytest.approx(0.4)  # -(pace-marcel)
    assert "surprise_ops" not in pitchers.columns or pd.isna(prow.get("surprise_ops"))


def test_target_filter_only():
    panel = pd.DataFrame([_panel_row(mlb_id=1, ret12=0.1), _panel_row(mlb_id=2, ret12=None)])
    out = build_class_hold_frame(panel, _events(), _info())
    assert len(out) == 1 and out.iloc[0]["mlb_id"] == 1


def test_stage_dummies_reference_is_rookie_year():
    rows = []
    for stage in ("prospect", "rookie_year", "sophomore", "established"):
        rows.append(_panel_row(mlb_id=1, stage=stage))
    out = build_class_hold_frame(pd.DataFrame(rows), _events(), _info())
    got = out[["prospect", "sophomore", "established"]].values.tolist()
    assert got == [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]
