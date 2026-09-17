# tests/test_hold_model.py
import numpy as np
import pandas as pd

from cardprice.hold_model import HOLD_FEATURES, build_hold_frame, horizon_importance


def make_panel(n=400, seed=7):
    rng = np.random.default_rng(seed)
    stages = rng.choice(["rookie_year", "sophomore", "established"], n)
    career_games = rng.integers(0, 800, n).astype(float)
    ops = rng.uniform(0.5, 1.0, n)
    df = pd.DataFrame(
        {
            "card_slug": [f"set/card{i % 40}" for i in range(n)],
            "grade": "ungraded",
            "entry_month": pd.to_datetime(
                rng.choice(pd.date_range("2021-04", "2025-09", freq="MS"), n)
            ),
            "mlb_id": [1000 + i % 40 for i in range(n)],
            "player_name": "P",
            "rookie_year": 2018,
            "card_type": rng.choice(["flagship", "bowman_1st"], n),
            "position": rng.choice(["OF", "IF", "C"], n),
            "career_games": career_games,
            "career_ops": ops,
            "career_home_runs": career_games * 0.1,
            "max_level_rank": 4,
            "rate_at_max_level": 0.9,
            "minor_games": 200.0,
            "career_stage": stages,
            "awards_to_date": rng.integers(0, 3, n).astype(float),
            "age": rng.uniform(21, 33, n),
            "age_at_debut": rng.uniform(20, 27, n),
            "price_level": rng.uniform(2, 5, n),
            "ret_3m": rng.normal(0, 0.1, n),
            "market_ret_3m": rng.normal(0, 0.05, n),
        }
    )
    # planted signal: 12m return driven by career_ops only
    df["ret_12m"] = 2.0 * (ops - ops.mean()) + rng.normal(0, 0.05, n)
    df["ret_6m"] = df["ret_12m"]
    df["ret_24m"] = df["ret_12m"]
    df["ret_36m"] = df["ret_12m"]
    return df


def test_build_hold_frame_derives_columns():
    frame = build_hold_frame(make_panel(40), 12)
    assert {
        "career_hr_rate",
        "log_career_games",
        "sophomore",
        "established",
        "bowman_1st",
        "years_since_rookie",
        "entry_year",
    } <= set(frame.columns)
    assert (frame["position"] != "P").all()
    assert frame[HOLD_FEATURES].notna().all().all()  # median imputation complete


def test_planted_signal_recovered_by_both_models():
    frame = build_hold_frame(make_panel(), 12)
    out = horizon_importance(frame, 12)
    assert out["lasso_stability"].idxmax() == "career_ops"
    assert out["lasso_stability"]["career_ops"] >= 0.8
    assert out["gbm_shap"].iloc[0]["feature"] == "career_ops"
    assert out["gbm_cv_r2"] > 0.5
