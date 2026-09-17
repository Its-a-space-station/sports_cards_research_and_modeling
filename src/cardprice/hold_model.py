# src/cardprice/hold_model.py
"""Hold-return modeling frame + per-horizon importance (LASSO stability / GBM-SHAP)."""

import numpy as np
import pandas as pd

from cardprice.career import season_year
from cardprice.features import standardize
from cardprice.model_gbm import gbm_cv_r2, gbm_shap_importance
from cardprice.model_lasso import lasso_path_summary, stability_selection

HOLD_FEATURES = [
    "career_ops",
    "career_hr_rate",
    "log_career_games",
    "max_level_rank",
    "rate_at_max_level",
    "log_minor_games",
    "awards_to_date",
    "age",
    "age_at_debut",
    "price_level",
    "ret_3m",
    "market_ret_3m",
    "sophomore",
    "established",
    "bowman_1st",
    "years_since_rookie",
]


def build_hold_frame(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = panel[(panel["position"] != "P") & panel[f"ret_{horizon}m"].notna()].copy()
    df["career_hr_rate"] = np.where(
        df["career_games"] > 0, df["career_home_runs"] / df["career_games"], 0.0
    )
    df["log_career_games"] = np.log1p(df["career_games"])
    df["log_minor_games"] = np.log1p(df["minor_games"])
    df["sophomore"] = (df["career_stage"] == "sophomore").astype(int)
    df["established"] = (df["career_stage"] == "established").astype(int)
    df["bowman_1st"] = (df["card_type"] == "bowman_1st").astype(int)
    df["years_since_rookie"] = df["entry_month"].map(season_year) - df["rookie_year"]
    df["entry_year"] = df["entry_month"].dt.year
    df[HOLD_FEATURES] = df[HOLD_FEATURES].fillna(df[HOLD_FEATURES].median())
    return df.reset_index(drop=True)


def horizon_importance(frame: pd.DataFrame, horizon: int, seed: int = 42) -> dict:
    X, _ = standardize(frame[HOLD_FEATURES])
    y = frame[f"ret_{horizon}m"]
    return {
        "lasso_stability": stability_selection(X, y, seed=seed),
        "lasso_path": lasso_path_summary(X, y, seed=seed),
        "gbm_shap": gbm_shap_importance(X, y, seed=seed),
        "gbm_cv_r2": gbm_cv_r2(X, y, seed=seed),
    }
