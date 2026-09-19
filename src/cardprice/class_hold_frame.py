# src/cardprice/class_hold_frame.py
"""Adapter from panel_class conventions to the P6c hold-model harness.

panel_class uses `month` (not entry_month), unprefixed career-to-date stats,
carries class_year in rookie_year, and includes prospect-stage rows and
pitchers. This module derives the feature frame; it deliberately does NOT
impute or scale — the gate imputes/scales from train years per fold, and the
importance runner documents its own full-frame-median caveat.
"""

import numpy as np
import pandas as pd

from cardprice.career import season_year

CLASS_HOLD_FEATURES = [
    "career_ops", "career_hr_rate", "log_career_games", "max_level_rank",
    "rate_at_max_level", "log_minor_games", "awards_to_date", "age", "age_at_debut",
    "price_level", "ret_3m", "market_ret_3m", "sophomore", "established", "prospect",
    "years_since_class", "surprise_ops", "draft_rank",
]
CLASS_PITCHER_FEATURES = [
    "career_era", "career_k_bb_pct", "log_career_games", "max_level_rank",
    "rate_at_max_level", "log_minor_games", "awards_to_date", "age", "age_at_debut",
    "price_level", "ret_3m", "market_ret_3m", "sophomore", "established", "prospect",
    "years_since_class", "surprise_era_flipped", "draft_rank",
]
PASSTHROUGH = [
    "card_slug", "grade", "mlb_id", "player_name", "rookie_year", "card_type",
    "position", "career_stage",
]


def _debuts(events: pd.DataFrame) -> pd.Series:
    d = events[events["event_type"] == "debut"]
    return d.groupby("mlb_id")["event_date"].min()


def build_class_hold_frame(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    info: pd.DataFrame,
    horizon: int = 12,
    group: str = "hitter",
) -> pd.DataFrame:
    """Feature frame for the class panel: entry_month/entry_year, target, features.

    Only the target NaN filter is applied (plus the position group filter);
    every other NaN is preserved for the gate's per-fold train imputation.
    """
    target = f"ret_{horizon}m"
    df = panel.copy()
    df["entry_month"] = df["month"]
    df["entry_year"] = df["entry_month"].dt.year
    df = df[df[target].notna()]
    df = df[df["position"] != "P"] if group == "hitter" else df[df["position"] == "P"]

    df["career_ops"] = df["ops"]
    df["career_era"] = df["era"]
    df["career_k_bb_pct"] = df["k_bb_pct"]
    games = df["games"].astype(float)
    df["career_hr_rate"] = np.where(games > 0, df["home_runs"] / games, 0.0)
    df["log_career_games"] = np.log1p(games)
    df["log_minor_games"] = np.log1p(df["minor_games"])
    df["sophomore"] = (df["career_stage"] == "sophomore").astype(int)
    df["established"] = (df["career_stage"] == "established").astype(int)
    df["prospect"] = (df["career_stage"] == "prospect").astype(int)
    df["years_since_class"] = df["entry_month"].map(season_year) - df["rookie_year"]

    debut_of = _debuts(events)
    birth = info.set_index("mlb_id")["birth_date"]
    debut_dates = df["mlb_id"].map(debut_of)
    birth_dates = df["mlb_id"].map(birth)
    df["age_at_debut"] = (debut_dates - birth_dates).dt.days / 365.25
    # look-ahead guard: a debut after the entry stamp hadn't happened yet
    df.loc[debut_dates > df["entry_month"], "age_at_debut"] = np.nan

    df["surprise_ops"] = pd.to_numeric(df["surprise_ops"], errors="coerce")
    df["surprise_era"] = pd.to_numeric(df["surprise_era"], errors="coerce")
    df["draft_rank"] = pd.to_numeric(df["draft_rank"], errors="coerce")
    df["surprise_era_flipped"] = -df["surprise_era"]

    keep = ["entry_month", "entry_year", target] + PASSTHROUGH + list(
        dict.fromkeys(CLASS_HOLD_FEATURES + CLASS_PITCHER_FEATURES)
    )
    return df[keep].reset_index(drop=True)
