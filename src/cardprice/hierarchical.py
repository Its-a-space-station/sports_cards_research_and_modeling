# src/cardprice/hierarchical.py
"""Hierarchical Bayesian consistency model: data prep + Bambi model wrapper.

Answers "which attributes matter, and are the effects consistent across draft
classes and positions?" with partial pooling: varying intercepts
(player ⊂ class:position ⊂ class_year) and varying surprise slopes by
class:position. Pure prep functions here; the model wrapper is Task 2.
"""

import numpy as np
import pandas as pd

FORMULA = (
    "y ~ surprise + career_stage + market_ret_3m + price_level "
    "+ (1|class_year) + (1|class_position) + (1|mlb_id) + (0 + surprise|class_position)"
)
PREDICTORS = ("surprise", "market_ret_3m", "price_level")
STAGES = ["prospect", "rookie_year", "sophomore", "established"]


def build_model_frame(panel: pd.DataFrame, group: str) -> tuple[pd.DataFrame, dict]:
    """(frame, scaling_params) for the hierarchical fit.

    Filters: ungraded, ret_12m non-null, group filter, surprise non-null
    (hitters: surprise_ops; pitchers: surprise := -surprise_era, the sign
    convention where positive = better-than-projection). Predictors z-scored
    (ddof=0); y unscaled. No other row filtering and no imputation.
    """
    df = panel[(panel["grade"] == "ungraded") & panel["ret_12m"].notna()].copy()
    if group == "hitter":
        df = df[df["position"] != "P"]
        df["surprise"] = pd.to_numeric(df["surprise_ops"], errors="coerce")
    elif group == "pitcher":
        df = df[df["position"] == "P"]
        df["surprise"] = -pd.to_numeric(df["surprise_era"], errors="coerce")
    else:
        raise ValueError(f"unknown group {group!r}")
    df = df[df["surprise"].notna()]

    params = {}
    for col in PREDICTORS:
        vals = pd.to_numeric(df[col], errors="coerce").astype(float)
        mean, sd = vals.mean(), vals.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            sd = 1.0
        df[col] = (vals - mean) / sd
        params[col] = (float(mean), float(sd))

    df["y"] = df["ret_12m"].astype(float)
    df["career_stage"] = pd.Categorical(df["career_stage"], categories=STAGES)
    df["class_year"] = df["rookie_year"].astype(int).astype(str)
    df["class_position"] = df["class_year"] + "_" + df["position"].astype(str)
    df["mlb_id"] = df["mlb_id"].astype(int).astype(str)
    keep = ["y", "surprise", "career_stage", "market_ret_3m", "price_level",
            "class_year", "class_position", "mlb_id"]
    return df[keep].reset_index(drop=True), params
