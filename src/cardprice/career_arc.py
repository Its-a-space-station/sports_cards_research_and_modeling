# src/cardprice/career_arc.py
"""Career-arc overlay: stage effects, per-entry-year consistency, position interaction.

Adapts the Plan-3 consistency module by column aliasing (it stays untouched):
its `excess_ret` is our absolute hold return; `games` our career_games;
`stats_season` our entry_year.
"""

import pandas as pd
import statsmodels.formula.api as smf

from cardprice.features import POSITION_MAP
from cardprice.model_consistency import (
    player_random_effects,
    position_interaction_test,
    season_consistency,
)

__all__ = [
    "consistency_adapter",
    "player_random_effects",
    "position_interaction_test",
    "season_consistency",
    "stage_effects",
    "years_curve",
]


def consistency_adapter(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = frame.copy()
    out["excess_ret"] = out[f"ret_{horizon}m"]
    out["games"] = out["career_games"]
    out["stats_season"] = out["entry_year"]
    out["position_group"] = out["position"].map(POSITION_MAP)
    return out


def stage_effects(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = frame.dropna(subset=[f"ret_{horizon}m", "age", "price_level", "market_ret_3m"]).copy()
    # rookie_year first so patsy uses it as the reference level (there are no
    # prospect rows — the stage is structurally empty in the panel).
    df["career_stage"] = pd.Categorical(
        df["career_stage"], categories=["rookie_year", "sophomore", "established"]
    )
    fit = smf.ols(
        f"ret_{horizon}m ~ C(career_stage) + age + price_level + market_ret_3m", data=df
    ).fit(cov_type="HC1")
    return pd.DataFrame(
        {
            "term": fit.params.index,
            "coef": fit.params.values,
            "se": fit.bse.values,
            "p_value": fit.pvalues.values,
            "n": len(df),
        }
    )


def years_curve(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return (
        frame.groupby("years_since_rookie")[f"ret_{horizon}m"]
        .agg(
            median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), n="size"
        )
        .reset_index()
    )
