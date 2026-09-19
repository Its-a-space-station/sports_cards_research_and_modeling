# src/cardprice/quintiles.py
"""Cross-sectional quintile spreads: do score-sorted baskets separate returns?"""

import numpy as np
import pandas as pd


def quintile_spread(
    frame: pd.DataFrame,
    score_col: str,
    target_col: str = "ret_12m",
    month_col: str = "entry_month",
    q: int = 5,
) -> pd.DataFrame:
    """Per month: mean target of top score quintile minus bottom quintile.

    NaN scores dropped per month; months with fewer than 2*q scored rows are
    skipped (insufficient cross-section for stable quintiles).
    """
    rows = []
    for month, g in frame.dropna(subset=[score_col]).groupby(month_col):
        if len(g) < 2 * q:
            continue
        ranks = g[score_col].rank(method="average")
        labels = np.ceil(ranks / len(g) * q).clip(1, q).astype(int)
        spread = g[target_col][labels == q].mean() - g[target_col][labels == 1].mean()
        rows.append({month_col: month, "spread": float(spread), "n_scored": len(g)})
    return pd.DataFrame(rows, columns=[month_col, "spread", "n_scored"])


def spread_summary(spreads: pd.DataFrame, n_boot: int = 10000, seed: int = 42) -> dict:
    """Block bootstrap over entry YEARS (year block = independence unit)."""
    s = spreads.dropna(subset=["spread"]).copy()
    if not len(s):
        return {"mean_spread": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_months": 0, "share_positive": np.nan}
    s["year"] = s["month"].dt.year if "month" in s.columns else s.iloc[:, 0].dt.year
    yearly = s.groupby("year")["spread"].mean()
    rng = np.random.default_rng(seed)
    boot = rng.choice(yearly.to_numpy(), size=(n_boot, len(yearly)), replace=True).mean(axis=1)
    return {
        "mean_spread": float(yearly.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_months": len(s),
        "share_positive": float((s["spread"] > 0).mean()),
    }
