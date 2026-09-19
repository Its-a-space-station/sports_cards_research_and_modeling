# tests/test_quintiles.py
import numpy as np
import pandas as pd
import pytest

from cardprice.quintiles import quintile_spread, spread_summary


def _frame(n=60, seed=3, planted=True):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(1, 13):
        month = pd.Timestamp(f"2023-{m:02d}-01")
        for _ in range(n):
            s = rng.normal(0, 1)
            ret = 0.4 * s + rng.normal(0, 0.2) if planted else rng.normal(0, 0.5)
            rows.append({"entry_month": month, "score": s, "ret_12m": ret})
    return pd.DataFrame(rows)


def test_quintile_spread_recovers_planted_signal():
    out = quintile_spread(_frame(), "score")
    assert len(out) == 12
    assert out["spread"].mean() > 0.5  # planted monotone relation
    shuffled = quintile_spread(_frame(planted=False), "score")
    assert abs(shuffled["spread"].mean()) < 0.2


def test_quintile_spread_nan_scores_excluded():
    df = _frame(n=10)
    df.loc[df.sample(frac=0.5, random_state=1).index, "score"] = np.nan
    out = quintile_spread(df, "score")
    assert (out["n_scored"] <= 5).all()


def test_spread_summary_year_block_bootstrap():
    spreads = pd.DataFrame(
        {"month": pd.to_datetime(
            ["2023-01-01", "2023-06-01", "2024-01-01", "2024-06-01", "2025-01-01", "2025-06-01"]),
         "spread": [0.5, 0.6, 0.2, 0.3, -0.4, -0.3], "n_scored": [50] * 6}
    )
    res = spread_summary(spreads)
    assert res["n_months"] == 6
    assert res["mean_spread"] == pytest.approx(0.15)
    assert res["ci_low"] < res["ci_high"]
    assert 0 <= res["share_positive"] <= 1
