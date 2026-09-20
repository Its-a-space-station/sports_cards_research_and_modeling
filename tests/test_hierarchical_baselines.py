# tests/test_hierarchical_baselines.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_hierarchical import (
    assign_folds,
    oof_lasso_predictions,
    player_fe_null_lpd,
    pooled_lasso_lpd,
)


def test_pooled_lasso_lpd_gaussian_math():
    y = pd.Series([0.0, 1.0, 2.0])
    pred = pd.Series([0.0, 1.0, 2.0])
    sigma = 1.0
    expected = float(np.mean([-0.5 * np.log(2 * np.pi * 1.0) - 0.0] * 3))
    assert pooled_lasso_lpd(pd.DataFrame(), y, pred, sigma) == pytest.approx(expected)


def test_player_fe_null_uses_oof_player_means():
    frame = pd.DataFrame(
        {"mlb_id": ["a", "a", "b", "b", "c"], "y": [1.0, 3.0, 0.0, 2.0, 5.0],
         "fold": [0, 1, 0, 1, 0]}
    )
    # player means computed out-of-fold; unseen player "c" -> global mean
    lpd = player_fe_null_lpd(frame, group_col="mlb_id", y_col="y", fold_col="fold")
    assert isinstance(lpd, float)
    assert np.isfinite(lpd)
    # in-sample (wrong) player means would give a higher LPD; OOF is lower/honest
    assert lpd < -0.5 * np.log(2 * np.pi * 0.5)


def test_oof_lasso_predictions_finite_aligned_deterministic():
    # Tiny synthetic frame (offline, seeded): one NaN predictor row exercises the
    # train-median imputation path; LassoCV is deterministic given the same folds.
    rng = np.random.default_rng(7)
    n = 60
    frame = pd.DataFrame(
        {
            "y": rng.normal(0, 1, n),
            "surprise": rng.normal(0, 1, n),
            "market_ret_3m": rng.normal(0, 1, n),
            "price_level": rng.normal(2, 0.3, n),
            "career_stage": pd.Categorical(
                rng.choice(["prospect", "rookie_year", "established"], n)
            ),
            "mlb_id": [str(i % 12) for i in range(n)],
        }
    )
    frame.loc[0, "market_ret_3m"] = np.nan
    frame["fold"] = assign_folds(n).to_numpy()
    p1 = oof_lasso_predictions(frame)
    p2 = oof_lasso_predictions(frame)
    assert len(p1) == n
    assert p1.notna().all() and np.isfinite(p1).all()
    assert (p1 == p2).all()
