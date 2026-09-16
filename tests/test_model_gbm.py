import numpy as np
import pandas as pd

from cardprice.model_gbm import gbm_cv_r2, gbm_shap_importance


def make_planted(n=400, seed=7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
    # nonlinear driver: y depends on x1^2, everything else noise
    y = X["x1"] ** 2 + rng.normal(scale=0.3, size=n)
    return X, y


def test_shap_ranks_planted_driver_top():
    X, y = make_planted()
    imp = gbm_shap_importance(X, y)
    assert imp.iloc[0]["feature"] == "x1"


def test_gbm_cv_r2_beats_noise_baseline():
    X, y = make_planted()
    r2 = gbm_cv_r2(X, y)
    assert r2 > 0.3  # clearly better than predicting the mean
