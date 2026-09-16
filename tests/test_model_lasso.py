import numpy as np
import pandas as pd

from cardprice.model_lasso import lasso_path_summary, stability_selection


def make_planted(n=300, seed=7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(8)})
    # true driver: x2 with coefficient 0.8; everything else noise
    y = 0.8 * X["x2"] + rng.normal(scale=0.5, size=n)
    return X, y


def test_stability_selection_recovers_planted_driver():
    X, y = make_planted()
    freq = stability_selection(X, y, n_boot=100, seed=42)
    assert freq.index[0] == "x2"
    assert freq["x2"] > 0.8
    # noise features selected far less often
    assert (freq.drop("x2") < 0.5).all()


def test_lasso_path_summary_coefficients():
    X, y = make_planted()
    out = lasso_path_summary(X, y)
    assert out.iloc[0]["feature"] == "x2"
    assert out.iloc[0]["coefficient"] > 0.3
    assert (out["coefficient"] != 0).all()
