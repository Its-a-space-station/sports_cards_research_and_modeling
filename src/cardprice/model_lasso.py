"""LASSO with stability selection: which stats drive card price changes."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV


def _fit_selected(X: pd.DataFrame, y: pd.Series, seed: int) -> pd.Index:
    """Fit LassoCV, then select features under the one-standard-error rule.

    The CV-MSE-optimal alpha is prediction-optimal but over-selects: noise
    features with spurious sample correlation survive bootstrap refits.
    The one-SE rule refits at the largest alpha whose mean CV MSE is within
    one standard error of the minimum, the standard choice for sparse
    model selection.
    """
    model = LassoCV(cv=5, random_state=seed, max_iter=10000).fit(X, y)
    mse = model.mse_path_.mean(axis=1)
    se = model.mse_path_.std(axis=1, ddof=1) / np.sqrt(model.mse_path_.shape[1])
    threshold = mse.min() + se[np.argmin(mse)]
    within = np.flatnonzero(mse <= threshold)
    alpha = model.alphas_[within[0]]  # alphas_ is descending: largest eligible
    refit = Lasso(alpha=alpha, max_iter=10000).fit(X, y)
    return X.columns[refit.coef_ != 0]


def stability_selection(
    X: pd.DataFrame, y: pd.Series, n_boot: int = 200, seed: int = 42
) -> pd.Series:
    rng = np.random.default_rng(seed)
    counts = pd.Series(0, index=X.columns, dtype=float)
    n = len(X)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        selected = _fit_selected(X.iloc[idx], y.iloc[idx], seed + b)
        counts[selected] += 1
    return (counts / n_boot).sort_values(ascending=False)


def lasso_path_summary(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> pd.DataFrame:
    model = LassoCV(cv=5, random_state=seed, max_iter=10000).fit(X, y)
    out = pd.DataFrame({"feature": X.columns, "coefficient": model.coef_})
    out = out[out["coefficient"] != 0]
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index)
