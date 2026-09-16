"""Gradient boosting + SHAP: nonlinear importance cross-check for the LASSO ranking."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import KFold


def _make_model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.8,
        random_state=seed,
        verbose=-1,
    )


def gbm_shap_importance(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> pd.DataFrame:
    model = _make_model(seed).fit(X, y)
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def gbm_cv_r2(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> float:
    kf = KFold(5, shuffle=True, random_state=seed)
    preds = pd.Series(np.nan, index=y.index)
    for train_idx, test_idx in kf.split(X):
        model = _make_model(seed).fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = model.predict(X.iloc[test_idx])
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot
