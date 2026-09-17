# src/cardprice/walkforward_holds.py
"""Year-grain walk-forward gate for hold returns: fit years < y, predict year y.

Block bootstrap over YEARS (overlapping holds are autocorrelated within and
across cards; the year block is the independence unit per the spec).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV


def walk_forward_years(
    frame: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    top_k: int = 5,
    min_train_years: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    target = f"ret_{horizon}m"
    years = sorted(frame["entry_year"].unique())
    rows = []
    for yi, y in enumerate(years):
        if yi < min_train_years:
            continue
        train = frame[frame["entry_year"] < y]
        test = frame[frame["entry_year"] == y]
        medians = train[feature_cols].median()
        X_train = train[feature_cols].fillna(medians)
        X_test = test[feature_cols].fillna(medians)
        mean, sd = X_train.mean(), X_train.std(ddof=0).replace(0, 1)
        X_train = (X_train - mean) / sd
        X_test = (X_test - mean) / sd
        model = LassoCV(cv=5, random_state=seed, max_iter=10000).fit(X_train, train[target])
        preds = model.predict(X_test)
        bench = test[target].median()
        ranked = test.assign(predicted=preds).nlargest(top_k, "predicted")
        for r in ranked.itertuples():
            rows.append(
                {
                    "entry_year": y,
                    "card_slug": r.card_slug,
                    "predicted": r.predicted,
                    "realized": getattr(r, target) - bench,
                }
            )
    return pd.DataFrame(rows)


def gate_evaluation_years(
    picks: pd.DataFrame, fee: float = 0.14, n_boot: int = 10000, seed: int = 42
) -> dict:
    yearly = picks.groupby("entry_year")["realized"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    boot = rng.choice(yearly, size=(n_boot, len(yearly)), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    return {
        "mean_excess": float(yearly.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "net_mean": float(yearly.mean() - fee),
        "net_ci_low": float(ci_low - fee),
        "n_years": len(yearly),
        "verdict": "PASS" if ci_low - fee > 0 else "FAIL",
    }
