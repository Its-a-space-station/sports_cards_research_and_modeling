"""Walk-forward evaluation: fit on months < m, predict month m, gate on realized excess."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV


def walk_forward(
    frame: pd.DataFrame, feature_cols: list[str], top_k: int = 2, min_train: int = 6, seed: int = 42
) -> pd.DataFrame:
    months = sorted(frame["month"].unique())
    rows = []
    for mi, m in enumerate(months):
        if mi < min_train:
            continue
        train = frame[frame["month"] < m]
        test = frame[frame["month"] == m]
        medians = train[feature_cols].median()
        X_train = train[feature_cols].fillna(medians)
        X_test = test[feature_cols].fillna(medians)
        model = LassoCV(cv=5, random_state=seed, max_iter=10000).fit(X_train, train["excess_ret"])
        preds = model.predict(X_test)
        ranked = test.assign(predicted=preds).nlargest(top_k, "predicted")
        for r in ranked.itertuples():
            rows.append(
                {
                    "month": m,
                    "card_slug": r.card_slug,
                    "predicted": r.predicted,
                    "realized": r.excess_ret,
                }
            )
    return pd.DataFrame(rows)


def gate_evaluation(
    picks: pd.DataFrame,
    frame: pd.DataFrame,
    fee_log: float = 0.14,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    monthly = []
    for m, grp in picks.groupby("month"):
        strategy = grp["realized"].mean()
        benchmark = frame[frame["month"] == m]["excess_ret"].median()
        monthly.append(strategy - benchmark)
    monthly = np.array(monthly)
    rng = np.random.default_rng(seed)
    boot = rng.choice(monthly, size=(n_boot, len(monthly)), replace=True).mean(axis=1)
    return {
        "mean_excess": float(monthly.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": float((boot <= 0).mean()),
        "net_of_fees_mean": float(monthly.mean() - fee_log),
        "n_months": len(monthly),
    }
