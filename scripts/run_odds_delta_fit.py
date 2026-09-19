# scripts/run_odds_delta_fit.py
"""Odds delta-fit evaluation vs the T2 baseline (descriptive only).

Per gate cell (the run_class_gate cell list: ungraded hitters 6/12/24/36,
psa_10 12m hitter, ungraded 12m pitcher), sliced from Task 3's stacked
class_hold_odds.parquet:
- baseline: walk-forward gate with the cell's T2 feature list, recomputed
  (must reconcile with T2's registered numbers within harness determinism),
- with_odds: the same walk-forward with the odds features appended
  (["has_market", "odds_level", "odds_delta_7d", "odds_delta_30d"]),
- per test year: spearman(predicted, realized) from the all_scores
  walk-forward and mean excess of the top-5 picks, for both; pooled deltas.

Everything here is DESCRIPTIVE: the pre-registered T2 verdict (registered
cell, base features) is untouched by this comparison.

Also re-runs the importance fits WITH the odds features on the registered
cell (same full-frame-median-imputation caveat as run_class_importance.py)
and writes the same long format to class_hold_importance_odds.csv.

Degenerate cells: 24m/36m hitters have zero has_market=1 rows (odds
snapshots cover only the 2025-2026 seasons; those cells' non-null targets
predate the odds era), so the with_odds features are identically zero and
the deltas are zero by construction — computed and reported as measured,
not suppressed. The 541 has_market=1 rows without a pre-entry snapshot keep
NaN odds features by design (541 is the post-psa_10-widening count; it was
446 before the psa_10 cell was added); the harness's per-fold train-median
imputation handles them as ordinary NaNs (never zero-filled here).
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cardprice.class_hold_frame import CLASS_HOLD_FEATURES, CLASS_PITCHER_FEATURES
from cardprice.features import standardize
from cardprice.model_gbm import gbm_cv_r2, gbm_shap_importance
from cardprice.model_lasso import lasso_path_summary, stability_selection
from cardprice.walkforward_holds import gate_evaluation_years, walk_forward_years

CELLS = [
    {"grade": "ungraded", "horizon": 12, "group": "hitter", "registered": True},
    {"grade": "ungraded", "horizon": 6, "group": "hitter", "registered": False},
    {"grade": "ungraded", "horizon": 24, "group": "hitter", "registered": False},
    {"grade": "ungraded", "horizon": 36, "group": "hitter", "registered": False},
    {"grade": "psa_10", "horizon": 12, "group": "hitter", "registered": False},
    {"grade": "ungraded", "horizon": 12, "group": "pitcher", "registered": False},
]
FEATURES = {"hitter": CLASS_HOLD_FEATURES, "pitcher": CLASS_PITCHER_FEATURES}
ODDS_COLS = ["has_market", "odds_level", "odds_delta_7d", "odds_delta_30d"]
IN_PARQUET = "data/processed/class_hold_odds.parquet"
OUT_JSON = "data/processed/class_odds_delta_fit.json"
OUT_IMPORTANCE_CSV = "data/processed/class_hold_importance_odds.csv"
LABEL = "descriptive"


def compare_feature_sets(
    frame: pd.DataFrame, base_features: list[str], odds_features: list[str], horizon: int
) -> dict:
    """Walk-forward both feature sets on one cell frame; return gate + deltas.

    Gate numbers come from the top-5 picks path (identical to the T2 gate
    runner); per-year Spearman uses the all_scores predictions (every test
    row scored, no selection).
    """
    out = {}
    for label, feats in (("baseline", base_features), ("with_odds", odds_features)):
        with warnings.catch_warnings():
            # LassoCV convergence chatter + ConstantInputWarning from Spearman
            # on constant predictions (all-zero lasso); a constant fit stays
            # honestly visible as NaN spearman in the output
            warnings.simplefilter("ignore")
            picks = walk_forward_years(frame, feats, horizon, top_k=5)
            scores = walk_forward_years(frame, feats, horizon, all_scores=True)
            spearman_by_year = {
                int(y): float(g["predicted"].corr(g["realized"], method="spearman"))
                for y, g in scores.groupby("entry_year")
            }
        out[label] = {
            "gate": gate_evaluation_years(picks),
            "yearly_mean_excess": {
                int(y): float(m)
                for y, m in picks.groupby("entry_year")["realized"].mean().items()
            },
            "spearman_by_year": spearman_by_year,
        }
    base, odds = out["baseline"], out["with_odds"]
    years = sorted(base["spearman_by_year"])
    out["delta"] = {
        "mean_excess": odds["gate"]["mean_excess"] - base["gate"]["mean_excess"],
        "spearman": float(
            np.nanmean(
                [odds["spearman_by_year"][y] - base["spearman_by_year"][y] for y in years]
            )
        ),
        "spearman_by_year": {
            y: odds["spearman_by_year"][y] - base["spearman_by_year"][y] for y in years
        },
        "yearly_mean_excess": {
            y: odds["yearly_mean_excess"][y] - base["yearly_mean_excess"][y] for y in years
        },
    }
    return out


def _cell_frame(stacked: pd.DataFrame, grade: str, group: str, horizon: int) -> pd.DataFrame:
    sub = stacked[
        (stacked["grade"] == grade)
        & (stacked["group"] == group)
        & (stacked["horizon"] == horizon)
    ]
    # defensive per-cell target filter, same as run_class_gate.run_cells (the
    # stacked parquet's builds already dropped that horizon's NaN targets)
    return sub[sub[f"ret_{horizon}m"].notna()]


def run_cells(stacked: pd.DataFrame, out_json: str = OUT_JSON) -> list[dict]:
    results = []
    for cell in CELLS:
        sub = _cell_frame(stacked, cell["grade"], cell["group"], cell["horizon"])
        if not len(sub):
            print(f"[skipped] {cell['grade']}/{cell['horizon']}m/{cell['group']}: no rows")
            continue
        base = FEATURES[cell["group"]]
        res = compare_feature_sets(sub, base, base + ODDS_COLS, cell["horizon"])
        note = None
        if int(sub["has_market"].sum()) == 0:
            note = (
                "zero has_market=1 rows: with_odds features identically zero, "
                "so deltas are zero by construction (2-season odds coverage)"
            )
        results.append({**cell, "label": LABEL, **res, "note": note})
        tag = "registered cell" if cell["registered"] else LABEL
        print(
            f"[{tag}] {cell['grade']}/{cell['horizon']}m/{cell['group']}: "
            f"baseline mean_excess={res['baseline']['gate']['mean_excess']:.4f} "
            f"with_odds={res['with_odds']['gate']['mean_excess']:.4f} "
            f"delta={res['delta']['mean_excess']:+.4f} "
            f"delta_spearman={res['delta']['spearman']:+.4f}"
            + (f" | {note}" if note else "")
        )
    Path(out_json).write_text(json.dumps({"cells": results}, indent=2, default=float))
    print(f"wrote {out_json}")
    return results


def run_importance_odds(stacked: pd.DataFrame, out_csv: str = OUT_IMPORTANCE_CSV) -> pd.DataFrame:
    """Importance fits WITH odds features on the registered cell (descriptive).

    Mirrors scripts/run_class_importance.py, including its full-frame median
    imputation caveat (pools information across time; descriptive rankings,
    not out-of-sample estimates).
    """
    sub = _cell_frame(stacked, "ungraded", "hitter", 12)
    feats = CLASS_HOLD_FEATURES + ODDS_COLS
    X = sub[feats]
    X = X.fillna(X.median())  # full-frame median imputation (caveat above)
    Z, _ = standardize(X)
    y = sub["ret_12m"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stab = stability_selection(Z, y)
        path = lasso_path_summary(Z, y)
        shap_imp = gbm_shap_importance(Z, y)
        r2 = gbm_cv_r2(Z, y)
    print(f"\n=== ungraded/hitter ret_12m WITH odds: {len(sub)} rows, gbm_cv_r2={r2:.3f} ===")
    print("lasso stability top10:", stab.round(2).head(10).to_dict())
    print("gbm shap top10:", shap_imp.head(10)["feature"].tolist())
    long_rows = (
        [("lasso_stability", f, v) for f, v in stab.items()]
        + [("lasso_path", r.feature, r.coefficient) for r in path.itertuples()]
        + [("gbm_shap", r.feature, r.mean_abs_shap) for r in shap_imp.itertuples()]
        + [("gbm_cv_r2", "__r2__", r2)]
    )
    rows = [
        {"grade": "ungraded", "group": "hitter", "horizon": 12,
         "method": m, "feature": f, "value": v}
        for m, f, v in long_rows
    ]
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    return out


def main() -> None:
    stacked = pd.read_parquet(IN_PARQUET)
    run_cells(stacked)
    run_importance_odds(stacked)


if __name__ == "__main__":
    main()
