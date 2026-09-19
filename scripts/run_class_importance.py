# scripts/run_class_importance.py
"""Hold-return importance fits on the class panel (descriptive only).

Per grade x horizon cell (hitters; plus the ungraded 12m pitcher cell):
full-frame median imputation -> features.standardize -> stability_selection +
lasso_path_summary + gbm_shap_importance + gbm_cv_r2.

CAVEAT: full-frame median imputation pools information across time, so these
fits are descriptive rankings, not out-of-sample estimates (the walk-forward
gate is the pre-registered test; same caveat as run_hold_modeling.py).
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cardprice.class_hold_frame import (
    CLASS_HOLD_FEATURES,
    CLASS_PITCHER_FEATURES,
    build_class_hold_frame,
)
from cardprice.features import standardize
from cardprice.model_gbm import gbm_cv_r2, gbm_shap_importance
from cardprice.model_lasso import lasso_path_summary, stability_selection

HORIZONS = (6, 12, 24, 36)
CELLS = [(g, "hitter", h) for g in ("ungraded", "psa_10") for h in HORIZONS]
CELLS.append(("ungraded", "pitcher", 12))
FEATURES = {"hitter": CLASS_HOLD_FEATURES, "pitcher": CLASS_PITCHER_FEATURES}
OUT_CSV = "data/processed/class_hold_importance.csv"


def main() -> None:
    panel = pd.read_parquet("data/processed/panel_class.parquet")
    events = pd.read_parquet("data/processed/events_class.parquet")
    info = pd.read_csv("data/reference/player_info_class.csv", parse_dates=["birth_date"])
    frames = {
        ("hitter", h): build_class_hold_frame(panel, events, info, horizon=h, group="hitter")
        for h in HORIZONS
    }
    frames[("pitcher", 12)] = build_class_hold_frame(
        panel, events, info, horizon=12, group="pitcher"
    )
    rows = []
    for grade, group, h in CELLS:
        sub = frames[(group, h)]
        sub = sub[sub["grade"] == grade]
        X = sub[FEATURES[group]]
        X = X.fillna(X.median())  # full-frame median imputation (caveat above)
        Z, _ = standardize(X)
        y = sub[f"ret_{h}m"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stab = stability_selection(Z, y)
            path = lasso_path_summary(Z, y)
            shap_imp = gbm_shap_importance(Z, y)
            r2 = gbm_cv_r2(Z, y)
        print(f"\n=== {grade}/{group} ret_{h}m: {len(sub)} rows, gbm_cv_r2={r2:.3f} ===")
        print("lasso stability top8:", stab.round(2).head(8).to_dict())
        print("gbm shap top5:", shap_imp.head(5)["feature"].tolist())
        long_rows = (
            [("lasso_stability", f, v) for f, v in stab.items()]
            + [("lasso_path", r.feature, r.coefficient) for r in path.itertuples()]
            + [("gbm_shap", r.feature, r.mean_abs_shap) for r in shap_imp.itertuples()]
            + [("gbm_cv_r2", "__r2__", r2)]
        )
        rows += [
            {"grade": grade, "group": group, "horizon": h, "method": m, "feature": f, "value": v}
            for m, f, v in long_rows
        ]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
