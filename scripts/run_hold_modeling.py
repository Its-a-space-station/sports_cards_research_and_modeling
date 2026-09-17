# scripts/run_hold_modeling.py
"""Per-horizon importance models on the multi-year panel (ungraded primary)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.hold_model import build_hold_frame, horizon_importance

HORIZONS = (6, 12, 24, 36)


def main() -> None:
    panel = pd.read_parquet("data/processed/panel_multiyear.parquet")
    rows = []
    for grade in ("ungraded", "psa_10"):
        sub = panel[panel["grade"] == grade]
        for h in HORIZONS:
            frame = build_hold_frame(sub, h)
            out = horizon_importance(frame, h)
            print(
                f"\n=== {grade} ret_{h}m: {len(frame)} rows, gbm_cv_r2={out['gbm_cv_r2']:.3f} ==="
            )
            print("lasso stability:", out["lasso_stability"].round(2).head(8).to_dict())
            print("gbm shap top5:", out["gbm_shap"].head(5)["feature"].tolist())
            for feat, v in out["lasso_stability"].items():
                rows.append(
                    {
                        "grade": grade,
                        "horizon": h,
                        "method": "lasso_stability",
                        "feature": feat,
                        "value": v,
                    }
                )
            for r in out["gbm_shap"].itertuples():
                rows.append(
                    {
                        "grade": grade,
                        "horizon": h,
                        "method": "gbm_shap",
                        "feature": r.feature,
                        "value": r.mean_abs_shap,
                    }
                )
            rows.append(
                {
                    "grade": grade,
                    "horizon": h,
                    "method": "gbm_cv_r2",
                    "feature": "__r2__",
                    "value": out["gbm_cv_r2"],
                }
            )
    pd.DataFrame(rows).to_csv("data/processed/hold_model_importance.csv", index=False)
    print("\nwrote data/processed/hold_model_importance.csv")


if __name__ == "__main__":
    main()
