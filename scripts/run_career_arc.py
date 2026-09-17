# scripts/run_career_arc.py
"""Career-arc overlay + consistency analyses on the multi-year panel.

For grade=ungraded (primary) and psa_10 (robustness), horizons 12m/36m:
stage effects (OLS, HC1), descriptive years-since-rookie curve, and the Plan-3
consistency battery (season_consistency / position_interaction_test /
player_random_effects) run on the aliased adapter frame with top_stat = the top
LASSO-stability feature from Task 2's real run (hold_model_importance.csv).
Note: the adapter's `excess_ret` is the ABSOLUTE hold return here, not
market-relative. career_stage has no prospect rows (structurally empty stage).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.career_arc import (
    consistency_adapter,
    player_random_effects,
    position_interaction_test,
    season_consistency,
    stage_effects,
    years_curve,
)
from cardprice.hold_model import build_hold_frame

HORIZONS = (12, 36)
MIN_STABILITY = 0.6
FALLBACK_STAT = "career_ops"
IMPORTANCE_CSV = "data/processed/hold_model_importance.csv"


def pick_top_stat(grade: str, horizon: int) -> tuple[str, float, bool]:
    """Top lasso_stability feature for grade/horizon; ties keep Task 2's CSV order.

    Returns (feature, stability, used_fallback). Falls back to career_ops when no
    feature reached MIN_STABILITY.
    """
    imp = pd.read_csv(IMPORTANCE_CSV)
    sel = imp[
        (imp["grade"] == grade) & (imp["horizon"] == horizon) & (imp["method"] == "lasso_stability")
    ]
    top = sel.sort_values("value", ascending=False, kind="stable").iloc[0]
    if top["value"] >= MIN_STABILITY:
        return str(top["feature"]), float(top["value"]), False
    return FALLBACK_STAT, float(top["value"]), True


def main() -> None:
    panel = pd.read_parquet("data/processed/panel_multiyear.parquet")
    for grade in ("ungraded", "psa_10"):
        sub = panel[panel["grade"] == grade]
        for h in HORIZONS:
            frame = build_hold_frame(sub, h)
            top_stat, stab, fell_back = pick_top_stat(grade, h)
            why = (
                f"top lasso_stability feature (stability={stab:.3f})"
                if not fell_back
                else (
                    f"fallback {FALLBACK_STAT}: no feature reached stability "
                    f">= {MIN_STABILITY} (best was {stab:.3f})"
                )
            )
            print(f"\n{'=' * 72}")
            print(f"grade={grade}  horizon={h}m  n={len(frame)}  top_stat={top_stat} — {why}")
            print(f"{'=' * 72}")
            print(f"career_stage counts: {frame['career_stage'].value_counts().to_dict()}")

            print(f"\n-- stage_effects (ref=rookie_year, HC1), ret_{h}m --")
            print(stage_effects(frame, h).round(4).to_string(index=False))

            print(f"\n-- years_curve, ret_{h}m --")
            print(years_curve(frame, h).round(4).to_string(index=False))

            adapter = consistency_adapter(frame, h)

            print(f"\n-- season_consistency(stat={top_stat}) --")
            print(season_consistency(adapter, top_stat).round(4).to_string(index=False))

            print(f"\n-- position_interaction_test(stat={top_stat}) --")
            print(position_interaction_test(adapter, top_stat))

            print(f"\n-- player_random_effects(stat={top_stat}) --")
            print(player_random_effects(adapter, top_stat))


if __name__ == "__main__":
    main()
