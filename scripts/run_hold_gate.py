# scripts/run_hold_gate.py
"""Year-grain walk-forward gate on the multi-year panel.

Per grade (ungraded first — 12m ungraded is THE pre-registered gate; psa_10 and
the 6/24/36m horizons are descriptive only): build_hold_frame →
walk_forward_years (top-5 picks per entry year) → gate_evaluation_years
(block bootstrap over years, net of 0.14 log fees). Prints every pick row and
the gate dicts. Prediction years with < 5 test cards or LassoCV convergence
issues are reported, not hidden.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.hold_model import HOLD_FEATURES, build_hold_frame
from cardprice.walkforward_holds import gate_evaluation_years, walk_forward_years

HORIZONS = (12, 6, 24, 36)
TOP_K = 5


def main() -> None:
    panel = pd.read_parquet("data/processed/panel_multiyear.parquet")
    for grade in ("ungraded", "psa_10"):
        sub = panel[panel["grade"] == grade]
        for h in HORIZONS:
            frame = build_hold_frame(sub, h)
            role = "PRE-REGISTERED GATE" if (grade == "ungraded" and h == 12) else "descriptive"
            print(f"\n{'=' * 72}")
            print(f"grade={grade}  horizon={h}m  n={len(frame)}  [{role}]")
            print(f"{'=' * 72}")
            per_year = frame.groupby("entry_year").size()
            print(f"test cards per entry_year: {per_year.to_dict()}")
            for y, n in per_year.items():
                if n < TOP_K:
                    print(f"WARNING: entry_year {y} has {n} < {TOP_K} test cards")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                picks = walk_forward_years(frame, HOLD_FEATURES, h, top_k=TOP_K)
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    print(f"WARNING: LassoCV convergence: {w.message}")
            print("\n-- picks --")
            print(picks.round(4).to_string(index=False))
            yearly = picks.groupby("entry_year")["realized"].mean()
            print("\n-- yearly mean excess --")
            print(yearly.round(4).to_string())
            gate = gate_evaluation_years(picks)
            print("\n-- gate (block bootstrap over years, fee=0.14) --")
            for k, v in gate.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
