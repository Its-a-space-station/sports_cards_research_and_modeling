# tests/test_odds_delta_fit.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_odds_delta_fit import compare_feature_sets


def _frame(seed=11, n=30):
    rng = np.random.default_rng(seed)
    rows = []
    for y in (2021, 2022, 2023, 2024, 2025):
        for c in range(n):
            s = rng.normal(0, 1)
            rows.append(
                {"entry_year": y, "entry_month": pd.Timestamp(f"{y}-06-01"), "mlb_id": c,
                 "card_slug": f"c{c}", "ret_12m": 0.5 * s + rng.normal(0, 0.15),
                 "f1": rng.normal(0, 1), "f2": rng.normal(0, 1),
                 "has_market": int(c % 3 == 0),
                 "odds_level": s * 0.5 * (c % 3 == 0),
                 "odds_delta_7d": rng.normal(0, 0.05) * (c % 3 == 0),
                 "odds_delta_30d": rng.normal(0, 0.1) * (c % 3 == 0)}
            )
    return pd.DataFrame(rows)


def test_compare_feature_sets_output_contract():
    out = compare_feature_sets(_frame(), ["f1", "f2"],
                               ["f1", "f2", "has_market", "odds_level",
                                "odds_delta_7d", "odds_delta_30d"], horizon=12)
    assert set(out) == {"baseline", "with_odds", "delta"}
    for k in ("baseline", "with_odds"):
        assert set(out[k]["gate"]) == {"mean_excess", "ci_low", "ci_high", "net_mean",
                                       "net_ci_low", "n_years", "verdict"}
        assert set(out[k]["spearman_by_year"]) == {2023, 2024, 2025}
    # planted signal inside odds_level should not hurt the with_odds fit
    assert out["delta"]["mean_excess"] > -0.5
