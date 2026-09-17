# tests/test_walkforward_holds.py
import numpy as np
import pandas as pd

from cardprice.walkforward_holds import gate_evaluation_years, walk_forward_years

FEATS = ["signal", "noise"]


def make_frame(seed=5):
    rows = []
    rng = np.random.default_rng(seed)
    for year in (2021, 2022, 2023, 2024):
        for i in range(60):
            sig = rng.normal(0, 1)
            rows.append(
                {
                    "entry_year": year,
                    "card_slug": f"set/y{year}c{i}",
                    "signal": sig,
                    "noise": rng.normal(0, 1),
                    "ret_12m": 0.3 * sig + rng.normal(0, 0.02),  # planted: signal orders returns
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_picks_high_signal_cards():
    picks = walk_forward_years(make_frame(), FEATS, 12, top_k=3, min_train_years=2)
    assert set(picks["entry_year"]) == {2023, 2024}
    # planted ordering: realized excess of picks should be strongly positive on average
    assert picks["realized"].mean() > 0.2


def test_gate_block_bootstrap_and_fees():
    picks = pd.DataFrame({"entry_year": [2022, 2023, 2024], "realized": [0.30, 0.25, 0.28]})
    out = gate_evaluation_years(picks, fee=0.14)
    assert out["n_years"] == 3
    assert abs(out["mean_excess"] - np.mean([0.30, 0.25, 0.28])) < 1e-9
    assert abs(out["net_mean"] - (out["mean_excess"] - 0.14)) < 1e-9
    assert out["net_ci_low"] > 0  # tight synthetic series clears 14% fees
    assert out["verdict"] == "PASS"
    bad = picks.assign(realized=[0.05, -0.10, 0.02])
    assert gate_evaluation_years(bad, fee=0.14)["verdict"] == "FAIL"
