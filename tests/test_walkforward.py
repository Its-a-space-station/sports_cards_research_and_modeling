import numpy as np
import pandas as pd

from cardprice.walkforward import gate_evaluation, walk_forward


def make_frame(n_months=14, n_cards=6, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    months = pd.date_range("2022-04-01", periods=n_months, freq="MS")
    for mi, m in enumerate(months):
        for ci in range(n_cards):
            ops = rng.normal(0.75, 0.05)
            # excess_ret strongly driven by ops -> learnable
            y = 4.0 * (ops - 0.75) + rng.normal(scale=0.02)
            rows.append(
                {
                    "card_slug": f"card/{ci}",
                    "month": m,
                    "ops": ops,
                    "games": 25.0,
                    "age": 23.0,
                    "excess_ret": y,
                    "mlb_id": ci,
                    "position_group": "IF",
                    "stats_season": 2022,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_ranks_and_lags():
    frame = make_frame()
    picks = walk_forward(frame, ["ops", "games", "age"], top_k=2, min_train=6)
    assert not picks.empty
    first_month = picks["month"].min()
    assert first_month == frame["month"].sort_values().unique()[6]  # min_train honored
    assert (picks.groupby("month").size() == 2).all()


def test_gate_passes_with_planted_signal():
    frame = make_frame()
    picks = walk_forward(frame, ["ops", "games", "age"], top_k=2, min_train=6)
    res = gate_evaluation(picks, frame, fee_log=0.14)
    assert res["n_months"] == 8
    assert res["mean_excess"] > 0
    assert res["ci_low"] > 0  # strong planted signal must clear the gate


def test_gate_fails_on_noise():
    frame = make_frame()
    rng = np.random.default_rng(99)
    frame["excess_ret"] = rng.normal(size=len(frame))  # destroy the signal
    picks = walk_forward(frame, ["ops", "games", "age"], top_k=2, min_train=6)
    res = gate_evaluation(picks, frame, fee_log=0.14)
    assert res["ci_low"] <= 0  # noise must not pass
