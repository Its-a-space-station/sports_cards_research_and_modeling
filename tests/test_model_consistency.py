import numpy as np
import pandas as pd

from cardprice.model_consistency import (
    player_random_effects,
    position_interaction_test,
    season_consistency,
)


def make_frame(n_per=40, seed=7, season_slopes=(0.5, 0.5, 0.5), pos_effect_if=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for si, season in enumerate([2023, 2024, 2025]):
        for pi, pos in enumerate(["IF", "OF"]):
            for _ in range(n_per):
                ops = rng.normal(0.75, 0.05)
                y = (
                    season_slopes[si] * (ops - 0.75)
                    + (pos_effect_if if pos == "IF" else 0.0) * (ops - 0.75)
                    + rng.normal(scale=0.1)
                )
                rows.append(
                    {
                        "excess_ret": y,
                        "ops": ops,
                        "games": 30.0,
                        "age": 23.0,
                        "stats_season": season,
                        "position_group": pos,
                        "mlb_id": hash((_, season)) % 50,
                    }
                )
    return pd.DataFrame(rows)


def test_season_consistency_stable_slopes():
    df = make_frame()
    out = season_consistency(df, "ops")
    assert len(out) == 3
    assert (out["coef"] > 0.2).all()  # sign/magnitude stable across seasons


def test_position_interaction_detects_none_when_absent():
    df = make_frame()
    res = position_interaction_test(df, "ops")
    assert res["p_value"] > 0.05  # no planted interaction -> not significant


def test_position_interaction_detects_present():
    df = make_frame(pos_effect_if=2.0)
    res = position_interaction_test(df, "ops")
    assert res["p_value"] < 0.01


def test_player_random_effects_keys():
    df = make_frame()
    res = player_random_effects(df, "ops")
    assert set(res) == {"random_intercept_var", "resid_var", "icc"}
    assert 0 <= res["icc"] <= 1
