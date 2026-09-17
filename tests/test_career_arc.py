# tests/test_career_arc.py
import numpy as np
import pandas as pd

from cardprice.career_arc import consistency_adapter, stage_effects, years_curve


def make_frame(n=600, seed=11):
    rng = np.random.default_rng(seed)
    stage = rng.choice(["rookie_year", "sophomore", "established"], n, p=[0.3, 0.3, 0.4])
    stage_effect = np.array(
        [{"rookie_year": 0.0, "sophomore": 0.25, "established": 0.05}[s] for s in stage]
    )
    return pd.DataFrame(
        {
            "career_stage": stage,
            "age": rng.uniform(21, 33, n),
            "price_level": rng.uniform(2, 5, n),
            "market_ret_3m": rng.normal(0, 0.05, n),
            "years_since_rookie": rng.integers(0, 6, n),
            "entry_year": rng.choice([2021, 2022, 2023, 2024, 2025], n),
            "career_games": rng.integers(0, 800, n).astype(float),
            "position": rng.choice(["OF", "1B", "SS"], n),
            "mlb_id": [2000 + i % 50 for i in range(n)],
            "ret_12m": stage_effect + rng.normal(0, 0.1, n),
        }
    )


def test_stage_effects_recovers_planted_effect():
    out = stage_effects(make_frame(), 12)
    soph = out[out["term"].str.contains("sophomore")].iloc[0]
    assert 0.15 < soph["coef"] < 0.35
    assert soph["p_value"] < 0.01


def test_stage_effects_rookie_year_is_reference():
    out = stage_effects(make_frame(), 12)
    stage_terms = out[out["term"].str.contains("career_stage")]["term"].tolist()
    assert not any("rookie_year" in t for t in stage_terms)  # reference level, not a term
    assert any("sophomore" in t for t in stage_terms)
    assert any("established" in t for t in stage_terms)


def test_consistency_adapter_columns():
    ad = consistency_adapter(make_frame(20), 12)
    assert {"excess_ret", "games", "stats_season", "position_group"} <= set(ad.columns)
    assert set(ad["position_group"]) <= {"OF", "IF", "P", "C"}


def test_years_curve_shape():
    out = years_curve(make_frame(), 12)
    assert {"years_since_rookie", "median", "n"} <= set(out.columns)
    assert out["n"].sum() == 600
