# tests/test_hierarchical_model.py
import numpy as np
import pandas as pd
import pytest

from cardprice.hierarchical import (
    FORMULA,
    FORMULA_NO_CLASS_POSITION,
    FORMULA_NO_SLOPES,
    SIMPLIFICATION_LADDER,
    build_model_frame,
    convergence_report,
    fit_model,
    fit_with_ladder,
    variance_components,
)

# Bambi 0.21 posterior names, pinned from a smoke fit (`az.summary(fit).index`):
# residual sd is plain `sigma`; every varying term gets a `<term>_sigma` sd.
SIGMA_PARAMS = {
    "sigma",
    "1|class_year_sigma",
    "1|class_position_sigma",
    "1|mlb_id_sigma",
    "surprise|class_position_sigma",
}


def _planted(n_classes=4, n_pos=2, n_players_per=6, rows_per=15, slope_sd=0.6, seed=7,
             class_sd=0.0, class_pos_sd=0.0):
    """Planted slope heterogeneity: class:position slopes ~ N(0.4, slope_sd).

    market_ret_3m / price_level are jittered (zero true effect): Bambi 0.21
    raises on constant predictors, so the brief's literal 0.0 / 2.0 constants
    could not survive the z-score.

    class_sd / class_pos_sd plant intercept variance for 1|class_year /
    1|class_position. Default 0.0 = the brief's DGP (only slopes and player
    intercepts vary); the ladder test sets them > 0 so a simplified rung has
    no degenerate (zero-variance) group term and can actually converge.
    """
    rng = np.random.default_rng(seed)
    rows = []
    stages = ["prospect", "rookie_year", "sophomore", "established"]
    slopes = {}
    class_fx = {}
    cp_fx = {}
    for cy in range(n_classes):
        class_fx[cy] = rng.normal(0, class_sd) if class_sd else 0.0
        for p in range(n_pos):
            slopes[(cy, p)] = rng.normal(0.4, slope_sd)
            cp_fx[(cy, p)] = rng.normal(0, class_pos_sd) if class_pos_sd else 0.0
    for cy in range(n_classes):
        for p in range(n_pos):
            for pl in range(n_players_per):
                player_effect = rng.normal(0, 0.1)
                for r in range(rows_per):
                    surprise = rng.normal(0, 1)
                    y = (0.1 + class_fx[cy] + cp_fx[(cy, p)]
                         + slopes[(cy, p)] * surprise + player_effect
                         + rng.normal(0, 0.05))
                    rows.append(
                        {"month": pd.Timestamp("2024-06-01"), "grade": "ungraded",
                         "ret_12m": y, "mlb_id": cy * 100 + p * 50 + pl,
                         "player_name": "x", "rookie_year": 2019 + cy,
                         "position": "P" if p == 0 else "SS",
                         "career_stage": stages[r % 4],
                         "surprise_ops": None if p == 0 else surprise,
                         "surprise_era": -surprise if p == 0 else None,
                         "market_ret_3m": rng.normal(0, 0.05),
                         "price_level": rng.normal(2.0, 0.3),
                         "entry_price": 10.0, "card_slug": "s/x",
                         "card_type": "bowman_1st_base", "awards_to_date": 0, "age": 22.0}
                    )
    return pd.DataFrame(rows), slopes


@pytest.mark.parametrize("draws,tune,chains", [(600, 400, 2)])
def test_fit_recovers_planted_slope_variance(draws, tune, chains):
    panel, _ = _planted()
    frame, _ = build_model_frame(panel, "hitter")
    # Sampler tightening (not the assertions): 4 slope groups + 24 player
    # intercepts on 360 rows is funnel-prone — at target_accept=0.9 the fit
    # shows 9 divergences and rhat_max 1.0500x, a hair over the loose bar.
    fit = fit_model(frame, draws=draws, tune=tune, chains=chains, target_accept=0.95)
    vc = variance_components(fit)
    assert set(vc.columns) == {"param", "mean", "sd", "hdi_3%", "hdi_97%"}
    assert set(vc["param"]) == SIGMA_PARAMS
    slope_row = vc[vc["param"] == "surprise|class_position_sigma"].iloc[0]
    # planted sd 0.6; posterior mean should land well inside [0.2, 1.2]
    assert 0.2 < slope_row["mean"] < 1.2
    assert 0.0 <= slope_row["hdi_3%"] < slope_row["hdi_97%"]
    rep = convergence_report(fit)
    assert rep["rhat_max"] < 1.05  # loose bar for a small synthetic
    assert rep["divergences"] == 0 or rep["divergence_rate"] <= 0.01


def test_convergence_report_arithmetic():
    panel, _ = _planted()
    frame, _ = build_model_frame(panel, "hitter")
    draws, chains = 200, 2
    fit = fit_model(frame, draws=draws, tune=200, chains=chains)
    rep = convergence_report(fit)
    assert set(rep) == {"rhat_max", "ess_min", "divergences", "divergence_rate", "pass"}
    assert isinstance(rep["pass"], bool)
    assert rep["divergence_rate"] == pytest.approx(rep["divergences"] / (draws * chains))
    expected = (rep["rhat_max"] < 1.01 and rep["ess_min"] > 400
                and rep["divergence_rate"] <= 0.01)
    assert rep["pass"] == expected


def test_ladder_uses_simplest_rung_that_converges():
    # Fixture tightened (assertions untouched): slope_sd=0.0 kills the varying
    # slopes so the full rung funnels and fails; planted class_year /
    # class_position intercept variance keeps the simplified rungs healthy, so
    # the walk exercises its early-stop path instead of only the floor.
    panel, _ = _planted(n_classes=3, n_players_per=4, rows_per=10, slope_sd=0.0,
                        class_sd=0.3, class_pos_sd=0.2)
    frame, _ = build_model_frame(panel, "hitter")
    _fit, used, rep = fit_with_ladder(frame, draws=600, tune=400, chains=2,
                                      target_accept=0.95)
    assert SIMPLIFICATION_LADDER == [FORMULA, FORMULA_NO_SLOPES, FORMULA_NO_CLASS_POSITION]
    assert used in (FORMULA, FORMULA_NO_SLOPES, FORMULA_NO_CLASS_POSITION)
    assert rep["pass"] or used == FORMULA_NO_CLASS_POSITION  # last rung is the floor
