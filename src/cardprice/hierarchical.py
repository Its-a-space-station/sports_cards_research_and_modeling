# src/cardprice/hierarchical.py
"""Hierarchical Bayesian consistency model: data prep + Bambi model wrapper.

Answers "which attributes matter, and are the effects consistent across draft
classes and positions?" with partial pooling: varying intercepts
(player ⊂ class:position ⊂ class_year) and varying surprise slopes by
class:position. `build_model_frame` prepares the frame; `fit_model` wraps the
Bambi NUTS fit, `convergence_report` is the scripted gate, and
`fit_with_ladder` walks SIMPLIFICATION_LADDER until a rung converges.

Missing-data behavior (Task 3 callers: read before judging row counts):
- NaN predictors (e.g. market_ret_3m, price_level) propagate through the
  z-score by design — no imputation. `fit_model` passes `dropna=True`, so
  Bambi listwise-deletes those rows at fit time and the fitted frame can be
  smaller than the frame handed over.
- NaN rookie_year / mlb_id (float-dtype NaN) make `build_model_frame` raise
  via `.astype(int)`; the caller must drop such rows first. This is kept as a
  fail-fast signal — a NaN-tolerant cast would silently pool a bogus "<NA>"
  group level into the partial pooling.
"""

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

FORMULA = (
    "y ~ surprise + career_stage + market_ret_3m + price_level "
    "+ (1|class_year) + (1|class_position) + (1|mlb_id) + (0 + surprise|class_position)"
)
FORMULA_NO_SLOPES = (
    "y ~ surprise + career_stage + market_ret_3m + price_level "
    "+ (1|class_year) + (1|class_position) + (1|mlb_id)"
)
FORMULA_NO_CLASS_POSITION = (
    "y ~ surprise + career_stage + market_ret_3m + price_level "
    "+ (1|class_year) + (1|mlb_id)"
)
SIMPLIFICATION_LADDER = [FORMULA, FORMULA_NO_SLOPES, FORMULA_NO_CLASS_POSITION]

# Declared prior interface: "common" covers the intercept + every fixed
# effect; "group" covers the sd (sigma hyperprior) of every varying term.
PRIORS = {"common": bmb.Prior("Normal", mu=0, sigma=1),
          "group": bmb.Prior("HalfNormal", sigma=1)}

PREDICTORS = ("surprise", "market_ret_3m", "price_level")
STAGES = ["prospect", "rookie_year", "sophomore", "established"]


def build_model_frame(panel: pd.DataFrame, group: str) -> tuple[pd.DataFrame, dict]:
    """(frame, scaling_params) for the hierarchical fit.

    Filters: ungraded, ret_12m non-null, group filter, surprise non-null
    (hitters: surprise_ops; pitchers: surprise := -surprise_era, the sign
    convention where positive = better-than-projection). Predictors z-scored
    (ddof=0); y unscaled. No other row filtering and no imputation.

    Raises via `.astype(int)` if rookie_year or mlb_id is a NaN float —
    callers must drop such rows first (see module docstring).
    """
    df = panel[(panel["grade"] == "ungraded") & panel["ret_12m"].notna()].copy()
    if group == "hitter":
        df = df[df["position"] != "P"]
        df["surprise"] = pd.to_numeric(df["surprise_ops"], errors="coerce")
    elif group == "pitcher":
        df = df[df["position"] == "P"]
        df["surprise"] = -pd.to_numeric(df["surprise_era"], errors="coerce")
    else:
        raise ValueError(f"unknown group {group!r}")
    df = df[df["surprise"].notna()]

    params = {}
    for col in PREDICTORS:
        vals = pd.to_numeric(df[col], errors="coerce").astype(float)
        mean, sd = vals.mean(), vals.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            sd = 1.0
        df[col] = (vals - mean) / sd
        params[col] = (float(mean), float(sd))

    df["y"] = df["ret_12m"].astype(float)
    df["career_stage"] = pd.Categorical(df["career_stage"], categories=STAGES)
    df["class_year"] = df["rookie_year"].astype(int).astype(str)
    df["class_position"] = df["class_year"] + "_" + df["position"].astype(str)
    df["mlb_id"] = df["mlb_id"].astype(int).astype(str)
    keep = ["y", "surprise", "career_stage", "market_ret_3m", "price_level",
            "class_year", "class_position", "mlb_id"]
    return df[keep].reset_index(drop=True), params


def fit_model(
    frame: pd.DataFrame,
    formula: str = FORMULA,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """Bambi NUTS fit; common effects Normal(0,1), group sigmas HalfNormal(1).

    Prior mapping (the module-level PRIORS dict): "common" covers the
    intercept + every fixed effect; "group" covers the sd of every varying
    term (1|class_year, 1|class_position, 1|mlb_id,
    surprise|class_position). Bambi 0.21 has no "group" priors key (it is
    ignored with a warning); the generic slot is "group_specific", which
    takes the full group-coefficient distribution — so the declared
    HalfNormal(1) lands on its sigma hyperprior. The residual sigma keeps
    Bambi's default HalfStudentT prior.

    `dropna=True`: rows with NaN in any model column are listwise-deleted
    by Bambi (default dropna=False would raise instead), so the fit can use
    fewer rows than `len(frame)`.
    """
    priors = {
        "common": PRIORS["common"],
        "group_specific": bmb.Prior("Normal", mu=0, sigma=PRIORS["group"]),
    }
    model = bmb.Model(formula, frame, priors=priors, dropna=True)
    return model.fit(draws=draws, tune=tune, chains=chains,
                     target_accept=target_accept, random_seed=random_seed,
                     progressbar=False)


def convergence_report(fit: az.InferenceData, max_divergence_rate: float = 0.01) -> dict:
    """Scripted gate: rhat<1.01 and ess>400 everywhere; divergences <= 1%."""
    summ = az.summary(fit)
    rhat_max = float(summ["r_hat"].max())
    ess_min = float(summ["ess_bulk"].min())
    div = int(fit.sample_stats.diverging.sum())
    total = int(fit.sample_stats.sizes["chain"] * fit.sample_stats.sizes["draw"])
    rate = div / total if total else 0.0
    return {"rhat_max": rhat_max, "ess_min": ess_min, "divergences": div,
            "divergence_rate": float(rate),
            "pass": bool(rhat_max < 1.01 and ess_min > 400 and rate <= max_divergence_rate)}


def fit_with_ladder(frame: pd.DataFrame, **fit_kwargs) -> tuple[az.InferenceData, str, dict]:
    """Walk SIMPLIFICATION_LADDER; stop at the first rung that converges.

    Priors are never adjusted — simplification is structural only. If even
    the floor rung fails the gate, it is returned anyway with its failing
    report, so the caller sees the honest state.
    """
    last = None
    for formula in SIMPLIFICATION_LADDER:
        fit = fit_model(frame, formula=formula, **fit_kwargs)
        rep = convergence_report(fit)
        last = (fit, formula, rep)
        if rep["pass"]:
            return last
    return last  # floor rung even if it still fails — reported honestly


def variance_components(fit: az.InferenceData) -> pd.DataFrame:
    """Posterior mean + 94% HDI for sigma parameters.

    Bambi 0.21 names: residual sd is plain `sigma`; every varying term gets
    a `<term>_sigma` sd (`1|class_year_sigma`, `1|class_position_sigma`,
    `1|mlb_id_sigma`, and `surprise|class_position_sigma` when the varying
    slopes are in the formula). The `*_offset` non-centered auxiliaries are
    not in the posterior group, so no exclusion is needed.
    """
    sigma_vars = [v for v in fit.posterior.data_vars if v == "sigma" or v.endswith("_sigma")]
    summ = az.summary(fit, var_names=sigma_vars, ci_prob=0.94, ci_kind="hdi")
    rows = [{"param": name, "mean": float(r["mean"]), "sd": float(r["sd"]),
             "hdi_3%": float(r["hdi94_lb"]), "hdi_97%": float(r["hdi94_ub"])}
            for name, r in summ.iterrows()]
    return pd.DataFrame(rows)
