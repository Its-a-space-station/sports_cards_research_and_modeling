# T4 — Hierarchical Bayesian Consistency Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the project's founding question with partial pooling instead of a binary: *which attributes matter for 12m raw card returns, and are the effects consistent across draft classes and positions?* — via a hierarchical Bayesian model (varying intercepts player ⊂ class:position ⊂ class; varying surprise slopes by class:position) with LOO comparison against the pooled LASSO and a player-fixed-effects null.

**Architecture:** New pure data-prep functions + a thin model module in `src/cardprice/hierarchical.py` (Bambi formula interface over PyMC; NUTS). Runner under `scripts/`. No changes to reviewed files.

**Tech Stack (spike-verified 2026-09-19 on Python 3.14.3):** `bambi==0.21.0`, `pymc==6.3.2`, `arviz==1.3.0` (with `pytensor==3.3.2`), on pandas 2.x / numpy 2.x from the worktree venv. **Dependency addition (spec-sanctioned):** these three are added to `[project.optional-dependencies].dev` pinned as shown — the only proper tool for partial pooling; the spike confirmed clean install + NUTS sampling on 3.14.

**Spec:** `docs/superpowers/specs/2026-09-17-t4-hierarchical-bayes-design.md`

## Pre-registered methodological gate (binding, per spec §4)

> **Convergence gate:** R̂ < 1.01 and ESS_bulk > 400 for every parameter; divergences ≤ 1 % of post-warmup draws. Pathological sampling → simplify structure in this exact ladder: (1) drop the varying slopes `(0 + surprise | class:position)`; (2) drop the `class:position` intercept level. **Never** tune priors to force convergence. **Decisional read (registered in advance):** surprise-slope variance ≈ 0 with a tight posterior → effects pool → future models simplify (a real, useful null); large, well-identified slope variance → position/class-specific modeling is justified. The read is reported for BOTH outcomes with the posterior variance components quoted.

## Global Constraints

- Ruff line-length **100**; `.venv/bin/ruff check src tests scripts` before every commit; **never** `ruff format`.
- Honest-golden discipline: goldens verified by hand/recomputation; a mismatch stops the run.
- **No look-ahead:** predictors are the already-vetted entry-month features (surprise/league/Marcel discipline from earlier threads); the model frame only z-scores them — no new information channels.
- Convergence checks are scripted and **fail the run loudly** (non-zero exit) — never a findings-doc afterthought.
- Every number in the findings doc traces to saved artifacts (`data/processed/hierarchical_*.csv` / `*.nc`, run logs) — no remembered numbers.
- Fits are seeded (`random_seed=42`); tests deterministic and offline (small synthetic frames only — no real-data fits in tests).
- The hierarchical fits are **descriptive inference** (consistency structure), not a trading gate; findings labeled as such.

## Data facts (measured 2026-09-19, panel_class.parquet)

- Target: `ret_12m`, **ungraded** rows only (consistent with every earlier thread).
- Model frame A (**hitters** primary): `position != "P"`, `surprise_ops` non-null → ~7.3k rows. Model frame B (**pitchers** secondary): `position == "P"`, `surprise_era_flipped := −surprise_era` non-null → ~3.7k rows.
- Group keys: `class_year` (panel's `rookie_year` column carries class year — 2015–2025, 11 levels); `class:position` (≤ ~30 cells); `player` = `mlb_id`.
- Predictors (z-scored, existing conventions): `surprise` (the varying-slope term), `career_stage` (categorical: prospect/rookie_year/sophomore/established), `market_ret_3m`, `price_level`.
- Surprise coverage is structural (post-debut, in-season, with ≥1 prior MLB season) — the findings doc must state that the consistency read applies to the *covered population*, not the whole universe.

---

### Task 1: Data prep + dependency pin (`src/cardprice/hierarchical.py` part 1)

**Files:**
- Modify: `pyproject.toml` (dev extra: add `bambi==0.21.0`, `pymc==6.3.2`, `arviz==1.3.0`)
- Create: `src/cardprice/hierarchical.py`
- Test: `tests/test_hierarchical_prep.py`

**Interfaces:**
- Consumes: `panel_class.parquet` (frame input), the conventions above.
- Produces (used by Tasks 2–4):
  - `build_model_frame(panel: pd.DataFrame, group: str) -> pd.DataFrame` — columns: `y` (= `ret_12m`), `surprise`, `career_stage` (category with reference `prospect`), `market_ret_3m`, `price_level`, `class_year` (str), `class_position` (str `"{class_year}_{position}"`), `mlb_id` (str). Filters: ungraded, target non-null, group filter, surprise non-null (hitters: `surprise_ops`; pitchers: `position == "P"` and `surprise := −surprise_era`). Scaling: `surprise`, `market_ret_3m`, `price_level` z-scored (ddof=0); `y` NOT scaled. Returns `(frame, scaling_params)` via tuple — scaling params dict `{col: (mean, sd)}` for back-transformation.
  - `FORMULA = "y ~ surprise + career_stage + market_ret_3m + price_level + (1|class_year) + (1|class_position) + (1|mlb_id) + (0 + surprise|class_position)"`
  - `PRIORS = {"common": bmb.Prior("Normal", mu=0, sigma=1), "group": bmb.Prior("HalfNormal", sigma=1)}` (Bambi prior spec; intercept gets Normal(0,1) via "common" too — Bambi distinguishes "common"/"group" terms; document the exact mapping used).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hierarchical_prep.py
import numpy as np
import pandas as pd
import pytest

from cardprice.hierarchical import build_model_frame


def _panel(n_per_class=20, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    stages = ["prospect", "rookie_year", "sophomore", "established"]
    for cy in (2019, 2020, 2021):
        for i in range(n_per_class):
            pos = "P" if i % 5 == 0 else "SS"
            so = rng.normal(0, 1) if pos != "P" else None
            se = rng.normal(0, 1) if pos == "P" else None
            rows.append(
                {"month": pd.Timestamp("2024-06-01"), "grade": "ungraded",
                 "ret_12m": rng.normal(0.1, 0.3), "mlb_id": cy * 100 + i,
                 "player_name": f"P{i}", "rookie_year": cy, "position": pos,
                 "career_stage": stages[i % 4], "surprise_ops": so,
                 "surprise_era": se, "market_ret_3m": rng.normal(0, 0.05),
                 "price_level": rng.normal(2, 0.3), "entry_price": 10.0,
                 "card_slug": f"s/p-{cy}-{i}", "card_type": "bowman_1st_base",
                 "awards_to_date": 0, "age": 22.0}
            )
    return pd.DataFrame(rows)


def test_hitters_frame_schema_scaling_and_sign():
    frame, params = build_model_frame(_panel(), "hitter")
    assert set(frame.columns) == {"y", "surprise", "career_stage", "market_ret_3m",
                                  "price_level", "class_year", "class_position", "mlb_id"}
    assert (frame["position" if "position" in frame else "class_position"].str.contains("SS")).all()
    # z-scored predictors
    assert frame["surprise"].mean() == pytest.approx(0, abs=1e-9)
    assert frame["surprise"].std(ddof=0) == pytest.approx(1, abs=1e-9)
    # y unscaled
    assert abs(frame["y"].mean() - 0.1) < 0.2
    # class keys
    assert set(frame["class_year"]) == {"2019", "2020", "2021"}
    assert frame["class_position"].str.contains("2020_SS").any()
    # scaling params present for back-transformation
    assert "surprise" in params and len(params["surprise"]) == 2


def test_pitchers_sign_flip_and_filter():
    frame, _ = build_model_frame(_panel(), "pitcher")
    raw = _panel()
    raw_p = raw[raw["position"] == "P"].reset_index(drop=True)
    assert len(frame) == len(raw_p)
    assert (frame["class_position"].str.contains("_P")).all()
    # surprise == -surprise_era (z-scored): verify the whole column by hand
    eras = raw_p["surprise_era"].astype(float)
    expected = ((-eras) - (-eras).mean()) / (-eras).std(ddof=0)
    got = frame.sort_values("mlb_id")["surprise"].to_numpy()
    assert got == pytest.approx(expected.to_numpy()[raw_p["mlb_id"].argsort()], abs=1e-9)


def test_target_and_grade_filters():
    panel = _panel()
    panel.loc[0, "grade"] = "psa_10"
    panel.loc[1, "ret_12m"] = None
    frame, _ = build_model_frame(panel, "hitter")
    assert len(frame) == len(panel[panel["position"] != "P"]) - 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hierarchical_prep.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement part 1 of `src/cardprice/hierarchical.py`**

```python
# src/cardprice/hierarchical.py
"""Hierarchical Bayesian consistency model: data prep + Bambi model wrapper.

Answers "which attributes matter, and are the effects consistent across draft
classes and positions?" with partial pooling: varying intercepts
(player ⊂ class:position ⊂ class_year) and varying surprise slopes by
class:position. Pure prep functions here; the model wrapper is Task 2.
"""

import numpy as np
import pandas as pd

FORMULA = (
    "y ~ surprise + career_stage + market_ret_3m + price_level "
    "+ (1|class_year) + (1|class_position) + (1|mlb_id) + (0 + surprise|class_position)"
)
PREDICTORS = ("surprise", "market_ret_3m", "price_level")
STAGES = ["prospect", "rookie_year", "sophomore", "established"]


def build_model_frame(panel: pd.DataFrame, group: str) -> tuple[pd.DataFrame, dict]:
    """(frame, scaling_params) for the hierarchical fit.

    Filters: ungraded, ret_12m non-null, group filter, surprise non-null
    (hitters: surprise_ops; pitchers: surprise := -surprise_era, the sign
    convention where positive = better-than-projection). Predictors z-scored
    (ddof=0); y unscaled. No other row filtering and no imputation.
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
```

- [ ] **Step 4: Run tests to verify they pass; install the pinned dev deps**

Run: `pytest tests/test_hierarchical_prep.py -v` → green; then `.venv/bin/pip install -q -e ".[dev]"` (picks up the pinned bambi/pymc/arviz); smoke-import: `.venv/bin/python -c "import bambi, pymc, arviz; print(bambi.__version__, pymc.__version__, arviz.__version__)"` → `0.21.0 6.3.2 1.3.0`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/cardprice/hierarchical.py tests/test_hierarchical_prep.py
git commit -m "feat: hierarchical model data prep + pinned bayes deps (bambi/pymc/arviz, spike-verified)"
```

---

### Task 2: Model wrapper + convergence gate (`hierarchical.py` part 2)

**Files:**
- Modify: `src/cardprice/hierarchical.py`
- Test: `tests/test_hierarchical_model.py`

**Interfaces:**
- Produces (used by Tasks 3–4):
  - `fit_model(frame: pd.DataFrame, formula: str = FORMULA, draws: int = 1000, tune: int = 1000, chains: int = 4, target_accept: float = 0.9, random_seed: int = 42) -> az.InferenceData`
  - `convergence_report(fit: az.InferenceData, max_divergence_rate: float = 0.01) -> dict` — `{"rhat_max": float, "ess_min": float, "divergences": int, "divergence_rate": float, "pass": bool}`; PASS iff rhat_max < 1.01 and ess_min > 400 and divergence_rate ≤ max_divergence_rate.
  - `SIMPLIFICATION_LADDER = [FORMULA, FORMULA_NO_SLOPES, FORMULA_NO_CLASS_POSITION]` with `FORMULA_NO_SLOPES = "y ~ surprise + career_stage + market_ret_3m + price_level + (1|class_year) + (1|class_position) + (1|mlb_id)"` and `FORMULA_NO_CLASS_POSITION = "y ~ surprise + career_stage + market_ret_3m + price_level + (1|class_year) + (1|mlb_id)"`.
  - `fit_with_ladder(frame, ...) -> tuple[az.InferenceData, str, dict]` — walk the ladder: fit FORMULA; if `convergence_report(...)` fails, refit the next rung; return (fit, formula_used, report_of_that_fit). **Never** adjust priors in the ladder.
  - `variance_components(fit) -> pd.DataFrame` — posterior mean + 94% HDI for every `sigma`-suffixed parameter (`1|class_year_sigma`, `1|class_position_sigma`, `1|mlb_id_sigma`, `surprise|class_position_sigma` when present) + `sigma` (residual).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hierarchical_model.py
import numpy as np
import pandas as pd
import pytest

from cardprice.hierarchical import (
    FORMULA,
    FORMULA_NO_CLASS_POSITION,
    FORMULA_NO_SLOPES,
    build_model_frame,
    convergence_report,
    fit_model,
    fit_with_ladder,
    variance_components,
)


def _planted(n_classes=4, n_pos=2, n_players_per=6, rows_per=15, slope_sd=0.6, seed=7):
    """Planted slope heterogeneity: class:position slopes ~ N(0.4, slope_sd)."""
    rng = np.random.default_rng(seed)
    rows = []
    stages = ["prospect", "rookie_year", "sophomore", "established"]
    slopes = {}
    for cy in range(n_classes):
        for p in range(n_pos):
            slopes[(cy, p)] = rng.normal(0.4, slope_sd)
    for cy in range(n_classes):
        for p in range(n_pos):
            for pl in range(n_players_per):
                player_effect = rng.normal(0, 0.1)
                for r in range(rows_per):
                    surprise = rng.normal(0, 1)
                    y = (0.1 + slopes[(cy, p)] * surprise + player_effect
                         + rng.normal(0, 0.05))
                    rows.append(
                        {"month": pd.Timestamp("2024-06-01"), "grade": "ungraded",
                         "ret_12m": y, "mlb_id": cy * 100 + p * 50 + pl,
                         "player_name": "x", "rookie_year": 2019 + cy,
                         "position": "P" if p == 0 else "SS",
                         "career_stage": stages[r % 4],
                         "surprise_ops": None if p == 0 else surprise,
                         "surprise_era": -surprise if p == 0 else None,
                         "market_ret_3m": 0.0, "price_level": 2.0,
                         "entry_price": 10.0, "card_slug": "s/x",
                         "card_type": "bowman_1st_base", "awards_to_date": 0, "age": 22.0}
                    )
    return pd.DataFrame(rows), slopes


@pytest.mark.parametrize("draws,tune,chains", [(400, 400, 2)])
def test_fit_recovers_planted_slope_variance(draws, tune, chains):
    panel, _ = _planted()
    frame, _ = build_model_frame(panel, "hitter")
    fit = fit_model(frame, draws=draws, tune=tune, chains=chains)
    vc = variance_components(fit)
    slope_row = vc[vc["param"] == "surprise|class_position_sigma"].iloc[0]
    # planted sd 0.6; posterior mean should land well inside [0.2, 1.2]
    assert 0.2 < slope_row["mean"] < 1.2
    rep = convergence_report(fit)
    assert rep["rhat_max"] < 1.05  # loose bar for a small synthetic
    assert rep["divergences"] == 0 or rep["divergence_rate"] <= 0.01


def test_convergence_report_arithmetic():
    panel, _ = _planted()
    frame, _ = build_model_frame(panel, "hitter")
    fit = fit_model(frame, draws=200, tune=200, chains=2)
    rep = convergence_report(fit)
    assert set(rep) == {"rhat_max", "ess_min", "divergences", "divergence_rate", "pass"}
    assert isinstance(rep["pass"], bool)


def test_ladder_uses_simplest_rung_that_converges():
    panel, _ = _planted(slope_sd=0.0)  # no heterogeneity -> full formula may struggle
    frame, _ = build_model_frame(panel, "hitter")
    fit, used, rep = fit_with_ladder(frame, draws=300, tune=300, chains=2)
    assert used in (FORMULA, FORMULA_NO_SLOPES, FORMULA_NO_CLASS_POSITION)
    assert rep["pass"] or used == FORMULA_NO_CLASS_POSITION  # last rung is the floor
```

(Implementer note: runtime is the constraint — keep synthetic frames tiny and draws/tune small; mark any test exceeding ~5 min for trimming. If the small-synthetic ladder test is flaky by construction, tighten the fixture (slope_sd=0.0 makes the first rung slow) rather than the assertions, and report what you changed.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hierarchical_model.py -v`
Expected: FAIL — import errors for missing functions.

- [ ] **Step 3: Implement part 2 of `src/cardprice/hierarchical.py`**

```python
import arviz as az
import bambi as bmb

FORMULA_NO_SLOPES = (
    "y ~ surprise + career_stage + market_ret_3m + price_level "
    "+ (1|class_year) + (1|class_position) + (1|mlb_id)"
)
FORMULA_NO_CLASS_POSITION = (
    "y ~ surprise + career_stage + market_ret_3m + price_level "
    "+ (1|class_year) + (1|mlb_id)"
)
SIMPLIFICATION_LADDER = [FORMULA, FORMULA_NO_SLOPES, FORMULA_NO_CLASS_POSITION]


def fit_model(
    frame: pd.DataFrame,
    formula: str = FORMULA,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """Bambi NUTS fit; common effects Normal(0,1), group sigmas HalfNormal(1)."""
    priors = {"common": bmb.Prior("Normal", mu=0, sigma=1),
              "group": bmb.Prior("HalfNormal", sigma=1)}
    model = bmb.Model(formula, frame, priors=priors)
    return model.fit(draws=draws, tune=tune, chains=chains,
                     target_accept=target_accept, random_seed=random_seed,
                     progressbar=False)


def convergence_report(fit: az.InferenceData, max_divergence_rate: float = 0.01) -> dict:
    """Scripted gate: rhat<1.01 and ess>400 everywhere; divergences <= 1%."""
    summ = az.summary(fit)
    rhat_max = float(summ["r_hat"].max())
    ess_min = float(summ["ess_bulk"].min())
    div = int(fit.sample_stats.diverging.sum())
    total = int(fit.posterior.sizes["chain"] * fit.posterior.sizes["draw"] * fit.posterior.sizes.get("chain", 1))
    total = int(fit.sample_stats.sizes["chain"] * fit.sample_stats.sizes["draw"])
    rate = div / total if total else 0.0
    return {"rhat_max": rhat_max, "ess_min": ess_min, "divergences": div,
            "divergence_rate": float(rate), "pass": bool(rhat_max < 1.01 and ess_min > 400 and rate <= max_divergence_rate)}


def fit_with_ladder(frame: pd.DataFrame, **fit_kwargs):
    """Walk SIMPLIFICATION_LADDER; stop at the first rung that converges.
    Priors are never adjusted — simplification is structural only."""
    last = None
    for formula in SIMPLIFICATION_LADDER:
        fit = fit_model(frame, formula=formula, **fit_kwargs)
        rep = convergence_report(fit)
        last = (fit, formula, rep)
        if rep["pass"]:
            return last
    return last  # floor rung even if it still fails — reported honestly


def variance_components(fit: az.InferenceData) -> pd.DataFrame:
    """Posterior mean + 94% HDI for sigma parameters."""
    summ = az.summary(fit, var_names=["~.(?!.*offset).*_sigma$"], filter_vars="regex")
    rows = [{"param": name, "mean": float(r["mean"]), "sd": float(r["sd"]),
             "hdi_3%": float(r["hdi_3%"]), "hdi_97%": float(r["hdi_97%"])}
            for name, r in summ.iterrows()]
    return pd.DataFrame(rows)
```

(Note for the implementer: the `convergence_report` body above has a redundant `total` line — keep the second assignment only (the first is dead); the regex in `variance_components` is a starting point — adapt to Bambi's actual parameter naming via `az.summary(fit).index` and pin the chosen names in the variance-components test; `~` for the residual sigma is included in the frame.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hierarchical_model.py -v` → green (may take several minutes); full `pytest` green; `ruff check src tests scripts` clean.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/hierarchical.py tests/test_hierarchical_model.py
git commit -m "feat: hierarchical model wrapper + scripted convergence ladder"
```

---

### Task 3: Real fits + baselines + LOO comparison (`scripts/run_hierarchical.py`)

**Files:**
- Create: `scripts/run_hierarchical.py`
- Test: `tests/test_hierarchical_baselines.py` (synthetic, small)

**Interfaces:**
- Consumes: Task 1's frame builder, Task 2's wrapper; `panel_class.parquet`.
- Produces:
  - `pooled_lasso_lpd(frame: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series, sigma: float) -> float` — mean Gaussian log predictive density of the pooled-LASSO out-of-fold predictions: `-0.5*log(2*pi*sigma^2) - (y-pred)^2/(2*sigma^2)`, sigma = OOF residual std (ddof=0). Documented as an ELPD approximation for baseline comparison.
  - `player_fe_null_lpd(frame) -> float` — same metric with player-mean predictions (out-of-fold player means; unseen player → global mean).
  - `loo_comparison(fit, frame) -> dict` — `{"hierarchical_elpd": float (arviz.loo), "lasso_elpd": float, "player_fe_elpd": float, "winner": "hierarchical"|"lasso"|"player_fe"}` by highest ELPD.
  - Runner outputs (artifacts): `data/processed/hierarchical_variance_components_{group}.csv`, `hierarchical_convergence_{group}.json`, `hierarchical_loo_{group}.json`, `hierarchical_fit_{group}.nc` (arviz `to_netcdf`), `hierarchical_slopes_{group}.csv` (per class_position shrunken surprise-slope posterior means + HDIs).
- Baseline LASSO definition (mirrors the registered harness's behavior): train-median imputation + z-scoring + LassoCV(cv=5), 5-fold out-of-fold predictions on the SAME model frame (features: surprise, market_ret_3m, price_level, career_stage dummies). Documented.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hierarchical_baselines.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_hierarchical import player_fe_null_lpd, pooled_lasso_lpd


def test_pooled_lasso_lpd_gaussian_math():
    y = pd.Series([0.0, 1.0, 2.0])
    pred = pd.Series([0.0, 1.0, 2.0])
    sigma = 1.0
    expected = float(np.mean([-0.5 * np.log(2 * np.pi * 1.0) - 0.0] * 3))
    assert pooled_lasso_lpd(pd.DataFrame(), y, pred, sigma) == pytest.approx(expected)


def test_player_fe_null_uses_oof_player_means():
    frame = pd.DataFrame(
        {"mlb_id": ["a", "a", "b", "b", "c"], "y": [1.0, 3.0, 0.0, 2.0, 5.0],
         "fold": [0, 1, 0, 1, 0]}
    )
    # player means computed out-of-fold; unseen player "c" -> global mean
    lpd = player_fe_null_lpd(frame, group_col="mlb_id", y_col="y", fold_col="fold")
    assert isinstance(lpd, float)
    assert np.isfinite(lpd)
    # in-sample (wrong) player means would give a higher LPD; OOF is lower/honest
    assert lpd < -0.5 * np.log(2 * np.pi * 0.5)
```

- [ ] **Step 2: Implement + test**

Implement the runner with the metric functions importable (per the test). Run tests green.

- [ ] **Step 3: Real fits (controller-adjacent, may run inside the agent if ≤ 90 min)**

Run: `.venv/bin/python scripts/run_hierarchical.py --group hitter` then `--group pitcher` (each fit: 4 chains × 1000/1000 on ~7.3k / ~3.7k rows — expected 20–60 min per fit on this machine; if a fit exceeds 90 min, stop at the ladder rung reached and report state — do NOT let it run away). Acceptance: convergence JSON written per group (pass or an honest ladder-rung report); variance components + slopes CSVs; LOO JSON with all three ELPDs. If convergence fails on the floor rung for either group, the runner reports it verbatim and continues to the other group (no silent success).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_hierarchical.py tests/test_hierarchical_baselines.py
git commit -m "feat: hierarchical real fits + baselines + LOO comparison"
```

---

### Task 4: Findings doc

**Files:**
- Create: `docs/findings/2026-09-19-t4-hierarchical-consistency.md`

- [ ] **Step 1: Write the findings doc**

Sections (every number from the artifacts — variance-components CSVs, convergence JSONs, LOO JSONs, slopes CSVs):

```markdown
# T4 — Hierarchical Bayesian Consistency Model

Date: 2026-09-19
Plan: docs/superpowers/plans/2026-09-19-t4-hierarchical-bayes.md
Spec: docs/superpowers/specs/2026-09-17-t4-hierarchical-bayes-design.md

## The question
Which attributes matter for 12m raw card returns, and are the effects consistent
across draft classes and positions? (P6c answered coarsely: player identity absorbs
81–97% of residual variance. This thread quantifies the STRUCTURE of that answer.)

## Model & gate
<formula; priors; the pre-registered convergence gate quoted verbatim; ladder used
per group (rung + why); convergence report values per group>

## Consistency verdict (pre-registered decisional read)
<surprise|class_position_sigma posterior (mean + 94% HDI) per group; whether it
pools (≈0 tight) or justifies class/position-specific effects; the other variance
components for context (class, class:position intercept, player, residual)>

## Which attributes matter (global effects)
<posterior mean + 94% HDI for surprise, career_stage levels, market_ret_3m,
price_level per group, back-transformed where meaningful>

## Baselines / LOO
<hierarchical vs pooled-LASSO vs player-FE-null ELPDs per group; winner;
the Gaussian-LPD approximation caveat for the baselines>

## Shrunken slopes
<top/bottom class_position surprise-slope cells with HDIs — who deviates from the pool>

## Caveats
<surprise-covered population only (structural 7.8%/3.9%); 2-season span of surprise
features; descriptive inference not a gate; LOO baseline approximation; runtime
constraints of the ladder; pitchers secondary>

## Reproduce
<exact commands>
```

- [ ] **Step 2: Final verification + commit**

Run: `pytest && ruff check src tests scripts` → green.
```bash
git add docs/findings/2026-09-19-t4-hierarchical-consistency.md
git commit -m "docs: T4 hierarchical-consistency findings"
```

---

## Self-Review

**Spec coverage:** Bambi/PyMC choice with fallback note ✓ (env spike passed — the spec's primary path is viable); model structure per spec (varying intercepts + varying slopes by class:position, regularizing priors) ✓; LOO vs pooled LASSO + player-FE null ✓ (with the documented Gaussian-LPD approximation for non-Bayesian baselines); pre-registered convergence gate + simplification ladder + never-tune-priors ✓; decisional read both directions ✓; posterior-snapshot/vc recovery tests ✓ (planted-variance recovery + fixed-seed small fits); findings template with convergence values, variance components, baselines, shrunken slopes, caveats ✓.

**Placeholder scan:** convergence_report's dead-line is called out explicitly for the implementer; variance-components regex has an adapt-then-pin instruction (parameter naming must match Bambi's actual output — the test pins behavior, not my guessed names) — no other soft spots.

**Type consistency:** `build_model_frame` returns (frame, params) and both model/baseline consumers use the same frame ✓; ladder rungs derive from FORMULA strings with distinct names ✓; LOO JSON schema matches the Task-4 table ✓; `surprise` sign convention matches T2-model's (`-surprise_era`) ✓.
