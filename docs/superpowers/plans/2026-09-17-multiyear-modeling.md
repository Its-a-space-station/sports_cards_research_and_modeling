# P6c: Multi-Year Hold Modeling + Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the research question on the multi-year panel — which entry-month attributes predict card returns over 6/12/24/36-month holds, whether effects are consistent across entry years and positions, and whether a walk-forward buy rule beats the universe median net of fees.

**Architecture:** Four thin layers on proven modules: (1) a parser-level tie-break fix + panel rebuild; (2) a hold-frame builder feeding the existing LASSO-stability / LightGBM-SHAP modules per horizon; (3) consistency + career-arc analyses adapting `model_consistency.py` by column aliasing; (4) a year-grain walk-forward gate (new module, same discipline as `walkforward.py`) with a block bootstrap over years, net of 14% fees — PASS/FAIL reported honestly, FAIL = stay research-stage.

**Tech Stack:** Python 3.11+, existing `cardprice` package (sklearn, lightgbm, shap, statsmodels). No new dependencies, no network access anywhere in this plan.

**Spec:** `docs/superpowers/specs/2026-09-16-multiyear-minors-design.md` (§6–§8)
**Predecessors:** P6a merged (980a6b2), P6b merged (b527bbd) — `data/processed/panel_multiyear.parquet` (5,553 rows / 59 cards; ungraded 2,838, psa_10 2,715).

## Facts established (from P6b artifacts + final review — verified on disk 2026-09-17)

- Panel contract: one row per (card_slug, grade, entry_month); identity cols `card_slug, grade, entry_month, mlb_id, player_name, rookie_year, card_type, position`; predictors `career_games, career_ops, career_home_runs` (hitters) / `career_era, career_k_bb_pct, career_innings_pitched` (pitchers), `max_level, max_level_rank, rate_at_max_level, minor_games, career_stage, awards_to_date, age, age_at_debut, price_level, ret_3m, market_ret_3m`; outcomes `ret_6m, ret_12m, ret_24m, ret_36m` (annualized log returns); plus `entry_price`. Non-NaN outcomes: 6m 4,869 · 12m 4,191 · 24m 2,953 · 36m 1,896; 36m entries ≤ 2023-09.
- **psa_10 chart duplicate keys (the Task 1 fix):** the chart parquet has 123 (slug, date) duplicate keys on psa_10 with differing prices (ungraded: zero). `series_month_ends`' "last within month" pick is file-order dependent on those cells (up to 51% off the month median). Fix = median-collapse duplicates per (card_slug, grade, date) before the month-end pick. Ungraded rows must be BIT-IDENTICAL after the rebuild (0 dupes) — the rebuild test asserts this against a saved pre-fix copy.
- **`prospect` career_stage is structurally EMPTY** (charts never start pre-debut). No prospect effects are estimable — do not build one, do not report one.
- Timing convention to state in the report: `ret_3m`/`market_ret_3m` include the entry month's own month-end print — **enter-at-month-end** convention (deliberate; P6b final review finding 3).
- Goldens from P6b (all must still hold after the Task 1 rebuild; ungraded ones must be byte-identical): Soto flagship ungraded ret_12m at 2021-04 = −0.2716796494005069; Bryant (592178) career_home_runs at 2021-04 = 142; Henderson (683002) career_stage at 2023-08 = sophomore; Judge (592450) awards_to_date at 2023-01 = 2; Skenes (694973) career_era non-null / career_ops null at 2024+ entries.
- Existing modules to reuse unchanged: `model_lasso.stability_selection` (1-SE rule, 200 bootstraps) + `lasso_path_summary`; `model_gbm.gbm_shap_importance` + `gbm_cv_r2`; `model_consistency.season_consistency` (groups by `stats_season`) / `position_interaction_test` (IF vs OF) / `player_random_effects`; `features.standardize`, `features.POSITION_MAP`; `walkforward.gate_evaluation` (fee_log=0.14 pattern to mirror, not reuse — its bootstrap is flat over months).
- Effective-N reality: the 12m gate has entry years 2021(Apr+)–2025(Sep) → prediction years 2023/2024/2025 = **3 yearly blocks**. The block bootstrap over 3 years is weak by construction — the report must say so (spec §7 power caveat).

## Global Constraints

- Python >= 3.11; no network access in this plan; raw snapshots untouched.
- Tests offline; lint `ruff check` / `ruff format --check` on touched files only; never repo-wide `ruff format`.
- **Ungraded is the primary series**; psa_10 appears only as robustness, and only AFTER the Task 1 tie-break fix. psa_9 stays excluded.
- **No pop columns** (single-snapshot look-ahead). No prospect-stage claims (empty stage).
- **Honest gate:** the walk-forward gate verdict is computed from the run's real output and reported as-is; FAIL is a legitimate, fully-reported outcome (Plans 4/5 precedent).
- Honest goldens: if a P6b golden changes after the Task 1 rebuild beyond psa_10-tie-break effects (i.e. any ungraded change), STOP and report.
- `data/raw/` and `data/processed/` are gitignored — commits contain code/tests/docs only; copy changed parquets back to the main checkout at merge time.

---

### Task 1: psa_10 duplicate tie-break + probe strengthening + panel rebuild

**Files:**
- Modify: `src/cardprice/multiyear.py` (`series_month_ends`)
- Modify: `tests/test_multiyear.py` (new dupe test), `tests/test_multiyear_panel.py` (strengthen probe fixture)

**Interfaces:**
- Consumes: existing `series_month_ends` contract (Task P6b-3).
- Produces (Tasks 2-4 consume): rebuilt `data/processed/panel_multiyear.parquet` with deterministic psa_10 month-ends; ungraded rows identical to the P6b panel in every column **except `market_ret_3m`** (amended during execution: the market median pools both grades per entry month, so correcting psa_10 prices legitimately moves the pooled median stamped onto ungraded rows — 1,062/2,838 rows, that column only. That drift IS the fix propagating. Every other ungraded column must be byte-identical.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_multiyear.py — append

def test_series_month_ends_collapses_duplicate_dates_by_median():
    # two prints on the same day with different prices (real psa_10 chart pattern)
    chart = pd.DataFrame(
        {
            "card_slug": ["set/card"] * 4,
            "grade": ["psa_10"] * 4,
            "date": [pd.Timestamp("2021-04-01")] * 2 + [pd.Timestamp("2021-05-01")] * 2,
            "price": [10.0, 20.0, 30.0, 30.0],
            "mlb_id": 1,
            "player_name": "T",
            "rookie_year": 2021,
            "card_type": "flagship",
        }
    )
    me = series_month_ends(chart)
    apr = me[me["month"] == pd.Timestamp("2021-04-01")].iloc[0]
    assert apr["price"] == 15.0  # median of the dupes, not file-order "last" (20.0)
    may = me[me["month"] == pd.Timestamp("2021-05-01")].iloc[0]
    assert may["price"] == 30.0
```

```python
# tests/test_multiyear_panel.py — edit tiny_universe() so the probe's inputs actually differ
# across the 2021-06-01 cut (currently vacuous: no post-cut data exists in the fixture).
# 1) Add a second, post-cut game row to the game_logs DataFrame (same columns as the first row):
#    {"mlb_id": 1, "group": "hitting", "season": 2021, "date": "2021-07-01",
#     "level": "mlb", "gamesPlayed": 1, "atBats": 3, "hits": 2, "doubles": 1,
#     "triples": 0, "homeRuns": 1, "rbi": 2, "baseOnBalls": 1, "strikeOuts": 0,
#     "hitByPitch": 0, "sacFlies": 0, "stolenBases": 0},
# 2) Replace the empty events frame with one post-cut award:
#    events = pd.DataFrame(
#        [{"mlb_id": 1, "event_date": pd.Timestamp("2021-11-01"),
#          "event_type": "award_win", "details": "fake award"}]
#    )
# The probe's assertions (predictor equality for entry_month <= cut) now bite: a lag
# leak on games or awards changes early entries' predictors between full and truncated
# builds, and the events-truncation branch (`events[...] < cut`) is exercised for real.
# Keep every other line of the fixture and both tests identical.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_multiyear.py::test_series_month_ends_collapses_duplicate_dates_by_median -v`
Expected: FAIL — April price comes out 20.0 (file-order last) instead of 15.0.

- [ ] **Step 3: Implement the tie-break**

In `src/cardprice/multiyear.py`, inside `series_month_ends`, BEFORE the month groupby:

```python
    # duplicate (slug, grade, date) prints exist on some psa_10 pages (chart
    # calibration artifacts) — collapse by median so the month-end pick is
    # order-independent
    df = (
        df.groupby(["card_slug", "grade", "date"], as_index=False)
        .agg(
            price=("price", "median"),
            mlb_id=("mlb_id", "first"),
            player_name=("player_name", "first"),
            rookie_year=("rookie_year", "first"),
            card_type=("card_type", "first"),
        )
    )
```

- [ ] **Step 4: Run tests, rebuild the panel, re-verify goldens**

Run: `python -m pytest tests/test_multiyear.py tests/test_multiyear_panel.py -v` — all PASS; full suite green; ruff clean.
Then the rebuild with the ungraded-identity check:

```bash
cp data/processed/panel_multiyear.parquet /tmp/panel_multiyear_prefix.parquet
python scripts/build_multiyear_panel.py
```

```python
# verify.py (run with the venv python; not committed)
import pandas as pd
old = pd.read_parquet("/tmp/panel_multiyear_prefix.parquet")
new = pd.read_parquet("data/processed/panel_multiyear.parquet")
ou = old[old.grade == "ungraded"].sort_values(["card_slug", "entry_month"]).reset_index(drop=True)
nu = new[new.grade == "ungraded"].sort_values(["card_slug", "entry_month"]).reset_index(drop=True)
# market_ret_3m pools both grades per entry month, so the psa_10 correction moves it
# legitimately; every OTHER ungraded column must be identical
cols = [c for c in ou.columns if c != "market_ret_3m"]
assert ou[cols].equals(nu[cols]), "ungraded rows changed beyond market_ret_3m — STOP and report"
moved = (ou["market_ret_3m"] != nu["market_ret_3m"]).sum()
print(f"ungraded market_ret_3m moved on {moved}/{len(ou)} rows (expected: subset of entries sharing a month with corrected psa_10 cards)")
```

Goldens to re-verify and report (expected: all unchanged — ungraded has 0 dupes, and the career/stage/awards predictors don't touch chart prices): Soto ungraded ret_12m at 2021-04 = −0.2716796494005069; Bryant career_home_runs 142 at 2021-04; Henderson sophomore at 2023-08; Judge awards 2 at 2023-01; Skenes pitcher path. Report new psa_10 row count + how many month-end prices moved vs the pre-fix panel. If ANY ungraded row differs, STOP and report — do not proceed.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/multiyear.py tests/test_multiyear.py tests/test_multiyear_panel.py
git commit -m "fix: deterministic psa_10 month-ends (median-collapse dupe dates); live truncation probe"
```

---

### Task 2: Hold-frame builder + per-horizon importance models

**Files:**
- Create: `src/cardprice/hold_model.py`
- Create: `scripts/run_hold_modeling.py`
- Test: `tests/test_hold_model.py`

**Interfaces:**
- Consumes: panel_multiyear.parquet (Task 1); `model_lasso.stability_selection`/`lasso_path_summary`; `model_gbm.gbm_shap_importance`/`gbm_cv_r2`; `features.standardize`, `features.POSITION_MAP`.
- Produces (Tasks 3-4 consume):
  - `HOLD_FEATURES = ["career_ops", "career_hr_rate", "log_career_games", "max_level_rank", "rate_at_max_level", "log_minor_games", "awards_to_date", "age", "age_at_debut", "price_level", "ret_3m", "market_ret_3m", "sophomore", "established", "bowman_1st", "years_since_rookie"]` (module constant, hitters only — pitcher rows are excluded from modeling; 5 pitchers is too thin, and their career columns differ).
  - `build_hold_frame(panel: pd.DataFrame, horizon: int) -> pd.DataFrame` — hitters (`position != "P"`), non-NaN `ret_{horizon}m`, plus derived columns: `career_hr_rate` = career_home_runs / career_games (0/0 → 0), `log_career_games` = log1p(career_games), `log_minor_games` = log1p(minor_games), `sophomore`/`established` = career_stage dummies (rookie_year is the reference; prospect absent from the data), `bowman_1st` = (card_type == "bowman_1st"), `years_since_rookie` = season_year(entry_month) − rookie_year, `entry_year` = entry_month.year. Median-impute HOLD_FEATURES (in-sample importance pass; the walk-forward re-imputes with train medians).
  - `horizon_importance(frame: pd.DataFrame, horizon: int, seed: int = 42) -> dict` — keys `lasso_stability` (Series), `lasso_path` (DataFrame), `gbm_shap` (DataFrame), `gbm_cv_r2` (float), fitted on standardized features via `features.standardize`, target `ret_{horizon}m`.
  - CLI writes `data/processed/hold_model_importance.csv` (rows: horizon, method, feature, value) and prints per-horizon summaries.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hold_model.py
import numpy as np
import pandas as pd

from cardprice.hold_model import HOLD_FEATURES, build_hold_frame, horizon_importance


def make_panel(n=400, seed=7):
    rng = np.random.default_rng(seed)
    stages = rng.choice(["rookie_year", "sophomore", "established"], n)
    career_games = rng.integers(0, 800, n).astype(float)
    ops = rng.uniform(0.5, 1.0, n)
    df = pd.DataFrame(
        {
            "card_slug": [f"set/card{i % 40}" for i in range(n)],
            "grade": "ungraded",
            "entry_month": pd.to_datetime(
                rng.choice(pd.date_range("2021-04", "2025-09", freq="MS"), n)
            ),
            "mlb_id": [1000 + i % 40 for i in range(n)],
            "player_name": "P",
            "rookie_year": 2018,
            "card_type": rng.choice(["flagship", "bowman_1st"], n),
            "position": rng.choice(["OF", "IF", "C"], n),
            "career_games": career_games,
            "career_ops": ops,
            "career_home_runs": career_games * 0.1,
            "max_level_rank": 4,
            "rate_at_max_level": 0.9,
            "minor_games": 200.0,
            "career_stage": stages,
            "awards_to_date": rng.integers(0, 3, n).astype(float),
            "age": rng.uniform(21, 33, n),
            "age_at_debut": rng.uniform(20, 27, n),
            "price_level": rng.uniform(2, 5, n),
            "ret_3m": rng.normal(0, 0.1, n),
            "market_ret_3m": rng.normal(0, 0.05, n),
        }
    )
    # planted signal: 12m return driven by career_ops only
    df["ret_12m"] = 2.0 * (ops - ops.mean()) + rng.normal(0, 0.05, n)
    df["ret_6m"] = df["ret_12m"]
    df["ret_24m"] = df["ret_12m"]
    df["ret_36m"] = df["ret_12m"]
    return df


def test_build_hold_frame_derives_columns():
    frame = build_hold_frame(make_panel(40), 12)
    assert {"career_hr_rate", "log_career_games", "sophomore", "established", "bowman_1st",
            "years_since_rookie", "entry_year"} <= set(frame.columns)
    assert (frame["position"] != "P").all()
    assert frame[HOLD_FEATURES].notna().all().all()  # median imputation complete


def test_planted_signal_recovered_by_both_models():
    frame = build_hold_frame(make_panel(), 12)
    out = horizon_importance(frame, 12)
    assert out["lasso_stability"].idxmax() == "career_ops"
    assert out["lasso_stability"]["career_ops"] >= 0.8
    assert out["gbm_shap"].iloc[0]["feature"] == "career_ops"
    assert out["gbm_cv_r2"] > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hold_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.hold_model'`.

- [ ] **Step 3: Implement hold_model.py + run_hold_modeling.py**

```python
# src/cardprice/hold_model.py
"""Hold-return modeling frame + per-horizon importance (LASSO stability / GBM-SHAP)."""

import numpy as np
import pandas as pd

from cardprice.career import season_year
from cardprice.features import standardize
from cardprice.model_gbm import gbm_cv_r2, gbm_shap_importance
from cardprice.model_lasso import lasso_path_summary, stability_selection

HOLD_FEATURES = [
    "career_ops", "career_hr_rate", "log_career_games", "max_level_rank",
    "rate_at_max_level", "log_minor_games", "awards_to_date", "age", "age_at_debut",
    "price_level", "ret_3m", "market_ret_3m", "sophomore", "established",
    "bowman_1st", "years_since_rookie",
]


def build_hold_frame(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = panel[(panel["position"] != "P") & panel[f"ret_{horizon}m"].notna()].copy()
    df["career_hr_rate"] = np.where(
        df["career_games"] > 0, df["career_home_runs"] / df["career_games"], 0.0
    )
    df["log_career_games"] = np.log1p(df["career_games"])
    df["log_minor_games"] = np.log1p(df["minor_games"])
    df["sophomore"] = (df["career_stage"] == "sophomore").astype(int)
    df["established"] = (df["career_stage"] == "established").astype(int)
    df["bowman_1st"] = (df["card_type"] == "bowman_1st").astype(int)
    df["years_since_rookie"] = df["entry_month"].map(season_year) - df["rookie_year"]
    df["entry_year"] = df["entry_month"].dt.year
    df[HOLD_FEATURES] = df[HOLD_FEATURES].fillna(df[HOLD_FEATURES].median())
    return df.reset_index(drop=True)


def horizon_importance(frame: pd.DataFrame, horizon: int, seed: int = 42) -> dict:
    X, _ = standardize(frame[HOLD_FEATURES])
    y = frame[f"ret_{horizon}m"]
    return {
        "lasso_stability": stability_selection(X, y, seed=seed),
        "lasso_path": lasso_path_summary(X, y, seed=seed),
        "gbm_shap": gbm_shap_importance(X, y, seed=seed),
        "gbm_cv_r2": gbm_cv_r2(X, y, seed=seed),
    }
```

```python
# scripts/run_hold_modeling.py
"""Per-horizon importance models on the multi-year panel (ungraded primary)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.hold_model import build_hold_frame, horizon_importance  # noqa: E402

HORIZONS = (6, 12, 24, 36)


def main() -> None:
    panel = pd.read_parquet("data/processed/panel_multiyear.parquet")
    rows = []
    for grade in ("ungraded", "psa_10"):
        sub = panel[panel["grade"] == grade]
        for h in HORIZONS:
            frame = build_hold_frame(sub, h)
            out = horizon_importance(frame, h)
            print(f"\n=== {grade} ret_{h}m: {len(frame)} rows, gbm_cv_r2={out['gbm_cv_r2']:.3f} ===")
            print("lasso stability:", out["lasso_stability"].round(2).head(8).to_dict())
            print("gbm shap top5:", out["gbm_shap"].head(5)["feature"].tolist())
            for feat, v in out["lasso_stability"].items():
                rows.append({"grade": grade, "horizon": h, "method": "lasso_stability",
                             "feature": feat, "value": v})
            for r in out["gbm_shap"].itertuples():
                rows.append({"grade": grade, "horizon": h, "method": "gbm_shap",
                             "feature": r.feature, "value": r.mean_abs_shap})
            rows.append({"grade": grade, "horizon": h, "method": "gbm_cv_r2",
                         "feature": "__r2__", "value": out["gbm_cv_r2"]})
    pd.DataFrame(rows).to_csv("data/processed/hold_model_importance.csv", index=False)
    print("\nwrote data/processed/hold_model_importance.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then the real run**

Run: `python -m pytest tests/test_hold_model.py -v` — 2 PASS (the planted-signal test is the anti-placebo gate: if it fails, the model code cannot detect a known signal and the real run is meaningless — STOP and report); full suite green; ruff clean.
Then: `python scripts/run_hold_modeling.py` (~5-15 min: stability selection is 200 bootstraps × 8 frames). Record: per grade × horizon — top-5 features by both methods (do they agree?), gbm_cv_r2 (negative R² is a meaningful, reportable result — Plans 4/5 precedent), and whether any feature is stable (≥0.6) at multiple horizons.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/hold_model.py scripts/run_hold_modeling.py tests/test_hold_model.py
git commit -m "feat: hold-return importance models per horizon"
```

---

### Task 3: Consistency + career-arc overlay

**Files:**
- Create: `src/cardprice/career_arc.py`
- Create: `scripts/run_career_arc.py`
- Test: `tests/test_career_arc.py`

**Interfaces:**
- Consumes: build_hold_frame (Task 2); `model_consistency.season_consistency` / `position_interaction_test` / `player_random_effects` (via column-aliasing adapter — modules stay untouched); statsmodels.
- Produces (Task 4 report consumes):
  - `consistency_adapter(frame: pd.DataFrame, horizon: int) -> pd.DataFrame` — aliases for the Plan-3 consistency functions: `excess_ret` := `ret_{horizon}m` (absolute hold return — documented), `games` := `career_games`, `stats_season` := `entry_year`, `position_group` := position mapped via `features.POSITION_MAP`.
  - `stage_effects(frame: pd.DataFrame, horizon: int) -> pd.DataFrame` — pooled OLS `ret_{h}m ~ C(career_stage) + age + price_level + market_ret_3m` (HC1 robust SEs), coefficient table per stage (rookie_year = reference).
  - `years_curve(frame: pd.DataFrame, horizon: int) -> pd.DataFrame` — median + IQR of `ret_{h}m` by `years_since_rookie` (descriptive arc table; small cells reported as-is with n).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_career_arc.py
import numpy as np
import pandas as pd

from cardprice.career_arc import consistency_adapter, stage_effects, years_curve


def make_frame(n=600, seed=11):
    rng = np.random.default_rng(seed)
    stage = rng.choice(["rookie_year", "sophomore", "established"], n, p=[0.3, 0.3, 0.4])
    stage_effect = np.array([{"rookie_year": 0.0, "sophomore": 0.25, "established": 0.05}[s] for s in stage])
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


def test_consistency_adapter_columns():
    ad = consistency_adapter(make_frame(20), 12)
    assert {"excess_ret", "games", "stats_season", "position_group"} <= set(ad.columns)
    assert set(ad["position_group"]) <= {"OF", "IF", "P", "C"}


def test_years_curve_shape():
    out = years_curve(make_frame(), 12)
    assert {"years_since_rookie", "median", "n"} <= set(out.columns)
    assert out["n"].sum() == 600
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_career_arc.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement career_arc.py + run_career_arc.py**

```python
# src/cardprice/career_arc.py
"""Career-arc overlay: stage effects, per-entry-year consistency, position interaction.

Adapts the Plan-3 consistency module by column aliasing (it stays untouched):
its `excess_ret` is our absolute hold return; `games` our career_games;
`stats_season` our entry_year.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from cardprice.features import POSITION_MAP
from cardprice.model_consistency import (
    player_random_effects,
    position_interaction_test,
    season_consistency,
)


def consistency_adapter(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = frame.copy()
    out["excess_ret"] = out[f"ret_{horizon}m"]
    out["games"] = out["career_games"]
    out["stats_season"] = out["entry_year"]
    out["position_group"] = out["position"].map(POSITION_MAP)
    return out


def stage_effects(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = frame.dropna(subset=[f"ret_{horizon}m", "age", "price_level", "market_ret_3m"])
    fit = smf.ols(
        f"ret_{horizon}m ~ C(career_stage) + age + price_level + market_ret_3m", data=df
    ).fit(cov_type="HC1")
    return pd.DataFrame(
        {
            "term": fit.params.index,
            "coef": fit.params.values,
            "se": fit.bse.values,
            "p_value": fit.pvalues.values,
            "n": len(df),
        }
    )


def years_curve(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return (
        frame.groupby("years_since_rookie")[f"ret_{horizon}m"]
        .agg(median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), n="size")
        .reset_index()
    )
```

`run_career_arc.py` prints, for grade=ungraded (primary) and psa_10 (robustness), horizon ∈ {12, 36} (the actionable + the arc horizon): stage_effects table; years_curve; `season_consistency(adapter, top_stat)` where top_stat = the top LASSO-stability feature from Task 2's real run for that grade/horizon (read it from `data/processed/hold_model_importance.csv`; if no feature reached ≥0.6 stability, use `career_ops` and say why); `position_interaction_test(adapter, top_stat)`; `player_random_effects(adapter, top_stat)`.

- [ ] **Step 4: Run tests, then the real run**

Run: `python -m pytest tests/test_career_arc.py -v` — 3 PASS; full suite green; ruff clean.
Then: `python scripts/run_career_arc.py` — record every table for the report. Note explicitly: whether the sophomore premium (if any) is stable across 12m/36m and across grades; position interaction p-value; player ICC (high ICC = player identity dominates stats — the Plan-4 finding's multi-year analog).

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/career_arc.py scripts/run_career_arc.py tests/test_career_arc.py
git commit -m "feat: career-arc overlay + consistency analyses"
```

---

### Task 4: Year-grain walk-forward gate + findings report

**Files:**
- Create: `src/cardprice/walkforward_holds.py`
- Create: `scripts/run_hold_gate.py`
- Create: `docs/findings/2026-09-17-multiyear-modeling.md` (the report — real numbers only)
- Test: `tests/test_walkforward_holds.py`

**Interfaces:**
- Consumes: build_hold_frame / HOLD_FEATURES (Task 2); sklearn LassoCV; numpy.
- Produces:
  - `walk_forward_years(frame, feature_cols, horizon: int, top_k: int = 5, min_train_years: int = 2, seed: int = 42) -> pd.DataFrame` — per entry_year y (with ≥ min_train_years earlier years present): train on entry_year < y, median-impute + standardize from TRAIN only, LassoCV fit, predict year y, take top_k by predicted `ret_{horizon}m`; row per pick: `entry_year, card_slug, predicted, realized` where realized = actual `ret_{horizon}m` − the year's universe median `ret_{horizon}m` (excess per year).
  - `gate_evaluation_years(picks: pd.DataFrame, fee: float = 0.14, n_boot: int = 10000, seed: int = 42) -> dict` — yearly mean excess series; block bootstrap over YEARS (resample years with replacement, mean of resampled yearly means); keys `mean_excess, ci_low, ci_high, net_mean (= mean − fee), net_ci_low (= ci_low − fee), n_years, verdict` where verdict = "PASS" iff `net_ci_low > 0` else "FAIL".
  - The gate is pre-registered on **12m, ungraded** (primary). 6/24/36m and psa_10 runs are descriptive only.

- [ ] **Step 1: Write the failing test**

```python
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
    picks = pd.DataFrame(
        {"entry_year": [2022, 2023, 2024], "realized": [0.30, 0.25, 0.28]}
    )
    out = gate_evaluation_years(picks, fee=0.14)
    assert out["n_years"] == 3
    assert abs(out["mean_excess"] - np.mean([0.30, 0.25, 0.28])) < 1e-9
    assert abs(out["net_mean"] - (out["mean_excess"] - 0.14)) < 1e-9
    assert out["net_ci_low"] > 0  # tight synthetic series clears 14% fees
    assert out["verdict"] == "PASS"
    bad = picks.assign(realized=[0.05, -0.10, 0.02])
    assert gate_evaluation_years(bad, fee=0.14)["verdict"] == "FAIL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_walkforward_holds.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement walkforward_holds.py + run_hold_gate.py**

```python
# src/cardprice/walkforward_holds.py
"""Year-grain walk-forward gate for hold returns: fit years < y, predict year y.

Block bootstrap over YEARS (overlapping holds are autocorrelated within and
across cards; the year block is the independence unit per the spec).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV


def walk_forward_years(
    frame: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    top_k: int = 5,
    min_train_years: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    target = f"ret_{horizon}m"
    years = sorted(frame["entry_year"].unique())
    rows = []
    for yi, y in enumerate(years):
        if yi < min_train_years:
            continue
        train = frame[frame["entry_year"] < y]
        test = frame[frame["entry_year"] == y]
        medians = train[feature_cols].median()
        X_train = train[feature_cols].fillna(medians)
        X_test = test[feature_cols].fillna(medians)
        mean, sd = X_train.mean(), X_train.std(ddof=0).replace(0, 1)
        X_train = (X_train - mean) / sd
        X_test = (X_test - mean) / sd
        model = LassoCV(cv=5, random_state=seed, max_iter=10000).fit(X_train, train[target])
        preds = model.predict(X_test)
        bench = test[target].median()
        ranked = test.assign(predicted=preds).nlargest(top_k, "predicted")
        for r in ranked.itertuples():
            rows.append(
                {
                    "entry_year": y,
                    "card_slug": r.card_slug,
                    "predicted": r.predicted,
                    "realized": getattr(r, target) - bench,
                }
            )
    return pd.DataFrame(rows)


def gate_evaluation_years(
    picks: pd.DataFrame, fee: float = 0.14, n_boot: int = 10000, seed: int = 42
) -> dict:
    yearly = picks.groupby("entry_year")["realized"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    boot = rng.choice(yearly, size=(n_boot, len(yearly)), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    return {
        "mean_excess": float(yearly.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "net_mean": float(yearly.mean() - fee),
        "net_ci_low": float(ci_low - fee),
        "n_years": len(yearly),
        "verdict": "PASS" if ci_low - fee > 0 else "FAIL",
    }
```

`run_hold_gate.py`: loads panel_multiyear.parquet; per grade (ungraded first), per horizon (12m is THE gate; 6/24/36 descriptive): build_hold_frame → walk_forward_years → gate_evaluation_years; prints every pick row + gate dicts. 12m ungraded expected prediction years: 2023, 2024, 2025 (2021 has only Apr+ entries and is a train-only year; 2025 entries truncate at 2025-09 for 12m). If a year has < 5 test cards or LassoCV fails to converge, report it, don't hide it.

- [ ] **Step 4: Run tests, then the gate, then write the findings report**

Run: `python -m pytest tests/test_walkforward_holds.py -v` — 2 PASS; full suite green; ruff clean.
Then: `python scripts/run_hold_gate.py` and record everything.
Then write `docs/findings/2026-09-17-multiyear-modeling.md` with REAL numbers from the three runs (hold modeling, career arc, gate). Mandatory sections:
1. **Verdict** — gate PASS/FAIL, net numbers, first sentence.
2. **What predicts hold returns** — per horizon, both methods, agreement/disagreement; gbm_cv_r2 incl. negatives.
3. **Career arc** — stage effects, years curve, ICC; state the enter-at-month-end convention; state that prospect is structurally unobservable.
4. **Consistency** — per-year coefficient table for the top stat; position interaction.
5. **The gate** — picks table, yearly excess, bootstrap CI, net-of-fees, and the power caveat in bold: 3 yearly blocks; overlapping holds autocorrelated; effective N ≈ cards × independent years ≪ row count.
6. **Limitations** — psa_10 robustness validity (post-tie-break), 6 absent players, 2021×bowman_1st empty cell, raw-condition noise handled by IQR-quarantine + medians, no prospect window, no autos.
7. **Recommendation** — research-stage continues or (only on PASS) the conversation reopens. No real-money trading regardless (spec §10).

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/walkforward_holds.py scripts/run_hold_gate.py tests/test_walkforward_holds.py docs/findings/2026-09-17-multiyear-modeling.md
git commit -m "feat: year-grain walk-forward gate + multi-year findings report"
```

---

## Done criteria for this plan

- `python -m pytest -q` all offline tests PASS (incl. planted-signal recovery, live truncation probe, gate math); ruff clean.
- Panel rebuilt with deterministic psa_10 month-ends; ungraded rows proven identical except the `market_ret_3m` pooled-median column (documented propagation of the psa_10 correction); all five P6b goldens re-verified unchanged.
- `data/processed/hold_model_importance.csv` — 2 grades × 4 horizons × 3 methods.
- Career-arc tables produced for ungraded + psa_10 at 12m/36m.
- `docs/findings/2026-09-17-multiyear-modeling.md` — gate verdict (pre-registered: 12m, ungraded, net of 14% fees, block bootstrap over years), real numbers throughout, power caveat stated, honest FAIL acceptable.
- This closes the spec's three-plan arc (P6a data, P6b panel, P6c modeling). Copy `panel_multiyear.parquet` back to the main checkout at merge time.
