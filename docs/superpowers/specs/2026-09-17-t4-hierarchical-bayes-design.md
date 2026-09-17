# Design: T4 — Hierarchical Bayesian Consistency Model

Date: 2026-09-17
Status: Approved (design stage)
Program: Research-agenda expansion, thread T4 of 5. Depends on T2 (`panel_class.parquet`); modeling-only, no new data collection.

## 1. Goal

Answer the project's founding question properly: **which attributes matter, and are the effects consistent across seasons and players at the same position?** P6c answered with a binary (player identity absorbs 81–97 % of residual variance). Hierarchical partial pooling answers with posterior distributions: how large the true effect variation across classes/positions/players is, with shrinkage instead of overfit fixed effects.

## 2. Decisions locked in brainstorming

- **Library:** Bambi (formula API over PyMC) — same NUTS inference engine, fraction of the code, easy to audit; PyMC direct as fallback if a needed structure isn't expressible.
- **Baselines:** pooled LASSO (validated module) and a player-fixed-effects-only null, compared by LOO-CV — the hierarchy must *predict* better out-of-sample, not just fit nicer.
- **statsmodels `mixedlm`:** sanity baseline only (weak varying-slope support, no real posterior).
- **New heavy dependency accepted** (`bambi`, `pymc`, `arviz`, pinned) — the only proper tool for partial pooling; installed in the worktree venv per convention.

## 3. Model

Unit: card × entry-month rows of `panel_class.parquet` (fallback `panel_multiyear.parquet` for a standalone dry run).

```
ret_12m ~ surprise + career_stage + market_regime + log_price_level   (global)
        + (1 | draft_class) + (1 | draft_class:position) + (1 | player)
        + (0 + surprise | draft_class:position)                        (varying slopes)
```

- Regularizing priors: Normal(0, 1) on scaled effects, Half-Normal on variance components.
- NUTS, 4 chains, target_accept 0.9; scaled predictors (existing panel z-scoring conventions).
- Primary read-out: **posterior distributions of the variance components** — how much do surprise effects actually vary across classes and positions? Secondary: per-position/class shrunken slope estimates (who pools, who doesn't).
- Fallback mapping for a standalone dry run on `panel_multiyear.parquet`: `rookie_year` plays the `draft_class` role, and the panel's existing career-to-date performance features replace `surprise` (which only exists in `panel_class.parquet`). The fallback validates mechanics, not the substantive question.

## 4. Quality gate (methodological, pre-registered)

- R̂ < 1.01 and ESS > 400 for every parameter; divergences ≤ 1 %.
- Pathological sampling → simplify structure (drop varying slopes first, then the position level) — **never** tune priors to force convergence.
- Decisional read, registered in advance:
  - Slope-variance ≈ 0 with tight posterior → effects pool → future models simplify (a real, useful null).
  - Large, well-identified slope variance → position/class-specific modeling justified.

## 5. Architecture

```
src/cardprice/model_hierarchical.py    data prep, model, convergence checks, LOO
data/processed/hierarchical_summary.csv
docs/findings/<date>-hierarchical-consistency.md
```

## 6. Testing

- Prior-predictive smoke test on synthetic data with planted variance components — the fitted model must recover them (largest first: player > class).
- Convergence checks scripted (fail the run, not just print).
- Posterior-summary golden snapshot on a fixed-seed subsample (catches silent model/data drift between runs).
- LOO comparison table regenerated from saved fits; no hand-copied numbers in the findings doc.
