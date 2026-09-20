# T4 — Hierarchical Bayesian Consistency Model

Date: 2026-09-19
Plan: docs/superpowers/plans/2026-09-19-t4-hierarchical-bayes.md
Spec: docs/superpowers/specs/2026-09-17-t4-hierarchical-bayes-design.md

Every number below is recomputed from the artifacts —
`data/processed/hierarchical_convergence_{hitter,pitcher}.json`,
`hierarchical_variance_components_{hitter,pitcher}.csv`,
`hierarchical_slopes_{hitter,pitcher}.csv`, `hierarchical_loo_{hitter,pitcher}.json`,
`hierarchical_fit_{hitter,pitcher}.nc` — and the Task-3 fit logs
(`.superpowers/sdd/2026-09-19-t4-hierarchical-bayes/task-3-fit-{hitter,pitcher}.log`),
not from memory. Artifacts in `data/processed/` are gitignored per repo convention.
Per the plan's global constraints this thread is **descriptive inference**, not a
trading gate.

## The question

Which attributes matter for 12m raw card returns, and are the effects consistent
across draft classes and positions? P6c answered coarsely — player identity absorbs
81–97% of residual variance at the 36m arc horizon (player random-effects ICC:
ungraded 0.283 at 12m → 0.806 at 36m; psa_10 0.814 → 0.966;
`docs/findings/2026-09-17-multiyear-modeling.md`). This thread quantifies the
STRUCTURE of that answer with partial pooling and posteriors.

## Model & gate

Rung-0 (full) formula:

```
y ~ surprise + career_stage + market_ret_3m + price_level
    + (1|class_year) + (1|class_position) + (1|mlb_id)
    + (0 + surprise|class_position)
```

Priors (pre-registered, never adjusted anywhere in the ladder): common effects
Normal(0, 1); group sigmas HalfNormal(1). Sampler (both groups, from the
convergence JSONs): NUTS, 4 chains × 1,000 tune + 1,000 draws, target_accept 0.9,
random_seed 42.

Pre-registered convergence gate (binding, spec §4 / plan): **R̂ < 1.01 and
ESS_bulk > 400 for every parameter; divergences ≤ 1% of post-warmup draws.**
Pathological sampling → simplify structure in the registered ladder: (1) drop the
varying slopes `(0 + surprise|class_position)`; (2) drop the `class_position`
intercept level. Never tune priors to force convergence.

Ladder walk, verbatim from the fit logs (4,000 post-warmup draws per fit):

| group | rung 0 (full) | rung 1 (no slopes) | rung 2 (floor) |
|---|---|---|---|
| hitter | FAIL — 10 div (0.25%) + R̂ > 1.01 and ESS < 100/chain warnings | FAIL — 7 div (0.18%) + R̂ > 1.01 warning | **PASS** |
| pitcher | FAIL — 56 div (1.40% > 1%) | FAIL — 91 div (2.28% > 1%) | **FAIL** |

Convergence reports of record (`hierarchical_convergence_{group}.json`):

| group | rung used | pass | rhat_max | ess_min | divergences | div rate |
|---|---|---|---|---|---|---|
| hitter | 2 (floor) | **true** | 1.0053 | 728 | 16 | 0.40% |
| pitcher | 2 (floor) | **false** | 1.0089 | 479 | 43 | **1.075%** |

**Hitter = clean floor pass:** R̂ 1.0053 < 1.01, ESS 728 > 400, 16 divergences =
0.40% ≤ 1%. **Pitcher = floor-rung gate FAILURE, verbatim:** R̂ 1.0089 ✓ and ESS
479 ✓ pass, but 43 divergences = **1.075% > 1.000%** — a borderline miss on the
divergence criterion alone. The runner's honest-failure path executed as designed
(WARNING logged, honest failing JSON written, every artifact still produced, the
other group proceeded). **Every pitcher conclusion below is INDICATIVE-ONLY.**

Rows (both groups, from the convergence JSONs): panel 93,969 rows → 0 NaN-key rows
dropped → model frames **2,718 hitter / 1,697 pitcher** (ungraded, `ret_12m`
non-null, surprise non-null) → bambi listwise-deletes **158 / 132** rows with NaN
predictors → **2,560 / 1,565** fitted rows. Fitted grouping factors: 10 class
years (2015–2024 — no 2025-class row carries a surprise value, since a 2025
rookie has no prior completed MLB season inside the panel window), 214 hitter /
143 pitcher players.

## Consistency verdict (pre-registered decisional read)

The registered read: slope-variance ≈ 0 with a tight posterior → effects pool →
future models simplify (a real, useful null); large, well-identified slope
variance → position/class-specific modeling is justified.

What the data delivered is **neither registered outcome — the slope question is
not identifiable at this granularity.** The full varying-slopes structure failed
the gate on BOTH groups (rungs 0 and 1 above), so no
`surprise|class_position_sigma` posterior exists at all, and the class:position
level itself does not survive the ladder. The decisional read, with the mandated
qualifier: **no IDENTIFIABLE class/class:position slope heterogeneity at the
granularity the covered data supports.** Convergence failure is not evidence of
absence — it is partly a power statement: 2,718 hitter / 1,697 pitcher rows after
filters (158 / 132 further listwise-deleted), spread over 73 hitter
class:position cells (median 29 fitted rows/cell, min 2) and 10 pitcher cells.

What IS identified — the floor-rung variance components
(`hierarchical_variance_components_{group}.csv`, posterior mean [94% HDI]):

| component | hitter | pitcher (INDICATIVE-ONLY) |
|---|---|---|
| class_year sd | 0.042 [0.0001, 0.091] | 0.053 [~0.00001, 0.132] |
| player (mlb_id) sd | 0.227 [0.195, 0.257] | 0.374 [0.325, 0.425] |
| residual sd | 0.437 [0.425, 0.449] | 0.377 [0.364, 0.390] |

The structure of the answer: **class-level variation ≈ 0 (the HDI touches zero on
both groups); player-level variation dominates the hierarchy** — player sd ≈ 5.4×
(hitter) / ≈ 7.1× (pitcher, indicative) the class sd. Variance shares implied by
the posterior-mean sigmas: hitter residual 78% / player 21% / class 0.7%; pitcher
(indicative) residual 50% / player 49% / class 1.0%. Consistent with P6c's coarse
"player identity absorbs 81–97%" — now quantified with posteriors. (Magnitudes
are not directly comparable: P6c's ICC is the player share of *residual* variance
at 36m — 0.283 at its own 12m horizon; these are variance components of total
outcome variance at 12m on the surprise-covered population.)

## Which attributes matter (global effects)

Common-effect posteriors, recomputed from the flattened netCDF3 archives
(reloaded with `xr.open_dataset(path, engine="scipy")`; 94% HDI over 4,000
draws). Predictors are z-scored — 1 z = 0.235 `surprise_ops` (hitter) / 4.675
−`surprise_era` (pitcher); 0.735 / 0.662 log-`price_level`; 0.0022 / 0.0023
`market_ret_3m`; y is unscaled `ret_12m`. `career_stage` has exactly two populated
levels in the covered population — established (reference) and sophomore
(1,881 / 679 fitted hitter rows; 1,198 / 367 pitcher rows): prospect and
rookie_year rows carry no surprise by construction, so only this one stage
contrast is identified.

Hitter (gate-pass):

| term | mean | sd | 94% HDI |
|---|---|---|---|
| Intercept | −0.139 | 0.028 | [−0.190, −0.088] |
| surprise (per z) | +0.011 | 0.010 | [−0.007, +0.029] |
| career_stage[sophomore] | +0.037 | 0.025 | [−0.011, +0.084] |
| market_ret_3m (per z) | −0.024 | 0.009 | [−0.041, −0.006] |
| price_level (per z) | −0.333 | 0.015 | [−0.361, −0.305] |

Pitcher (INDICATIVE-ONLY — floor-rung gate failure):

| term | mean | sd | 94% HDI |
|---|---|---|---|
| Intercept | −0.083 | 0.043 | [−0.165, −0.001] |
| surprise (per z) | +0.023 | 0.013 | [−0.001, +0.049] |
| career_stage[sophomore] | +0.039 | 0.031 | [−0.022, +0.095] |
| market_ret_3m (per z) | +0.007 | 0.010 | [−0.013, +0.027] |
| price_level (per z) | −0.417 | 0.014 | [−0.442, −0.390] |

Read: entry `price_level` is the dominant global effect in both groups (higher
entry price → lower subsequent 12m return, HDIs far from zero). The surprise
coefficient is ≈ 0 with the HDI spanning zero in both groups — a low-power null
consistent with T2-model's registered surprise finding, measured on the same
structurally covered subsample. `market_ret_3m` is a small negative hitter
effect, null for pitchers (indicative); the sophomore-vs-established contrast
spans zero in both.

## Baselines / LOO

Per-observation mean ELPD, highest wins (`hierarchical_loo_{group}.json`):

| group | hierarchical | pooled LASSO (5-fold OOF) | player-FE null (OOF) | winner |
|---|---|---|---|---|
| hitter | **−0.6243** | −0.7192 | −0.7524 | hierarchical |
| pitcher (INDICATIVE-ONLY) | **−0.4951** | −0.8532 | −0.8478 | hierarchical |

Required caveats, exactly as the artifacts record them:

- **(a) Gaussian plug-in approximation** — verbatim from both LOO JSONs:
  "Baseline ELPDs are a Gaussian plug-in approximation: a constant sigma (OOF
  residual std, ddof=0) in a Gaussian log predictive density, not a Bayesian
  posterior predictive. Baselines score ALL model-frame rows (median-imputed
  predictors); the hierarchical ELPD is PSIS-LOO on dropna-complete rows only,
  normalized to a per-observation mean. Comparison is indicative, not exact."
- **(b) Pareto-k warnings:** `az.loo` emitted Pareto-k > 0.7 warnings on BOTH
  groups (both fit logs) — PSIS-LOO is partially unreliable here. The margins are
  large (hitter +0.095 per-obs ELPD over LASSO; pitcher +0.353 over the
  next-best baseline), so the winner is probably robust — but the artifacts
  record point ELPDs only, no SE and no pareto_k values, so no formal uncertainty
  statement about the margins is possible from these artifacts.
- **(c)** The pitcher LOO is INDICATIVE-ONLY per the floor-rung gate failure.

## Shrunken slopes

None to report. Both `hierarchical_slopes_{group}.csv` files are header-only
(`class_position,mean,sd,hdi_3%,hdi_97%`) — **because** the winning rung on both
groups is the floor rung, which has no `class_position` term, so no per-cell
shrunken surprise slopes exist. The rungs that would have produced them (0 and 1)
failed the convergence gate on both groups. The empty CSVs are the artifact form
of the consistency verdict, stated plainly rather than silently.

## Caveats

- **Surprise-covered population only.** Surprise coverage is structural — a value
  needs ≥ 1 prior completed MLB season (Marcel prior) plus in-season pace
  strictly before the entry month, making it a mostly post-debut feature: 7.8%
  (`surprise_ops`) / 3.9% (`surprise_era`) of the 93,969 panel rows panel-wide
  (T2-data findings), narrowing to 2,718 / 1,697 model-frame rows after the
  ungraded + `ret_12m`-non-null + group filters, and 2,560 / 1,565 after listwise
  deletion. The consistency read applies to the covered population, not the
  universe; pre-debut and rookie-entry rows — exactly where a surprise term would
  be most interesting — are structurally absent.
- **Span of the covered rows:** surprise-covered entry months run **2021–2025**
  in both groups, concentrated in 2024–2025 (hitter 1,850/2,718 = 68%; pitcher
  1,116/1,697 = 66%).
- **Descriptive inference, not a gate.** Per the plan's global constraints this
  thread is descriptive; nothing trades on it. The pitcher half is additionally
  INDICATIVE-ONLY (floor-rung gate failure).
- **LOO approximation and reliability:** Gaussian plug-in baselines; Pareto-k >
  0.7 warnings on both groups; point ELPDs only — see Baselines / LOO.
- **Rung margins are seed-dependent.** The pitcher floor-rung divergence rate
  (1.075% vs the 1.000% threshold) is a borderline miss that a different seed
  could flip either way; which rungs pass or fail is likewise seed-sensitive
  (hitter's floor-rung pass margins are comfortable by comparison: ESS 728 vs
  400, 0.40% vs 1.00%). All fits are seeded `random_seed=42`; baselines use fold
  seed 42.
- **Flattened netCDF3 archives.** `hierarchical_fit_{group}.nc` is one flattened
  netCDF3 file per group (xarray scipy engine; posterior/sample_stats vars bare,
  `observed__` / `loglik__` prefixed), because arviz 1.3's DataTree
  `to_netcdf` requires a NETCDF4 backend that is not installed and new
  dependencies were prohibited. Reload recipe (recorded verbatim in each
  convergence JSON's `fit_netcdf_format` field):
  `xr.open_dataset(path, engine="scipy")` — `az.from_netcdf` will NOT parse these
  files. Consequence: group-level posterior frames must be re-assembled by hand
  from the flat variable namespace; the variance-components CSVs stand as the
  reviewed artifact of record for the variance read.

## Reproduce

Exact commands, in order (worktree root):

```bash
# 1. Panel prep (writes data/processed/panel_class.parquet, 93,969 rows;
#    full upstream pipeline: docs/findings/2026-09-19-t2-data.md §Reproduce)
.venv/bin/python scripts/build_class_panel.py

# 2. Fits (each walks the 3-rung ladder and writes all five artifacts per group;
#    logs of record: .superpowers/sdd/2026-09-19-t4-hierarchical-bayes/task-3-fit-{hitter,pitcher}.log)
.venv/bin/python scripts/run_hierarchical.py --group hitter
.venv/bin/python scripts/run_hierarchical.py --group pitcher
```

Sampler settings (both groups, from the convergence JSONs): draws 1,000, tune
1,000, chains 4, target_accept 0.9, random_seed 42.

Verification at this state: `.venv/bin/python -m pytest -q` →
**271 passed, 6 deselected, 26 warnings in 127.80s** (6 deselected = `live`
network markers; warnings pre-existing upstream noise).
