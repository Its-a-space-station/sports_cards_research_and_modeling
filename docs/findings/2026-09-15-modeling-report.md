# Modeling Findings — 2026-09-15

Runner: `scripts/run_modeling.py` (full stdout: `docs/findings/2026-09-15-modeling-run-output.txt`). All models
validated on synthetic planted-signal data in Tasks 2–5; this is the first real-data pass.

## 1. Dataset summary

| | monthly | weekly |
|---|---|---|
| raw rows | 325 | 184 |
| rows after filters | **281** | **89** |
| dropped: horizon gap | 31 (`months_since_prev > 2`) | 46 (`days_since_prev > 21`) |
| dropped: `excess_ret` NaN | 13 (first observed month per card) | 49 |
| cards / players | 13 / 13 | 11 / 11 (2 cards lost all rows to filters) |
| hitter rows (cards) | 254 (11) | 61 (9) |
| pitcher rows (cards) | 27 (2) — descriptive only, excluded from all fits | 28 (2) |
| seasons | 2022–2026 | 2024–2026 |
| span | 2022-11 → 2026-09 (35 months) | 2024-09-23 → 2026-09-07 (31 weeks) |
| grades | all PSA 10 | mixed: psa_10 46, psa_9 27, sgc_10 9, psa_8 4, sgc_9.5 2, sgc_9 1 |

Hitter rows by season — monthly: {2022: 4, 2023: 34, 2024: 69, 2025: 85, 2026: 62};
weekly: {2024: 5, 2025: 3, 2026: 53}. Monthly position split: OF 149 / IF 105 (no catcher
rows survive filters). Horizon rules were applied exactly as specified in Plan 3's review:
monthly keeps gaps ≤ 2 months, weekly keeps gaps ≤ 21 days; both then require a realized
next-period excess return.

## 2. Which stats matter most

Side-by-side, monthly (254 hitter rows, features standardized to unit population SD before
fitting):

| feature | LASSO stability freq | LASSO path coef (cv-min) | GBM mean \|SHAP\| |
|---|---|---|---|
| age | 0.00 | 0 (zeroed) | **0.0296** |
| years_since_rookie | 0.01 | 0 | 0.0290 |
| form_ops_delta | 0.01 | 0 | 0.0238 |
| ops | **0.02** | 0 | 0.0222 |
| home_runs | 0.00 | 0 | 0.0204 |
| games | 0.00 | 0 | 0.0190 |
| form_games | 0.00 | 0 | 0.0164 |
| strikeouts | 0.00 | 0 | 0.0161 |
| slg | 0.00 | 0 | 0.0159 |
| playoff | 0.00 | 0 | 0.0079 |

Weekly (61 hitter rows): stability selection — games 0.11, form_ops_delta 0.10,
years_since_rookie 0.08, slg 0.08, strikeouts 0.06, age 0.06, ops 0.04, home_runs 0.04,
form_games 0.02, playoff 0.00. GBM SHAP — form_games 0.034, age 0.034, ops 0.033,
years_since_rookie 0.033, games 0.021, home_runs 0.015, and exactly 0.000 for slg,
strikeouts, form_ops_delta, playoff. The weekly LASSO path at cv-min is again empty.

**Agreement:** all three monthly methods say the same essential thing — *nothing is a
stable driver*. The best stability frequency is 0.02 (ops); the cv-min LASSO path is
**empty** on both grains (every coefficient zeroed), which is consistent with the
near-zero stability frequencies rather than contradictory. SHAP's ranking is nearly flat
(0.008–0.030 monthly), i.e. the GBM also finds no dominant feature; its top-4 (age,
years_since_rookie, form_ops_delta, ops) overlaps the stability top-3 (ops,
form_ops_delta, years_since_rookie) on career-stage and recent-form features.

**Disagreement:** SHAP ranks `age` #1 monthly while stability selection never picks it
(0.00) — the GBM is likely exploiting small-sample nonlinearities that no linear model
replicates. On weekly, stability ranks `form_ops_delta` #2 (0.10) while SHAP gives it
exactly 0.000. With 61 rows, neither is trustworthy; flagged, not reconciled.

**Effect sizes in concrete terms** (there are no nonzero LASSO coefficients, so these
come from the per-season OLS fits of Section 3, raw-scale coef × feature SD):
1 SD higher OPS (0.154) ≈ +2.3% monthly excess return in 2024 (coef 0.151, p = 0.127 —
not significant); 1 SD higher form-OPS delta (0.213) ≈ +4.7% in 2024 (coef 0.222,
p = 0.163 — not significant). Every other season is smaller or flips sign.

**Answer to "which stats matter most":** none, at detectable strength. If a weak prior is
needed for future work, the only features any method keeps circling are `ops`,
`form_ops_delta`, and the career-stage pair (`age`, `years_since_rookie`).

## 3. Consistency across seasons and positions

Per-season OLS (`excess_ret ~ stat + games + age`, HC1 robust SEs), monthly hitter rows:

**ops**

| season | coef | SE | p | n |
|---|---|---|---|---|
| 2022 | — | — | — | 4 (too few, skipped) |
| 2023 | +0.215 | 0.489 | 0.660 | 34 |
| 2024 | +0.151 | 0.099 | 0.127 | 69 |
| 2025 | −0.007 | 0.142 | 0.962 | 85 |
| 2026 | +0.055 | 0.242 | 0.819 | 62 |

**form_ops_delta**

| season | coef | SE | p | n |
|---|---|---|---|---|
| 2022 | — | — | — | 0 |
| 2023 | +0.152 | 0.191 | 0.425 | 22 |
| 2024 | +0.222 | 0.159 | 0.163 | 48 |
| 2025 | +0.066 | 0.062 | 0.292 | 63 |
| 2026 | −0.014 | 0.066 | 0.830 | 57 |

Position interaction (joint F-test on `stat × position_group`, IF vs OF), monthly:
ops p = 0.815; form_ops_delta p = 0.219. Weekly: ops p = 0.882; form_ops_delta p = 0.209.

Player random intercepts (mixed model, `excess_ret ~ ops + games + age` per mlb_id):
monthly **ICC ≈ 1.5e-6** (random-intercept var 5.0e-8 — the MLE hit the boundary of the
parameter space, i.e. the player-level variance component is effectively zero);
weekly ICC = 0.024, likewise negligible.

Weekly per-season fits are dominated by tiny-n seasons and are reported, not read:
ops 2024 coef −0.694 with SE 3.083 on n = 5 (design matrix rank-deficient —
statsmodels raised `SingularMatrixWarning`; this coefficient is unidentified and
meaningless), 2025 skipped (n = 3), 2026 +0.176 (SE 0.298, p = 0.554, n = 53);
form_ops_delta 2026 +0.233 (SE 0.140, p = 0.097, n = 51).

**Verdict — Q1 (which stats matter):** no stat shows a stable, replicable effect on next
-month card excess returns at these sample sizes. The descriptive rankings weakly favor
production (`ops`, `form_ops_delta`) and career stage (`age`, `years_since_rookie`), but
stability selection selects everything at ≤ 2% frequency monthly and the LASSO path is
empty.

**Verdict — Q2 (consistency across seasons/positions):** there is no detectable effect to
be inconsistent. The ops coefficient is positive in 3 of 4 fittable monthly seasons (2025
is the exception at −0.007) and no season is significant; position interactions are null
on both grains (p ≥ 0.21); player-level ICC is ≈ 0, so card returns do not persistently
load on player identity beyond the measured covariates. Nothing here supports
season- or position-specific signal — but power is too low to rule small effects out.

## 4. Buy-signal gate

Walk-forward (monthly grain, hitters only, `HITTER_FEATURES`, top-2 picks per month,
min 6 training months, train-window median imputation + train-window standardization,
LassoCV at cv-min refit each month):

| metric | value |
|---|---|
| n_months (test months) | 29 |
| mean monthly excess over median-card benchmark[^1] | **+0.0191** (+1.9%) |
| bootstrap 95% CI | **[−0.0005, +0.0395]** |
| bootstrap p (share of means ≤ 0) | 0.028 |
| net of fees (0.14 log round-trip) | **−0.1209** (−12.1%/month) |

The reported p = 0.028 is one-sided (share of bootstrap means ≤ 0), while the 95% CI is
two-sided — the matching two-sided p is ≈ 0.056, so a CI whose lower bound sits just
under zero alongside p = 0.028 is consistent, not contradictory.

[^1]: Benchmark universe disclosure: the benchmark is the median realized `excess_ret`
of all **hitter** cards in the test month — the walk-forward frame is hitters-only, so
this is not literally the "median realized excess_ret of ALL cards that month" the plan
text specifies. Rerun with the all-cards benchmark (pitchers included): mean **+0.0185**,
bootstrap 95% CI **[−0.0037, +0.0427]**, p = 0.061 — gate verdict unchanged (FAIL).

**GATE VERDICT: FAIL.** Both pass conditions fail: `ci_low` = −0.0005 is not > 0, and
`net_of_fees_mean` = −0.121 is far below 0. The pre-fee point estimate is positive
(+1.9%/mo) but its CI's lower bound sits just under zero, and a ~14 log-point round-trip
fee burden is an order of magnitude larger than the estimated edge.

**Pipeline decision:** do NOT wire this LASSO ranking signal into the trading pipeline as
-is. The research-first gate did its job — a marginal, fee-dominated signal stays on the
bench. Revisit only if a future feature set (Plan 5 events, Savant quality-of-contact)
lifts the pre-fee edge well clear of the fee hurdle.

## 5. Limitations

- **Sample size / power.** 254 monthly hitter rows across 11 hitter cards and 5 seasons;
  weekly has 61 hitter rows with two of three seasons at n ≤ 5. Only very large effects
  could reach significance; nulls here are "not detected", not "absent".
- **PSA-10-only monthly history.** The monthly panel is a single grade; results do not
  generalize across the grade ladder (the weekly panel mixes grades but is too sparse to
  estimate grade effects).
- **Sparse weekly data.** 184 raw rows → 89 after filters; 2024 and 2025 are effectively
  unmodelable (n = 5 and 3 hitter rows), and the 2024 ops fit was rank-deficient
  (coefficient reported with its SE and explicitly not interpreted).
- **Pop controls unavailable historically.** PSA pop counts are only known as of scrape
  date; using them in backtests would be look-ahead, so supply-side scarcity is
  uncontrolled.
- **3 unresolved seed cards** carried through from panel construction; **SP-parallel pop
  entries** remain unnormalized. Both shrink the effective universe slightly.
- **Pitchers are descriptive-only** (27 monthly / 28 weekly rows, 2 cards) — excluded
  from every fit.
- **Monthly grain limits within-season timing resolution** — debut/award/call-up spikes
  arrive mid-month and are averaged away; this motivates the event study instead.
- **Event spikes not modeled** (Plan 5). All features here are level/form stats; playoff
  is the only event indicator and is never selected.
- **Convergence warnings, noted not papered over:** many bootstrap Lasso refits raised
  sklearn `ConvergenceWarning` with tiny duality gaps (~1e-5–1e-4, near-zero signal on
  small resamples); the monthly mixed model converged on the boundary (ICC ≈ 0 is a real
  boundary estimate, not a failure). Neither invalidates the null findings.

## 6. Next steps

1. **Plan 5 — event study:** abnormal returns around debuts, milestones, awards, and
   playoff runs, plus a mean-reversion test. Monthly level-stats have no detectable edge;
   event-driven moves are the remaining hypothesized alpha.
2. **Weekly accumulation cadence:** keep the weekly scrape running; at the current rate
   each additional season adds ~50+ usable hitter rows. Re-run this modeling pass when
   weekly hitter rows ≥ 200.
3. **Universe expansion:** more cards per player, more players, and more grades per card
   would raise power and enable grade-relative signals; resolve the 3 seed cards and
   normalize SP-parallel pop entries.
4. **Savant features if signal stays weak:** quality-of-contact (barrel %, xwOBA, hard-hit
   %) as an overlay — but only after the fee-aware gate framework is reused unchanged to
   judge them.
