# Multi-Year Hold Modeling — Findings (2026-09-17, P6c)

Branch `multiyear-modeling` · Plans P6a (data) / P6b (panel) / P6c (modeling) · Spec: `docs/superpowers/specs/2026-09-16-multiyear-minors-design.md`

## 1. Verdict

**The pre-registered gate FAILS: a walk-forward LASSO buying the top-5 ungraded cards per entry year does not beat the universe median net of 14% round-trip fees over 12-month holds.** Mean yearly excess +0.242 log, block-bootstrap 95% CI over years [−0.142, +0.655]; net of fees the mean drops to +0.102 and the CI lower bound to **−0.282 ≤ 0 → FAIL**. The verdict rests on 3 yearly blocks (prediction years 2023/2024/2025) and is weakly powered by construction — see §5. Research stage continues; no trading pipeline (§7).

Gate registration (pre-registered in the P6c plan before any modeling run): 12m horizon, ungraded series, top-5 picks per entry year, block bootstrap over years, net of 0.14 log fees, PASS iff net_ci_low > 0. All other grade × horizon cells are descriptive only.

## 2. What predicts hold returns

In-sample importance pass (Task 2; `data/processed/hold_model_importance.csv`, 2 grades × 4 horizons × 3 methods; hitters only):

gbm_cv_r2 by grade × horizon — all eight positive, rising monotonically with horizon, psa_10 above ungraded everywhere:

| grade | 6m | 12m | 24m | 36m |
|---|---|---|---|---|
| ungraded | 0.426 | 0.578 | 0.748 | 0.813 |
| psa_10 | 0.596 | 0.743 | 0.828 | 0.852 |

Top-5 features (LASSO stability | GBM SHAP), top-5 overlap:

- **ungraded 6m** — price_level(1.0), ret_3m(1.0), log_career_games(.72), age(.57), max_level_rank(.56) | price_level, ret_3m, log_career_games, age, career_hr_rate → 4/5
- **ungraded 12m** — price_level(1.0), ret_3m(1.0), max_level_rank(.99), age(.98), market_ret_3m(.97) | price_level, ret_3m, years_since_rookie, career_ops, age → 3/5
- **ungraded 24m** — max_level_rank(1.0), price_level(1.0), ret_3m(1.0), career_ops(1.0), age(1.0) | price_level, ret_3m, age_at_debut, career_ops, age → 4/5
- **ungraded 36m** — age(1.0), price_level(1.0), career_ops(.98), ret_3m(.98), age_at_debut(.96) | age_at_debut, price_level, ret_3m, career_ops, rate_at_max_level → 4/5
- **psa_10 6m** — log_career_games(1.0), price_level(1.0), market_ret_3m(1.0), sophomore(.95), ret_3m(.90) | price_level, log_career_games, ret_3m, age, age_at_debut → 3/5
- **psa_10 12m** — log_career_games(1.0), price_level(1.0), market_ret_3m(1.0), ret_3m(1.0), sophomore(1.0) | price_level, log_career_games, ret_3m, age_at_debut, age → 3/5
- **psa_10 24m** — log_career_games(1.0), price_level(1.0), market_ret_3m(1.0), ret_3m(.99), sophomore(.94) | price_level, ret_3m, log_career_games, age, age_at_debut → 3/5
- **psa_10 36m** — log_career_games(1.0), price_level(1.0), market_ret_3m(1.0), established(1.0), age(.97) | price_level, log_career_games, log_minor_games, age, market_ret_3m → 4/5

Agreement/disagreement: 3–4 of 5 top features overlap at every cell — no wild disagreement. `price_level` and trailing `ret_3m` are top-2 by both methods nearly everywhere (only miss: GBM top-5 omits ret_3m at psa_10/36m, where LASSO has it at 0.925). Systematic split: LASSO favors `market_ret_3m` and the career-stage dummies; GBM favors `age`/`age_at_debut` — linear vs nonlinear cuts of the same career clock. `price_level` is stable ≥0.99 in 8/8 fits, `ret_3m` ≥0.90 in 8/8.

Honesty caveat on this section: these R² are strikingly high for return modeling because the pass is in-sample — median imputation and standardization use the full frame (Task 2, by design). The walk-forward gate (§5) re-standardizes from train years only but inherits `build_hold_frame`'s full-frame median imputation (the walk-forward's own train-median fillna is armed but finds no NaNs as composed), so it is honest out-of-sample on standardization, not on imputation; it fails at the registered horizon regardless. The dominant "predictors" are price persistence (level + 3m momentum), not player stats.

## 3. Career arc

Task 3 (`scripts/run_career_arc.py`; OLS with HC1, career_stage reference = rookie_year; `excess_ret` = absolute hold return, market enters only as the `market_ret_3m` control).

Stage effects (coef / p), ungraded:

| term | 12m (n=1998) | 36m (n=901) |
|---|---|---|
| sophomore | +0.174 / 0.089 | −0.068 / 0.317 |
| established | +0.093 / 0.359 | **−0.253 / 0.0002** |
| age | +0.026 / <0.001 | +0.040 / <0.001 |
| price_level | −0.079 / <0.001 | −0.051 / <0.001 |
| market_ret_3m | +0.447 / 0.002 | +0.059 / 0.479 |

psa_10:

| term | 12m (n=1921) | 36m (n=875) |
|---|---|---|
| sophomore | **+0.408 / <0.001** | −0.057 / 0.475 |
| established | +0.309 / 0.0009 | **−0.269 / 0.0005** |
| age | +0.035 / <0.001 | +0.048 / <0.001 |
| price_level | −0.078 / <0.001 | −0.024 / 0.004 |
| market_ret_3m | +1.180 / <0.001 | +0.439 / <0.001 |

Reading: a sophomore premium exists at 12m (both grades; psa_10 significant) but vanishes or flips at 36m, where established-vs-rookie is significantly **negative** in both grades. Early-career cards bought in the rookie window outperform over 12m and underperform over 36m, conditional on age/price/market. The rookie_year reference cell is nearly empty (31/29 cards at 12m; **8 per grade at 36m**) — all stage contrasts hang on that tiny base.

Years-since-rookie curves (median 12m log return, ungraded): y0 −0.290 (n=45) · y1 −0.049 (264) · y2 −0.195 (262) · y3 −0.165 (219) · y4 −0.393 (213) · y5 0.000 (239) · y6 −0.186 (245) · y7 −0.176 (200) · y8 −0.173 (152) · y9 −0.051 (104) · y10 +0.038 (55). Mostly negative medians at every cell in all four grade × horizon frames — the typical card loses value post-entry regardless of arc point; no clean "arc" shape emerges (likely partly survivorship/entry-at-first-price construction, not disentangled here).

Player random-effects ICC (share of residual variance absorbed by player identity): ungraded 0.283 (12m) → 0.806 (36m); psa_10 0.814 (12m) → 0.966 (36m). At the arc horizon, identity dominates and stats add little — the multi-year analog of the Plan-4 finding.

Conventions stated explicitly: **enter-at-month-end** — `ret_3m`/`market_ret_3m` include the entry month's own month-end print, i.e. the strategy enters at the month-end close, not the month-open. The **prospect stage is structurally unobservable**: no card's SCP chart history begins at/before its player's debut month, so `career_stage == "prospect"` has zero rows and no prospect-window effect can be estimated from this data (P6b structural finding).

## 4. Consistency

Per-year coefficients for the top LASSO-stability stat per cell (ties broken by Task-2 CSV row order; two of four picks are price/market covariates, not player stats — reported as selected):

ungraded 12m, `price_level` (stability 1.000, tied with ret_3m): 2021 −0.187 (p<0.001, n=228) · 2022 −0.155 (p<0.001, n=356) · 2023 −0.067 (p<0.001, n=443) · 2024 −0.015 (p=0.439, n=534) · 2025 −0.013 (p=0.522, n=437). Monotone decay to zero.

ungraded 36m, `age` (0.995, tied with price_level): 2021 +0.095 (p<0.001, n=228) · 2022 +0.062 (p<0.001, n=356) · 2023 +0.020 (p=0.013, n=317). Same sign, magnitude decays ~5×.

psa_10 12m, `log_career_games` (1.000, 3-way tie): 2021 +0.908 (p<0.001, n=232) · 2022 +0.381 (p<0.001, n=343) · 2023 +0.107 (p=0.236, n=420) · 2024 +0.326 (p=0.002, n=509) · 2025 +0.078 (p=0.312, n=417). Non-monotone; significant in 3 of 5 years.

psa_10 36m, `log_career_games` (1.000, 4-way tie): 2021 +0.383 (p<0.001, n=232) · 2022 +0.142 (p=0.002, n=343) · 2023 **−0.080 (p=0.045, n=300)** — sign flip in the most recent complete cohort.

Position interaction (IF vs OF × stat): F=0.266 p=0.606 (ungraded 12m) · F=0.421 p=0.516 (ungraded 36m) · F=2.627 p=0.105 (psa_10 12m) · F=3.867 **p=0.0496** (psa_10 36m) — null everywhere except one marginal cell.

Summary: no effect is consistent across entry years at the magnitude the pooled fit implies; coefficients decay (ungraded) or flip sign (psa_10 36m). Position interactions are essentially absent.

## 5. The gate

Pre-registered cell: **12m, ungraded**, top-5 picks per entry year, train on entry years < y (train-years-only standardization, LassoCV cv=5; feature imputation uses full-frame medians from `build_hold_frame` — the walk-forward's own train-median fillna is armed but finds no NaNs as composed), realized = pick return − entry-year universe median, block bootstrap (10,000 resamples) over years, fee 0.14 log. Imputation detail: on this frame 126/1,998 rows (≈6%) had NaN `price_level`/`ret_3m` and 48 had NaN `market_ret_3m`, all filled with full-frame medians; any resulting bias is toward PASS, so the FAIL verdict below is conservative. Prediction years 2023/2024/2025 (2021 is a train-only year — Apr+ entries; 2025 entries truncate at 2025-09 for 12m). No year had <5 test cards (2021: 228 · 2022: 356 · 2023: 443 · 2024: 534 · 2025: 437); LassoCV emitted no convergence warnings.

Picks (all 15 rows):

| entry_year | card_slug | predicted | realized |
|---|---|---|---|
| 2023 | 2015-bowman-chrome/carlos-correa-110 | 0.1634 | +0.8250 |
| 2023 | 2021-topps-chrome-update/adolis-garcia-usc64 | 0.1266 | +0.6250 |
| 2023 | 2021-topps-chrome-update/adolis-garcia-usc64 | 0.0881 | +0.5719 |
| 2023 | 2015-bowman-chrome/carlos-correa-110 | 0.0626 | +0.7832 |
| 2023 | 2020-topps-chrome/randy-arozarena-49 | 0.0488 | +0.4680 |
| 2024 | 2021-topps-chrome-update/adolis-garcia-usc64 | 0.0432 | +0.4214 |
| 2024 | 2015-topps-chrome/kris-bryant-112 | −0.0005 | +0.0744 |
| 2024 | 2024-topps-chrome/wyatt-langford-122 | −0.0088 | −0.2741 |
| 2024 | 2021-topps-chrome-update/adolis-garcia-usc64 | −0.0091 | +0.4147 |
| 2024 | 2021-topps-chrome-update/adolis-garcia-usc64 | −0.0156 | +0.4247 |
| 2025 | 2015-topps-chrome/kris-bryant-112 | 0.1366 | +0.6844 |
| 2025 | 2015-topps-chrome/kris-bryant-112 | 0.1288 | +0.7246 |
| 2025 | 2021-topps-chrome-update/adolis-garcia-usc64 | 0.0698 | −1.2464 |
| 2025 | 2020-bowman-chrome/randy-arozarena-11 | 0.0608 | +0.1573 |
| 2025 | 2021-topps-chrome-update/adolis-garcia-usc64 | 0.0528 | −1.0281 |

(Panel rows are card × entry_month, so the top-5 are row-level picks: the same card can be picked at several entry months within a year. Distinct cards picked: 3 per year — correa/garcia/arozarena in 2023, garcia/bryant/langford in 2024, bryant/garcia/arozarena in 2025. Concentration is high: 5 distinct players across all 15 picks.)

Yearly mean excess: 2023 **+0.655** · 2024 **+0.212** · 2025 **−0.142** (two adolis-garcia picks at −1.25/−1.03 sank the most recent year).

Gate dict: `mean_excess=+0.2417, ci_low=−0.1416, ci_high=+0.6546, net_mean=+0.1017, net_ci_low=−0.2816, n_years=3, verdict=FAIL`.

**Power caveat: this verdict rests on 3 yearly blocks — the block bootstrap over 3 years is weak by construction and its CI spans the full range of the observed yearly means. Overlapping holds are autocorrelated within and across cards, so the year block is the independence unit: effective N ≈ cards × independent years ≪ the 1,998-row frame (with pick concentration, effectively ~3 distinct cards × 3 years for the strategy leg). A PASS or FAIL at this power would both be fragile; FAIL is the honest read of a posterior that includes large losses net of fees.**

Descriptive cells (not the gate; horizon/grade were pre-registered, so none of these change the verdict):

| grade × horizon | n_years | mean_excess | net_ci_low | verdict |
|---|---|---|---|---|
| ungraded 6m | 4 (2023–2026) | +0.636 | +0.207 | PASS (descriptive; 2026 is a partial entry year, 156 test cards) |
| ungraded 24m | 2 | −0.033 | −0.458 | FAIL |
| ungraded 36m | 1 | −0.288 | −0.428 | FAIL (single block; CI degenerates to the point) |
| psa_10 6m | 4 | +0.309 | −0.041 | FAIL |
| psa_10 12m | 3 | +0.146 | −0.184 | FAIL |
| psa_10 24m | 2 | +0.181 | −0.116 | FAIL |
| psa_10 36m | 1 | +0.275 | +0.135 | PASS (descriptive; single block, degenerate CI — uninformative) |

Read: short-horizon momentum picks did beat the median over 6m in this sample (descriptive), but the edge decays with horizon and never survives fees at the registered 12m horizon; the 36m "PASS" is a single-year artifact with a degenerate CI and carries no evidential weight.

## 6. Limitations

- **psa_10 robustness validity is conditional on the Task-1 tie-break fix**: 123 duplicate (slug, date) psa_10 chart keys (4.4% of its month-cells) are now collapsed deterministically (median of last-date prices, commit `dc47689`); ungraded rows are byte-identical to P6b except the pooled `market_ret_3m` column (documented propagation). All psa_10 numbers here are post-fix.
- **6 universe players are fully absent from price data** (Alvarez, Franco, Rutschman, Strider, Harris II, Anthony — 2022 class hardest hit), and the **2021 × bowman_1st cell is empty** (all 4 unresolved). Cohort coverage is incomplete.
- **Raw-condition noise** in the underlying sales is handled upstream by IQR-quarantine (4×IQR fence within card-grade-month, flag-never-delete, `validate.quarantine_outliers`) plus month-end **median** collapse — the panel never sees individual sales.
- **No prospect window**: structurally unobservable (§3) — the earliest career stage in the data is rookie_year.
- **No autos/parallels** (spec §10 non-goal); flagship + bowman_1st base only.
- Pick concentration (§5): the strategy leg is ~3 distinct cards per year, so idiosyncratic player risk dominates the yearly means.
- rookie_year reference n=8 at 36m makes the 36m stage contrasts fragile (§3).
- 12m-vs-36m comparisons mix different cohorts (36m frames end at entry year 2023), not just horizons.
- In-sample importance numbers (§2) are optimistic by construction (full-sample imputation/standardization).

## 7. Recommendation

The gate FAILS at the pre-registered cell, so per the spec (§7: "FAIL = stay research-stage, reported honestly") **the project stays research-stage**; the trading conversation does not reopen (that option was reserved for a PASS, and spec §10 rules out real-money trading this round regardless of outcome). The descriptive 6m PASS does not change this: the 12m horizon was registered in advance precisely to prevent horizon-shopping after seeing results.

If the arc continues, the honest next questions are: (a) does the 6m momentum effect survive a registered 6m gate with more independent years (it needs a longer price history — currently 4 blocks, one partial); (b) can the persistence features (price_level, ret_3m) be separated from the player-stat signal the research question asks about; (c) does expanding the card universe beyond 59 cards / ~35 priced players change the yearly-block distribution. None of these are trading preparations.
