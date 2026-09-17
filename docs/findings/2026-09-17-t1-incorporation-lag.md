# T1 — Price Incorporation Lag (Daily Event Study)

Date: 2026-09-17
Plan: docs/superpowers/plans/2026-09-17-t1-incorporation-lag.md
Spec: docs/superpowers/specs/2026-09-17-t1-incorporation-lag-design.md

## Headline

On the pre-registered primary sample (`ungraded/all` — effectively 48 established-star
breakout events, 69 kept event-card pairs) **no statistically significant post-event
repricing is detectable at daily grain**: the pooled curve's bootstrap CI floor never
clears the baseline for 3 consecutive bins (`lag_days = null`; pre-period clean,
`pre_ok = true`, coverage share 0.857), the half-life fit sits below the noise floor
(`half_life_days = NaN`, amplitude NaN), and the per-event adjustment distribution is
insufficient-data 62.8% / no_adjustment 20.9% / intermediate 11.6% / fast 4.7%, with
0.0% late. The spec §8 gate therefore lands **intermediate — reported as measured**:
reaction timing is neither shown dead (fast share ≪ 80%) nor shown plausible
(late share ≪ 20%). The binding constraint is observability, not speed — at this
universe's sale density, zero prospect-stratum and zero debut events survive the
evidence rules, so the verdict covers established-star breakouts only, and T3's
anticipation-vs-reaction framing cannot be settled from this sample alone.

## Data & method

- Events: `detect_breakouts` over 35,152 game-log rows (z ≥ 2.5 vs trailing 30-game
  baseline; hitter composite / pitcher game score), merged with dated debuts and
  deduped at ±14 d per player → **866 events** (833 breakout, 33 debut; levels
  {mlb 646, aaa 60, aa 50, a_plus 45, a 32, n/a 33}). 590 fall after the SCP tracking
  floor 2021-03 (572 breakout, 18 debut); the 276 pre-floor events have no sales
  coverage and land entirely in the drop logs. Strata by career stage at event date:
  595 established / 271 prospect.
- Sales: 7,243 per-sale records, 2021-03 → 2026-09, median 2 sales/card/month.
- Evidence rules (recalibrated 2026-09-17, user-approved): event-anchored baseline =
  median of same-grade-class sales strictly before the event over 56 d (≥ 3 required);
  collection window ±21 d (≥ 3 sales required); drops counted per run below.
- Market adjustment: each sale's relative price divided by the all-universe daily
  median relative index of its grade class. Sales on days with no index reading are
  left unadjusted (NaN), excluded from curves and classification, and counted:
  **17 unadjusted sales (ungraded runs), 40 (psa_10 runs)** — market coverage is good
  but not total.
- Curve: pooled daily median of adjusted relative prices in integer bins ±14 d;
  95% CIs from cluster bootstrap (2,000 resamples by event, seed 42). 519 ungraded and
  312 psa_10 sales fall inside the curve bins.
- Kept samples: ungraded 69 event-card pairs / 48 distinct events; psa_10 66 pairs /
  56 events. Prospect stratum kept **0** pairs in both grade classes, so each
  `all` run is numerically identical to its `established` run (verified: curves equal).

## Results (verbatim from lag_summary.json)

| run | kept pairs (events) | drop_log (baseline/window) | unadj. sales | lag_days | pre_ok (cover) | half-life (ampl.) | adjustment shares |
|---|---|---|---|---|---|---|---|
| ungraded/all | 69 (48) | 1282 / 17 | 17 | null | true (0.857) | NaN (NaN) | insuff .628 · no_adj .209 · interm .116 · fast .047 · late .000 |
| ungraded/established | 69 (48) | 864 / 17 | 17 | null | true (0.857) | NaN (NaN) | identical to ungraded/all |
| ungraded/prospect | 0 | 418 / 0 | 0 | — | — | — | skipped (kept < 5) |
| psa_10/all | 66 (56) | 1222 / 29 | 40 | null | true (0.929) | 0.5 d (0.025) | insuff .824 · interm .078 · no_adj .059 · fast .039 · late .000 |
| psa_10/established | 66 (56) | 818 / 29 | 40 | null | true (0.929) | 0.5 d (0.025) | identical to psa_10/all |
| psa_10/prospect | 0 | 404 / 0 | 0 | — | — | — | skipped (kept < 5) |

Adjustment-share denominators: shares are over the events with ≥ 1 market-adjusted
sale — **43 of 48** distinct events (ungraded) and **51 of 56** (psa_10); 5 events per
grade class have every window sale on a NaN-index day and so drop out of the
classification entirely (not even as `insufficient`; raw counts reconcile — ungraded
27 insuff / 9 no_adj / 5 interm / 2 fast, psa_10 42 insuff / 4 interm / 3 no_adj /
2 fast). Segment definitions: pre = all sales before the event, early = [0, 3.5) d,
late = ≥ 7 d (uncapped).

Key curve bins, median [CI] (n sales), from lag_curves.csv:

| bin | ungraded/all | psa_10/all |
|---|---|---|
| -1 | 1.142 [0.932, 1.767] (11) | 0.984 [0.891, 1.037] (11) |
| 0 | 0.962 [0.788, 1.355] (15) | 1.022 [0.847, 1.120] (7) |
| +1 | 0.968 [0.489, 1.237] (32) | 0.969 [0.759, 1.166] (9) |
| +3 | 0.793 [0.584, 1.107] (14) | 1.364 [0.926, 1.648] (7) |
| +7 | 0.945 [0.575, 2.501] (25) | 1.000 [0.659, 1.133] (7) |
| +14 | 1.017 [0.705, 2.861] (11) | 0.901 [0.835, 1.856] (6) |

Reading: the psa_10 median path fits a small, fast move (amplitude ≈ 2.5%,
half-life 0.5 d), but with 6–11 sales per bin the bootstrap CIs are far too wide for
the lag estimator to fire — `lag_days` is null in every run. `pre_ok` is true
everywhere (pre-event CIs cover 1.0 at shares 0.857/0.929), so the nulls are not a
broken-baseline artifact.

## Gate verdict (spec §8, pre-registered)

Primary sample `ungraded/all`, verbatim from the run:

```
GATE (spec §8, primary = ungraded/all):
  fast share (>=80% adjusted within 72h): 0.046
  late share (>=20% adjusting at >=7d):   0.000
  VERDICT: intermediate -> report as measured
```

Neither gate condition is near threshold. The dominant per-event class is
`insufficient` (62.8%) — a statement about sale density, not about repricing speed.
Consequence for T3: the anticipation-vs-reaction decision stays **open**; it should be
revisited after T2 widens the cross-section (or denser price data exists), not
decided from T1's established-only, density-limited sample. No retrofitted narrative.

## Caveats

- **Prospect/debut unobservability (first-class finding):** at this universe's sale
  density (median 2 sales/card/month), zero prospect-stratum and zero debut events
  survive the evidence rules under any parameter setting (probe 2026-09-17: 48/590
  post-floor events kept, 100% established breakouts). Re-confirmed by this run:
  prospect kept = 0 pairs in both grade classes; debut events kept = 0 in both.
  The gate verdict therefore applies to
  established-star breakouts only; prospect-window repricing speed remains unmeasured
  and needs T2's breadth (or denser price data).
- Recalibrated density rules (baseline 56d, window ±21d, min 3 sales; user-approved
  2026-09-17) — original spec values kept only 42 pairs.
- best_offer mix, mix-of-grades handled via grade classes, 39-player universe,
  SCP floor 2021-03.

## Reproduce

`python -m cardprice.lag_study` (venv, repo root); artifacts in data/processed/ (gitignored).
