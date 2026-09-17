# Design: Minor-League Data + Multi-Year Hold Modeling

Date: 2026-09-16
Status: Approved (design stage)
Supersedes/extends: `2026-09-14-mlb-card-price-panel-design.md` (original within-season design; that spec's completed work stands)

## 1. Goal

Extend the research platform in two directions:

1. **Minor-league performance data** as model predictors (prospect windows, pre-debut, level-tagged).
2. **Multi-year hold modeling** (holds < 5 years): predict annualized card returns over 6/12/24/36-month holds from information knowable at entry, and map how returns distribute across the career arc (prospect → rookie → sophomore → established).

Both serve the same practical question: which cards to buy and hold for under 5 years, based on measurable player performance.

## 2. Decisions locked in brainstorming

- **Objective:** buy-and-hold return model first; career-arc timing overlay second.
- **Price series: raw (ungraded) is PRIMARY**, PSA 10 secondary (robustness check). Rationale: raw has better liquidity for mid-tier players, fits Bowman 1st base cards (which trade mostly raw), and avoids PSA-10 pop-inflation bias in multi-year comparisons. Technical note: raw carries per-sale condition noise — handled by existing IQR quarantine + median aggregation.
- **Universe expands** to ~30–40 players: hobby-relevant rookies from classes 2015–2024 (e.g. 2015 Bryant/Correa, 2016 Seager/Bregman, 2017 Judge/Bellinger, 2018 Soto/Acuña, 2019 Tatis/Alonso/Vlad Jr., 2020 Lewis/Arozarena, 2021 García/India, plus the existing 2022–2025 seed).
- **Card types per player:** flagship Topps Chrome (or Chrome Update where the flagship rookie lives there) base RC + Bowman 1st base chrome prospect card, when SCP has them. Autos deferred pending liquidity.
- **Minors as predictors + Bowman 1st cards** (user's explicit choice).
- **Approach A:** expanded-universe buy-and-hold model + career-arc overlay. Event/momentum hybrid rejected (overlaps Plan 5's finer-grain null).

## 3. Data depth (measured live 2026-09-15/16)

- **Player stats (not binding):** MLB Stats API minor-league game logs verified to **2011** (Trout AA 91 G); AAA/AA/A+/A via `sportId=11/12/13/14` (per-level calls — the plural `sportIds=` form returns partial data). Henderson 2022 AAA: 65 G verified. Soto minors 2017-2018 verified.
- **Card prices (binding): SCP monthly chart floor = 2021-03.** Verified: Soto 2018 Topps Chrome Update #HMT55 — all six chart buckets (incl. ungraded) start 2021-03-01 → 2026-09-01, 67 monthly points. Cards released 2022+ are tracked from release. Deeper free sources (130point, PSA APR) are bot-blocked.
- **Pop counts:** current snapshots only; excluded from regressions (look-ahead), descriptive context only.
- **Consequence:** the multi-year price panel spans **2021-03 → 2026-09 (~5.5 years)**. 36-month holds computable for entries through 2023-09; 24-month through 2024-09; 12-month through 2025-09. Pre-2021 rookie-season price action is lost for pre-2021 rookie classes (caveat, not blocker: hold windows from 2021 onward are fully measured).

## 4. Architecture

No new architecture — the existing layers generalize:

```
ingest/    collect_stats.py  + minor-league levels (sportId loop, level column)
           collect_prices.py + expanded card catalog (set-family resolver:
                                 topps-chrome, topps-chrome-update, bowman-chrome)
           stats_api.py     + season-level stats for pre-2021 seasons (cheap career-to-date)
features/  panel.py         + career-to-date features, career_stage, hold-return outcomes
           ungraded series  = primary ("used" chart bucket + ungraded sales tables)
model/     hold-return models (6/12/24/36m), career-arc overlay, walk-forward gate
```

## 5. Data plan

- **Minors:** per player, per season 2011–2026, per level (AAA/AA/A+/A), game logs via `stats=gameLog&sportId=<level>`. Storage keys `{mlb_id}_{group}_{season}_{level}`; level column added to game_logs. Pre-2021 MLB: season-level lines via `stats=season` (game logs unnecessary where no prices exist).
- **Cards:** resolver extended to set families per player-class year: `baseball-cards-<year>-topps-chrome`, `-topps-chrome-update`, `-bowman-chrome`. Same no-guess rules, same parallel exclusion (refractors/autos excluded), same ≥5s politeness. Bowman 1st for the prospect window.
- **Prices:** monthly chart (ungraded + psa_10 series both kept) + recent sales. Same SCP collectors; raw snapshots immutable; reparse-over-snapshots pattern for any parser change.

## 6. Panel & features (multi-year)

Unit of observation: **card × entry-month** (entry months: all months with a valid price, 2021-04+).

**Outcome:** annualized log return `ln(price_{t+h} / price_t) / (h/12)` for h ∈ {6, 12, 24, 36} months; ungraded series primary, psa_10 as robustness. Months where the horizon runs past the data end are excluded per-hold-length (not fabricated).

**Entry-month predictors (all knowable at entry):**
- Career-to-date MLB line (OPS/wRC-style sums, HR rate; ERA/K-BB% for pitchers) through the last day before entry.
- Minors pedigree: best level-adjusted season (highest level reached, OPS/ERA there), age at debut (or age now if pre-debut), current minor-league form for pre-debut players.
- `career_stage` categorical: `prospect` (pre-debut), `rookie_year`, `sophomore` (year 2), `established` (year 3+).
- Awards-to-date count; All-Star/award events from the existing registry.
- Entry price level (log of trailing 3-month median price); market regime (universe median trailing 3-month return).
- Pop columns stay OUT (single-snapshot look-ahead).

**Career-arc overlay:** pooled card-month excess returns ~ career_stage + years_since_debut (+ performance interactions) — the buy-pre-peak/sell-post-peak question.

## 7. Estimation & gate (unchanged discipline)

- LASSO stability selection (1-SE rule), LightGBM+SHAP, mixed-effects per player/season — same validated modules.
- Walk-forward gate, harder than before: fit on entry years < y, predict entry year y; PASS iff top-k picks beat the universe median with a block-bootstrap CI (over years) clearing zero **net of ~14% round-trip fees**. FAIL = stay research-stage, reported honestly (as Plan 4 did).
- Power caveat written into the report: overlapping holds are autocorrelated; effective sample ≈ (cards × independent years), far smaller than row count.

## 8. Testing

- Same regime: golden rows (minors reconstruction vs known lines — e.g. Henderson 2022 AAA: 65 G), no-look-ahead invariants at entry-month grain, planted-signal tests for any new model code, horizon/overlap disclosure tests.
- Liquidity report rerun for the expanded universe (raw + graded, flagship + Bowman) decides the final card list; autos considered only if base liquidity is clearly sufficient.

## 9. Plan shape (three implementation plans)

- **P6a — data expansion:** minors collectors, season-stats backfill, universe catalog (~35 players), resolver for set families, collection runs, liquidity report.
- **P6b — multi-year panel:** career-to-date + career_stage features, hold-return outcome builder, ungraded-primary panel, golden verification.
- **P6c — modeling + report:** hold-return models per horizon, career-arc overlay, walk-forward gate, findings report `docs/findings/`.

## 10. Explicit non-goals

- No autos/parallels (unless liquidity proves base is thick AND autos are separately validated later).
- No basketball, no pre-2015 players (price floor 2021 makes their early windows unmeasurable anyway).
- No real-money trading regardless of gate outcome this round; a PASS reopens that conversation with the user.
- No Savant pitch-level features (still deferred from Plan 3; reconsider only if models are weak).
