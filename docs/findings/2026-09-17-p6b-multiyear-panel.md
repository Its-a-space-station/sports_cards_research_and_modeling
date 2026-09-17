# P6b: Multi-Year Panel — Completion Report (2026-09-17)

Plan: `docs/superpowers/plans/2026-09-17-multiyear-panel.md` · Branch `multiyear-panel` (8 commits + report) · Final review: MERGE-READY.

## What was built

- **Events + info** (`scripts/collect_universe_events.py`): `data/processed/events_universe.parquet` — 39 debuts (MLB-level logs only) + 25 award_wins (MVP/Cy Young/ROY, 2015–2026); `data/reference/player_info_universe.csv` — 39 players, 0 null birth dates. Verified: Judge 4 wins (ROY 2017, MVP 2022/2024/2025), Henderson ROY 2023.
- **Career features** (`src/cardprice/career.py`): `career_to_date` (career MLB line through lag date), `minors_pedigree` (max level + OPS/ERA there + minor games), `career_stage` (prospect/rookie_year/sophomore/established; Nov/Dec roll to next season), `awards_to_date`.
- **Outcomes** (`src/cardprice/multiyear.py`): month-end series per (card, grade), annualized log returns at 6/12/24/36m with honest NaN truncation, trailing price level + 3-month return, market regime.
- **Panel** (`scripts/build_multiyear_panel.py` → `data/processed/panel_multiyear.parquet`): **5,553 rows / 59 cards** (ungraded 2,838, psa_10 2,715; psa_9 excluded by design). Non-NaN outcomes: 6m 4,869 · 12m 4,191 · 24m 2,953 · 36m 1,896 (36m entries ≤ 2023-09, exactly the data-end truncation).

## Verification (all reproduced independently by the final reviewer)

- Soto flagship ungraded ret_12m at 2021-04: −0.2716796494005069 — panel == hand-computed from chart, exact.
- Bryant career HR at 2021-04 = **142** (26+39+29+13+31+4; the plan's original 147 was my typo — 2020 was the 60-game COVID season). 314 post-lag games correctly excluded — the lag discipline is empirically load-bearing.
- Henderson sophomore at 2023-08 (his cards have no chart data before that); Judge awards_to_date = 2 at 2023-01 (2024/2025 MVPs correctly excluded); Skenes pitcher path clean (career_era non-null, AAA ERA 0.988 as rate_at_max_level).
- No-look-ahead: lag-boundary test + truncation probe + real-data goldens. Suite 137 passed, ruff clean.

## Structural finding (matters for interpretation)

**The `prospect` career stage is empty** — no card's chart history begins at/before its player's debut month (SCP tracking starts at rookie-product release, post-debut). Prospect-window returns are unobservable from this data; P6c cannot estimate prospect effects. This is the same structural wall Plan 5 hit (post-debut observability), now measured at the monthly grain.

## Carry-overs for P6c (must-carry, in order)

1. **psa_10 chart duplicate tie-break** (Important): 123 (slug, date) duplicate keys with differing prices on the psa_10 series (4.4% of its month-cells; ungraded has zero). The inherited "last within month" pick is file-order dependent there (deviates up to 51% from month median). P6c Task 1: define the tie-break (median of last-date prices), rebuild the panel, re-verify goldens — before quoting any psa_10 robustness number. Ungraded primary analysis is unaffected.
2. Prospect stage empty — do not attempt prospect-effect estimates.
3. Overlapping holds are autocorrelated — effective N ≈ cards × independent years; report it. Ungraded primary; no pop columns (single-snapshot look-ahead).
4. State the timing convention explicitly: `ret_3m`/`market_ret_3m` include the entry month's own month-end print — enter-at-month-end convention.
5. Strengthen the truncation-probe fixture (add a post-cut game + non-empty events) before any modeling refactors.

## Deferred minors (triaged at final review — all safe to defer)

STAGES unused · `_group_of` two-way heuristic (no two-way players in universe) · `_month_idx` style · truthiness-vs-0.0 price (min entry_price 0.35) · empty-input column-less frames · silent NaN age/position (0 nulls in panel) · duplicated month-index scaffolding in multiyear.py · ENTRY_FLOOR comment wording (2021-04..06 entries honestly get NaN trailing features).

## Test state at merge

- Offline suite: **137 passed**; ruff clean on all touched files; live suite unaffected (no new live tests; 6 remain from P6a and earlier).
