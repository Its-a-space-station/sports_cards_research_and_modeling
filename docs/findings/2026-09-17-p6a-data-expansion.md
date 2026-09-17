# P6a: Data Expansion — Completion Report (2026-09-17)

Plan: `docs/superpowers/plans/2026-09-16-data-expansion.md` · Branch `data-expansion` (10+ commits) · Final review: MERGE-READY.

## What was built

- **Minor-league game logs** (`stats_api.py`): `fetch_game_log(..., sport_id=)` + `fetch_minor_league_logs` + `MINOR_LEAGUE_LEVELS` (AAA=11, AA=12, A+=13, A=14). One call per level — plural `sportIds=` returns partial data and is banned.
- **Universe catalog** (`data/reference/players_universe.csv`, 39 players, rookie classes 2015–2025, incl. deliberate busts Franco/Kelenic/Walker) and set-family resolver (`scripts/resolve_universe.py`) → `cards_universe.csv` (78 rows = 39 × {flagship, bowman_1st}).
- **Resolution results:** mlb_id **39/39**; flagship **32/39**; bowman_1st **27/39**. All URLs from real fetched SCP anchors (no-guess rule held; final reviewer audited every URL-producing path).
- **Stats collection:** `data/processed/game_logs_universe.parquet` — **35,152 game rows / 39 players** (mlb 27,321 · aaa 2,541 · aa 2,313 · a_plus 1,653 · a 1,324). Goldens verified from raw snapshots: Henderson 2022 AAA = 65, Bryant 2015 MLB = **151** (plan corrected from erroneous 145 probe), Soto 2017 A = 23.
- **Price collection:** `universe_sales.parquet` (7,243 rows) + `universe_chart_monthly.parquet` (14,930 rows) — **59 card pages, 100% of resolved URLs**, 0 challenge failures. Ungraded is a first-class series (`normalize_grade` at weekly/liquidity layer; scp_parse untouched). `card_type` flows catalog → parquets → selection.
- **Modeling selection:** `data/reference/cards_modeling.csv` — **143 (card, series) rows / 59 cards / 33 players**; rule `sales_per_week ≥ 0.3 OR n_chart_points ≥ 36`; grade split ungraded 59 / psa_10 55 / psa_9 29.

## Misses (honest, unpatched)

- 19 unresolved URLs: 7 flagship + 12 bowman_1st. The 7 flagship misses (Alvarez, D. Williams, Franco, Rutschman, Strider, Harris II, Anthony) are genuine SCP catalog gaps — surname absent even from full console listings.
- 6 players fully absent from price data (Alvarez, Franco, Rutschman, Strider, Harris II, Anthony); 2022 class hardest hit. 2021 × bowman_1st cell empty (all 4 unresolved).
- SCP's plain `bowman-chrome` consoles mostly lack Prospects/Draft cards, so several "bowman_1st" picks are rookie-year Bowman Chrome base, not true earlier 1sts (e.g. Franco bowman_1st is EMPTY — his 1st lives in 2019 Bowman Chrome Prospects, outside the plain slug). Full audit trail in the Task 2 report.

## Carry-overs for P6b (from final review)

1. **psa_9 has no chart history** — chart label is grader-agnostic `grade_9`, sales parse to `psa_9`; they never join. Do NOT rename labels (would be wrong). P6b sources psa_9 monthly prices from weekly sales (`weekly_price_series`) instead.
2. **Universe parquets are a strict superset of seed** (all 13 seed slugs ⊂ 59 universe slugs; seed lacks `card_type`). Never concatenate universe with `scp_sales.parquet`/`scp_chart_monthly.parquet`.
3. **`key:cib` rows** (16, 3 cards) persist in the chart parquet; re-apply the `key:` filter in any new consumer (as `run_universe_liquidity.py` does).
4. Add a min-`n_sales` companion filter when using `sales_per_week` (span floor = 1 day creates burst artifacts, e.g. Witt 14/wk from 2 sales).

## Deferred minors (triaged at final review — all safe to defer)

- float-mlb_id snapshot-key risk (latent; disk keys clean int) · `normalize_grade` pd.NA gap (unreachable today) · `reparse_snapshots.py` is seed-only (point at cards_universe.csv + add card_type if ever used on universe snapshots) · meta-merge fan-out (no dup URLs today) · offline-test sleep ~1.2s · `META_COLS` declarative-only in both collectors.

## Test state at merge

- Offline suite: **125 passed**; ruff clean on all touched files; live suite **6/6** (needs `PLAYWRIGHT_BROWSERS_PATH=.pw-browsers`).
