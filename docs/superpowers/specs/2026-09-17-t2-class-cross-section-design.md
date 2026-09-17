# Design: T2 — Class-Wide Cross-Section + Surprise Features

Date: 2026-09-17
Status: Approved (design stage)
Program: Research-agenda expansion, thread T2 of 5 (the data backbone for T3 and T4; T5 consumes its outputs).
Extends: `2026-09-16-multiyear-minors-design.md` (same panel conventions, wider universe + expectations term).

## 1. Goal

Replace the 39-player, partially outcome-selected universe with **entire prospect classes**, and add the missing covariate class our nulls pointed to: **expectations**. Models then regress returns on *surprise* (performance relative to expectation) rather than raw performance.

Rationale (from P6c findings): season-cumulative stats barely predict returns — consistent with a market that prices expectations. The tradeable object is the innovation; this thread builds the instrument to measure it, plus the cross-sectional breadth (Grinold: IR = IC × √breadth) that n = 39 could never provide.

## 2. Decisions locked in brainstorming

- **Universe:** full 1st Bowman Chrome **Prospect Auto checklist**, classes **2015–2025** (~1,300 players). Checklist fixed at print time → includes busts → no ex-post selection.
- **Era flag:** `price_visible_breakout` = first pro season ≥ 2020. SCP's price archive floor is **2021-03** (verified 2026-09-17: global sales/chart floor 2021-03-13 / 2021-03-01; even a 2015 card's first observable sale is 2021-05). Classes 2015–2019 contribute late-window rows only; the early-prediction core is classes 2020–2025.
- **Priced card:** base **non-auto** 1st Bowman Chrome, **raw** (deepest sales per player; consistent with the raw-prices decision). The auto may be added later as a second `card_type`.
- **Data budget:** free-first; anything paywalled stops and asks.
- **Gate shape:** same pre-registered form as P6c — plus a cross-sectional quintile-spread test.

## 3. Data depth (known constraints)

- **Prices:** SCP per-sale + monthly charts, floor 2021-03. Cards released 2022+ tracked from release.
- **Stats:** MLB Stats API game logs (MLB + minors) verified deep (minors to 2011); 2015–2025 collection is volume, not feasibility.
- **Expectations:** preseason Steamer/ZiPS (FanGraphs historical pages, 2021–2026); dated MLB Pipeline Top-100 + Baseball America Top-100 via Wayback captures (2015–2025). Each expectation row carries source + as-of date.
- **Checklist source:** SCP set pages (keeps slug convention), audited against a second free source (e.g. Cardboard Connection checklists).

## 4. Architecture

```
src/cardprice/checklist.py        1st Bowman auto checklists 2015-2025 -> universe
src/cardprice/collect_prices.py   extended: base non-auto card per player (batched, resumable)
src/cardprice/collect_stats.py    extended: ~1,300 players, MLB+minors 2015-2026
src/cardprice/expectations.py     projections + dated rankings -> expectations.parquet
src/cardprice/multiyear.py        extended: surprise features -> panel_class.parquet
data/processed/: class_universe.parquet, class_sales.parquet,
                 class_chart_monthly.parquet, class_liquidity.csv,
                 game_logs_class.parquet, expectations.parquet, panel_class.parquet
```

## 5. Data plan

- **Universe assembly:** checklist rows (player, class year, card number) → `mlb_id` via the existing MLB API search pattern. Unmapped players (name collisions, never-affiliated) logged to an audit CSV with reason — never silently dropped. Audit table (like the GemRate spike's) committed to the findings doc.
- **Prices:** base non-auto 1st Bowman Chrome per player → `class_sales.parquet` (per-sale), `class_chart_monthly.parquet` (ungraded + psa_10), `class_liquidity.csv`. Batched, resumable SCP scrape, ≥ 5 s politeness (existing collector conventions); ~1,300 cards ≈ hours unattended.
- **Stats:** extend game-log collection to all mapped players, MLB + minors, 2015–2026 → `game_logs_class.parquet` (existing rate-limited client + resumable cache).
- **Expectations:** per player-season: preseason projection (Steamer preferred, ZiPS fallback), prospect rank (Pipeline primary, BA secondary), source, as-of date. Pre-debut players: prospect rank **is** the expectations term.

## 6. Surprise panel & features

Extend the multiyear builder → `panel_class.parquet` (card × entry-month, same conventions as `panel_multiyear.parquet`):

- `surprise_ops` / `surprise_era`: actual pace − preseason projection at each month-end (pace computed from game logs strictly before entry).
- `prospect_rank`, `rank_change` (dated ranking revisions; pre-debut players).
- Career-stage, career-to-date stats, awards-to-date, market regime, entry price level — as in the existing panel.
- `price_visible_breakout` era flag column on every row.
- No-look-ahead invariants at entry-month grain, as before.

## 7. Modeling & gate

- Pooled LASSO + LightGBM/SHAP on 12 m ungraded returns ~ surprise + stage + regime, walk-forward by entry year — same validated modules.
- **Pre-registered gate (same shape as P6c):** PASS iff top-k picks beat the universe median with block-bootstrap CI (over years) clearing zero **net of ~14 % round-trip fees**. FAIL = research stage continues, reported honestly.
- **Cross-sectional quintile spread:** top vs bottom surprise-quintile baskets per entry month — the breadth evaluation this universe exists for.
- Power note in report: entry years still ≤ 4–5 independent; the gain is players per year (~10–30×), not years.

## 8. Error handling

- Checklist↔SCP mismatches: exact-match rules as in `pop_gemrate.py` (no-guess); ambiguous → audit CSV, manual review, documented in findings.
- Collector per-card failures: row written with empty fields, run continues, failures summarized.
- Wayback gaps: missing ranking capture → that source-season NaN, never interpolated from later lists (look-ahead).

## 9. Testing

- Golden checklist rows: known names per class (e.g. 2016 class ⊃ Guerrero Jr.; 2019 class ⊃ Witt Jr.) — exact names verified against published checklists during planning, never assumed.
- Golden sales/chart counts for 3 known cards against captured fixtures.
- Hand-computed surprise goldens (one hitter pace-vs-projection, one pitcher).
- Panel join invariants: no duplicated card-months; era-flag goldens (Bryant 2015 → late-window).
- No-look-ahead tests: expectations as-of dates strictly before entry month.

## 10. Plan shape

Two implementation plans, mirroring the P6a/P6b split:

- **T2-data:** checklist builder → universe assembly → price/stat collectors → expectations collection → `panel_class.parquet`.
- **T2-model:** surprise models, walk-forward gate, quintile-spread evaluation, findings doc.

T3, T4, T5 are single plans each (T3's spike is its plan's task 0).
