# Design: T2 — Class-Wide Cross-Section + Surprise Features

Date: 2026-09-17
Status: Approved (design stage)
Program: Research-agenda expansion, thread T2 of 5 (the data backbone for T3 and T4; T5 consumes its outputs).
Extends: `2026-09-16-multiyear-minors-design.md` (same panel conventions, wider universe + expectations term).

## 1. Goal

Replace the 39-player, partially outcome-selected universe with **entire prospect classes**, and add the missing covariate class our nulls pointed to: **expectations**. Models then regress returns on *surprise* (performance relative to expectation) rather than raw performance.

Rationale (from P6c findings): season-cumulative stats barely predict returns — consistent with a market that prices expectations. The tradeable object is the innovation; this thread builds the instrument to measure it, plus the cross-sectional breadth (Grinold: IR = IC × √breadth) that n = 39 could never provide.

## 2. Decisions locked in brainstorming

- **Universe:** full 1st Bowman **auto checklists from both families** — Bowman Chrome (BCAP/CPA, international-signee-heavy) and Bowman Draft (CDA, draftee-heavy) — classes **2015–2025** (~1,300 players), unioned per player (a player's class = year of their *earliest* 1st Bowman auto across families). Checklists fixed at print time → includes busts → no ex-post selection. *Amended 2026-09-17 per spike `2026-09-17-scp-checklists.md`: originally "1st Bowman Chrome Prospect auto checklist" — Chrome-only would silently drop drafted players. **Further corrected after the full pull:** SCP's draft-auto checklists may be incomplete (Witt Jr. absent from SCP's 2019 Draft auto set; his earliest SCP-listed auto is 2020 Chrome CPA-BWJ). Classes derive strictly from data (earliest auto across families), and a Cardboard Connection completeness audit of one draft year is required (Task 2).*
- **Era flag:** `price_visible_breakout` = first pro season ≥ 2020. SCP's price archive floor is **2021-03** (verified 2026-09-17: global sales/chart floor 2021-03-13 / 2021-03-01; even a 2015 card's first observable sale is 2021-05). Classes 2015–2019 contribute late-window rows only; the early-prediction core is classes 2020–2025.
- **Priced card:** base **non-auto** 1st Bowman (Chrome `BCP` or Draft `BDC` base, matching the family of the player's 1st auto), **raw** (deepest sales per player; consistent with the raw-prices decision). The auto may be added later as a second `card_type`.
- **Data budget:** free-first; anything paywalled stops and asks.
- **Gate shape:** same pre-registered form as P6c — plus a cross-sectional quintile-spread test.
- **Expectations term (amended 2026-09-17, user-approved):** pre-debut = MLB Pipeline Top-100 rank (+ FG "The Board" dated draft-board FV as secondary); post-debut = **in-house Marcel-style lagged projection** (5/4/3 season weights, regression to the mean) computed from our own season stats. *Replaces the original Steamer/ZiPS + BA Top-100 design: both are paywalled + Cloudflare-protected (probe `2026-09-17-expectations-sources.md`). FG membership for true Steamer/ZiPS is recorded as an optional paid upgrade, not purchased.*

## 3. Data depth (known constraints)

- **Prices:** SCP per-sale + monthly charts, floor 2021-03. Cards released 2022+ tracked from release.
- **Stats:** MLB Stats API game logs (MLB + minors) verified deep (minors to 2011); 2015–2025 collection is volume, not feasibility (~1,300 players ≈ 7–9 h at 0.3 s/call → collectors **must be resumable**: skip-if-snapshot-exists + incremental parquet flushes — amended 2026-09-17).
- **Expectations (amended 2026-09-17 per probe):** MLB Pipeline Top-100 2015–2025 — free, dated, plain-HTTP (mlb.com `/milb/prospects/YYYY/top100/` for 2020–2025, news-article full lists for 2015–2019; Akamai 403s intermittently → retry/backoff). FG "The Board" dated draft boards (secondary, draft classes). Marcel projections computed in-house at panel time. BA Top-100 dropped (paywall + CF); Wayback deferred (archive.org rate-limited at probe time; not on the critical path). Each expectation row carries source + as-of date.
- **Checklist source (spike-verified 2026-09-17):** SCP auto set pages per year × family, slugs discovered via one GET of `/brand/baseball-cards/bowman` (never guessed — slug forms drift: `-prospect-autograph` / `-prospects-autographs` / `-prospects-autograph`); `?exclude-variants=true` single GET covers most years, cursor-POST pagination loop for >150-entry sets. Parsed from `table#games_table`; base = anchors without `[Parallel]` bracket. Cross-audit vs Cardboard Connection (2015 verified: CC ⊂ SCP). SCP quirks: duplicate numbers across players, spelling-variant rows → dedupe + audit.

## 4. Architecture

```
scripts/build_class_checklists.py  Chrome+Draft auto set pages (brand-page slug
                                   discovery, cursor-POST pagination) -> checklists
scripts/resolve_class_universe.py  mlb_id mapping + 1st-Bowman family/year +
                                   base-card resolution (incremental CSV, audit)
src/cardprice/collect_prices.py    extended: resumable (skip-if-snapshot-exists,
                                   incremental flush) — amendment 2026-09-17
scripts/collect_universe_stats.py  extended: same resumability treatment
src/cardprice/expectations.py      Pipeline lists + FG draft boards -> expectations.parquet
src/cardprice/multiyear.py         extended: surprise features -> panel_class.parquet
data/reference/: class_checklists.parquet, cards_class_universe.csv
data/processed/: class_sales.parquet, class_chart_monthly.parquet,
                 class_liquidity.csv, game_logs_class.parquet,
                 expectations.parquet, panel_class.parquet
```

## 5. Data plan

- **Universe assembly:** checklist rows from BOTH families (Chrome + Draft) per year → union per player, class = year of earliest 1st Bowman auto → `mlb_id` via the existing MLB API search pattern. Unmapped players (name collisions, never-affiliated) logged to an audit CSV with reason — never silently dropped. Audit table (like the GemRate spike's) committed to the findings doc.
- **Prices:** base non-auto 1st Bowman per player (`BCP` Chrome or `BDC` Draft, matching the family of the player's 1st auto; earliest year wins) → `class_sales.parquet` (per-sale), `class_chart_monthly.parquet` (ungraded + psa_10), `class_liquidity.csv`. Resumable SCP scrape, ≥ 5 s politeness (existing collector conventions); ~1,300 cards ≈ hours unattended.
- **Stats:** extend game-log collection to all mapped players, MLB + minors, 2015–2026 → `game_logs_class.parquet` (existing rate-limited client + resumable cache).
- **Expectations:** per player-season: Pipeline Top-100 rank (primary; as-of = list publication date, preseason), FG draft-board FV (secondary, draft classes), source, as-of date. Pre-debut players: prospect rank **is** the expectations term. Post-debut: Marcel-style lagged projection (5/4/3 weights, regression to mean) computed from our own `season_stats` at panel-build time — fully in-house, no external dependency.

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

- Checklist↔SCP mismatches: exact-match rules as in `pop_gemrate.py` (no-guess); ambiguous → audit CSV, manual review, documented in findings. Known SCP quirks (spike-verified): duplicate card numbers across two players; spelling-variant duplicate rows — dedupe rules must be explicit.
- Collector per-card failures: row written with empty fields, run continues, failures summarized; runs are resumable (skip-if-snapshot-exists) so crashes never restart from zero.
- Pipeline gaps: a missing year/list page after retries → that source-season NaN, never interpolated from later lists (look-ahead).

## 9. Testing

- Golden checklist rows: known names per class (e.g. 2016 class ⊃ Guerrero Jr. via Bowman Chrome CPA — exact names verified against published checklists, never assumed). Negative golden: Witt Jr. must NOT appear in the 2019 Chrome checklist. Witt's own class derives from data: earliest auto across families per SCP listing (2020 Chrome CPA-BWJ as of the 2026-09-17 pull; draft completeness audited in Task 2).
- Golden sales/chart counts for 3 known cards against captured fixtures.
- Hand-computed surprise goldens (one hitter pace-vs-projection, one pitcher).
- Panel join invariants: no duplicated card-months; era-flag goldens (Bryant 2015 → late-window).
- No-look-ahead tests: expectations as-of dates strictly before entry month.

## 10. Plan shape

Two implementation plans, mirroring the P6a/P6b split:

- **T2-data:** checklist builder → universe assembly → price/stat collectors → expectations collection → `panel_class.parquet`.
- **T2-model:** surprise models, walk-forward gate, quintile-spread evaluation, findings doc.

T3, T4, T5 are single plans each (T3's spike is its plan's task 0).
