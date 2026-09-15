# Design: Within-Season MLB Player Performance vs. Graded Rookie Card Prices

Date: 2026-09-14
Status: Approved (design stage)

## 1. Goal

Research-first project: quantify how a MLB player's in-season performance correlates
with the price of his graded rookie cards (PSA 9/10), determine which stats matter
most in a multivariate setting, and test whether those effects are consistent across
seasons and positions. If — and only if — walk-forward validation shows a real
predictive edge, extend the same pipeline into a buy-early signal for real money.

## 2. Decisions locked in brainstorming

- **Sport:** Baseball (MLB) first; framework kept sport-agnostic enough to extend to NBA later.
- **Card segment:** Graded modern rookies — PSA 9 and PSA 10 flagship rookies
  (Topps Chrome / Bowman Chrome 1st autos), chosen for liquidity and within-season
  performance sensitivity.
- **Data budget:** Free sources only.
- **Approach:** Weekly panel model (core) + event-study overlay (validation).
- **Mode:** Research first; trading pipeline only if results justify it.

## 3. Data sources (verified 2026-09-14)

### Prices
- **Primary: SportsCardsPro** — server-rendered, per-card per-grade individual sold
  listings (date, title, price) back to ~2019, pre-filtered for reprints/lots, grade
  parsed. Polite scraping (no login wall).
- **Backfill: 130point.com** — multi-marketplace (eBay, Goldin, PWCC/Fanatics,
  Heritage), multi-year own DB, recovers true Best Offer accepted prices that eBay
  hides. JSON backend; low-rate access, ToS-gray.
- **Cross-check (manual):** eBay Price Guide (~2y history) and Terapeak Product
  Research (~3y history) for validating specific card-grade windows.
- **Dead ends (do not build on):** eBay Finding API (decommissioned 2025-02),
  Marketplace Insights API (approval-gated, 90-day depth), public eBay sold search
  (90-day cap, login-walled, hides Best Offer prices).

### Player stats
- **Primary: MLB Stats API** (`statsapi.mlb.com`, no key) via the maintained
  `python-mlb-statsapi` wrapper. `stats=gameLog` per player-season (hitting +
  pitching), 2018–present, updated overnight. Season-to-date stats at any date are
  reconstructed from game logs.
- **Supplement: Baseball Savant** CSV exports (`csv=true`) for xwOBA, barrel%,
  sprint speed; 2015–present. Retro-revised, so raw pulls are versioned.
- **Avoid:** FanGraphs automation (hard-blocked, 403s; pybaseball's FanGraphs
  functions are currently broken) and bulk Baseball-Reference scraping (rate-limited
  to 20 req/min; ToS restricts bulk collection and AI training use).

### Population counts
- **GemRate free UI** (multi-grader pop counts + pop growth over time) via light,
  low-volume browser automation or manual capture for the bounded card set;
  snapshotted monthly. PSA pop report and PSA APR sit behind Incapsula — not a bulk
  pipeline. GemRate Partner API is commercial — out of budget.

### Known pitfalls the design must handle
- COVID 2020–21 boom/crash is a structural break → sample starts 2022.
- Auction sales average ~16.5% below BIN for identical items → sale-type control.
- Best Offer accepted prices hidden on eBay → prefer 130point-recovered prices.
- Thin liquidity → mark low-sale weeks missing; exclude chronically illiquid cards.
- Pop counts are submission counts with crack-resubmit inflation and selection bias
  → treat as controls, never as ground-truth supply.
- Strong hobby seasonality (October peak, winter floor) → de-seasonalize against a
  market index built from the tracked universe.

## 4. Architecture

Python project `cardprice-mlb`, four decoupled layers, each independently runnable
and testable.

```
ingest/    collectors -> immutable dated raw snapshots (JSON/CSV) in data/raw/
  stats_api.py      MLB Stats API game logs, hitters+pitchers, 2022-2026
  savant.py         Savant CSV pulls (xwOBA, barrel%, sprint speed, whiff%)
  prices_scp.py     SportsCardsPro polite scraper (primary prices)
  prices_130p.py    130point backfill + Best Offer recovery
  pop_gemrate.py    monthly pop-count snapshots for the bounded card set
features/  raw snapshots -> analysis panel (parquet in data/processed/)
  season-to-date stat reconstruction at any date
  weekly card price series (median per card-grade-week, sale-type indicators)
  card x week panel + market index for de-seasonalizing
model/     panel estimation + buy-signal derivation (walk-forward)
events/    event-study overlay reusing the same panel
validate/  post-ingest data-quality checks
```

Storage: raw JSON/CSV snapshots in `data/raw/` (never overwritten), cleaned parquet
in `data/processed/`. No database at this scale.

## 5. Scope guardrails

- **Universe:** ~40–60 players — top MLB rookies/sophomores 2022–2026 with flagship
  rookie cards in PSA 9 and PSA 10, selected for sales liquidity. Roughly 100–150
  card-grade series.
- **Seasons:** 2022, 2023, 2024, 2025, 2026 (five independent post-break seasons;
  cross-season consistency is a core question).
- **Price hygiene:** separate auction vs. BIN; drop lots/reprints (SCP pre-filter +
  title re-audit); never mix raw and graded; weeks with <2 sales = missing (no
  interpolation).
- **Cadence:** weekly in-season pulls (stats overnight; SCP lags 1–2 days). This
  cadence later becomes the live buy-signal pipeline with no rework.

## 6. Modeling spec

Unit of observation: `card × week`.

**Outcome:** week-over-week log price change minus the same-week median change
across all tracked cards (removes market seasonality).

**Predictor blocks** (all lagged to be known at week start — no look-ahead):
- Season-to-date performance: OPS/wRC+-style line, xwOBA, barrel% (hitters);
  ERA/FIP-style line, K-BB%, whiff% (pitchers).
- Recent form: last-14-day stats minus season-to-date stats ("hot streak" delta).
- Static: age, position, draft slot / prospect rank, card set, PSA 9 vs 10.
- Supply: pop count and pop growth at snapshot date.
- Controls: auction-vs-BIN share, playoff-week dummy, award-announcement windows.

**Estimation, three passes:**
1. **LASSO** with stability selection (bootstrap resamples; a stat counts only if
   selected in >80% of fits) → the multivariate importance ranking.
2. **Gradient boosting** (XGBoost or LightGBM) + SHAP → nonlinearities and
   interactions; second importance ranking to cross-check LASSO.
3. **Hierarchical / mixed-effects model** — random slopes per player, position, and
   season → reports the *variance* of each stat's coefficient across groups,
   directly answering consistency across seasons and positions.

**Buy-signal derivation (gated):** walk-forward validation — fit through week *t*,
predict week *t+1* drift from current stats. Success criterion: predicted
top-decile cards outperform the panel median by a statistically significant margin
across held-out seasons, net of ~13% eBay seller fees. If not met, the project
stops at the research stage and this is reported honestly.

**Event-study overlay:** events = MLB debuts/call-ups, 3+ HR games, no-hitters,
All-Star selections, award announcements, playoff series wins. Cumulative abnormal
card returns in (−7d, +1d, +7d, +21d) windows vs. panel-model prediction. Validates
causality and tests the documented 2–3 week mean-reversion of event spikes —
informing the sell side, not just the buy side.

## 7. Data quality & error handling

- Raw snapshots immutable and dated; collectors idempotent (re-runs fill gaps,
  never overwrite).
- `validate/` after every ingest: price outliers (>4σ within card-grade-month
  quarantined, not deleted), title re-parse audit for reprints/lots, stale-series
  report (no sales in 3+ weeks → flagged illiquid, excluded from panel).
- Statcast retro-revisions: features always rebuilt from latest raw snapshot, never
  patched in place.

## 8. Testing

- Unit tests: season-to-date reconstruction (rebuild a known player's line at a
  known date, assert against official season total); title/grade parser;
  auction/BIN classifier.
- Golden sample: one player, one season, hand-checked prices — the full feature
  pipeline must reproduce it exactly.
- Model validation is walk-forward only — no random train/test split (time leakage
  is the classic failure mode for this kind of model).

## 9. Explicit non-goals (this phase)

- No real-money trading, no automated purchasing.
- No NBA implementation (design only keeps the door open).
- No paid data sources, no PSA/GemRate commercial APIs.
- No database infrastructure; no web dashboard.
