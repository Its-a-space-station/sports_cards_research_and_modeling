# T3 — Betting-Odds Deltas as Features

Date: 2026-09-19
Plan: docs/superpowers/plans/2026-09-19-t3-odds-deltas.md
Spec: docs/superpowers/specs/2026-09-17-t3-odds-deltas-design.md
Spike (governs sources): docs/superpowers/spikes/2026-09-19-odds-history.md

Every number below is recomputed from the artifacts — `data/processed/odds_snapshots.parquet`,
`data/reference/odds_events_registry.csv`, `data/reference/odds_unmapped_audit.csv`,
`data/processed/class_hold_odds.parquet`, `data/processed/class_odds_delta_fit.json`,
`data/processed/class_hold_importance_odds.csv`, `data/processed/odds_spike_events.parquet`,
`data/processed/odds_lag_summary.json` — not from run-log memory. Artifacts in
`data/processed/` are gitignored per repo convention; the two `data/reference/` CSVs are committed.
Per spec §2 there is **no separate odds gate**: all evaluation output is descriptive, and the
registered T2 verdict (ungraded/12m/hitter, base features, PASS) stands untouched.

## Headline

- **Delta-fit: with_odds adds exactly nothing anywhere — and the design guarantees that null.**
  All six cells: delta_mean_excess +0.0000; the only nonzero movement anywhere is the 6m cell's
  2026 Spearman fold (+0.0055, no top-5 pick change). Odds snapshots span entry years 2025–2026
  only, and the walk-forward trains on years < y, so the odds columns have zero train variance in
  every fold except that one. **This null is a coverage/design artifact, not evidence that odds
  carry no signal** — the comparison becomes informative only when ≥ 1 full market season sits
  inside the training window of a cell with evaluable test years.
- **Importance:** only `has_market` registers at all (lasso stability 0.05, tied-5th; shap #16);
  `odds_level`/`odds_delta_7d`/`odds_delta_30d` are never selected. price_level and ret_3m dominate
  every method exactly as in the T2 base fit.
- **Odds-spike lag:** 111 spike events over 40 players, but only **13 event-card pairs** survive
  T1's evidence rules in the primary cell — lag_days null, pre-period check failed, 12/13 events
  classified insufficient. In spec §8 language: **intermediate — reported as measured**, with the
  binding constraint again being sale density (T1's first-class finding, reproduced on a second,
  independent event type).

## Data

**Coverage (registry, 18 rows):** 12/12 Polymarket award cells + 6/6 Kalshi series, no missing
cells. 2025 cells (AL/NL × MVP/CY/ROY, 26–32 outcomes each) collected through 2025-11-11..14
(award settlement — market reality, not a gap); 2026 cells (46–69 Polymarket outcomes; 30–53
Kalshi markets) through 2026-09-19. Snapshots: **49,543 rows** (38,375 Polymarket / 11,168
Kalshi), UTC-daily, 2025-03-25 → 2026-09-19, `implied_prob` ∈ [0.0002, 0.9995], zero NaT.

**Price semantics per source.** Polymarket rows are vig-normalized: per (event, date) the
implied probs sum to exactly 1.0000 (min = max, reverified). Kalshi rows are raw single-outcome
binary closes — `implied_prob == raw_price` on every Kalshi row (reverified), no vig removal —
flagged by `source`. The two are **not interchangeable**: per player-day, taking the max
|pm − kalshi| across that day's same-player market pairs (5,007 player-days carry both
sources), the divergence averages **4.9pp with max 0.97** — against a 0.10 spike floor.

**Mapping (exact normalized-name only, never fuzzy).** 32,472 mapped rows over **178 distinct
players** (107 Polymarket, 156 Kalshi, 85 both); 17,071 rows carry no mlb_id. Audit:
**441 unmapped rows / 200 distinct labels / 0 fetch gaps** — 22 rows are single-letter
placeholder outcomes (Polymarket's unused slots), the remaining 419 are real player names
genuinely absent from both info CSVs (Ohtani, Betts, J. Ramírez, Ben Rice, Fried, Murakami… —
pre-class veterans or players outside the class/universe; every last-name near-miss inspected
during collection was a *different* player). Consequence: pre-class stars' odds are collected
but can never join the card panel (recorded, not an error).

**Ceilings and caveats.** Free coverage ceiling = **2 seasons** (2025 full, 2026 through
09-19): Polymarket has no earlier award markets and Kalshi's settled 2025 markets are not
retrievable unauthenticated — Kalshi is a 2026-only **cross-source check**, not a second
history. The Odds API historical is paid: recorded as an option for the user, never called
(spec §2 — escalation, not purchase). Volume semantics differ by source and must not be mixed:
Polymarket = gamma lifetime market volume repeated per date row; Kalshi = daily candle
contracts.

## Delta-fit (descriptive; T2 registered verdict untouched)

Recomputed baselines (top-5 picks path, identical harness calls to `run_class_gate`) reconcile
**exactly (4dp) with T2's registered/found numbers in all six cells** — checked before reading
any with_odds number. Delta = with_odds − baseline; delta_spearman = mean over test years of
per-year spearman(predicted, realized) deltas (all_scores path). From
`class_odds_delta_fit.json`:

| cell | baseline mean_excess | with_odds | delta_mean_excess | delta_spearman |
|---|---|---|---|---|
| ungraded/12m/hitter (T2 registered) | 2.6714 | 2.6714 | +0.0000 | +0.0000 |
| ungraded/6m/hitter | 1.6881 | 1.6881 | +0.0000 | **+0.0014** (2026 fold: +0.0055) |
| ungraded/24m/hitter † | 0.9668 | 0.9668 | +0.0000 | +0.0000 |
| ungraded/36m/hitter † | 0.4949 | 0.4949 | +0.0000 | +0.0000 |
| psa_10/12m/hitter | 0.3134 | 0.3134 | +0.0000 | +0.0000 |
| ungraded/12m/pitcher | 1.0131 | 1.0131 | +0.0000 | +0.0000 |

† anticipated-degenerate cells: zero has_market=1 rows (their non-null targets predate the odds
era), so with_odds features are identically zero and the deltas are zero *by construction* —
computed and reported as measured, with the JSON carrying the one-line note.

**The mechanism of the zero deltas (the honest headline).** has_market=1 rows exist only in
entry years 2025–2026 (1,096 of 113,471 stacked rows: 829 in 2025, 267 in 2026 — the 2-season
snapshot ceiling), and a walk-forward test fold for year y trains on years < y. The odds
columns therefore have **zero train variance in every fold except test-year 2026** — and only
the 6m cell has 2026 test rows (12m+ targets from 2026 entries don't exist yet). With a
constant-zero train column the harness standardizes with sd→1 and the lasso coefficient is
exactly 0, so with_odds predictions are bitwise the baseline's (the deltas are exact 0.0, not
rounding dust). The single fold where an odds coefficient could even be estimated — 6m, test
2026, training through the 2025 market rows — moves that year's Spearman by +0.0055 and no
top-5 pick, hence gate delta +0.0000 there too. **The null is a coverage/design artifact, not
evidence of no signal**; it cannot detect odds signal until a market season falls inside a
training window of an evaluable cell.

NaN discipline: 541 has_market=1 rows without a pre-entry snapshot (early-season entries) keep
NaN odds features by design — never zero-filled; the harness's per-fold train-median imputation
handles them as ordinary NaNs.

## Importance with odds

Registered cell (ungraded/12m/hitter, 26,640 rows), `CLASS_HOLD_FEATURES + odds`, same
full-frame median imputation caveat as T2's importance runner (descriptive rankings, not
out-of-sample). From `class_hold_importance_odds.csv`:

- **lasso_stability:** price_level 1.00, ret_3m 1.00, log_career_games 0.28, log_minor_games
  0.10, then **has_market 0.05 tied-5th** (with career_hr_rate); odds_level/odds_delta_7d/
  odds_delta_30d all 0.00 — never selected.
- **lasso_path:** has_market 6th by |coef| (+0.008); the three odds level/delta columns
  absent (zero on the full path).
- **gbm_shap:** has_market **#16 (0.0002)**; the other three bottom-tied at #19 (0.0000).
  price_level 0.144 and ret_3m 0.087 dominate; draft_rank keeps its #5 slot.
- **gbm_cv_r2:** 0.3969 with odds vs 0.3971 base-features — unchanged within noise.

Expected arithmetic: only 208/26,640 rows in this cell have a market, and 100 of those carry
NaN odds features (median-imputed here). Quote as a coverage artifact, not evidence about
odds' true information content.

## Odds-spike lag (descriptive; T1 machinery verbatim)

**Events.** `odds_spike_events` over the **Polymarket** rows (mapped players): |1-day delta| ≥
max(0.10, 3 × rolling-30d σ of daily deltas), largest |delta| kept per 7-day same-direction
cluster. **111 events over 40 players** (61 in 2025, 50 in 2026; 2025-05-03 → 2026-09-08;
strata: 76 established / 35 prospect) → `odds_spike_events.parquet`. Kalshi rows are excluded
from event detection by design: raw unnormalized closes, 2026-only, and mixing sources in one
daily-diff series injects cross-source artifacts — in a mixed-source probe (reviewed spike
detection over the two-source frame, keeping the higher-implied_prob row per player-day), 37 of
132 events have a different source than the previous daily observation, and the count is
tie-break-sensitive (15–37 of 123–136 events across stated per-day tie-breaks; see Data). A
Kalshi-only run would add 40 events / 23 players (2026 only); not used.

**Sales & config.** Union frame = class_sales ∪ universe_sales deduped on (card_slug,
sale_date, title, price): **54,425 rows** (47,892 + 7,243 − 710 duplicate rows; the frames
share the SCP source and overlap on some bowman_1st cards). T1's recalibrated defaults:
56d event-anchored baseline (≥ 3 sales), ±21d window (≥ 3 sales), market index per grade class
over the union frame, cluster bootstrap (2,000 resamples, seed 42). Primary cell =
ungraded/all. From `odds_lag_summary.json`:

| run | kept pairs (events) | drop_log (baseline/window) | lag_days | pre_ok (cover) | adjustment classes |
|---|---|---|---|---|---|
| ungraded/all | 13 (13) | 127 / 3 | null | **false** (0.286) | insuff 12 · fast 1 |
| ungraded/established | 13 (13) | 81 / 2 | null | false (0.286) | identical to ungraded/all |
| ungraded/prospect | 0 | 46 / 1 | — | — | **too few event-card pairs (0), skipped** |
| psa_10/all | 14 (14) | 111 / 11 | null | **false** (0.571) | insuff 12 · no_adj 1 · interm 1 |
| psa_10/established | 9 (9) | 70 / 11 | 3 | false (0.357) | insuff 8 · no_adj 1 |
| psa_10/prospect | 5 (5) | 41 / 0 | 7 | false (0.357) | insuff 4 · interm 1 |

Zero sales went unadjusted for market-index NaN days. Curve sales in bins: 100 (ungraded/all),
116 (psa_10/all); key bins of the primary curve — t=0: 0.997 [0.997, 0.997] (n=5); t=+1: 1.080
[0.925, 1.944] (n=9); t=+3: 1.734 [0.695, 8.748] (n=4); t=+7: 1.081 [0.865, 1.208] (n=4) — wide,
handful-of-sales bins, and the pre-period is not clean.

**Verdict in spec §8 language (descriptive — T3 has no registered lag gate), primary
ungraded/all:** fast share (≥ 80% adjusted within 72h) = 0.077; late share (≥ 20% adjusting at
≥ 7d) = 0.000 → **intermediate — reported as measured**. Neither reaction-timing-dead nor
timing-edge-plausible is established.

**Two honest qualifiers, both first-class:**

1. **The yield assumption did not hold — verify, don't assume.** Award-candidate cards are
   more liquid than the class median, but the kept-pair yield is *worse* than T1's in absolute
   terms (13 pairs / 13 events vs T1's 69 / 48): the 111 events map to only 143 candidate
   event-card pairs (median 1 card per player — one 1st-Bowman card each), and only 16 of
   those pairs have ≥ 3 baseline sales inside 56 days. The binding constraint is again sale
   density at daily grain — T1's observability finding, reproduced on a second, independent
   event type (and, as in T1, zero prospect-stratum pairs survive in the ungraded class).
2. **Every lag read carries `pre_ok = false`.** estimate_lag's own baseline-sanity check fails
   in all runs (pre-period CI coverage of 1.0 at shares 0.286–0.571 vs the calibrated 0.8
   bar), so even the null in the primary cell and the 3d/7d `lag_days` values in the thin
   psa_10 substrata (5–9 pairs) are fragile, as-measured numbers — not evidence of 3–7 day
   incorporation.

## Caveats

- **2 seasons of odds** (2025 full, 2026 through 09-19) — the free ceiling; every odds-side
  null above is conditioned on it. The delta-fit becomes informative only once ≥ 1 full market
  season sits inside walk-forward training windows.
- **Longshot-dominated books:** over half of mapped Polymarket snapshot rows price below 2%
  implied probability (share < 0.02: CY 0.61, MVP 0.56, ROY 0.55), and Kalshi's ROY series are
  its thinnest (30 listed markets each — 28 (AL) / 30 (NL) carrying any traded candle — vs
  42–53 in its MVP/CY series). Most listed candidates can never produce a 0.10 absolute spike, so spike
  coverage concentrates on contenders — ROY-only players still contributed 40 of the 111
  events.
- **No sportsbook odds:** prediction-market prices only (Polymarket + Kalshi); The Odds API
  historical is paid — recorded as an option, never called.
- **has_market default semantics:** has_market=0 ⇒ all odds features 0.0 is the modeling choice
  "no listed award market ≈ zero priced expectation", not data; 541 has_market=1 rows keep NaN
  odds features (no pre-entry snapshot) by design — never zero-filled, harness train-median
  imputation per fold.
- **Cross-season carryover (F1 doc-debt from the Task-3 review):** `month_grain_features`
  filters `has_market` by season but takes `odds_level` from *any* pre-entry snapshot, so **30
  rows** (entry year 2026, 10 players) carry has_market=0 with non-zero odds features inherited
  from 2025 snapshots. Code untouched (semantics intended — the level snapshot always predates
  entry, so no look-ahead); the module docstrings now document this exception (final-review wave).
- **Non-class candidates' odds are panel-invisible:** pre-class stars (Judge, Ohtani,
  Raleigh…) have collected odds but no class card rows; 17,071 unmapped snapshot rows are
  expected and audited, not a join bug.
- **Strata fallback:** `assign_strata` uses class game logs (level == "mlb"); players absent
  from those logs fall to `career_stage`'s "prospect" — acceptable, noted.
- **Kalshi cross-source role only:** raw closes, 2026-only, excluded from spike detection
  (player-day max |pm − kalshi| averages 4.9pp, max 0.97, over 5,007 paired player-days;
  mixed-source spike events switch source vs the previous day at 15–37 of 123–136 events
  depending on the per-day source tie-break).

## Reproduce

From the repo root (worktree venv: `.venv/bin/python`; artifacts in `data/processed/` are
gitignored and rebuilt by these steps):

```bash
# 1. Odds snapshots (full backfill, ~73 min cold / ~7 min warm-resume; live)
PLAYWRIGHT_BROWSERS_PATH=.pw-browsers .venv/bin/python scripts/collect_odds.py

# 2. Odds features into the hold frame (6 cells, stacked)
.venv/bin/python scripts/build_odds_features.py

# 3. Delta-fit + importance with odds (descriptive)
.venv/bin/python scripts/run_odds_delta_fit.py

# 4. Odds-spike lag study (descriptive; writes odds_spike_events.parquet + odds_lag_summary.json)
.venv/bin/python scripts/run_odds_lag.py
```

**Prospective weekly refresh (cron creation is the controller's step — documented here, not
created by this task):** Wednesdays 09:23 local, in the *main* checkout
(`/Users/tomcruise/sports_cards_research_and_modeling` — note it has no venv; create/reuse one
per the repo convention / worktree venv recipe first if absent):

```bash
cd /Users/tomcruise/sports_cards_research_and_modeling \
  && PLAYWRIGHT_BROWSERS_PATH=.pw-browsers .venv/bin/python scripts/collect_odds.py --since 14d \
  && .venv/bin/python scripts/build_odds_features.py \
  && .venv/bin/python scripts/run_odds_delta_fit.py
```

`--since 14d` skips *fetching* chunks ending before today−14; snapshot-resume loads existing
chunks from `data/raw/odds/`, so only the moved trailing chunks and Kalshi candles are
refetched. Commit any changed `data/reference/*.csv` (audit/registry) after the refresh. The
first real `--since` run against the populated store should be eyeballed (collector concern
C5). Delete the cron if the collector is retired. Rerun step 4 too if refreshed lag numbers
are wanted.

Verification at this state: `.venv/bin/python -m pytest -q` → 262 passed;
`.venv/bin/ruff check src tests scripts` → clean.
