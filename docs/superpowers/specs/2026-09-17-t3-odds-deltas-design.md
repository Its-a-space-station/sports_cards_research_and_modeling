# Design: T3 — Betting-Odds Deltas as Features

Date: 2026-09-17
Status: Approved (design stage)
Program: Research-agenda expansion, thread T3 of 5. Depends on T1 (daily event frame) and T2 (`panel_class.parquet`); runs after both.
Related: T2 builds the fallback expectations term (ranking/projection revisions), so T3 degrades safely.

## 1. Goal

Add market-priced expectations — award-futures odds (MVP / Cy Young / ROY) — as features. Odds are a liquid, daily-updated forecast of exactly the events our event study found price-relevant; their **deltas** are a candidate leading indicator of card repricing, and their levels let us sharpen T2's surprise term (actual pace − *market* expectation).

## 2. Decisions locked in brainstorming

- **Spike-first:** a time-boxed feasibility spike precedes any production code; historical award-futures availability is the thread's existential risk.
- **Budget:** free-first (program-wide). No paid odds data without explicit user sign-off; the paid option (e.g. The Odds API historical) is documented and escalated, never purchased silently.
- **Framing depends on T1's verdict:** if T1 shows ≥ 80 % adjustment within 72 h, odds deltas are evaluated as *anticipation* features (do odds move before prices?), not reaction signals.
- **No separate gate:** T3 is a feature-add evaluated by delta-fit inside T2's existing walk-forward harness; a standalone gate would double-count evidence.

## 3. Phase 0 — feasibility spike (time-boxed)

Enumerate and probe, in order:

1. **Kalshi** public API — historical candlesticks for any MLB award / season-player markets.
2. **Polymarket** CLOB API — price history for sports-award markets.
3. **The Odds API** free tier — current odds only (starts prospective collection; historical is paid → escalate).
4. Public archives (Wayback captures of odds pages) — spot-check depth.

**Pass criterion:** ≥ 2 full seasons of dated award-futures prices for MLB players, at weekly-or-better cadence.
Spike output: `docs/superpowers/spikes/<date>-odds-history.md` with verdict, verbatim response excerpts, and a coverage table (markets × weeks × players).

## 4. If the spike passes

```
src/cardprice/collect_odds.py     -> data/processed/odds_snapshots.parquet
                                     (player, market, date, implied_prob, source)
src/cardprice/features.py         odds_level, odds_delta_7d, odds_delta_30d
```

- Implied probabilities vig-normalized (unit-tested).
- Merge into `panel_class.parquet` at month grain and into T1's daily event frame.
- Evaluation: incremental importance (LASSO stability + LightGBM gain) and walk-forward delta-fit vs the T2 baseline model.
- **Prospective collection starts regardless of retro verdict** (free sources, weekly snapshot) so future seasons have odds history.

## 5. If the spike fails

- Expectations term = ranking/projection **revision deltas** (already built in T2) + Kalshi/Polymarket where coverage exists (likely 2024+).
- Paid historical-odds option written up and escalated to the user as a decision.
- Findings doc records the null availability result — it is itself useful evidence about what the market could have known.

## 6. Error handling & testing

- Parser fixtures from captured API payloads (committed, like the GemRate fixtures).
- Missing-market weeks: forward-fill ≤ 2 weeks, then NaN — never fabricated.
- Source conflicts (two books disagree): keep per-source rows; features use the primary source, secondary as robustness.
- Tests: vig-normalization golden (hand-computed), forward-fill limit, month-grain merge invariants (no duplicated player-months).

## 7. Explicitly out

- Scraping ToS-hostile odds portals.
- Pre-game / in-play odds.
- Any paid data without user sign-off.
