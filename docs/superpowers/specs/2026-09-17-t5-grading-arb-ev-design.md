# Design: T5 — Grading-Arbitrage EV Model

Date: 2026-09-17
Status: Approved (design stage)
Program: Research-agenda expansion, thread T5 of 5. Runs after T2 (consumes `class_sales.parquet`, `class_liquidity.csv`); independent of T1/T3/T4.

## 1. Goal

Estimate the expected value of the **buy-raw → grade → sell-PSA-10** pipeline per card: a *manufacturing* edge that converts prediction into production. Rank cards by EV with honest uncertainty; identify which inputs the answer is most fragile to.

Explicitly a **desk model**, labeled as such — not realized P&L, not submission advice.

## 2. Decisions locked in brainstorming

- **Scope:** EV model + sensitivity analysis only. No portfolio optimization, no historical backtest, no BGS/SGC routes (all deferrable).
- **Data:** existing GemRate collector (`pop_gemrate.py`, built and fixture-tested in the 2026-09-15 spike) extended to the T2 liquidity-filtered subset; paired raw + PSA-10 sale prices from `class_sales.parquet`.
- **Budget:** free-first; PSA fee schedules are public web data.

## 3. Data

- **Pop / grade distributions:** extend pop collection to cards with ≥ 24 sales in the trailing year (keeps the ~35 s/card scrape ≈ 2 h and only models cards liquid enough to trade). Full PSA grade counts (10 / 9 / ≤ 8) appended monthly to `data/reference/pop_snapshots.csv` (existing idempotent pattern).
- **Prices:** per card, per grade — empirical trailing-12-month sale-price distribution from `class_sales.parquet` (raw = ungraded bucket; graded = PSA 10).
- **Costs:** PSA service-tier fees + shipping as a **dated config file** (`data/reference/grading_fees.json`, fees change — a date is part of the schema; unknown tier → validation error).

## 4. Model (`src/cardprice/grading_ev.py`)

Per card, Monte Carlo (seeded, deterministic):

1. Draw grade-outcome probabilities from a **Dirichlet posterior** over the card's observed PSA grade counts (fewer total submissions graded → wider uncertainty, automatically). Limitation, stated in the findings doc: historical pop counts reflect past submitters' selection (well-centered copies get submitted first), so they are an optimistic-biased estimate of a new submission's outcome probabilities; sensitivity arm (a) covers this.
2. Draw the post-grade sale price from the card's empirical per-grade distribution; draw the raw acquisition cost from the ungraded distribution.
3. EV = E[sale proceeds − raw cost − grading fee − shipping − marketplace fees (~14 %)]; also report **P(profit > 0)** and a 90 % CI.

**Sensitivity (tornado per card):** (a) gem-rate estimation error (posterior spread), (b) pop inflation — PSA-10 supply growth from our monthly snapshots eroding the graded premium, (c) price slippage −10 % / −20 %, (d) fee-tier changes.

## 5. Architecture & outputs

```
src/cardprice/grading_ev.py       Monte Carlo EV + sensitivity
data/reference/grading_fees.json  dated PSA fee config
data/processed/grading_ev.csv     ranked: card, EV, P(profit>0), 90% CI, top sensitivities
docs/findings/<date>-grading-arb-ev.md
```

## 6. Error handling

- Cards with < 5 graded sales in 12 m: EV computed but flagged `thin_price_evidence`; never silently treated as certain.
- GemRate match failures: existing no-guess rules; ambiguous → audit CSV, excluded from the ranked list, counted in findings.
- Stale pop snapshot (> 45 days) → warning column, not exclusion (pops move slowly).

## 7. Testing

- Golden EV on the Henderson fixture card (spike numbers: **PSA-specific** — PSA 10 pop 2,070 of PSA total 3,129 → PSA gem rate 0.6616), hand-verified. Outcome probabilities use PSA counts only; GemRate's all-grader headline (2,996 / 4,798 = 0.6244) is never mixed into PSA outcome probabilities.
- Dirichlet unit tests: empty counts → prior; large counts → MLE.
- Fee-config validation: dated tiers; unknown service level rejected.
- Monte Carlo seeded; tests deterministic.
