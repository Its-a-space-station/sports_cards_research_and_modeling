# Design: T1 — Price Incorporation Lag (Daily Event Study)

Date: 2026-09-17
Status: Approved (design stage)
Program: Research-agenda expansion, thread T1 of 5. Execution order: **T1 → T2 → T3 → T4, T5 after T2** (locked with user 2026-09-17).
Extends: `docs/findings/2026-09-17-multiyear-modeling.md` (monthly-grain nulls: price persistence dominates; no stable stat effects). T1 asks whether the monthly grain itself hid a fast, tradeable repricing process.

## 1. Goal

Measure how quickly card prices incorporate on-field information at **daily** grain, using per-sale records. The answer gates the program's timing premise:

- If prices fully adjust within ~72 h, reaction-timing strategies are dead; T3 (odds deltas) reframes from *reaction* to *anticipation*.
- If a meaningful share of events take ≥ 7 days to adjust, a timing edge is plausible and later threads proceed as designed.

## 2. Decisions locked in brainstorming

- **Approach:** sale-level event study around statistically-detected breakout games. Blind change-point detection on daily medians is a secondary robustness check only — daily series are too thin (measured 2026-09-17: median 2 sales/card/month; 75 % of card-months have < 5 sales) for reliable unsupervised detection. Sale-level pooling across events is the design response to that thinness.
- **Data budget:** free-first (user decision, program-wide). T1 needs no new collection at all.
- **Sequencing:** T1 first — cheapest thread, and its verdict reframes T3.

## 3. Data (existing only)

- `data/processed/universe_sales.parquet` — 7,243 per-sale records (`sale_date`, `price`, `grade`, `best_offer`, …), 2021-03 → 2026-09.
- `data/processed/game_logs_universe.parquet` — 35,152 game-log rows (MLB + minors) for breakout detection.
- `data/processed/events_universe.parquet` — dated MLB debuts (39) as a second, pre-dated event class.
- Universe limitation: 39 players. Event counts will be modest; CIs reported honestly. T2 expands the cross-section later; T1's job is the lag *measurement*, not breadth.

## 4. Architecture

```
src/cardprice/breakout_events.py   detection: game logs -> breakout events table
src/cardprice/lag_study.py         alignment, lag estimation, robustness checks
data/processed/breakout_events.parquet
data/processed/lag_curves.csv
docs/findings/<date>-incorporation-lag.md
```

No new dependencies (pandas/numpy/scipy stack already present).

## 5. Breakout event definition

- **Hitters:** single-game composite score (total bases + walks + stolen bases); z ≥ 2.5 vs the player's trailing 30-game baseline (min 20 PA in baseline).
- **Pitchers:** Bill James game score, z ≥ 2.5 vs trailing 30-game (min 5 GS).
- **Debuts:** existing dated debut events, second event class (mechanism differs: pure information shock, no performance ambiguity).
- Overlapping ±14 d windows per player deduped — keep the first event, drop overlapping later ones.
- Strata: `prospect` (pre-debut / rookie year) vs `established`, per the panel's career-stage definition — repricing speed plausibly differs by fame level.

## 6. Lag estimation

- Normalize each sale by its card's event-anchored baseline: median of same-grade-class sales strictly before the event over the trailing **56 days** (relative index; 1.0 = baseline). Recalibrated from 28 d on 2026-09-17 (see note below).
- Sales windows collected at **±21 d**; pooled daily median relative-price curve read on **±14 d bins** per stratum; bootstrap CIs clustered by event.
- **Lag** = first day the CI floor clears 1.0 and holds for 3 consecutive days.
- **Half-life**: exponential-adjustment fit to the post-event path.
- Both reported as a **distribution across events**, not one pooled point estimate.

## 7. Bias controls

- Subtract the all-universe daily median index (market drift confound).
- Recompute within grade buckets (ungraded / PSA 10) — hype events shift the *mix* of what sells; a level shift driven by mix is not repricing.
- Event-cards with < 3 sales in the window are dropped and counted (report the count). Recalibrated from < 5 on 2026-09-17 (see note below).
- `best_offer=True` sales flagged; weekday/weekend composition noted.

**Recalibration note (2026-09-17, user-approved):** the original density rules (28 d baseline, ±14 d window, ≥ 5 window sales) kept only 42 event-card pairs of 590 post-floor events on real data — and **zero** prospect-stratum or debut events under any parameter setting (median 2 sales/card-month). Calibrated values (56 d baseline, ±21 d collection window, ≥ 3 window sales) keep 69 pairs / 48 events / 740 sales, still 100 % established breakouts. **Prospect-window and debut repricing are unobservable at this universe's sale density** — a first-class finding of this thread; the §8 gate reads on the established-heavy `ungraded/all` sample, and prospect-window lag measurement is deferred to T2's breadth (or denser price data).

## 8. Pre-registered decision gate

- **≥ 80 % of events fully adjust within 72 h** → reaction-timing dead; T3 reframes to anticipation; findings doc says so plainly.
- **Meaningful share (≥ 20 %) takes ≥ 7 d** → timing edge plausible; T3 proceeds as a reaction-lag study.
- Intermediate outcomes reported as measured; no retrofitted narratives.

## 9. Testing

- Synthetic fixture: planted event with a known 2-day lag must be recovered within ±1 day.
- Golden breakout rows from known real games (hand-picked multi-HR / high-K games), verified against the game logs.
- Unit tests: window dedupe, market-index subtraction, grade-bucket split, event-anchored baseline normalization (no look-ahead: baseline uses sales strictly before the EVENT date; a per-sale trailing baseline would absorb the repricing being measured — wording corrected in the T1 plan).
- Honest-golden discipline: mismatches stop the run; never fudge.
