# Event Study Findings — 2026-09-16

Runner: `scripts/run_event_study.py` (full stdout: `docs/findings/2026-09-16-event-study-output.txt`).
Engine and registry validated on synthetic planted-signal panels in Tasks 1–3; this is the real-data
pass and the final plan of the five-plan arc. Predecessor: `docs/findings/2026-09-15-modeling-report.md`
(Plan 4 gate FAIL — continuous stats show no tradeable signal; this plan tests the discrete-event channel).

Method in one paragraph: events are matched to the card's panel rows (`mlb_id` join) in a
`[-1, +2]`-month window around the event month; the event-month CAR is the sum of `excess_ret`
(market-median-adjusted log returns) at offset 0 over rows passing the horizon rule
(`months_since_prev <= 2`); significance is a 5,000-draw permutation test (seed 42) of the mean CAR
against pseudo-events drawn from all 312 non-null monthly `excess_ret` cells (pool mean −0.0048,
SD 0.1974). Weekly is the same machinery on the sparse weekly panel and is descriptive only.

## 1. Event registry summary

134 events across 16 players, written to `data/processed/events.parquet` (game-log events derived
offline; playoff and award events fetched live from the MLB Stats API on 2026-09-16 — no fetch
failures, no WARN fallback):

| event_type | registry n | rule |
|---|---|---|
| debut | 16 | first game in the loaded logs (all 16 players debuted 2022+) |
| four_hit_game | 67 | hitter game, hits ≥ 4 (and not a 3-HR game) |
| ten_k_game | 27 | pitcher game, strikeouts ≥ 10 |
| playoff_appearance | 17 | one per player-season with postseason games |
| award_win | 6 | MVP / Cy Young / ROY winners in the seed set |
| three_hr_game | 1 | hitter game, home runs ≥ 3 |

Per player (rows sorted by total):

| player | debut | four_hit | ten_k | playoff | award | 3-HR | total |
|---|---|---|---|---|---|---|---|
| Spencer Strider | 1 | 0 | 19 | 2 | 0 | 0 | 22 |
| Julio Rodriguez | 1 | 13 | 0 | 2 | 1 | 0 | 17 |
| Bobby Witt Jr | 1 | 11 | 0 | 1 | 0 | 0 | 13 |
| Jackson Chourio | 1 | 10 | 0 | 2 | 0 | 0 | 13 |
| Gunnar Henderson | 1 | 7 | 0 | 2 | 1 | 0 | 11 |
| Paul Skenes | 1 | 0 | 7 | 0 | 2 | 0 | 10 |
| Adley Rutschman | 1 | 6 | 0 | 2 | 0 | 0 | 9 |
| Jackson Merrill | 1 | 6 | 0 | 2 | 0 | 0 | 9 |
| Corbin Carroll | 1 | 3 | 0 | 1 | 1 | 0 | 6 |
| Anthony Volpe | 1 | 2 | 0 | 2 | 0 | 0 | 5 |
| Jacob Wilson | 1 | 3 | 0 | 0 | 0 | 0 | 4 |
| Jordan Walker | 1 | 3 | 0 | 0 | 0 | 0 | 4 |
| Wyatt Langford | 1 | 3 | 0 | 0 | 0 | 0 | 4 |
| Nick Kurtz | 1 | 0 | 0 | 0 | 1 | 1 | 3 |
| Roki Sasaki | 1 | 0 | 1 | 1 | 0 | 0 | 3 |
| Roman Anthony | 1 | 0 | 0 | 0 | 0 | 0 | 1 |

Awards: J-Rod 2022 AL ROY (2022-11-14), Carroll 2023 NL ROY and Henderson 2023 AL ROY
(both 2023-11-13), Skenes 2024 NL ROY (2024-11-18), Skenes 2025 NL Cy Young (2025-11-12),
Kurtz 2025 AL ROY (2025-11-10). The single three-HR event is Kurtz's 4-HR game on 2025-07-25.

**Date-accuracy evidence.** Golden debuts assert against the real parquet and pass
(`tests/test_events_gamelogs.py::test_golden_real_debuts`): Henderson 2022-08-31 and Skenes
2024-05-11 match their known MLB debut dates; the same values are in `events.parquet`. All six
award events carry the exact announcement date from the API payload — the Nov-15 fallback was
never triggered (no `(announcement date approximated: Nov 15)` marker in any `details` field).
Playoff event dates are first-postseason-game dates from the player game log and spot-check
correctly (Carroll 2023-10-03 NLWC G1; Volpe 2024-10-05 ALDS G1). Doubleheaders collapse to one
player-day before milestone rules apply (tested in Task 1).

**Coverage caveat for everything below:** 3 of the 16 registry players (Rutschman, Strider,
Roman Anthony) have no card in the price panel, so their 32 events — including Strider's 19
ten-K games — never enter the study. The study universe is the 13 paneled players only.

## 2. Do events move card prices?

Monthly, event-month CAR (offset 0), permutation p (5,000 draws, seed 42):

| event_type | n (events w/ offset-0 CAR) | mean CAR | p | null SD | ~5%-detectable \|CAR\| |
|---|---|---|---|---|---|
| four_hit_game | 50 | −0.0054 | 0.843 | 0.028 | 0.055 |
| playoff_appearance | 13 | −0.0662 | 0.196 | 0.054 | 0.107 |
| ten_k_game | 6 | −0.0503 | 0.445 | 0.079 | 0.158 |
| award_win | 5[^1] | −0.0278 | 0.695 | 0.086 | 0.173 |
| debut | **0** | — | — | — | — |
| three_hr_game | **0** | — | — | — | — |

[^1]: One of the five award CARs is a placeholder 0.0: Skenes' 2024 ROY month (2024-11) is his
card's first panel month, whose `excess_ret` is NaN and sums to 0.0. The sixth award event
(Kurtz 2025 AL ROY, announced 2025-11-10) drops out entirely because November 2025 precedes
his card's first panel month (2025-12). The four informative award
CARs are −0.131 (J-Rod), −0.212 (Carroll), −0.123 (Henderson), +0.327 (Skenes Cy Young);
their mean is −0.0347 — the conclusion is unchanged either way.

**Verdict: no detectable event-month price response for any measurable event type.** Every
measurable mean CAR is *negative* (−0.005 to −0.066) and none approaches significance
(smallest p = 0.196). The best-powered row is four_hit_game (n = 50): an event-month effect of
roughly ±5.5 log-points would have been detectable at 5%, and the observed mean is −0.005 —
so for that event class the null is a moderately informative one. The others are underpowered:
ten_k needs ~±16 points and awards ~±17 points to show up at these n's; their nulls are
"not detected", not "absent".

**The two highest-stakes event types are structurally unobservable in this panel** (n = 0, not
a null result — no test could run):

- **debut**: for all 13 paneled players, the first PSA-10 panel month is **3 or more months
  after the debut month** (e.g. Witt debuts 2022-04, first price 2022-10; Skenes debuts
  2024-05, first price 2024-11; Kurtz debuts 2025-04, first price 2025-12). Cards must be
  printed, pulled, graded, and traded before a PSA-10 price series exists, so the entire
  debut window `[-1, +2]` always precedes the first observation.
- **three_hr_game**: the only registry event (Kurtz 2025-07-25) precedes Kurtz's first panel
  month (2025-12) by five months.

**Pooled weekly study (2026-dominated window, descriptive):** n = 5 events landed in covered
weeks; mean CAR +0.0348, p = 0.801. Composition warning: three of the five weekly CARs are
placeholder 0.0s (first-observation weeks with NaN `excess_ret`: Volpe 2024-04, Langford
2025-07, Merrill playoff 2024-10). Only two are informative — Skenes 10-K game week of
2026-07-25 (+0.152) and Merrill 4-hit game week of 2026-07-19 (+0.022) — mean +0.087 on n = 2.
Directionally positive but far from inferential.

**Multiple testing:** five p-values were computed (four monthly event types + one weekly
pooled). Bonferroni at 5 tests gives α = 0.010; the smallest observed p is 0.196. **Nothing
survives correction — and nothing was close before correction either**, so the conclusion is
robust to however the family is defined.

## 3. Mean reversion

Setup: for events with a positive event-month CAR, the "giveback" is CAR[0,+1] − CAR[0], i.e.
the month-after excess return; negative giveback = the spike faded.

| event_type | positive spikes | mean next-month giveback | share giving back any |
|---|---|---|---|
| four_hit_game | 24 | **+0.0359** | 46% |
| playoff_appearance | 5 | +0.0446 | 20% |
| award_win | 1 | +0.1393 | 0% |
| ten_k_game | 1 | +0.0464 | 0% |

**Verdict: no reversion visible at the monthly grain — descriptively, mild continuation.**
Mean next-month giveback is *positive* in all four groups (prices rose further on average),
and only the four_hit group has a meaningful count: there, 46% of positive spikes gave back
some value the next month — essentially a coin flip — with a mean of +3.6 points. Given
Section 2's null (mean event-month CAR ≈ 0), these "spikes" are mostly ordinary positive
noise months, so this is close to a regression-to-the-mean measurement with the noise winning.

Against the hobby claim that event spikes revert within 2–3 weeks: this dataset **cannot
confirm or refute it**. A spike that fades over 2–3 weeks lives inside one calendar month —
the event month's CAR nets the rise and the fade against each other, and the next-month cell
only catches whatever is left. The weekly panel is the right grain but has only 2 informative
event-week observations. The one with follow-up data (Skenes' 10-K game, event week +0.152)
shows mixed signs over the next three covered weeks (−0.05/+0.05 across the two grade rows,
then 0.00/+0.19, then 0.00/−0.13) — no clean fade, but n = 1 event: anecdote, not evidence.

## 4. Implications

- **What this adds to the Plan 4 null.** Plan 4's gate FAILED for continuous stats (empty
  LASSO path, no stable feature, fee-aware edge −12.1%/month). The discrete-event channel is
  not different: the best-powered event class (four_hit, n = 50) shows a −0.005 event-month
  CAR against a ±0.055 detectable band, and no event type shows a detectable price response.
  Combined arc conclusion: across 13 cards, 2022–2026, neither *how well* a player plays
  (levels/form) nor *discrete on-field shocks* (milestones, awards, playoff appearances)
  produce a measurable, fee-beating signal in PSA-10 rookie card prices at the monthly grain.
  The two halves are consistent with an efficient-enough market at this grain — or with
  effects smaller than this dataset can see; the power caveats are real (Section 5).
- **Does "buy the breakout game" have any support?** No. The one adequately-powered milestone
  class (4-hit games) shows no event-month bump. The headline retail narrative — buy the
  debut, buy the 3-HR game — is untestable here for a structural reason: PSA-10 price history
  does not exist until months after those events. If the profitable version of that trade
  exists, it lives in raw/ungraded or just-graded copies in the days-to-weeks after the event,
  which this dataset never observes.
- **Sell timing.** There is no statistical support for "sell into the event spike" either:
  next-month giveback after positive event months is on average *positive*, not negative. But
  with 24 usable spikes in the best group and a grain too coarse for the 2–3-week claim, this
  is absence of evidence, not evidence of absence.
- **What would sharpen it:** (1) weekly price history with real depth — the weekly panel is
  131/184 rows in 2026 alone; another 1–2 seasons of accumulation turns the weekly event study
  from n = 2 informative events into dozens and directly addresses the 2–3-week reversion
  claim; (2) a bigger universe — more cards per player (base, parallels, grades) and more
  players would lift event counts per type (ten_k_game is n = 6 of 27 registry events —
  19 are Strider's, lost to the unmatched-card problem; resolving the 3 unmatched seed cards
  matters more for events than it did for levels); (3) earlier price discovery — SCP chart data starts when graded
  supply trades, so debut-window questions need a different source (raw-card sales,
  presale/prospect products) entirely; (4) raw-grade and low-pop parallel panels to test
  whether the event response lives in a different market segment than PSA-10 base.

## 5. Limitations

- **Power / event counts.** After losing 3 unmatched players and pre-panel events: four_hit
  50, playoff 13, ten_k 6, award 5, debut 0, three_hr 0; weekly pooled has 2 informative
  events. Only the four_hit null is a tight one (±0.055); everything else detects only very
  large effects. three_hr_game (1 registry event) and debut (unobservable) are reported as
  descriptive/structural, not inferential.
- **Structural censoring of early-career events.** First PSA-10 observation is ≥3 months
  after debut for all 13 paneled players, so the highest-hypothesized-alpha events are the
  least observable; the study is silent on them, not negative on them.
- **Announcement-date approximation for awards — not triggered, but noted.** All six award
  events carried exact API announcement dates; had the payload lacked dates, the Nov-15
  fallback would have placed events at month grain only. Award *leaks* (voting results
  rumored before announcement) are unmodeled.
- **Monthly grain.** A spike-and-fade inside one month is invisible; the reversion test
  measures only what survives into the next calendar month. The 2–3-week hobby reversion
  claim is outside this grain's resolution.
- **Universe.** 13 cards, one card per player, PSA-10 only on the monthly grain; 3 of 16
  registry players unmatched to any card (costing 32 events, 19 of them Strider ten-K games).
  Results do not generalize across grades, parallels, or players. The weekly panel can carry
  two grade rows (e.g. PSA 10 and PSA 9) for one card-week under a single `card_slug`; every
  reported weekly event-week CAR happened to come from a single row, so no result is affected,
  but weekly CAR windows spanning multiple offsets mix grades within an event.
- **Multiple testing.** Five p-values computed; Bonferroni α = 0.05/5 = 0.010. Nothing
  survives (minimum p = 0.196), and no result was marginal enough for the correction to
  change any conclusion.
- **Placeholder-zero CARs.** Events whose only in-window rows have NaN `excess_ret` (first
  observed month/week for a card) sum to a 0.0 CAR that is kept by the significance test
  (one case in award_win, three in the weekly pool). Flagged in Section 2; no conclusion
  changes when they are excluded, but future runs should treat them as missing, not zero.
- **Permutation-pool simplification (carried from Task 3).** The null pool includes actual
  event-window cells; with ≤50 of 312 cells affected this mildly widens the null, biasing
  toward non-rejection — conservative given the all-null outcome.
