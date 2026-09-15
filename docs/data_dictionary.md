# Data Dictionary: Analysis Panels

Built by `scripts/build_panels.py` (re-run after `scripts/run_price_pipeline.py`):

```bash
python scripts/run_price_pipeline.py   # refresh scp_weekly.parquet from sales
python scripts/build_panels.py         # write both panel parquets
```

Sources: `data/processed/scp_chart_monthly.parquet`, `data/processed/scp_weekly.parquet`,
`data/processed/game_logs.parquet`, `data/reference/player_info.csv`,
`data/reference/cards_seed.csv`.

## Conventions (both panels)

- **Lag rule (all stats):** every performance stat is a cumulative season-to-date line
  through `period_start - 1 day` (for month M: through the last day of M-1; for week W
  — weeks start Monday — through the preceding Sunday). Rate stats (avg/obp/slg/ops,
  era/whip/k_bb_pct) are always recomputed from summed counting stats over that window,
  never averaged from per-game rates. No stat ever uses a game played on or after the
  period start (no look-ahead; enforced by `tests/test_golden_panel.py`).
- **Debut-period inclusion rule:** a period is kept for a card only if the player's
  first game of `stats_season` is on or before the period end (monthly) resp. on or
  before the lag date (weekly). The debut period itself is kept with stats as of the
  lag date, which may be all zeros/NaN.
- **Legitimate NaNs:** rate stats are NaN when their denominator is zero at the lag
  date (pre-debut / debut-lag rows, e.g. a pitcher with 0 outs). Hitting stat columns
  are NaN on pitcher rows and pitching stat columns are NaN on hitter rows (structural).
  `log_ret`/`excess_ret` are NaN for each series' first period (no prior price).
- **Market-median outcome:** `market_median_ret` is the cross-sectional median of
  `log_ret` over all panel rows in the same period (month resp. week). `excess_ret =
  log_ret - market_median_ret` is the card's market-relative outcome for the period.
- **Current shapes:** panel_monthly 325 rows / 13 cards, 2022-10 → 2026-09;
  panel_weekly 184 rows / 13 cards, 2022-12-12 → 2026-09-07.

## Return horizons

Sales are sparse, so `log_ret` spans the gap between **observed** periods, which is
often longer than one period. The horizon columns disclose this per row. Measured on
the 2026-09-15 build:

- **Weekly (`days_since_prev`, n=135 returns):** only 30.4% are true 7-day returns;
  median horizon 14 days; 65.9% are ≤ 21 days; long tail out to 1281 days (multi-year).
  Heterogeneous horizons also contaminate `market_median_ret`/`excess_ret`, which pool
  these returns per week.
- **Monthly (`months_since_prev`, n=312 returns):** 90.1% (281) are one-month returns;
  28 are 3-month (mostly the off-season Dec→Mar gap); 3 rows are 4–5-month gaps.

**Downstream rule for Plan 4:** filter or weight by horizon — for the weekly primary
analysis use rows with `days_since_prev <= 21`; for the monthly panel flag rows with
`months_since_prev > 2`. Do not treat raw `log_ret` as a fixed-horizon return.

## panel_monthly.parquet

One row per card per calendar month with a PSA-10 chart price point (the card's
rookie season onward). Grade is always `psa_10`.

| Column | Type | Definition | Lag rule |
|---|---|---|---|
| card_slug | str | SCP card identifier `set/player-N` | static |
| mlb_id | int64 | MLB Stats API player id | static |
| player_name | str | Player display name | static |
| month | datetime64 | Month start (first day of month) | period key |
| price | float64 | Last PSA-10 chart price point within the month (USD) | same-month (outcome period) |
| rookie_year | int64 | Card's rookie-year season | static |
| stats_season | int64 | Season the stats are drawn from (= `month.year`; rolls each calendar year) | per period |
| set_slug | str | SCP set identifier | static |
| grade | str | Always `psa_10` in this panel | static |
| playoff | int64 | 1 if month is October, else 0 | deterministic |
| form_games | int64 | Games played in the 14 days ending at the lag date | lagged |
| games | int64 | Season-to-date games played (pitchers: appearances) | lagged (through month-1 end) |
| ops | float64 | Season-to-date OPS (hitters) | lagged; NaN if 0 PA denominator |
| avg | float64 | Season-to-date batting avg = hits/at_bats (hitters) | lagged; NaN if 0 AB |
| obp | float64 | Season-to-date OBP (hitters) | lagged; NaN if 0 PA denominator |
| slg | float64 | Season-to-date slugging (hitters) | lagged; NaN if 0 AB |
| home_runs | float64 | Season-to-date home runs (hitters) | lagged |
| strikeouts | float64 | Season-to-date strikeouts (hitters) | lagged |
| form_ops_delta | float64 | Last-14-day OPS (rebuilt from counting-stat diffs) minus season-to-date OPS (hitters) | lagged; NaN if 0 AB in window |
| form_era_delta | float64 | Last-14-day ERA minus season-to-date ERA (pitchers) | lagged; NaN if 0 outs in window |
| age | float64 | Player age at month start, (month − birth_date)/365.25, rounded to 2 dp | deterministic |
| position | str | Field position from player_info.csv | static |
| era | float64 | Season-to-date ERA (pitchers) | lagged; NaN if 0 IP |
| whip | float64 | Season-to-date WHIP (pitchers) | lagged; NaN if 0 IP |
| k_bb_pct | float64 | (K − BB) / batters faced, season-to-date (pitchers) | lagged; NaN if 0 BF |
| innings_pitched | float64 | Season-to-date IP, MLB thirds notation converted via outs (pitchers) | lagged |
| log_ret | float64 | ln(price_t / price_{t-1}) within card, ordered by month | outcome; NaN first month |
| months_since_prev | float64 | Whole months between this row's month and the previous observed month for the card (Jun→Oct = 4) | NA for each card's first month; >1 on sparse gaps |
| market_median_ret | float64 | Median of log_ret across all cards that month | outcome |
| excess_ret | float64 | log_ret − market_median_ret | outcome; NaN first month |

## panel_weekly.parquet

One row per card × grade per week (Monday week start) from
`scp_weekly.parquet`; covers all graded sales series (psa_10, psa_9, sgc_10, …),
not just PSA 10.

| Column | Type | Definition | Lag rule |
|---|---|---|---|
| card_slug | str | SCP card identifier `set/player-N` | static |
| grade | str | Grade of the sales series (e.g. `psa_10`, `sgc_9.5`) | static |
| mlb_id | int64 | MLB Stats API player id | static |
| player_name | str | Player display name | static |
| week | datetime64 | Week start (Monday) | period key |
| price | float64 | Median sale price that week (USD) | same-week (outcome period) |
| n_sales | int64 | Number of sales in the week | same-week |
| best_offer_share | float64 | Share of the week's sales accepted via best offer | same-week |
| rookie_year | int64 | Card's rookie-year season | static |
| stats_season | int64 | max(rookie_year, week.year); rolls each calendar year | per period |
| set_slug | str | SCP set identifier | static |
| playoff | int64 | 1 if week starts in October, else 0 | deterministic |
| form_games | int64 | Games played in the 14 days ending at the lag date (Sunday before week start) | lagged |
| games | int64 | Season-to-date games played (pitchers: appearances) | lagged (through preceding Sunday) |
| ops | float64 | Season-to-date OPS (hitters) | lagged; NaN if 0 PA denominator |
| avg | float64 | Season-to-date batting avg (hitters) | lagged; NaN if 0 AB |
| obp | float64 | Season-to-date OBP (hitters) | lagged; NaN if 0 PA denominator |
| slg | float64 | Season-to-date slugging (hitters) | lagged; NaN if 0 AB |
| home_runs | float64 | Season-to-date home runs (hitters) | lagged |
| strikeouts | float64 | Season-to-date strikeouts (hitters) | lagged |
| form_ops_delta | float64 | Last-14-day OPS minus season-to-date OPS (hitters) | lagged; NaN if 0 AB in window |
| form_era_delta | float64 | Last-14-day ERA minus season-to-date ERA (pitchers) | lagged; NaN if 0 outs in window |
| age | float64 | Player age at week start, (week − birth_date)/365.25, rounded to 2 dp | deterministic |
| position | str | Field position from player_info.csv | static |
| era | float64 | Season-to-date ERA (pitchers) | lagged; NaN if 0 IP |
| whip | float64 | Season-to-date WHIP (pitchers) | lagged; NaN if 0 IP |
| k_bb_pct | float64 | (K − BB) / batters faced, season-to-date (pitchers) | lagged; NaN if 0 BF |
| innings_pitched | float64 | Season-to-date IP via outs conversion (pitchers) | lagged |
| log_ret | float64 | ln(price_t / price_{t-1}) within card × grade, ordered by week | outcome; NaN first week of series |
| days_since_prev | float64 | Days between this row's week and the previous observed week for the card × grade | NA for each series' first week; >7 on sparse gaps |
| market_median_ret | float64 | Median of log_ret across all card × grade rows that week | outcome |
| excess_ret | float64 | log_ret − market_median_ret | outcome; NaN first week of series |
