# P6b: Multi-Year Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the multi-year panel — one row per (card, series, entry month) with strictly-lagged predictors (career-to-date MLB line, minors pedigree, career stage, awards-to-date, price level, market regime) and annualized log-return outcomes over 6/12/24/36-month holds.

**Architecture:** New focused modules reusing proven machinery: `career.py` (career-to-date + pedigree + stage, built on `season_stats.py`'s cumulative-line functions), `multiyear.py` (month-ends per series + hold-return builder), a small events/info collector (reuses `events.fetch_award_events`, `stats_api.fetch_player_info`), and a CLI assembler. Ungraded is the primary series; psa_10 the robustness check; psa_9 excluded (see Facts).

**Tech Stack:** Python 3.11+, existing `cardprice` package. No new dependencies, no new network sources.

**Spec:** `docs/superpowers/specs/2026-09-16-multiyear-minors-design.md` (§6 panel/features, §9 plan shape)
**Predecessor:** P6a merged (`main` @ 980a6b2) — universe data collected and verified.

## Facts established (from P6a artifacts — verified on disk 2026-09-16/17)

- `data/processed/game_logs_universe.parquet` — 35,152 rows / 39 players; columns from `game_log_to_frame` (`mlb_id, group, season, date` + raw camelCase stat cols like `homeRuns`, `inningsPitched`) plus `level` ∈ {`mlb`, `aaa`, `aa`, `a_plus`, `a`}. Goldens: Henderson 2022 AAA = 65 rows, Bryant 2015 MLB = 151, Soto 2017 A = 23.
- `data/processed/universe_chart_monthly.parquet` — 14,930 rows / 59 cards; meta cols `player_name, mlb_id, rookie_year, set_slug, card_slug, card_type`; `grade` includes `ungraded` (2,863 rows), `psa_10`, `grade_9`, `key:cib` (16 rows / 3 cards). Monthly floor 2021-03; data end 2026-09-01.
- `data/reference/cards_modeling.csv` — 143 (card_slug, grade) rows / 59 cards / 33 players: 59 ungraded + 55 psa_10 + 29 psa_9. **psa_9 has zero chart points** (chart label `grade_9` is grader-agnostic; sales label `psa_9` is PSA-specific — they never join, and renaming would be wrong). psa_9 stays OUT of this panel; if wanted later it must be sourced from weekly sales medians.
- **Universe parquets are a strict superset of the seed parquets** (all 13 seed slugs ⊂ 59 universe slugs; seed lacks `card_type`). Never concatenate `universe_*` with `scp_*` parquets.
- Horizon truncation at data end 2026-09: 6m holds → entries ≤ 2026-03; 12m → ≤ 2025-09; 24m → ≤ 2024-09; 36m → ≤ 2023-09. Entries with a horizon past the end get NaN for that hold, never a fabricated value.
- `events.fetch_award_events(mlb_ids, seasons)` (events.py) loops awards × seasons league-wide (≈72 calls for 2015–2026), skips 404 (unannounced), returns columns `mlb_id, event_date, event_type, details` with `event_type == "award_win"`. `events.events_from_game_logs(mlb_logs)` emits `debut` rows offline (logs cover every universe player's debut — all rookie_years ≥ 2015).
- `stats_api.fetch_player_info(mlb_ids)` returns `mlb_id, name, birth_date, position` in ONE batched call.
- `season_stats.hitting_to_date(game_log_df, through: date) -> dict` / `pitching_to_date(...)` rebuild cumulative lines (rates from summed counting stats) for ANY game-log frame — feeding it career logs (all seasons) yields a career-to-date line unchanged.
- Lag convention (from panel.py): predictors at entry month `m` use data through `m - 1 day` only.
- Existing panel/market pattern (panel.py): universe median return per period via `groupby("month")["log_ret"].median()`.

## Global Constraints

- Python >= 3.11; free sources only; raw snapshots immutable (new dated files only).
- Politeness: ≥0.3s between MLB Stats API calls; never plural `sportIds=`. No SCP access in this plan.
- Tests offline by default; `live` marker for network tests.
- Lint: `ruff check` / `ruff format --check` on touched files only; never repo-wide `ruff format` (it rewrites markdown-embedded code in plan docs).
- **No look-ahead:** every predictor must be computable from data strictly before the entry month. Structural test in Task 4 (truncation probe) enforces this end-to-end.
- **Honest goldens:** golden values below carry real-world expectations; verify each against BOTH the parquet and the player's published stat line. If reality disagrees, STOP and report — never edit data to fit a golden.
- `data/raw/` and `data/processed/` are gitignored — commits contain code/tests/reference CSVs only; copy new parquets back to the main checkout at merge time.
- `grade == "ungraded"` normalization lives in weekly/liquidity; the chart parquet already emits `ungraded`. Filter `key:`-prefixed grades in any chart consumer (mirrors `run_universe_liquidity.py`).
- **Spec narrowings (scope discipline, not gaps):** career "HR rate" is carried as `career_home_runs` + `career_games` (P6c derives the rate); the spec's "current minor-league form for pre-debut players" is proxied by `rate_at_max_level` + `minor_games` through the lag date (per-level current-season splits deferred — P6c revisits only if models are weak); "All-Star/award events" means the registry's `award_win` rows (MVP/Cy Young/ROY) — no All-Star fetcher exists and none is built here.

---

### Task 1: Universe events + player info

**Files:**
- Create: `scripts/collect_universe_events.py`
- Test: `tests/test_collect_universe_events.py`

**Interfaces:**
- Consumes: `events.fetch_award_events`, `events.events_from_game_logs`, `events.EVENT_COLUMNS`; `stats_api.fetch_player_info`; `data/processed/game_logs_universe.parquet`; `data/reference/cards_universe.csv`.
- Produces (Tasks 2-4 consume):
  - `data/reference/player_info_universe.csv` — `mlb_id, name, birth_date, position` for all 39 players (committed).
  - `data/processed/events_universe.parquet` — columns `EVENT_COLUMNS` (`mlb_id, event_date, event_type, details`); `event_type` ∈ {`debut`, `award_win`}. Debut rows come from MLB-level (`level == "mlb"`) game logs only — never from minor-league rows.
  - `build_universe_events(game_logs: pd.DataFrame, awards: pd.DataFrame) -> pd.DataFrame` — pure function, unit-tested offline.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_universe_events.py
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from collect_universe_events import build_universe_events


def test_debut_from_mlb_level_only():
    game_logs = pd.DataFrame(
        [
            {"mlb_id": 1, "group": "hitting", "season": 2017, "date": "2017-05-01", "level": "a"},
            {"mlb_id": 1, "group": "hitting", "season": 2019, "date": "2019-04-01", "level": "mlb"},
            {"mlb_id": 1, "group": "hitting", "season": 2019, "date": "2019-04-20", "level": "mlb"},
        ]
    )
    game_logs["date"] = pd.to_datetime(game_logs["date"])
    awards = pd.DataFrame(
        [{"mlb_id": 1, "event_date": pd.Timestamp("2019-11-11"), "event_type": "award_win", "details": "AL ROY 2019"}]
    )
    out = build_universe_events(game_logs, awards)
    debut = out[out["event_type"] == "debut"].iloc[0]
    assert debut["event_date"] == pd.Timestamp("2019-04-01")  # NOT the 2017 minor-league game
    assert set(out["event_type"]) == {"debut", "award_win"}
    assert list(out.columns) == ["mlb_id", "event_date", "event_type", "details"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collect_universe_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect_universe_events'`.

- [ ] **Step 3: Implement collect_universe_events.py**

```python
# scripts/collect_universe_events.py
"""Universe event registry (debut + award_win) and player info for the 39 players."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.events import EVENT_COLUMNS, events_from_game_logs, fetch_award_events  # noqa: E402
from cardprice.stats_api import fetch_player_info  # noqa: E402


def build_universe_events(game_logs: pd.DataFrame, awards: pd.DataFrame) -> pd.DataFrame:
    mlb_logs = game_logs[game_logs["level"] == "mlb"]
    debuts = events_from_game_logs(mlb_logs)[lambda d: d["event_type"] == "debut"]
    out = pd.concat([debuts, awards], ignore_index=True)
    return (
        out[EVENT_COLUMNS]
        .sort_values(["mlb_id", "event_date"])
        .reset_index(drop=True)
    )


def main() -> None:
    cards = pd.read_csv("data/reference/cards_universe.csv")
    mlb_ids = sorted(cards["mlb_id"].dropna().astype(int).unique())
    print(f"{len(mlb_ids)} players")

    info = fetch_player_info(mlb_ids)
    info.to_csv("data/reference/player_info_universe.csv", index=False)

    game_logs = pd.read_parquet("data/processed/game_logs_universe.parquet")
    awards = fetch_award_events(mlb_ids, list(range(2015, 2027)))
    events = build_universe_events(game_logs, awards)
    events.to_parquet("data/processed/events_universe.parquet", index=False)
    print(events.groupby("event_type").size())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then the collector (network, ~2-3 min)**

Run: `python -m pytest tests/test_collect_universe_events.py -v` — 1 PASS.
Then: `python scripts/collect_universe_events.py`
Expected: 39 players in player_info_universe.csv (birth_date non-null for all 39 — check it; a null birth_date breaks age features downstream, so report any). Events: 39 debut rows + award_win rows. Sanity-check award counts against well-known facts: Judge (592450) ≥ 2 (2017 AL ROY, 2022 AL MVP); Henderson (683002) ≥ 1 (2023 AL ROY); report the actual event_type counts from the script output — never fabricate a count you didn't see. Note: 2026 awards unannounced → 404-skip is expected behavior.

- [ ] **Step 5: Commit**

```bash
git add scripts/collect_universe_events.py tests/test_collect_universe_events.py data/reference/player_info_universe.csv
git commit -m "feat: universe events registry + player info"
```

---

### Task 2: Career features module

**Files:**
- Create: `src/cardprice/career.py`
- Test: `tests/test_career.py`

**Interfaces:**
- Consumes: `season_stats.hitting_to_date` / `pitching_to_date`; game_logs_universe.parquet (`level` column); events_universe.parquet; player_info_universe.csv (Task 1).
- Produces (Tasks 3-4 consume):
  - `LEVEL_RANK = {"a": 1, "a_plus": 2, "aa": 3, "aaa": 4}` (module constant).
  - `season_year(month: pd.Timestamp) -> int` — baseball-season year for a month: Nov/Dec roll to next year (`year + 1 if month.month >= 11 else year`). Jan/Feb map to the upcoming season (same year).
  - `career_to_date(game_logs_mlb: pd.DataFrame, mlb_id: int, as_of: date) -> dict` — cumulative MLB line through `as_of` (all seasons; `hitting_to_date`/`pitching_to_date` on filtered logs). Empty career → dict from an empty frame (zeros/None — the season_stats behavior; do not special-case).
  - `minors_pedigree(game_logs_minors: pd.DataFrame, mlb_id: int, as_of: date) -> dict` — keys `max_level` (name or None), `max_level_rank` (int, 0 if none), `rate_at_max_level` (career-to-date OPS for hitters / ERA for pitchers at the highest level reached, through `as_of`; None if no minors), `minor_games` (total minor games through `as_of`).
  - `career_stage(game_logs_mlb: pd.DataFrame, mlb_id: int, entry_month: pd.Timestamp) -> str` — `"prospect"` if no MLB game strictly before `entry_month`; else index = `season_year(entry_month) - debut_season` → 0 `rookie_year`, 1 `sophomore`, ≥2 `established` (debut_season = season of first MLB game in the logs).
  - `awards_to_date(events: pd.DataFrame, mlb_id: int, as_of) -> int` — count of `award_win` rows with `event_date <= as_of`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_career.py
from datetime import date

import pandas as pd

from cardprice.career import (
    awards_to_date,
    career_stage,
    career_to_date,
    minors_pedigree,
    season_year,
)


def make_logs(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


BRYANT_2015 = {  # one game standing in for a season's worth of counting stats
    "mlb_id": 592178, "group": "hitting", "season": 2015, "level": "mlb",
    "gamesPlayed": 151, "atBats": 559, "hits": 154, "doubles": 31, "triples": 5,
    "homeRuns": 26, "rbi": 99, "baseOnBalls": 77, "strikeOuts": 199,
    "hitByPitch": 10, "sacFlies": 3, "stolenBases": 13,
}


def test_season_year_rolls_offseason():
    assert season_year(pd.Timestamp("2023-04-01")) == 2023
    assert season_year(pd.Timestamp("2023-01-01")) == 2023
    assert season_year(pd.Timestamp("2022-12-01")) == 2023


def test_career_to_date_sums_across_seasons():
    logs = make_logs(
        [
            {"date": "2015-04-17", **BRYANT_2015},
            {"date": "2016-04-03", **{**BRYANT_2015, "season": 2016, "homeRuns": 39}},
            {"date": "2017-04-02", **{**BRYANT_2015, "season": 2017, "homeRuns": 29}},
        ]
    )
    out = career_to_date(logs, 592178, date(2017, 12, 31))
    assert out["home_runs"] == 94
    assert out["games"] == 453  # 3 synthetic rows x 151
    # strictly-lagged: through 2016 only sees 2 seasons
    out16 = career_to_date(logs, 592178, date(2016, 12, 31))
    assert out16["home_runs"] == 65


def test_minors_pedigree_levels():
    logs = make_logs(
        [
            {"mlb_id": 1, "group": "hitting", "season": 2017, "date": "2017-06-01", "level": "a",
             "gamesPlayed": 23, "atBats": 80, "hits": 20, "doubles": 5, "triples": 0,
             "homeRuns": 3, "rbi": 10, "baseOnBalls": 10, "strikeOuts": 15,
             "hitByPitch": 1, "sacFlies": 1, "stolenBases": 2},
            {"mlb_id": 1, "group": "hitting", "season": 2018, "date": "2018-05-01", "level": "aa",
             "gamesPlayed": 40, "atBats": 150, "hits": 45, "doubles": 10, "triples": 1,
             "homeRuns": 8, "rbi": 30, "baseOnBalls": 20, "strikeOuts": 30,
             "hitByPitch": 2, "sacFlies": 2, "stolenBases": 5},
        ]
    )
    out = minors_pedigree(logs, 1, date(2018, 6, 1))
    assert out["max_level"] == "aa" and out["max_level_rank"] == 3
    assert out["minor_games"] == 63
    assert out["rate_at_max_level"] is not None  # OPS at AA
    empty = minors_pedigree(logs, 999, date(2018, 6, 1))
    assert empty["max_level"] is None and empty["max_level_rank"] == 0
    assert empty["rate_at_max_level"] is None and empty["minor_games"] == 0


def test_career_stage_henderson_shape():
    logs = make_logs(
        [
            {"mlb_id": 683002, "group": "hitting", "season": 2022, "date": "2022-08-31", "level": "mlb"},
            {"mlb_id": 683002, "group": "hitting", "season": 2023, "date": "2023-03-30", "level": "mlb"},
        ]
    )
    assert career_stage(logs, 683002, pd.Timestamp("2022-06-01")) == "prospect"
    assert career_stage(logs, 683002, pd.Timestamp("2022-09-01")) == "rookie_year"
    assert career_stage(logs, 683002, pd.Timestamp("2023-04-01")) == "sophomore"
    assert career_stage(logs, 683002, pd.Timestamp("2025-04-01")) == "established"
    assert career_stage(logs, 683002, pd.Timestamp("2022-12-01")) == "sophomore"  # offseason rolls forward


def test_awards_to_date_counts():
    events = pd.DataFrame(
        [
            {"mlb_id": 592450, "event_date": pd.Timestamp("2017-11-13"), "event_type": "award_win", "details": "AL ROY"},
            {"mlb_id": 592450, "event_date": pd.Timestamp("2022-11-16"), "event_type": "award_win", "details": "AL MVP"},
            {"mlb_id": 592450, "event_date": pd.Timestamp("2022-08-31"), "event_type": "debut", "details": "x"},
        ]
    )
    assert awards_to_date(events, 592450, date(2017, 12, 31)) == 1
    assert awards_to_date(events, 592450, date(2023, 1, 1)) == 2
    assert awards_to_date(events, 592450, date(2016, 1, 1)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_career.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.career'`.

- [ ] **Step 3: Implement career.py**

```python
# src/cardprice/career.py
"""Career-to-date features at an entry date: MLB line, minors pedigree, career stage, awards.

All functions take an as_of/entry_month and use only data through that date —
the no-look-ahead contract lives here.
"""

from datetime import date

import pandas as pd

from cardprice.season_stats import hitting_to_date, pitching_to_date

LEVEL_RANK = {"a": 1, "a_plus": 2, "aa": 3, "aaa": 4}
STAGES = ("prospect", "rookie_year", "sophomore", "established")


def season_year(month: pd.Timestamp) -> int:
    """Baseball-season year for a calendar month: Nov/Dec roll to next season."""
    return month.year + 1 if month.month >= 11 else month.year


def _group_of(logs: pd.DataFrame) -> str:
    return logs["group"].iloc[0]


def career_to_date(game_logs_mlb: pd.DataFrame, mlb_id: int, as_of: date) -> dict:
    sub = game_logs_mlb[game_logs_mlb["mlb_id"] == mlb_id]
    if not len(sub):
        return {}
    fn = hitting_to_date if _group_of(sub) == "hitting" else pitching_to_date
    return fn(sub, as_of)


def minors_pedigree(game_logs_minors: pd.DataFrame, mlb_id: int, as_of: date) -> dict:
    sub = game_logs_minors[game_logs_minors["mlb_id"] == mlb_id]
    sub = sub[sub["date"].dt.date <= as_of]
    out = {"max_level": None, "max_level_rank": 0, "rate_at_max_level": None, "minor_games": 0}
    if not len(sub):
        return out
    out["minor_games"] = int(sub["gamesPlayed"].sum())
    top = max(sub["level"].unique(), key=lambda lv: LEVEL_RANK[lv])
    out["max_level"] = top
    out["max_level_rank"] = LEVEL_RANK[top]
    at_top = sub[sub["level"] == top]
    fn = hitting_to_date if _group_of(at_top) == "hitting" else pitching_to_date
    line = fn(at_top, as_of)
    out["rate_at_max_level"] = line.get("ops") if fn is hitting_to_date else line.get("era")
    return out


def career_stage(game_logs_mlb: pd.DataFrame, mlb_id: int, entry_month: pd.Timestamp) -> str:
    sub = game_logs_mlb[game_logs_mlb["mlb_id"] == mlb_id]
    before = sub[sub["date"] < entry_month]
    if not len(before):
        return "prospect"
    debut_season = int(sub["season"].min())
    idx = season_year(entry_month) - debut_season
    # any game before entry implies debut_season <= season_year(entry_month), so idx >= 0
    return "rookie_year" if idx == 0 else "sophomore" if idx == 1 else "established"


def awards_to_date(events: pd.DataFrame, mlb_id: int, as_of: date) -> int:
    wins = events[
        (events["mlb_id"] == mlb_id)
        & (events["event_type"] == "award_win")
        & (events["event_date"].dt.date <= as_of)
    ]
    return len(wins)
```

(The clamp note: `idx` can't be negative when `before` is non-empty — any game before entry means debut_season ≤ season_year(entry_month); season_year rolls Nov/Dec forward precisely so an offseason entry after a debut lands on sophomore. If you find a counterexample during testing, STOP and report it rather than clamping silently.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_career.py -v` — 5 PASS. Full suite stays green; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/career.py tests/test_career.py
git commit -m "feat: career features (career-to-date, minors pedigree, career stage, awards)"
```

---

### Task 3: Hold-return outcome builder

**Files:**
- Create: `src/cardprice/multiyear.py`
- Test: `tests/test_multiyear.py`

**Interfaces:**
- Consumes: `data/processed/universe_chart_monthly.parquet` (Task P6a).
- Produces (Task 4 consumes):
  - `HORIZONS = (6, 12, 24, 36)` (module constant, months).
  - `series_month_ends(chart: pd.DataFrame) -> pd.DataFrame` — one row per (card_slug, grade, month): last price point within the month + meta (`mlb_id, player_name, rookie_year, card_type`). Filters: `grade ∈ {"ungraded", "psa_10"}` and drops `key:`-prefixed grades (belt-and-suspenders; the grade filter already excludes them).
  - `hold_returns(me: pd.DataFrame, horizons: tuple = HORIZONS) -> pd.DataFrame` — one row per (card_slug, grade, entry_month) with `entry_price` and `ret_{h}m` columns: annualized log return `ln(price_{t+h} / price_t) / (h/12)`; NaN when the target month has no price point (gap or past data end) — never fabricated. Entry months restricted to ≥ 2021-04 (need 3 trailing months for Task 4's price-level feature).
  - `trailing_price_level(me: pd.DataFrame) -> pd.Series`-producing helper folded into hold_returns? NO — keep separate: `trailing_features(me: pd.DataFrame) -> pd.DataFrame` with per (card_slug, grade, month): `price_level` = ln of the median of the 3 prior month-end prices (NaN if < 3 prior points), `ret_3m` = ln(p_t / p_{t-3}) exact-month-match else NaN.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiyear.py
import numpy as np
import pandas as pd

from cardprice.multiyear import HORIZONS, hold_returns, series_month_ends, trailing_features


def make_chart(slug="set/card", grade="ungraded", start="2021-01", months=66, price0=10.0, growth=0.01):
    dates = pd.date_range(start, periods=months, freq="MS")
    n = len(dates)
    return pd.DataFrame(
        {
            "card_slug": slug,
            "grade": grade,
            "date": dates,
            "price": [price0 * np.exp(growth * i) for i in range(n)],
            "mlb_id": 1,
            "player_name": "Test Player",
            "rookie_year": 2021,
            "card_type": "flagship",
        }
    )


def test_series_month_ends_filters_and_aggregates():
    chart = pd.concat(
        [
            make_chart(grade="ungraded"),
            make_chart(grade="psa_10"),
            make_chart(grade="grade_9"),       # grader-agnostic label: excluded
            make_chart(grade="key:cib"),       # uncalibrated: excluded
        ]
    )
    me = series_month_ends(chart)
    assert set(me["grade"]) == {"ungraded", "psa_10"}
    assert len(me) == 2 * 66
    assert {"card_type", "mlb_id", "rookie_year"} <= set(me.columns)


def test_hold_returns_exact_and_truncated():
    me = series_month_ends(make_chart(months=30))  # 2021-01 .. 2023-06
    out = hold_returns(me)
    row = out[out["entry_month"] == pd.Timestamp("2021-04-01")].iloc[0]
    # growth=0.01 log-linear: annualized return == 0.12 at every horizon
    for h in HORIZONS:
        assert abs(row[f"ret_{h}m"] - 0.12) < 1e-9
    # truncation: last month-end is 2023-06; 12m needs <= 2022-06
    last12 = out[out["ret_12m"].notna()]["entry_month"].max()
    assert last12 == pd.Timestamp("2022-06-01")
    assert out[out["entry_month"] > pd.Timestamp("2022-06-01")]["ret_12m"].isna().all()
    # entries before 2021-04 excluded
    assert out["entry_month"].min() == pd.Timestamp("2021-04-01")


def test_hold_returns_gap_month_is_nan_not_fabricated():
    chart = make_chart(months=30)
    chart = chart[chart["date"] != pd.Timestamp("2021-10-01")]  # punch a hole
    out = hold_returns(series_month_ends(chart))
    row = out[out["entry_month"] == pd.Timestamp("2021-04-01")].iloc[0]
    assert pd.isna(row["ret_6m"])   # 2021-04 + 6m = 2021-10 = the hole
    assert not pd.isna(row["ret_12m"])


def test_trailing_features():
    me = series_month_ends(make_chart(months=30))
    tf = trailing_features(me)
    row = tf[tf["month"] == pd.Timestamp("2021-04-01")].iloc[0]
    assert abs(row["ret_3m"] - 0.03) < 1e-9  # 3 months of 0.01 growth
    # first 3 months have insufficient history
    early = tf[tf["month"] <= pd.Timestamp("2021-03-01")]
    assert early["price_level"].isna().all() and early["ret_3m"].isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_multiyear.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.multiyear'`.

- [ ] **Step 3: Implement multiyear.py**

```python
# src/cardprice/multiyear.py
"""Multi-year hold panel pieces: month-end series, hold-return outcomes, trailing features.

Ungraded is the primary series; psa_10 the robustness check. psa_9 is excluded:
its chart label (grade_9) is grader-agnostic and never joins sales psa_9.
"""

import numpy as np
import pandas as pd

HORIZONS = (6, 12, 24, 36)
PANEL_SERIES = ("ungraded", "psa_10")
ENTRY_FLOOR = pd.Timestamp("2021-04-01")  # 3 trailing months exist from the 2021-03 chart floor
META_COLS = ["mlb_id", "player_name", "rookie_year", "card_type"]


def series_month_ends(chart: pd.DataFrame) -> pd.DataFrame:
    df = chart[chart["grade"].isin(PANEL_SERIES)].copy()
    df = df[~df["grade"].astype(str).str.startswith("key:")]
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    return (
        df.sort_values("date")
        .groupby(["card_slug", "grade", "month"])
        .agg(price=("price", "last"), **{c: (c, "first") for c in META_COLS})
        .reset_index()
    )


def _month_idx(s: pd.Series) -> pd.Series:
    return s.dt.year * 12 + s.dt.month


def hold_returns(me: pd.DataFrame, horizons: tuple = HORIZONS) -> pd.DataFrame:
    rows = []
    for (slug, grade), grp in me.groupby(["card_slug", "grade"]):
        price_by_idx = {_month_idx(pd.Series([r.month])).iloc[0]: r.price for r in grp.itertuples()}
        for r in grp.itertuples():
            if r.month < ENTRY_FLOOR:
                continue
            t = _month_idx(pd.Series([r.month])).iloc[0]
            row = {"card_slug": slug, "grade": grade, "entry_month": r.month, "entry_price": r.price}
            for h in horizons:
                p2 = price_by_idx.get(t + h)
                row[f"ret_{h}m"] = np.log(p2 / r.price) / (h / 12) if p2 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def trailing_features(me: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (slug, grade), grp in me.groupby(["card_slug", "grade"]):
        grp = grp.sort_values("month")
        price_by_idx = {_month_idx(pd.Series([r.month])).iloc[0]: r.price for r in grp.itertuples()}
        for r in grp.itertuples():
            t = _month_idx(pd.Series([r.month])).iloc[0]
            prior = [price_by_idx.get(t - k) for k in (1, 2, 3)]
            known = [p for p in prior if p]
            level = np.log(np.median(known)) if len(known) == 3 else np.nan
            p3 = price_by_idx.get(t - 3)
            rows.append(
                {
                    "card_slug": slug,
                    "grade": grade,
                    "month": r.month,
                    "price_level": level,
                    "ret_3m": np.log(r.price / p3) if p3 else np.nan,
                }
            )
    return pd.DataFrame(rows)
```

(`_month_idx` on a one-element Series is clunky but keeps the mapping explicit; a `r.month.year * 12 + r.month.month` inline is equally acceptable — pick one, use it consistently. `if p2` is safe: prices are strictly positive after the parser's `price <= 0` filter.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_multiyear.py -v` — 4 PASS. Full suite green; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/multiyear.py tests/test_multiyear.py
git commit -m "feat: hold-return outcome builder + trailing features"
```

---

### Task 4: Panel assembly, no-look-ahead probe, verification

**Files:**
- Create: `scripts/build_multiyear_panel.py`
- Test: `tests/test_multiyear_panel.py`

**Interfaces:**
- Consumes: `series_month_ends`, `hold_returns`, `trailing_features` (Task 3); `career_to_date`, `minors_pedigree`, `career_stage`, `awards_to_date` (Task 2); universe parquets + events_universe.parquet + player_info_universe.csv + cards_modeling.csv (P6a/Task 1).
- Produces (P6c consumes): `data/processed/panel_multiyear.parquet` — one row per (card_slug, grade, entry_month) with columns:
  - identity: `card_slug, grade, entry_month, mlb_id, player_name, rookie_year, card_type, position`
  - predictors (all strictly before entry): career line cols (`career_games, career_ops` hitters / `career_era, career_k_bb_pct` pitchers, plus `career_home_runs` hitters / `career_innings_pitched` pitchers), `max_level, max_level_rank, rate_at_max_level, minor_games, career_stage, awards_to_date, age, age_at_debut` (debut known and ≤ lag → age at debut; pre-debut → age at entry, i.e. age-now semantics per the spec), `price_level, ret_3m, market_ret_3m`
  - outcomes: `ret_6m, ret_12m, ret_24m, ret_36m`
  - `build_panel(me, outcomes, trailing, game_logs, events, info) -> pd.DataFrame` — pure function (the selected-universe restriction happens in main() via the `me` frame, not here).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiyear_panel.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_multiyear_panel import build_panel

from cardprice.multiyear import hold_returns, series_month_ends, trailing_features


def tiny_universe():
    chart = pd.DataFrame(
        {
            "card_slug": "set/card",
            "grade": "ungraded",
            "date": pd.date_range("2021-01", periods=30, freq="MS"),
            "price": [10.0 * np.exp(0.01 * i) for i in range(30)],
            "mlb_id": 1,
            "player_name": "Test Player",
            "rookie_year": 2021,
            "card_type": "flagship",
        }
    )
    game_logs = pd.DataFrame(
        [
            {"mlb_id": 1, "group": "hitting", "season": 2021, "date": "2021-04-01",
             "level": "mlb", "gamesPlayed": 1, "atBats": 4, "hits": 1, "doubles": 0,
             "triples": 0, "homeRuns": 0, "rbi": 0, "baseOnBalls": 0, "strikeOuts": 1,
             "hitByPitch": 0, "sacFlies": 0, "stolenBases": 0},
        ]
    )
    game_logs["date"] = pd.to_datetime(game_logs["date"])
    events = pd.DataFrame(columns=["mlb_id", "event_date", "event_type", "details"])
    info = pd.DataFrame(
        [{"mlb_id": 1, "name": "Test Player", "birth_date": pd.Timestamp("1998-01-01"), "position": "OF"}]
    )
    cards = pd.DataFrame(
        [{"card_slug": "set/card", "grade": "ungraded", "player_name": "Test Player",
          "mlb_id": 1, "rookie_year": 2021, "card_type": "flagship"}]
    )
    return chart, game_logs, events, info, cards


def test_panel_rows_and_lag():
    chart, game_logs, events, info, cards = tiny_universe()
    me = series_month_ends(chart)
    panel = build_panel(me, hold_returns(me), trailing_features(me), game_logs, events, info)
    row = panel[panel["entry_month"] == pd.Timestamp("2021-05-01")].iloc[0]
    # lag discipline: entry 2021-05 sees games through 2021-04-30 -> the Apr 1 game counts
    assert row["career_games"] == 1
    assert row["career_stage"] == "rookie_year"
    assert row["awards_to_date"] == 0
    assert row["age"] == round((pd.Timestamp("2021-05-01") - pd.Timestamp("1998-01-01")).days / 365.25, 2)
    # pre-debut entry: the Apr 1 game is NOT strictly before entry month 2021-04-01,
    # and lag (Mar 31) sees zero games -> prospect with an empty career line
    pre = panel[panel["entry_month"] == pd.Timestamp("2021-04-01")]
    assert len(pre) == 1  # entry exists: 2021-04 has a month-end price
    assert pre.iloc[0]["career_stage"] == "prospect"
    assert pre.iloc[0]["career_games"] == 0
    assert pre.iloc[0]["age_at_debut"] == pre.iloc[0]["age"]  # pre-debut: age-now semantics
    # outcomes present and consistent with Task 3
    assert "ret_12m" in panel.columns
    # market regime column is the universe median trailing return for that month
    assert "market_ret_3m" in panel.columns


def test_no_lookahead_truncation_probe():
    chart, game_logs, events, info, cards = tiny_universe()
    me = series_month_ends(chart)
    full = build_panel(me, hold_returns(me), trailing_features(me), game_logs, events, info)
    cut = pd.Timestamp("2021-06-01")
    truncated_logs = game_logs[game_logs["date"] < cut]
    truncated_events = events[events["event_date"] < cut] if len(events) else events
    part = build_panel(
        me, hold_returns(me), trailing_features(me), truncated_logs, truncated_events, info
    )
    pred_cols = [
        "career_games", "career_stage", "awards_to_date", "age", "age_at_debut",
        "max_level_rank", "minor_games", "price_level", "ret_3m", "market_ret_3m",
    ]
    early_full = full[full["entry_month"] <= cut].sort_values("entry_month")
    early_part = part[part["entry_month"] <= cut].sort_values("entry_month")
    for c in pred_cols:
        a, b = early_full[c].to_numpy(), early_part[c].to_numpy()
        assert len(a) == len(b)
        assert all((x == y) or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b)), c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_multiyear_panel.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement build_multiyear_panel.py**

```python
# scripts/build_multiyear_panel.py
"""Assemble the multi-year panel: card x entry-month with lagged predictors + hold outcomes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.career import awards_to_date, career_stage, career_to_date, minors_pedigree  # noqa: E402
from cardprice.multiyear import hold_returns, series_month_ends, trailing_features  # noqa: E402

HITTER_CAREER = {"games": "career_games", "ops": "career_ops", "home_runs": "career_home_runs"}
PITCHER_CAREER = {
    "games": "career_games",
    "era": "career_era",
    "k_bb_pct": "career_k_bb_pct",
    "innings_pitched": "career_innings_pitched",
}


def build_panel(me, outcomes, trailing, game_logs, events, info) -> pd.DataFrame:
    mlb_logs = game_logs[game_logs["level"] == "mlb"]
    minor_logs = game_logs[game_logs["level"] != "mlb"]
    tf = trailing.rename(columns={"month": "entry_month"})
    df = outcomes.merge(tf, on=["card_slug", "grade", "entry_month"], how="left", validate="one_to_one")
    meta = me[["card_slug", "grade", "month", "mlb_id", "player_name", "rookie_year", "card_type"]]
    df = df.merge(
        meta, left_on=["card_slug", "grade", "entry_month"], right_on=["card_slug", "grade", "month"],
        how="left", validate="one_to_one",
    ).drop(columns="month")
    market = (
        tf.groupby("entry_month")["ret_3m"].median().rename("market_ret_3m").reset_index()
    )
    df = df.merge(market, on="entry_month", how="left", validate="many_to_one")
    debuts = (
        events[events["event_type"] == "debut"][["mlb_id", "event_date"]]
        .rename(columns={"event_date": "debut_date"})
    )

    rows = []
    for r in df.itertuples(index=False):
        lag = (r.entry_month - pd.Timedelta(days=1)).date()
        line = career_to_date(mlb_logs, int(r.mlb_id), lag)
        ped = minors_pedigree(minor_logs, int(r.mlb_id), lag)
        row = r._asdict()
        for src, dst in (HITTER_CAREER | PITCHER_CAREER).items():
            row[dst] = line.get(src) if line else None
        row.update(
            max_level=ped["max_level"],
            max_level_rank=ped["max_level_rank"],
            rate_at_max_level=ped["rate_at_max_level"],
            minor_games=ped["minor_games"],
            career_stage=career_stage(mlb_logs, int(r.mlb_id), r.entry_month),
            awards_to_date=awards_to_date(events, int(r.mlb_id), lag),
        )
        p_info = info[info["mlb_id"] == r.mlb_id]
        if len(p_info):
            birth = p_info.iloc[0]["birth_date"]
            row["age"] = round((r.entry_month - birth).days / 365.25, 2)
            row["position"] = p_info.iloc[0]["position"]
            d = debuts[debuts["mlb_id"] == r.mlb_id]
            if len(d) and d.iloc[0]["debut_date"].date() <= lag:
                row["age_at_debut"] = round((d.iloc[0]["debut_date"] - birth).days / 365.25, 2)
            else:
                row["age_at_debut"] = row["age"]  # pre-debut: age-now semantics (spec §6)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    chart = pd.read_parquet("data/processed/universe_chart_monthly.parquet")
    game_logs = pd.read_parquet("data/processed/game_logs_universe.parquet")
    events = pd.read_parquet("data/processed/events_universe.parquet")
    info = pd.read_csv("data/reference/player_info_universe.csv", parse_dates=["birth_date"])
    cards = pd.read_csv("data/reference/cards_modeling.csv")

    me = series_month_ends(chart)
    # restrict to the liquidity-selected (card, series) universe
    sel = cards[cards["grade"].isin(["ungraded", "psa_10"])][["card_slug", "grade"]].drop_duplicates()
    me = me.merge(sel, on=["card_slug", "grade"], how="inner", validate="many_to_one")

    panel = build_panel(
        me, hold_returns(me), trailing_features(me), game_logs, events, info
    )
    panel.to_parquet("data/processed/panel_multiyear.parquet", index=False)
    print(f"panel_multiyear: {len(panel)} rows, {panel['card_slug'].nunique()} cards")
    print("\nrows per horizon (non-NaN):")
    for h in (6, 12, 24, 36):
        print(f"  ret_{h}m: {panel[f'ret_{h}m'].notna().sum()}")
    print("\nrows per career_stage:")
    print(panel["career_stage"].value_counts())
    print("\nrows per grade:")
    print(panel["grade"].value_counts())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then build + verify the real panel**

Run: `python -m pytest tests/test_multiyear_panel.py -v` — 2 PASS; full suite green; ruff clean.
Then: `python scripts/build_multiyear_panel.py`
Verification checklist (do all, report each result):
1. **Golden row:** from `universe_chart_monthly.parquet` directly compute by hand (filter the card, print the two month prices) the 12m annualized log return for Juan Soto's flagship card (`baseball-cards-2018-topps-chrome-update/juan-soto-hmt55`, ungraded) at entry 2021-04; assert the panel's `ret_12m` for that row matches to 1e-9. If that exact card/month is absent from the selected universe, pick any selected ungraded card with a full 2021-04..2022-04 window, say which, and pin that.
2. **Goldens on predictors (at panel-realistic entry months ≥ 2021-04):** Bryant (592178) `career_home_runs` at entry 2021-04 == **147** (his published 2015–2020 lines: 26+39+29+13+31+9; lag = 2021-03-31, and 2021 games must NOT count) — verify against the panel AND the published totals; a mismatch means the lag discipline is broken, STOP. Henderson (683002) `career_stage` at 2023-04 == "sophomore". Judge (592450) `awards_to_date` at 2023-01 == 2 (2017 AL ROY + 2022 AL MVP) — subject to Task 1's events actually containing them; if an expected award row is missing from events_universe.parquet, STOP and report (don't patch the golden).
3. **Coverage table:** rows per career_stage × grade; entries per year. Flag if any of prospect/rookie_year/sophomore/established is empty or if 36m rows == 0.
4. **No look-ahead in the wild:** assert `panel[panel.career_stage == "prospect"]["career_games"].fillna(0).max() == 0` and every row's `entry_price > 0`.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_multiyear_panel.py tests/test_multiyear_panel.py
git commit -m "feat: multi-year panel assembly + verification"
```

---

## Done criteria for this plan

- `python -m pytest -q` all offline tests PASS; ruff clean on touched files.
- `data/reference/player_info_universe.csv` (39 players, no null birth_date) and `data/processed/events_universe.parquet` (39 debuts + award_wins).
- `data/processed/panel_multiyear.parquet` — card × entry-month, ungraded + psa_10 only, predictor columns populated, 4 horizon outcome columns with honest NaN truncation (36m entries ≤ 2023-09).
- Golden verifications recorded: Soto (or substitute) ret_12m row, Bryant career HR 94, Henderson stage, Judge awards; no-look-ahead truncation probe green; prospect rows have zero career games.
- Coverage table (career_stage × grade, entries per year) in the final report; empty cells flagged, not patched.
- Next: P6c — hold-return models per horizon (LASSO 1-SE / GBM+SHAP / mixed-effects), career-arc overlay, walk-forward gate net of 14% fees, findings report. P6c's brief must carry: (a) overlapping holds are autocorrelated — effective N ≈ cards × independent years, report it; (b) ungraded primary, psa_10 robustness; (c) no pop columns (single-snapshot look-ahead).
