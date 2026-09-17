# T1 — Price Incorporation Lag (Daily Event Study) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how fast card prices incorporate on-field information at daily grain, via a sale-level event study around statistically-detected breakout games and dated debuts.

**Architecture:** Two new modules in `src/cardprice/`: `breakout_events.py` (game logs → event table) and `lag_study.py` (sales + events → relative-price windows → pooled lag curves → lag/half-life/adjustment-class estimates). Pure analysis on existing parquets; CLI runner writes artifacts to `data/processed/` (gitignored) and prints the summary that feeds `docs/findings/`.

**Tech Stack:** pandas 2.x, numpy 2.x, pytest. **No scipy** (not a declared dependency) — the half-life fit is pure-numpy grid search. No other new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-t1-incorporation-lag-design.md`

## Global Constraints

- Ruff line-length **100**; run `ruff check src tests` before every commit.
- Tests run with `pytest` (addopts already exclude `live`); never add network access to tests.
- Honest-golden discipline: a golden mismatch stops the run; never adjust expected values to make a failing test pass — investigate and report instead.
- No look-ahead: baselines use only data **strictly before** their anchor date.
- Seed every RNG; tests must be deterministic.
- **Spec correction folded into this plan (apply in Task 5):** the baseline anchor is the **event date** (sales strictly before the event), not each sale's own trailing median — a per-sale trailing baseline would absorb the very repricing being measured. Spec §9's line "baseline uses sales strictly before the sale date" is corrected to "strictly before the event date".
- Sales `bucket` column is contaminated with non-card vocabulary (`manual-only`, `cib`, …) — **never use it**; grade classes come from the `grade` column only (`NaN` = ungraded).
- Data artifacts under `data/processed/` are gitignored — numbers go into the findings doc, artifacts stay local.

## Existing code this plan reuses (do not re-implement)

- `cardprice.career.career_stage(game_logs_mlb, mlb_id, entry_month)` → `"prospect" | "rookie_year" | "sophomore" | "established"`, tested, no-look-ahead by construction. `game_logs_mlb` must be the `level == "mlb"` subset.
- `data/reference/player_info_universe.csv` (`mlb_id,name,birth_date,position`) — id → name map for readable output.
- Input schemas (verified 2026-09-17):
  - `universe_sales.parquet`: `sale_date` (datetime64), `title`, `price` (float, no NaNs), `list_price`, `best_offer` (bool), `grade` (string: `psa_10`, `psa_9`, `sgc_10`, …, NaN = ungraded), `bucket` (contaminated, ignore), `player_name`, `mlb_id` (int), `rookie_year`, `set_slug`, `card_slug`, `card_type` (`flagship` | `bowman_1st`).
  - `game_logs_universe.parquet`: `mlb_id`, `group` (`hitting` | `pitching`), `season`, `date` (datetime64), `level` (`mlb`, `aaa`, `aa`, `a_plus`, `a`), hitter counting stats (`totalBases`, `baseOnBalls`, `stolenBases`, `plateAppearances`, …), pitcher counting stats (`outs`, `strikeOuts`, `hits`, `earnedRuns`, `runs` = runs **allowed** in pitching rows, `baseOnBalls`, `gamesStarted`, …).
  - `events_universe.parquet`: `mlb_id`, `event_date` (datetime64), `event_type` (`debut` | `award_win`), `details`.

---

### Task 1: Breakout event detection (`breakout_events.py`)

**Files:**
- Create: `src/cardprice/breakout_events.py`
- Test: `tests/test_breakout_events.py`

**Interfaces:**
- Consumes: `game_logs_universe.parquet`, `events_universe.parquet` (paths only; this task reads nothing itself).
- Produces (used by Tasks 2–5):
  - `hitter_game_score(row: pd.Series) -> float` — `totalBases + baseOnBalls + stolenBases`, NaN → 0.
  - `pitcher_game_score(row: pd.Series) -> float` — Bill James classic, starts only meaningful.
  - `detect_breakouts(game_logs: pd.DataFrame, z_min: float = 2.5, baseline_games: int = 30, min_pa: int = 20, min_gs: int = 5) -> pd.DataFrame` — columns: `mlb_id` (int), `event_date` (datetime64), `event_type` (`"breakout"`), `group` (`hitting`|`pitching`), `level` (str), `score` (float), `z` (float), `baseline_n` (int).
  - `merge_debuts(breakouts: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame` — appends `event_type == "debut"` rows from the events registry with `group`/`level`/`score`/`z`/`baseline_n` = NaN.
  - `dedupe_overlaps(events: pd.DataFrame, window_days: int = 14) -> pd.DataFrame` — per `mlb_id`, keep first event, drop later events within ±`window_days` of a kept one.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_breakout_events.py
import numpy as np
import pandas as pd

from cardprice.breakout_events import (
    dedupe_overlaps,
    detect_breakouts,
    hitter_game_score,
    merge_debuts,
    pitcher_game_score,
)


def _hitting_row(date, tb, bb=0, sb=0, pa=4, mlb_id=1, level="mlb"):
    return {
        "mlb_id": mlb_id, "group": "hitting", "season": 2024, "date": pd.Timestamp(date),
        "level": level, "totalBases": tb, "baseOnBalls": bb, "stolenBases": sb,
        "plateAppearances": pa,
    }


def _pitching_row(date, outs, k, h, er, r, bb, gs=1, mlb_id=2, level="mlb"):
    return {
        "mlb_id": mlb_id, "group": "pitching", "season": 2024, "date": pd.Timestamp(date),
        "level": level, "outs": outs, "strikeOuts": k, "hits": h, "earnedRuns": er,
        "runs": r, "baseOnBalls": bb, "gamesStarted": gs,
    }


def test_hitter_game_score_sums_counting_stats():
    row = pd.Series({"totalBases": 7, "baseOnBalls": 2, "stolenBases": 1})
    assert hitter_game_score(row) == 10.0


def test_hitter_game_score_nan_is_zero():
    row = pd.Series({"totalBases": 4, "baseOnBalls": np.nan, "stolenBases": np.nan})
    assert hitter_game_score(row) == 4.0


def test_pitcher_game_score_bill_james_golden():
    # 7 IP (21 outs), 9 K, 4 H, 1 BB, 2 R (2 ER).
    # 50 + 21 outs + 2*3 (innings completed after the 4th: (21-12)//3) + 9 K
    # - 2*4 H - 4*2 ER - 2*0 unearned - 1 BB = 50+21+6+9-8-8-0-1 = 69
    row = pd.Series({"outs": 21, "strikeOuts": 9, "hits": 4, "earnedRuns": 2,
                     "runs": 2, "baseOnBalls": 1})
    assert pitcher_game_score(row) == 69.0


def test_pitcher_game_score_unearned_runs_cost_two():
    # same as above but one run unearned: -4*1 ER -2*1 unearned = -6 instead of -8 -> 71
    row = pd.Series({"outs": 21, "strikeOuts": 9, "hits": 4, "earnedRuns": 1,
                     "runs": 2, "baseOnBalls": 1})
    assert pitcher_game_score(row) == 71.0


def test_detect_breakouts_finds_planted_z():
    rows = []
    # 30 quiet games alternating tb 1,2 (mean 1.5, sd ~0.51), then a 14-TB monster
    dates = pd.date_range("2024-04-01", periods=31, freq="D")
    for i, d in enumerate(dates[:-1]):
        rows.append(_hitting_row(d, tb=1 + i % 2))
    rows.append(_hitting_row(dates[-1], tb=14))
    out = detect_breakouts(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0]["event_date"] == dates[-1]
    assert out.iloc[0]["z"] >= 2.5
    assert out.iloc[0]["group"] == "hitting"
    assert out.iloc[0]["baseline_n"] == 30


def test_detect_breakouts_respects_min_baseline():
    # only 10 prior games: detection must still work (baseline = trailing UP TO
    # 30 games); alternating 1,2 keeps baseline sd > 0
    rows = [
        _hitting_row(d, tb=1 + i % 2)
        for i, d in enumerate(pd.date_range("2024-04-01", periods=10, freq="D"))
    ]
    rows.append(_hitting_row(pd.Timestamp("2024-04-11"), tb=12))
    out = detect_breakouts(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0]["baseline_n"] == 10


def test_detect_breakouts_skips_relief_appearances_as_events_and_baselines():
    rows = []
    dates = pd.date_range("2024-04-01", periods=8, freq="7D")
    for d in dates[:-1]:  # 7 mediocre starts
        rows.append(_pitching_row(d, outs=15, k=3, h=6, er=3, r=3, bb=2))
    # a relief gem must NOT be an event (gamesStarted=0)
    rows.append(_pitching_row(dates[-1], outs=9, k=8, h=0, er=0, r=0, bb=0, gs=0))
    out = detect_breakouts(pd.DataFrame(rows))
    assert len(out) == 0


def test_detect_breakouts_pitcher_start_detected():
    rows = []
    dates = pd.date_range("2024-04-01", periods=8, freq="7D")
    # 7 mediocre starts alternating K=3/4 (game scores 44/45, sd > 0)
    for i, d in enumerate(dates[:-1]):
        rows.append(_pitching_row(d, outs=15, k=3 + i % 2, h=6, er=3, r=3, bb=2))
    # monster: 50 + 27 outs + 2*5 ((27-12)//3) + 14 K - 2*2 H - 0 - 0 - 1 BB = 96
    rows.append(_pitching_row(dates[-1], outs=27, k=14, h=2, er=0, r=0, bb=1))
    out = detect_breakouts(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0]["group"] == "pitching"
    assert out.iloc[0]["z"] >= 2.5


def test_detect_breakouts_constant_baseline_skipped():
    # sd == 0 baseline must not produce events (division guard)
    rows = [_hitting_row(d, tb=2) for d in pd.date_range("2024-04-01", periods=31, freq="D")]
    out = detect_breakouts(pd.DataFrame(rows))
    assert len(out) == 0


def test_merge_debuts_appends_debut_rows():
    breakouts = pd.DataFrame(
        {"mlb_id": [1], "event_date": pd.to_datetime(["2024-05-01"]),
         "event_type": ["breakout"], "group": ["hitting"], "level": ["mlb"],
         "score": [12.0], "z": [3.1], "baseline_n": [30]}
    )
    events = pd.DataFrame(
        {"mlb_id": [1, 2], "event_date": pd.to_datetime(["2023-04-02", "2024-03-30"]),
         "event_type": ["debut", "award_win"], "details": ["x", "y"]}
    )
    out = merge_debuts(breakouts, events)
    assert len(out) == 2
    debut = out[out["event_type"] == "debut"].iloc[0]
    assert debut["mlb_id"] == 1 and pd.isna(debut["z"])
    # award_win rows are NOT events for this study
    assert "award_win" not in set(out["event_type"])


def test_dedupe_overlaps_keeps_first_drops_within_window():
    events = pd.DataFrame(
        {"mlb_id": [1, 1, 1, 2],
         "event_date": pd.to_datetime(
             ["2024-05-01", "2024-05-10", "2024-06-01", "2024-05-02"]),
         "event_type": ["breakout"] * 4}
    )
    out = dedupe_overlaps(events, window_days=14)
    kept = out.sort_values(["mlb_id", "event_date"])
    assert list(kept["event_date"]) == [
        pd.Timestamp("2024-05-01"),  # 05-10 is +9d -> dropped
        pd.Timestamp("2024-06-01"),  # +31d from kept 05-01 -> kept
        pd.Timestamp("2024-05-02"),  # different player
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_breakout_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cardprice.breakout_events'`

- [ ] **Step 3: Implement `src/cardprice/breakout_events.py`**

```python
# src/cardprice/breakout_events.py
"""Breakout-game detection: statistically extreme single-game performances.

A breakout is a game whose score sits z >= z_min above the player's trailing
baseline at the same level (hitters: TB+BB+SB composite; pitchers: Bill James
game score, starts only). Baselines are strictly trailing (no look-ahead).
"""

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "mlb_id", "event_date", "event_type", "group", "level", "score", "z", "baseline_n",
]


def _v(x) -> float:
    return 0.0 if pd.isna(x) else float(x)


def hitter_game_score(row: pd.Series) -> float:
    """Total bases + walks + stolen bases; NaN components count as zero."""
    return _v(row.get("totalBases")) + _v(row.get("baseOnBalls")) + _v(row.get("stolenBases"))


def pitcher_game_score(row: pd.Series) -> float:
    """Bill James game score: 50 + outs + 2*(IP completed after 4th) + K
    - 2*H - 4*ER - 2*unearned - BB. `runs` is runs allowed (pitching rows)."""
    outs, k = _v(row.get("outs")), _v(row.get("strikeOuts"))
    h, er = _v(row.get("hits")), _v(row.get("earnedRuns"))
    r, bb = _v(row.get("runs")), _v(row.get("baseOnBalls"))
    bonus = 2.0 * max(0, int((outs - 12) // 3))
    return 50.0 + outs + bonus + k - 2.0 * h - 4.0 * er - 2.0 * max(0.0, r - er) - bb


def detect_breakouts(
    game_logs: pd.DataFrame,
    z_min: float = 2.5,
    baseline_games: int = 30,
    min_pa: int = 20,
    min_gs: int = 5,
) -> pd.DataFrame:
    """Detect breakout games per (mlb_id, level).

    Baseline = up to `baseline_games` trailing games at the same level
    (hitters: games with PA > 0; pitchers: starts only). Hitter baseline is
    usable when it totals >= `min_pa` plate appearances and >= 5 games;
    pitcher baseline needs >= `min_gs` starts. Baselines with sd == 0 are
    skipped (z undefined).
    """
    logs = game_logs.sort_values(["mlb_id", "level", "date"])
    rows: list[dict] = []
    for (mlb_id, level), sub in logs.groupby(["mlb_id", "level"], sort=False):
        group = sub["group"].iloc[0]
        if group == "hitting":
            scores = sub.apply(hitter_game_score, axis=1).to_numpy(dtype=float)
            usable = (sub["plateAppearances"].fillna(0) > 0).to_numpy()
            pa = sub["plateAppearances"].fillna(0).to_numpy(dtype=float)
            event_ok = np.ones(len(sub), dtype=bool)
        else:
            starts = (sub["gamesStarted"].fillna(0) == 1).to_numpy()
            scores = (
                sub.apply(pitcher_game_score, axis=1).where(starts).to_numpy(dtype=float)
            )
            usable = starts
            pa = np.zeros(len(sub))
            event_ok = starts
        dates = sub["date"].to_numpy()
        for i in range(len(sub)):
            if not event_ok[i] or np.isnan(scores[i]):
                continue
            base, pa_sum, gs_n = [], 0.0, 0
            j = i - 1
            while j >= 0 and len(base) < baseline_games:
                if usable[j] and not np.isnan(scores[j]):
                    base.append(scores[j])
                    pa_sum += pa[j]
                    gs_n += 1
                j -= 1
            base = np.asarray(base)
            if group == "hitting":
                if pa_sum < min_pa or len(base) < 5:
                    continue
            elif gs_n < min_gs:
                continue
            if len(base) < 2 or base.std(ddof=1) == 0:
                continue
            z = (scores[i] - base.mean()) / base.std(ddof=1)
            if z >= z_min:
                rows.append(
                    {
                        "mlb_id": int(mlb_id), "event_date": dates[i],
                        "event_type": "breakout", "group": group, "level": level,
                        "score": float(scores[i]), "z": float(z), "baseline_n": int(len(base)),
                    }
                )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def merge_debuts(breakouts: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Append dated MLB debuts as a second event class (pure info shocks)."""
    debuts = events[events["event_type"] == "debut"].copy()
    for col in ("group", "level", "score", "z", "baseline_n"):
        debuts[col] = np.nan
    out = pd.concat([breakouts, debuts[EVENT_COLUMNS]], ignore_index=True)
    return out.sort_values(["mlb_id", "event_date"]).reset_index(drop=True)


def dedupe_overlaps(events: pd.DataFrame, window_days: int = 14) -> pd.DataFrame:
    """Per player: keep the first event, drop later events within +/-window_days
    of any kept event (their windows would double-count the same repricing)."""
    keep_idx = []
    for _, sub in events.sort_values("event_date").groupby("mlb_id", sort=False):
        kept_dates: list[pd.Timestamp] = []
        for idx, row in sub.iterrows():
            d = row["event_date"]
            if all(abs((d - k).days) > window_days for k in kept_dates):
                keep_idx.append(idx)
                kept_dates.append(d)
    return events.loc[keep_idx].sort_values(["mlb_id", "event_date"]).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_breakout_events.py -v`
Expected: 10 passed. Then `ruff check src/cardprice/breakout_events.py tests/test_breakout_events.py` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/breakout_events.py tests/test_breakout_events.py
git commit -m "feat: breakout-game detection (hitter composite + Bill James game score, trailing z)"
```

---

### Task 2: Event-anchored sale windows (`lag_study.py` part 1)

**Files:**
- Create: `src/cardprice/lag_study.py`
- Test: `tests/test_lag_study.py`

**Interfaces:**
- Consumes: Task 1's events table (`mlb_id`, `event_date`, …) and the sales parquet schema.
- Produces (used by Tasks 3–5):
  - `GRADE_CLASSES = {"ungraded": None, "psa_10": "psa_10"}` — filter spec: `ungraded` keeps `grade.isna()`, `psa_10` keeps `grade == "psa_10"`.
  - `event_sale_windows(sales, events, grade_class="ungraded", baseline_days=28, window_days=14, min_baseline=3, min_window=5) -> tuple[pd.DataFrame, dict]`
    - Returns `(windows, drop_log)`. `windows` columns: `event_id` (int, positional index into `events`), `mlb_id`, `card_slug`, `event_date` (datetime64), `sale_date` (datetime64), `t_days` (float, sale − event in days), `rel_price` (float, price ÷ baseline).
    - Baseline = **median price of the card's same-grade-class sales with `sale_date` in `[event_date − baseline_days, event_date)`** (strictly pre-event). Requires ≥ `min_baseline` baseline sales.
    - Window = sales with `sale_date` in `[event_date − window_days, event_date + window_days]`. Requires ≥ `min_window` window sales.
    - `drop_log`: `{"dropped_baseline": int, "dropped_window": int, "kept": int}` counting **event-card pairs**.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lag_study.py
import numpy as np
import pandas as pd
import pytest

from cardprice.lag_study import event_sale_windows


def _sales(rows):
    return pd.DataFrame(rows, columns=["sale_date", "price", "grade", "mlb_id", "card_slug"])


def _events():
    return pd.DataFrame(
        {"mlb_id": [1], "event_date": pd.to_datetime(["2024-06-15"]),
         "event_type": ["breakout"], "group": ["hitting"], "level": ["mlb"],
         "score": [12.0], "z": [3.0], "baseline_n": [30]}
    )


def _card_sales(card="a/x", base_price=100.0, n_base=5, n_pre_in_window=3, post=()):
    """Baseline sales at days -20..-16 (inside 28d baseline, OUTSIDE the ±14d
    window), in-window pre-event sales at days -3,-2,-1, then post-event sales
    from day +1."""
    rows = []
    for i in range(n_base):
        rows.append((pd.Timestamp("2024-05-26") + pd.Timedelta(days=i), base_price, None, 1, card))
    for i in range(n_pre_in_window):
        rows.append((pd.Timestamp("2024-06-12") + pd.Timedelta(days=i), base_price, None, 1, card))
    for j, price in enumerate(post):
        rows.append((pd.Timestamp("2024-06-16") + pd.Timedelta(days=j), price, None, 1, card))
    return rows


def test_rel_price_normalized_by_pre_event_median():
    sales = _sales(_card_sales(base_price=100.0, post=[130.0, 132.0, 128.0, 131.0, 129.0]))
    windows, drop_log = event_sale_windows(sales, _events())
    assert drop_log["kept"] == 1
    post = windows[windows["t_days"] > 0]
    assert np.isclose(post["rel_price"].mean(), 1.30, atol=0.02)
    pre = windows[windows["t_days"] < 0]
    assert np.isclose(pre["rel_price"].median(), 1.0)


def test_baseline_never_uses_post_event_sales():
    # baseline sales at 100 (some inside the ±14d window); post-event sales at
    # 200. If the baseline leaked post-event sales, rel would be pulled below 2.
    rows = [(pd.Timestamp("2024-06-01"), 100.0, None, 1, "a/x"),
            (pd.Timestamp("2024-06-05"), 100.0, None, 1, "a/x"),
            (pd.Timestamp("2024-06-10"), 100.0, None, 1, "a/x")]
    rows += [(pd.Timestamp("2024-06-16") + pd.Timedelta(days=j), 200.0, None, 1, "a/x")
             for j in range(5)]
    windows, _ = event_sale_windows(_sales(rows), _events())
    post = windows[windows["t_days"] > 0]
    assert (post["rel_price"] == 2.0).all()


def test_min_baseline_sales_rule():
    # only 2 pre-event sales total -> baseline too thin (in-window pre sales
    # would count toward the baseline too, so none are planted here)
    sales = _sales(_card_sales(n_base=2, n_pre_in_window=0, post=[130.0] * 5))
    windows, drop_log = event_sale_windows(sales, _events())
    assert drop_log["kept"] == 0 and drop_log["dropped_baseline"] == 1
    assert len(windows) == 0


def test_min_window_sales_rule():
    # 5 baseline sales but nothing inside the ±14d window -> dropped_window
    sales = _sales(_card_sales(n_base=5, n_pre_in_window=0, post=[]))
    windows, drop_log = event_sale_windows(sales, _events(), min_window=6)
    assert drop_log["kept"] == 0 and drop_log["dropped_window"] == 1


def test_grade_class_filters():
    rows = _card_sales(base_price=100.0, post=[130.0] * 5)
    rows = [(d, p, "psa_10", m, c) for (d, p, g, m, c) in rows]  # all graded
    windows, drop_log = event_sale_windows(_sales(rows), _events(), grade_class="ungraded")
    assert drop_log["kept"] == 0
    windows10, drop_log10 = event_sale_windows(_sales(rows), _events(), grade_class="psa_10")
    assert drop_log10["kept"] == 1
    assert (windows10["rel_price"] > 0).all()


def test_window_bounds():
    # sales at exactly -14d and +14d are inside; +15d is outside
    rows = _card_sales(n_base=5, n_pre_in_window=0, post=[130.0] * 3)
    rows.append((pd.Timestamp("2024-06-01"), 99.0, None, 1, "a/x"))   # exactly -14d
    rows.append((pd.Timestamp("2024-06-29"), 130.0, None, 1, "a/x"))   # exactly +14d
    rows.append((pd.Timestamp("2024-06-30"), 130.0, None, 1, "a/x"))   # +15d -> outside
    windows, drop_log = event_sale_windows(_sales(rows), _events())
    assert drop_log["kept"] == 1
    assert windows["t_days"].min() == pytest.approx(-14.0)
    assert windows["t_days"].max() == pytest.approx(14.0)
```

Note in the plan for the implementer: `sale_date` and `event_date` are both normalized to midnight in the source data; `t_days` is still computed as a float timedelta so intraday timestamps (if they ever appear) keep working.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lag_study.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cardprice.lag_study'`

- [ ] **Step 3: Implement part 1 of `src/cardprice/lag_study.py`**

```python
# src/cardprice/lag_study.py
"""Price incorporation lag: sale-level event study at daily grain.

Each sale is normalized by its card's EVENT-ANCHORED baseline (median of
same-grade-class sales strictly before the event) so post-event repricing is
visible instead of absorbed by a moving baseline. Pooled curves + cluster
bootstrap live in part 2 (Tasks 3-4).
"""

import numpy as np
import pandas as pd

GRADE_CLASSES = {"ungraded": None, "psa_10": "psa_10"}

WINDOW_COLUMNS = [
    "event_id", "mlb_id", "card_slug", "event_date", "sale_date", "t_days", "rel_price",
]


def _grade_mask(sales: pd.DataFrame, grade_class: str) -> pd.Series:
    grade = GRADE_CLASSES[grade_class]
    return sales["grade"].isna() if grade is None else (sales["grade"] == grade)


def event_sale_windows(
    sales: pd.DataFrame,
    events: pd.DataFrame,
    grade_class: str = "ungraded",
    baseline_days: int = 28,
    window_days: int = 14,
    min_baseline: int = 3,
    min_window: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Align a card's sales around each event; normalize by pre-event baseline.

    Returns (windows, drop_log); drop_log counts event-card pairs dropped for
    too few baseline sales / too few window sales, and kept pairs.
    """
    s = sales[_grade_mask(sales, grade_class) & sales["price"].notna()]
    rows: list[dict] = []
    drop_log = {"dropped_baseline": 0, "dropped_window": 0, "kept": 0}
    by_card = {c: g.sort_values("sale_date") for c, g in s.groupby("card_slug")}
    cards_of = s.groupby("mlb_id")["card_slug"].unique()
    for event_id, e in enumerate(events.itertuples()):
        for card in cards_of.get(e.mlb_id, []):
            g = by_card.get(card)
            if g is None:
                continue
            d = g["sale_date"]
            pre = g[(d >= e.event_date - pd.Timedelta(days=baseline_days)) & (d < e.event_date)]
            if len(pre) < min_baseline:
                drop_log["dropped_baseline"] += 1
                continue
            baseline = float(pre["price"].median())
            win = g[(d >= e.event_date - pd.Timedelta(days=window_days))
                    & (d <= e.event_date + pd.Timedelta(days=window_days))]
            if len(win) < min_window:
                drop_log["dropped_window"] += 1
                continue
            drop_log["kept"] += 1
            t = (win["sale_date"] - e.event_date).dt.total_seconds() / 86400.0
            for sale_date, t_days, price in zip(win["sale_date"], t, win["price"]):
                rows.append(
                    {
                        "event_id": event_id, "mlb_id": e.mlb_id, "card_slug": card,
                        "event_date": e.event_date, "sale_date": sale_date,
                        "t_days": float(t_days), "rel_price": float(price) / baseline,
                    }
                )
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS), drop_log
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lag_study.py -v`
Expected: 6 passed. `ruff check src/cardprice/lag_study.py tests/test_lag_study.py` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/lag_study.py tests/test_lag_study.py
git commit -m "feat: event-anchored sale windows with grade-class filtering and drop accounting"
```

---

### Task 3: Market index + adjustment (`lag_study.py` part 2)

**Files:**
- Modify: `src/cardprice/lag_study.py`
- Test: `tests/test_lag_study.py` (append)

**Interfaces:**
- Consumes: Task 2's `windows` frame; the sales schema.
- Produces (used by Tasks 4–5):
  - `market_relative_index(sales, grade_class="ungraded", baseline_days=28, min_cards=3, max_gap_days=3) -> pd.Series` — index = calendar day (normalized), values = median across cards of (card's daily median price ÷ card's trailing baseline at that date, baseline strictly pre-date); days with < `min_cards` contributing cards are NaN; gaps ≤ `max_gap_days` linearly interpolated.
  - `adjust_for_market(windows: pd.DataFrame, mkt: pd.Series) -> tuple[pd.DataFrame, int]` — adds `rel_adj = rel_price / mkt[sale_day]`; returns (frame, n_unadjusted) where `rel_adj` is NaN when the market index is NaN.

- [ ] **Step 1: Write the failing tests (append to `tests/test_lag_study.py`)**

```python
from cardprice.lag_study import adjust_for_market, market_relative_index


def _market_sales():
    rows = []
    # card A: baseline 100 (days -30..-26 of the probe day), then 130 on probe days
    for i in range(5):
        rows.append((pd.Timestamp("2024-05-20") + pd.Timedelta(days=i), 100.0, None, 1, "a/x"))
    # card B: baseline 50, flat (no event)
    for i in range(5):
        rows.append((pd.Timestamp("2024-05-20") + pd.Timedelta(days=i), 50.0, None, 2, "b/y"))
    # card C: baseline 200, flat
    for i in range(5):
        rows.append((pd.Timestamp("2024-05-20") + pd.Timedelta(days=i), 200.0, None, 3, "c/z"))
    probe = pd.Timestamp("2024-06-15")
    rows.append((probe, 130.0, None, 1, "a/x"))
    rows.append((probe, 50.0, None, 2, "b/y"))
    rows.append((probe, 200.0, None, 3, "c/z"))
    return _sales(rows), probe


def test_market_index_hand_computed():
    sales, probe = _market_sales()
    mkt = market_relative_index(sales)
    # at probe: A rel = 130/100 = 1.3, B = 1.0, C = 1.0 -> median = 1.0
    assert mkt.loc[probe] == 1.0
    probe2 = probe + pd.Timedelta(days=1)
    # only card A trades next day: < min_cards contributing -> the day never
    # enters as data; interpolation (limit 3, from probe's 1.0) fills it
    sales2 = pd.concat([sales, _sales([(probe2, 130.0, None, 1, "a/x")])])
    mkt2 = market_relative_index(sales2)
    assert mkt2.loc[probe2] == 1.0


def test_adjust_for_market_divides():
    sales, probe = _market_sales()
    mkt = market_relative_index(sales)
    # pretend a window sale of card A at probe with rel_price 1.3
    windows = pd.DataFrame(
        {"event_id": [0], "mlb_id": [1], "card_slug": ["a/x"],
         "event_date": [probe], "sale_date": [probe], "t_days": [0.0], "rel_price": [1.3]}
    )
    out, n_unadj = adjust_for_market(windows, mkt)
    assert out["rel_adj"].iloc[0] == 1.3 / 1.0
    assert n_unadj == 0


def test_adjust_for_market_missing_index_is_nan_not_fabricated():
    windows = pd.DataFrame(
        {"event_id": [0], "mlb_id": [1], "card_slug": ["a/x"],
         "event_date": [pd.Timestamp("2024-06-15")],
         "sale_date": [pd.Timestamp("2019-01-01")],  # far outside any market coverage
         "t_days": [0.0], "rel_price": [1.3]}
    )
    sales, _ = _market_sales()
    mkt = market_relative_index(sales)
    out, n_unadj = adjust_for_market(windows, mkt)
    assert pd.isna(out["rel_adj"].iloc[0])
    assert n_unadj == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lag_study.py -v`
Expected: FAIL — `ImportError: cannot import name 'market_relative_index'`

- [ ] **Step 3: Implement part 2 (append to `src/cardprice/lag_study.py`)**

```python
def market_relative_index(
    sales: pd.DataFrame,
    grade_class: str = "ungraded",
    baseline_days: int = 28,
    min_cards: int = 3,
    max_gap_days: int = 3,
) -> pd.Series:
    """Daily cross-card median of (card daily median price / card trailing baseline).

    A card contributes to a day only when it both trades that day and has >= 1
    baseline sale in the trailing `baseline_days` (strictly pre-date). Days
    with < `min_cards` contributing cards are NaN; NaN gaps of <= `max_gap_days`
    are linearly interpolated.
    """
    s = sales[_grade_mask(sales, grade_class) & sales["price"].notna()].copy()
    s["d"] = s["sale_date"].dt.normalize()
    per_day: dict[pd.Timestamp, list[float]] = {}
    for _, g in s.groupby("card_slug"):
        g = g.sort_values("d")
        days = g["d"].to_numpy()
        prices = g["price"].to_numpy(dtype=float)
        daily = g.groupby("d")["price"].median()
        for day, med in daily.items():
            lo = np.datetime64(day - pd.Timedelta(days=baseline_days))
            i0 = np.searchsorted(days, lo, side="left")
            i1 = np.searchsorted(days, np.datetime64(day), side="left")  # strictly pre-date
            if i1 - i0 < 1:
                continue
            base = float(np.median(prices[i0:i1]))
            if base > 0:
                per_day.setdefault(day, []).append(float(med) / base)
    idx = pd.date_range(s["d"].min(), s["d"].max(), freq="D")
    mkt = pd.Series(
        [float(np.median(per_day[d])) if len(per_day.get(d, [])) >= min_cards else np.nan
         for d in idx],
        index=idx,
    )
    return mkt.interpolate(limit=max_gap_days, limit_direction="both")


def adjust_for_market(windows: pd.DataFrame, mkt: pd.Series) -> tuple[pd.DataFrame, int]:
    """rel_adj = rel_price / market_index(sale day). NaN index -> NaN, counted."""
    out = windows.copy()
    day = out["sale_date"].dt.normalize()
    m = day.map(mkt)
    out["rel_adj"] = out["rel_price"] / m
    n_unadj = int(out["rel_adj"].isna().sum())
    return out, n_unadj
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lag_study.py -v`
Expected: 9 passed. `ruff check src/cardprice/lag_study.py tests/test_lag_study.py` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/lag_study.py tests/test_lag_study.py
git commit -m "feat: daily market-relative index and market adjustment for event windows"
```

---

### Task 4: Pooled curves, lag estimator, half-life, adjustment classes (`lag_study.py` part 3)

**Files:**
- Modify: `src/cardprice/lag_study.py`
- Test: `tests/test_lag_study.py` (append)

**Interfaces:**
- Consumes: Task 3's adjusted windows (`rel_adj`).
- Produces (used by Task 5):
  - `pooled_lag_curve(windows, value_col="rel_adj", bin_min=-14, bin_max=14, n_boot=2000, seed=42) -> pd.DataFrame` — columns: `t_bin` (int; bin k covers `[k−0.5, k+0.5)` days), `median`, `ci_lo`, `ci_hi`, `n_sales`. Bootstrap resamples `event_id`s with replacement (cluster bootstrap).
  - `estimate_lag(curve, threshold=1.0, hold_bins=3) -> dict` — `{"lag_days": int | None, "pre_ok": bool}`; first bin k ≥ 0 where `ci_lo` stays > threshold for `hold_bins` consecutive bins; `pre_ok` = all pre-bins' CIs cover 1.0.
  - `fit_half_life(curve, h_min=0.25, h_max=14.0, h_step=0.25) -> dict` — `{"half_life_days": float, "amplitude": float}`; fits `median − 1 = A·(1 − 2^(−t/h))` on bins t ≥ 0 by grid search over h with A solved by least squares; returns both NaN when Σ post-bin (median−1) ≤ 0 **or** the fitted amplitude ≤ 0.01 (noise-level move).
  - `classify_adjustment(windows, early_hi=3.5, late_lo=7.0, no_move=0.02, late_move=0.05, fast_ratio=0.8) -> pd.DataFrame` — per `event_id`: `pre_med` (t ∈ [−14, 0)), `early_med` (t ∈ [0, 3.5)), `late_med` (t ∈ [7, 14]), `cls` ∈ `{"fast", "intermediate", "late", "no_adjustment", "insufficient"}` (each segment needs ≥ 2 sales else `insufficient`). `late − pre ≤ no_move` → `no_adjustment`; `(early−pre)/(late−pre) ≥ fast_ratio` → `fast`; `|early−pre| ≤ no_move and late−pre ≥ late_move` → `late`; else `intermediate`.
  - `assign_strata(events, game_logs_mlb) -> pd.Series` — `"prospect"` when `career.career_stage(game_logs_mlb, mlb_id, event_date)` ∈ {`prospect`, `rookie_year`}, else `"established"`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_lag_study.py`)**

```python
from cardprice.lag_study import (
    classify_adjustment,
    estimate_lag,
    fit_half_life,
    pooled_lag_curve,
)


def _planted_windows(event_id, baseline=1.0, jump=0.3, jump_day=2.0, n_per_day=3, seed=0):
    """Synthetic adjusted windows: rel_adj ~ N(1, .01) pre, ramps to 1+jump at jump_day."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(-14, 15):
        level = baseline if t < jump_day else baseline + jump
        for _ in range(n_per_day):
            rows.append({"event_id": event_id, "t_days": t + rng.uniform(-0.3, 0.3),
                         "rel_adj": level + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_pooled_lag_curve_bins_and_cis():
    windows = pd.concat([_planted_windows(0, seed=1), _planted_windows(1, seed=2)])
    curve = pooled_lag_curve(windows, n_boot=200, seed=42)
    row = curve[curve["t_bin"] == 5].iloc[0]
    assert row["median"] > 1.2
    assert row["ci_lo"] > 1.0
    pre = curve[curve["t_bin"] == -5].iloc[0]
    assert pre["ci_lo"] <= 1.0 <= pre["ci_hi"]


def test_estimate_lag_recovers_planted_two_day_lag():
    windows = pd.concat([_planted_windows(e, jump_day=2.0, seed=e) for e in range(6)])
    curve = pooled_lag_curve(windows, n_boot=200, seed=42)
    res = estimate_lag(curve)
    assert res["lag_days"] is not None and res["lag_days"] <= 3
    assert res["pre_ok"]


def test_estimate_lag_none_when_no_adjustment():
    windows = pd.concat([_planted_windows(e, jump=0.0, seed=e) for e in range(6)])
    curve = pooled_lag_curve(windows, n_boot=200, seed=42)
    res = estimate_lag(curve)
    assert res["lag_days"] is None


def test_fit_half_life_planted():
    windows = pd.concat([_planted_windows(e, jump=0.3, jump_day=0.0, seed=e) for e in range(6)])
    curve = pooled_lag_curve(windows, n_boot=200, seed=42)
    res = fit_half_life(curve)
    assert res["half_life_days"] <= 1.0  # jump is immediate at t=0
    assert 0.2 < res["amplitude"] < 0.4


def test_fit_half_life_no_move_is_nan():
    windows = pd.concat([_planted_windows(e, jump=0.0, seed=e) for e in range(3)])
    curve = pooled_lag_curve(windows, n_boot=100, seed=42)
    res = fit_half_life(curve)
    assert np.isnan(res["half_life_days"])


def test_classify_adjustment_fast_late_flat():
    fast = _planted_windows(0, jump_day=1.0, seed=1)
    slow = _planted_windows(1, jump_day=9.0, seed=2)
    flat = _planted_windows(2, jump=0.0, seed=3)
    out = classify_adjustment(pd.concat([fast, slow, flat])).set_index("event_id")
    assert out.loc[0, "cls"] == "fast"
    assert out.loc[1, "cls"] in {"late", "intermediate"}
    assert out.loc[2, "cls"] == "no_adjustment"


def test_classify_adjustment_insufficient_segment():
    rows = [{"event_id": 0, "t_days": 1.0, "rel_adj": 1.3}]  # one sale total
    out = classify_adjustment(pd.DataFrame(rows))
    assert out.iloc[0]["cls"] == "insufficient"
```

For `assign_strata`, a focused test using a tiny synthetic game-log frame:

```python
from cardprice.lag_study import assign_strata


def test_assign_strata_uses_career_stage_at_event_date():
    logs = pd.DataFrame(
        {"mlb_id": [1, 1], "group": ["hitting", "hitting"], "season": [2023, 2023],
         "date": pd.to_datetime(["2023-04-01", "2023-04-02"]), "level": ["mlb", "mlb"]}
    )
    events = pd.DataFrame(
        {"mlb_id": [1, 1, 2],
         "event_date": pd.to_datetime(["2022-06-01", "2023-06-01", "2024-06-01"])}
    )
    strata = assign_strata(events, logs)
    assert list(strata) == ["prospect", "prospect", "prospect"]
    # mlb_id 2 has no MLB logs at all -> prospect; pre-debut 2022 event -> prospect;
    # 2023-06 is rookie_year (debut season 2023) -> prospect stratum
    events2 = pd.DataFrame({"mlb_id": [1], "event_date": pd.to_datetime(["2026-06-01"])})
    assert list(assign_strata(events2, logs)) == ["established"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lag_study.py -v`
Expected: FAIL — `ImportError: cannot import name 'pooled_lag_curve'`

- [ ] **Step 3: Implement part 3 (append to `src/cardprice/lag_study.py`)**

```python
from cardprice.career import career_stage  # module-top import with the others


def pooled_lag_curve(
    windows: pd.DataFrame,
    value_col: str = "rel_adj",
    bin_min: int = -14,
    bin_max: int = 14,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Pooled daily median curve with cluster-bootstrap CIs (resample events)."""
    w = windows.dropna(subset=[value_col]).copy()
    w["t_bin"] = np.floor(w["t_days"] + 0.5).astype(int)
    w = w[(w["t_bin"] >= bin_min) & (w["t_bin"] <= bin_max)]
    rng = np.random.default_rng(seed)
    event_ids = w["event_id"].unique()
    boots = {k: [] for k in range(bin_min, bin_max + 1)}
    grouped = {e: g for e, g in w.groupby("event_id")}
    for _ in range(n_boot):
        sample = rng.choice(event_ids, size=len(event_ids), replace=True)
        frame = pd.concat([grouped[e] for e in sample])
        meds = frame.groupby("t_bin")[value_col].median()
        for k in boots:
            if k in meds.index:
                boots[k].append(meds[k])
    rows = []
    for k in range(bin_min, bin_max + 1):
        obs = w[w["t_bin"] == k][value_col]
        b = np.asarray(boots[k])
        rows.append(
            {
                "t_bin": k,
                "median": float(obs.median()) if len(obs) else np.nan,
                "ci_lo": float(np.percentile(b, 2.5)) if len(b) >= 20 else np.nan,
                "ci_hi": float(np.percentile(b, 97.5)) if len(b) >= 20 else np.nan,
                "n_sales": int(len(obs)),
            }
        )
    return pd.DataFrame(rows)


def estimate_lag(curve: pd.DataFrame, threshold: float = 1.0, hold_bins: int = 3) -> dict:
    """First post-event bin whose CI floor clears `threshold` for `hold_bins`
    consecutive bins. None when adjustment never reaches significance."""
    c = curve.set_index("t_bin")
    pre = c.loc[c.index < 0]
    pre_ok = bool(((pre["ci_lo"] <= threshold) & (pre["ci_hi"] >= threshold)).all())
    post_bins = sorted(c.index[c.index >= 0])
    for k in post_bins:
        seq = [k + j for j in range(hold_bins)]
        if all(b in c.index for b in seq) and all(c.loc[b, "ci_lo"] > threshold for b in seq):
            return {"lag_days": int(k), "pre_ok": pre_ok}
    return {"lag_days": None, "pre_ok": pre_ok}


def fit_half_life(
    curve: pd.DataFrame,
    h_min: float = 0.25,
    h_max: float = 14.0,
    h_step: float = 0.25,
) -> dict:
    """Fit median-1 = A*(1 - 2^(-t/h)) on post bins (t >= 0); grid over h, OLS for A."""
    post = curve[curve["t_bin"] >= 0].dropna(subset=["median"])
    t = post["t_bin"].to_numpy(dtype=float)
    y = (post["median"] - 1.0).to_numpy(dtype=float)
    if y.sum() <= 0 or len(t) < 5:
        return {"half_life_days": np.nan, "amplitude": np.nan}
    best = None
    for h in np.arange(h_min, h_max + 1e-9, h_step):
        g = 1.0 - np.power(2.0, -t / h)
        a = float((y * g).sum() / (g * g).sum())
        sse = float(((y - a * g) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, h, a)
    if best[2] <= 0.01:  # fitted move is noise-level -> report no adjustment
        return {"half_life_days": np.nan, "amplitude": np.nan}
    return {"half_life_days": float(best[1]), "amplitude": float(best[2])}


def classify_adjustment(
    windows: pd.DataFrame,
    early_hi: float = 3.5,
    late_lo: float = 7.0,
    no_move: float = 0.02,
    late_move: float = 0.05,
    fast_ratio: float = 0.8,
) -> pd.DataFrame:
    """Per-event adjustment class from adjusted relative prices (see gate in spec §8)."""
    rows = []
    for event_id, g in windows.dropna(subset=["rel_adj"]).groupby("event_id"):
        pre = g[g["t_days"] < 0]["rel_adj"]
        early = g[(g["t_days"] >= 0) & (g["t_days"] < early_hi)]["rel_adj"]
        late = g[(g["t_days"] >= late_lo)]["rel_adj"]
        if min(len(pre), len(early), len(late)) < 2:
            rows.append({"event_id": event_id, "pre_med": np.nan, "early_med": np.nan,
                         "late_med": np.nan, "cls": "insufficient"})
            continue
        pre_med, early_med, late_med = pre.median(), early.median(), late.median()
        move = late_med - pre_med
        if move <= no_move:
            cls = "no_adjustment"
        elif (early_med - pre_med) / move >= fast_ratio:
            cls = "fast"
        elif abs(early_med - pre_med) <= no_move and move >= late_move:
            cls = "late"
        else:
            cls = "intermediate"
        rows.append({"event_id": event_id, "pre_med": float(pre_med),
                     "early_med": float(early_med), "late_med": float(late_med), "cls": cls})
    return pd.DataFrame(rows)


def assign_strata(events: pd.DataFrame, game_logs_mlb: pd.DataFrame) -> pd.Series:
    """'prospect' when career stage at the event is prospect/rookie_year, else 'established'."""
    out = []
    for e in events.itertuples():
        stage = career_stage(game_logs_mlb, e.mlb_id, pd.Timestamp(e.event_date))
        out.append("prospect" if stage in ("prospect", "rookie_year") else "established")
    return pd.Series(out, index=events.index, name="stratum")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lag_study.py -v`
Expected: 16 passed. Full suite: `pytest` → all green. `ruff check src tests` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/lag_study.py tests/test_lag_study.py
git commit -m "feat: pooled lag curves, lag/half-life estimators, per-event adjustment classes"
```

---

### Task 5: CLI runner, real-data run, findings doc

**Files:**
- Modify: `src/cardprice/lag_study.py` (add `main()`)
- Modify: `docs/superpowers/specs/2026-09-17-t1-incorporation-lag-design.md` (one-line baseline-anchor correction)
- Create: `docs/findings/2026-09-17-t1-incorporation-lag.md`

**Interfaces:**
- Consumes: everything above; `cardprice.career`; the three input parquets; `data/reference/player_info_universe.csv`.
- Produces: `data/processed/breakout_events.parquet`, `data/processed/lag_curves.csv`, `data/processed/lag_summary.json` (all gitignored), terminal summary → findings doc.

- [ ] **Step 1: Add the CLI to `lag_study.py`**

```python
def main() -> None:
    """Run the lag study end to end on the universe data."""
    import argparse
    import json

    from cardprice.breakout_events import dedupe_overlaps, detect_breakouts, merge_debuts

    ap = argparse.ArgumentParser()
    ap.add_argument("--sales", default="data/processed/universe_sales.parquet")
    ap.add_argument("--game-logs", default="data/processed/game_logs_universe.parquet")
    ap.add_argument("--events", default="data/processed/events_universe.parquet")
    ap.add_argument("--out-events", default="data/processed/breakout_events.parquet")
    ap.add_argument("--out-curves", default="data/processed/lag_curves.csv")
    ap.add_argument("--out-summary", default="data/processed/lag_summary.json")
    args = ap.parse_args()

    sales = pd.read_parquet(args.sales)
    logs = pd.read_parquet(args.game_logs)
    registry = pd.read_parquet(args.events)

    events = merge_debuts(detect_breakouts(logs), registry)
    events = dedupe_overlaps(events)
    events["stratum"] = assign_strata(events, logs[logs["level"] == "mlb"]).to_numpy()
    events.to_parquet(args.out_events, index=False)
    print(f"events: {len(events)} "
          f"({events['event_type'].value_counts().to_dict()}, "
          f"levels {events['level'].value_counts(dropna=False).to_dict()})")

    summary = {"n_events": int(len(events)), "runs": {}}
    curves_all = []
    for grade_class in ("ungraded", "psa_10"):
        mkt = market_relative_index(sales, grade_class=grade_class)
        for stratum in ("all", "prospect", "established"):
            ev = events if stratum == "all" else events[events["stratum"] == stratum]
            windows, drop_log = event_sale_windows(sales, ev, grade_class=grade_class)
            windows, n_unadj = adjust_for_market(windows, mkt)
            key = f"{grade_class}/{stratum}"
            print(f"{key}: events={len(ev)} drop_log={drop_log} unadjusted_sales={n_unadj}")
            if drop_log["kept"] < 5:
                print(f"{key}: too few event-card pairs, skipped")
                summary["runs"][key] = {"skipped": True, "drop_log": drop_log}
                continue
            curve = pooled_lag_curve(windows)
            lag = estimate_lag(curve)
            hl = fit_half_life(curve)
            cls = classify_adjustment(windows)
            shares = cls["cls"].value_counts(normalize=True).round(4).to_dict()
            curve["grade_class"], curve["stratum"] = grade_class, stratum
            curves_all.append(curve)
            summary["runs"][key] = {
                "drop_log": drop_log, "n_unadjusted": n_unadj, "lag": lag,
                "half_life": hl, "adjustment_shares": shares,
                "n_event_card_pairs": int(windows["event_id"].nunique()),
            }
            print(f"{key}: lag={lag} half_life={hl} shares={shares}")

    if curves_all:
        pd.concat(curves_all).to_csv(args.out_curves, index=False)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    primary = summary["runs"].get("ungraded/all", {})
    if primary.get("adjustment_shares"):
        s = primary["adjustment_shares"]
        fast = s.get("fast", 0.0)
        late = s.get("late", 0.0)
        print("\nGATE (spec §8, primary = ungraded/all):")
        print(f"  fast share (>=80% adjusted within 72h): {fast:.3f}")
        print(f"  late share (>=20% adjusting at >=7d):   {late:.3f}")
        if fast >= 0.8:
            print("  VERDICT: reaction-timing DEAD -> reframe T3 to anticipation")
        elif late >= 0.2:
            print("  VERDICT: timing edge PLAUSIBLE -> proceed as designed")
        else:
            print("  VERDICT: intermediate -> report as measured")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Correct the spec's baseline-anchor wording**

Edit `docs/superpowers/specs/2026-09-17-t1-incorporation-lag-design.md` §9:

- old: `- Unit tests: window dedupe, market-index subtraction, grade-bucket split, trailing-median normalization (no look-ahead: baseline uses sales strictly before the sale date).`
- new: `- Unit tests: window dedupe, market-index subtraction, grade-bucket split, event-anchored baseline normalization (no look-ahead: baseline uses sales strictly before the EVENT date; a per-sale trailing baseline would absorb the repricing being measured — wording corrected in the T1 plan).`

- [ ] **Step 3: Run the study on real data**

Run (from repo root, venv active): `python -m cardprice.lag_study`
Expected: prints event counts, per-run drop logs, lag/half-life/shares, and the gate verdict; writes the three artifacts. Sanity-check the printed numbers before writing findings: event count plausible (tens to low hundreds), `pre_ok` True on the primary curve, drop counts reported (not hidden).

- [ ] **Step 4: Write the findings doc**

Create `docs/findings/2026-09-17-t1-incorporation-lag.md`:

```markdown
# T1 — Price Incorporation Lag (Daily Event Study)

Date: 2026-09-17
Plan: docs/superpowers/plans/2026-09-17-t1-incorporation-lag.md
Spec: docs/superpowers/specs/2026-09-17-t1-incorporation-lag-design.md

## Headline
<one paragraph: the measured lag distribution, the gate verdict, what it means for T3>

## Data & method
<event counts by type/level/stratum; drop_log tables; sales coverage caveats;
median 2 sales/card/month thinness; cluster bootstrap; market adjustment;
unadjusted-sale counts>

## Results (verbatim from lag_summary.json)
<per grade_class × stratum: lag_days, pre_ok, half-life, amplitude, adjustment shares;
curves table or key bins (-1, 0, +1, +3, +7, +14)>

## Gate verdict (spec §8, pre-registered)
<fast/late shares on ungraded/all; verdict; consequence for T3 framing>

## Caveats
<thinness, 39-player universe, prospect-stratum event scarcity, best_offer mix,
mix-of-grades handled via grade classes, SCP floor 2021-03>

## Reproduce
`python -m cardprice.lag_study` (venv, repo root); artifacts in data/processed/ (gitignored).
```

Rules: every number in the doc comes from `lag_summary.json`/the run output — no hand-computed or remembered numbers; if a number looks wrong, stop and investigate rather than publishing it.

- [ ] **Step 5: Final verification + commit**

Run: `pytest && ruff check src tests`
Expected: all green.
```bash
git add src/cardprice/lag_study.py docs/superpowers/specs/2026-09-17-t1-incorporation-lag-design.md docs/findings/2026-09-17-t1-incorporation-lag.md
git commit -m "feat: T1 incorporation-lag study — CLI run, gate verdict, findings"
```

---

## Self-Review

**Spec coverage:** §3 data (Tasks 1–2 consume exactly the three parquets) ✓; §5 breakout definition incl. debuts, dedupe (Task 1) ✓; §6 lag estimation + half-life + distribution-across-events (Tasks 2, 4 — distribution via `classify_adjustment`) ✓; §7 bias controls: market index (Task 3), grade classes (Task 2), <5-sale rule + counted drops (Task 2), best_offer/weekday noted as caveats in the findings template (Task 5) ✓; §8 gate operationalized via `classify_adjustment` shares + verdict printer (Tasks 4–5) ✓; §9 testing incl. planted-lag synthetic (Tasks 1–4) ✓. Strata prospect/established via existing `career_stage` (Task 4) ✓.

**Placeholder scan:** no TBD/TODO; every code step carries full code; every test step carries full test code.

**Type consistency:** `event_id` is the positional index into the events frame passed to `event_sale_windows` — Task 5 subsets `events` BEFORE calling, so ids stay positional per run; `assign_strata` returns a Series indexed like `events`, written back with `.to_numpy()` after `reset_index(drop=True)` in Task 1's outputs — implementer must ensure `events` is reset-indexed before assignment (noted here deliberately). `classify_adjustment` consumes `rel_adj` (added by Task 3) — Task 4's synthetic frames carry `rel_adj` directly ✓. `GRADE_CLASSES` values: `None` ↔ `grade.isna()` ✓.
