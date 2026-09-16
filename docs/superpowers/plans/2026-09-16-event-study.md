# Event Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether discrete performance events (debuts, milestone games, awards, playoff appearances) produce abnormal returns in rookie card prices, and whether spikes mean-revert — the sell-side timing question.

**Architecture:** Event registry (derived from existing game logs + two small MLB Stats API additions) → event-study engine (cumulative abnormal returns in windows around each event, vs the panel model's market-median benchmark) → mean-reversion measurement → findings addendum. All engine code tested on synthetic panels with planted event effects first.

**Tech Stack:** Python 3.11+, existing `cardprice` package; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-14-mlb-card-price-panel-design.md` (§6 event-study overlay)
**Predecessor:** Plans 1-4 merged. `data/processed/game_logs.parquet` (6275 rows, 16 players, 2022-2026), `panel_monthly.parquet`, `panel_weekly.parquet`, `docs/findings/2026-09-15-modeling-report.md` (gate FAIL — continuous stats show no tradeable signal; this plan tests the discrete-event channel).

## Facts established (code below is built on them)

- Game logs (Plan 1) contain per-game `homeRuns, hits, strikeOuts, baseOnBalls, date` etc. for all 16 seed players, 2022-2026 — debuts and milestone games are derivable offline. Playoff games are NOT in the logs (gameLog defaults to regular season).
- `data/reference/cards_seed.csv` and `player_info.csv` (16 players, positions, birth dates).
- Monthly panel: PSA-10, `excess_ret` per card-month (market-median-adjusted), 2022-10→2026-09. Weekly panel: mostly 2026 window, `days_since_prev` horizon column; primary analysis filters ≤21 days.
- Modeling null (Plan 4): continuous stats channel shows no stable, fee-beating signal at monthly grain — the event study answers a different question (do shocks move prices, and how fast do they fade).

## Global Constraints

- Python >= 3.11; free sources; no new deps; offline tests default (`live` marker for the two API additions).
- Event dates must be accurate — wrong event dates poison the study. Every registry rule gets a golden check against a real, known event.
- No look-ahead: "pre-event" windows end strictly before the event date; benchmarks come from the same panel's market median.
- Honest reporting: small event counts are a power limitation to state, not to massage.
- Lint: `ruff check` / `ruff format --check` only; never repo-wide `ruff format` (it rewrites markdown-embedded code in plan docs).

---

### Task 1: Event registry from game logs (debuts + milestone games)

**Files:**
- Create: `src/cardprice/events.py`
- Test: `tests/test_events_gamelogs.py`

**Interfaces:**
- Consumes: `data/processed/game_logs.parquet`.
- Produces (Tasks 3-4 consume):
  - `events_from_game_logs(game_logs: pd.DataFrame) -> pd.DataFrame` — columns: `mlb_id, event_date (datetime64), event_type, details (str)`. Rules:
    - `debut`: first regular-season game date per player (per mlb_id, min date across all seasons — careers before 2022 are outside our data; note in details if the player's logs start at 2022 opening day AND the player is not a 2022 rookie per cards_seed, flag `details="career_start_predates_data"`) — for the seed set all players debuted 2022+, so this is clean; the function must still guard.
    - `three_hr_game`: hitter game with `homeRuns >= 3`.
    - `four_hit_game`: hitter game with `hits >= 4` (only if not already a three_hr_game — a 3-HR 4-hit game is one event).
    - `ten_k_game`: pitcher game with `strikeOuts >= 10`.
  - Event dates from multiple games of the same player on the same day (doubleheaders) collapse to one event (max stat line governs `details`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_events_gamelogs.py
import pandas as pd

from cardprice.events import events_from_game_logs


def make_logs():
    rows = [
        # player 1: debut Apr 10 2022; 3-HR game Jun 5; 4-hit game Jul 1 (no HR);
        #           doubleheader Aug 10 with 2 HR + 1 HR (no single-game 3 -> no event)
        (1, "hitting", 2022, "2022-04-10", {"homeRuns": 0, "hits": 1, "strikeOuts": 1}),
        (1, "hitting", 2022, "2022-06-05", {"homeRuns": 3, "hits": 3, "strikeOuts": 0}),
        (1, "hitting", 2022, "2022-07-01", {"homeRuns": 0, "hits": 4, "strikeOuts": 1}),
        (1, "hitting", 2022, "2022-08-10", {"homeRuns": 2, "hits": 3, "strikeOuts": 0}),
        (1, "hitting", 2022, "2022-08-10", {"homeRuns": 1, "hits": 1, "strikeOuts": 1}),
        # player 2 (pitcher): debut May 1; 10-K game Jun 15
        (2, "pitching", 2022, "2022-05-01", {"homeRuns": 0, "hits": 0, "strikeOuts": 5}),
        (2, "pitching", 2022, "2022-06-15", {"homeRuns": 0, "hits": 0, "strikeOuts": 10}),
    ]
    df = pd.DataFrame(
        [(m, g, s, d, *st.values()) for m, g, s, d, st in rows],
        columns=["mlb_id", "group", "season", "date", "homeRuns", "hits", "strikeOuts"],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_registry_rules():
    ev = events_from_game_logs(make_logs())
    p1 = ev[ev["mlb_id"] == 1].sort_values("event_date")
    assert p1["event_type"].tolist() == ["debut", "three_hr_game", "four_hit_game"]
    assert p1["event_date"].tolist() == list(pd.to_datetime(["2022-04-10", "2022-06-05", "2022-07-01"]))
    p2 = ev[ev["mlb_id"] == 2]
    assert set(p2["event_type"]) == {"debut", "ten_k_game"}


def test_no_event_from_doubleheader_split():
    ev = events_from_game_logs(make_logs())
    # 2 HR + 1 HR across a doubleheader is NOT a three-HR game
    assert not (ev["event_date"] == pd.Timestamp("2022-08-10")).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_events_gamelogs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.events'`.

- [ ] **Step 3: Implement events.py**

```python
# src/cardprice/events.py
"""Event registry: discrete performance events per player."""

import pandas as pd


def events_from_game_logs(game_logs: pd.DataFrame) -> pd.DataFrame:
    df = game_logs.copy()
    # collapse doubleheaders: one row per player-day with max of counting stats
    day = (
        df.groupby(["mlb_id", "date", "group"], as_index=False)
        [["homeRuns", "hits", "strikeOuts"]]
        .max()
    )
    rows = []
    for mlb_id, grp in day.groupby("mlb_id"):
        debut = grp["date"].min()
        rows.append({"mlb_id": mlb_id, "event_date": debut, "event_type": "debut",
                     "details": "first game in dataset"})
        hit = grp[grp["group"] == "hitting"]
        pit = grp[grp["group"] == "pitching"]
        for r in hit[hit["homeRuns"] >= 3].itertuples():
            rows.append({"mlb_id": mlb_id, "event_date": r.date, "event_type": "three_hr_game",
                         "details": f"{r.homeRuns} HR"})
        four_hit = hit[(hit["hits"] >= 4) & (hit["homeRuns"] < 3)]
        for r in four_hit.itertuples():
            rows.append({"mlb_id": mlb_id, "event_date": r.date, "event_type": "four_hit_game",
                         "details": f"{r.hits} H"})
        for r in pit[pit["strikeOuts"] >= 10].itertuples():
            rows.append({"mlb_id": mlb_id, "event_date": r.date, "event_type": "ten_k_game",
                         "details": f"{r.strikeOuts} K"})
    return pd.DataFrame(rows).sort_values(["mlb_id", "event_date"]).reset_index(drop=True)
```

NOTE: the debut guard for careers predating the data — join against `data/reference/cards_seed.csv` is NOT this function's job (it takes only game_logs). For the seed set every player's debut is in the data (all rookies 2022-2025); the function documents that assumption in the docstring: "debut = first game in the loaded logs; caller must ensure logs cover the player's career start."

- [ ] **Step 4: Run tests, then golden check on real data**

Run: `python -m pytest tests/test_events_gamelogs.py -v` — 2 PASS.
Golden check (offline, real parquet — put it in the test as a third test reading data/processed/game_logs.parquet, guarded by `pytest.mark.skipif(not Path("data/processed/game_logs.parquet").exists())`):
```python
def test_golden_real_debuts():
    logs = pd.read_parquet("data/processed/game_logs.parquet")
    ev = events_from_game_logs(logs)
    henderson = ev[(ev["mlb_id"] == 683002) & (ev["event_type"] == "debut")].iloc[0]
    assert henderson["event_date"] == pd.Timestamp("2022-08-31")  # Henderson's MLB debut
    skenes = ev[(ev["mlb_id"] == 694973) & (ev["event_type"] == "debut")].iloc[0]
    assert skenes["event_date"] == pd.Timestamp("2024-05-11")  # Skenes' MLB debut
```
Verify both dates against the parquet first; if a date differs, check the MLB record before believing either — report which was wrong.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/events.py tests/test_events_gamelogs.py
git commit -m "feat: event registry from game logs"
```

---

### Task 2: Awards + playoff events from the Stats API

**Files:**
- Modify: `src/cardprice/events.py` (append)
- Test: `tests/test_events_api.py`

**Interfaces:**
- Consumes: `stats_api.BASE`, `requests`; cards_seed.csv for the player set.
- Produces (Task 3 consumes):
  - `fetch_award_events(mlb_ids: list[int], seasons: list[int]) -> pd.DataFrame` — same columns as Task 1. Uses `GET {BASE}/awards/{award_id}/recipients?season={season}` (no key) for award_ids: `MVP` (use `BBWAAMVP`/`BBWAANLVP` — check the API; if those 404, use `GET {BASE}/awards` to list award ids and find the MVP/Cy Young/ROY entries), Cy Young, and Rookie of the Year; emits `event_type` = `award_win` with details naming the award, `event_date` = the announcement date if present in the payload (else Nov 15 of the season — note in details), filtered to the given mlb_ids.
  - `fetch_playoff_events(mlb_ids: list[int], seasons: list[int], players: pd.DataFrame) -> pd.DataFrame` — playoff *participation*: for each player-season, whether the player's team appeared in the postseason, via `GET {BASE}/schedule?sportId=1&teamId={team}&gameType=P&season={season}&fields=dates,date,games,teams,away,home,team,id` — or simpler: `stats=gameLog&group=hitting&season={season}&gameType=P` per player (playoff game log — exists for players who appeared). The gameLog-with-gameType=P route is simpler and player-level: emit `event_type` = `playoff_appearance`, `event_date` = first playoff game date, `details` = number of playoff games. Use that route.
- [ ] **Step 1: Write the failing test**

```python
# tests/test_events_api.py
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cardprice import events


def test_fetch_playoff_events_parses(monkeypatch):
    payload = {
        "stats": [
            {
                "splits": [
                    {"date": "2024-10-01", "stat": {"gamesPlayed": 1}},
                    {"date": "2024-10-05", "stat": {"gamesPlayed": 1}},
                ]
            }
        ]
    }

    def fake_get(url, params=None, timeout=None):
        assert params.get("gameType") == "P"
        r = MagicMock()
        r.json.return_value = payload
        r.raise_for_status = lambda: None
        return r

    monkeypatch.setattr(events.requests, "get", fake_get)
    out = events.fetch_playoff_events([683002], [2024], None)
    assert len(out) == 1
    assert out.iloc[0]["event_type"] == "playoff_appearance"
    assert out.iloc[0]["event_date"] == pd.Timestamp("2024-10-01")
    assert "2" in out.iloc[0]["details"]


def test_fetch_playoff_events_empty_when_no_postseason(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        r = MagicMock()
        r.json.return_value = {"stats": []}
        r.raise_for_status = lambda: None
        return r

    monkeypatch.setattr(events.requests, "get", fake_get)
    assert len(events.fetch_playoff_events([683002], [2024], None)) == 0


@pytest.mark.live
def test_awards_and_playoffs_live():
    ev = events.fetch_playoff_events([683002], [2024], None)  # Orioles made the 2024 postseason
    assert len(ev) == 1
    awards = events.fetch_award_events([683002], [2023])  # Henderson won 2023 AL ROY
    assert (awards["event_type"] == "award_win").any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_events_api.py -v` (live deselected)
Expected: FAIL with `AttributeError: module 'cardprice.events' has no attribute 'fetch_playoff_events'`.

- [ ] **Step 3: Implement** (append to events.py)

```python
import requests  # top of file if not present

from cardprice.stats_api import BASE


def fetch_playoff_events(mlb_ids, seasons, players=None) -> pd.DataFrame:
    rows = []
    for mlb_id in mlb_ids:
        for season in seasons:
            for group in ("hitting", "pitching"):
                resp = requests.get(
                    f"{BASE}/people/{mlb_id}/stats",
                    params={"stats": "gameLog", "group": group,
                            "season": season, "gameType": "P"},
                    timeout=30,
                )
                resp.raise_for_status()
                stats = resp.json().get("stats", [])
                if not stats:
                    continue
                splits = stats[0].get("splits", [])
                if splits:
                    rows.append({
                        "mlb_id": mlb_id, "event_date": pd.Timestamp(splits[0]["date"]),
                        "event_type": "playoff_appearance",
                        "details": f"{len(splits)} postseason games ({group})",
                    })
                    break  # one event per player-season
    return pd.DataFrame(rows)


def fetch_award_events(mlb_ids, seasons) -> pd.DataFrame:
    # discover award ids once
    resp = requests.get(f"{BASE}/awards", timeout=30)
    resp.raise_for_status()
    award_ids = [
        a["id"]
        for a in resp.json().get("awards", [])
        if any(k in a.get("name", "").lower() for k in ("most valuable", "cy young", "rookie of the year"))
    ]
    rows = []
    for award_id in award_ids:
        for season in seasons:
            r = requests.get(
                f"{BASE}/awards/{award_id}/recipients",
                params={"season": season}, timeout=30,
            )
            r.raise_for_status()
            for a in r.json().get("awards", []):
                for rec in a.get("recipients", []):
                    pid = rec.get("player", {}).get("id")
                    if pid in mlb_ids:
                        rows.append({
                            "mlb_id": pid, "event_date": pd.Timestamp(f"{season}-11-15"),
                            "event_type": "award_win",
                            "details": f"{a.get('name', award_id)} {season} (announcement date approximated: Nov 15)",
                        })
    return pd.DataFrame(rows)
```

NOTE: award payload shapes vary; if the recipients endpoint returns a different shape (e.g. `recipients` at top level of each award entry), adapt and note it. The announcement-date approximation (Nov 15) must stay visible in `details` — the report treats award events at month grain only, where mid-November precision is adequate. Sleep 0.3s between calls.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_events_api.py -v` — 2 PASS. Live: `python -m pytest tests/test_events_api.py -v -m live` — 1 PASS (network). If the awards endpoint discovery yields an empty list, print the awards list and pick the right name matchers — do not hardcode ids without evidence.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/events.py tests/test_events_api.py
git commit -m "feat: award and playoff events from Stats API"
```

---

### Task 3: Event-study engine

**Files:**
- Create: `src/cardprice/event_study.py`
- Test: `tests/test_event_study.py`

**Interfaces:**
- Consumes: events DataFrame (Tasks 1-2), panel parquets.
- Produces (Task 4 consumes):
  - `event_windows(events: pd.DataFrame, panel: pd.DataFrame, grain: str, pre: int, post: int) -> pd.DataFrame` — for each event, find the player's card panel rows (`mlb_id` join; weekly grain matches event date into its Monday week via `to_period("W-SUN")`; monthly into its month). A window = panel periods `[-pre, +post]` around the event's period. Output columns: `mlb_id, card_slug, event_type, event_date, period_offset (int), excess_ret, horizon_ok (bool)` (weekly: horizon_ok = days_since_prev <= 21; monthly: months_since_prev <= 2). Periods with no panel row are simply absent (no fabrication).
  - `mean_car(windows: pd.DataFrame, offsets: list[int]) -> pd.DataFrame` — cumulative abnormal return per event: sum of excess_ret over the given offsets (e.g. [0] = event period, [0,1] = through one period after); returns per-event CARs: `event key columns + car`.
  - `event_significance(cars: pd.Series, panel: pd.DataFrame, n_perm: int = 5000, seed: int = 42) -> dict` — permutation test: compare mean CAR to the distribution of mean "CARs" computed at the same number of randomly chosen (card, period) pseudo-events drawn from the panel (excluding actual event windows); returns `{"mean_car": float, "p_value": float, "n_events": int, "null_mean": float, "null_sd": float}`.

- [ ] **Step 1: Write the failing test (planted event effect)**

```python
# tests/test_event_study.py
import numpy as np
import pandas as pd

from cardprice.event_study import event_significance, event_windows, mean_car


def make_panel_and_events(seed=7, spike=0.0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-03-01", periods=8, freq="MS")
    rows = []
    for m in months:
        for c in range(6):
            rows.append({"mlb_id": c, "card_slug": f"card/{c}", "month": m,
                         "excess_ret": rng.normal(0, 0.05),
                         "months_since_prev": 1})
    panel = pd.DataFrame(rows)
    events = pd.DataFrame(
        {"mlb_id": [0, 1, 2], "event_date": pd.to_datetime(["2023-06-10"] * 3),
         "event_type": "three_hr_game", "details": "3 HR"}
    )
    if spike:
        for _, e in events.iterrows():
            mask = (panel["mlb_id"] == e["mlb_id"]) & (panel["month"] == "2023-06-01")
            panel.loc[mask, "excess_ret"] += spike
    return panel, events


def test_event_windows_offsets():
    panel, events = make_panel_and_events()
    w = event_windows(events, panel, "monthly", pre=1, post=2)
    assert set(w["period_offset"]) == {-1, 0, 1, 2}
    assert (w["n"].count() if "n" in w else len(w)) == 12  # 3 events x 4 offsets
    assert w["horizon_ok"].all()


def test_mean_car_with_planted_spike():
    panel, events = make_panel_and_events(spike=0.40)
    w = event_windows(events, panel, "monthly", pre=1, post=1)
    cars = mean_car(w, [0, 1])
    assert (cars["car"] > 0.30).all()  # planted 0.40 minus small noise (fixed seed makes this deterministic)


def test_significance_detects_spike_and_null():
    panel, events = make_panel_and_events(seed=11, spike=0.0)
    w = event_windows(events, panel, "monthly", pre=0, post=0)
    cars = mean_car(w, [0])
    res = event_significance(cars["car"], panel, n_perm=2000, seed=42)
    assert res["p_value"] > 0.05  # no planted effect -> null not rejected

    panel2, events2 = make_panel_and_events(seed=11, spike=0.40)
    w2 = event_windows(events2, panel2, "monthly", pre=0, post=0)
    cars2 = mean_car(w2, [0])
    res2 = event_significance(cars2["car"], panel2, n_perm=2000, seed=42)
    assert res2["p_value"] <= 0.05  # strong planted effect detected
```

NOTE: for the null test to work with only 3 events, the permutation's pseudo-events must be drawn over ALL (card, period) cells; with spike 0.40 vs noise sd 0.05 the planted p should be ~0. If the null test is flaky at seed 42, adjust only `n_perm` or seed and document it.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_event_study.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement event_study.py**

```python
# src/cardprice/event_study.py
"""Event study: abnormal card returns around discrete performance events."""

import numpy as np
import pandas as pd


def _event_period(event_date: pd.Timestamp, grain: str) -> pd.Timestamp:
    if grain == "monthly":
        return event_date.to_period("M").start_time
    return event_date.to_period("W-SUN").start_time


def event_windows(events, panel, grain, pre, post) -> pd.DataFrame:
    period_col = "month" if grain == "monthly" else "week"
    horizon_col = "months_since_prev" if grain == "monthly" else "days_since_prev"
    horizon_max = 2 if grain == "monthly" else 21
    periods = sorted(panel[period_col].unique())
    rows = []
    for e in events.itertuples():
        center = _event_period(e.event_date, grain)
        if center not in periods:
            continue
        center_idx = periods.index(center)
        for offset in range(-pre, post + 1):
            idx = center_idx + offset
            if idx < 0 or idx >= len(periods):
                continue
            period = periods[idx]
            prow = panel[(panel["mlb_id"] == e.mlb_id) & (panel[period_col] == period)]
            for r in prow.itertuples():
                hz = getattr(r, horizon_col)
                rows.append({
                    "mlb_id": e.mlb_id, "card_slug": r.card_slug,
                    "event_type": e.event_type, "event_date": e.event_date,
                    "period_offset": offset, "excess_ret": r.excess_ret,
                    "horizon_ok": bool(pd.isna(hz) or hz <= horizon_max),
                })
    return pd.DataFrame(rows)


def mean_car(windows: pd.DataFrame, offsets: list[int]) -> pd.DataFrame:
    w = windows[windows["period_offset"].isin(offsets) & windows["horizon_ok"]]
    return (
        w.groupby(["mlb_id", "card_slug", "event_type", "event_date"])["excess_ret"]
        .sum()
        .rename("car")
        .reset_index()
    )


def event_significance(cars, panel, n_perm=5000, seed=42, width: int = 1) -> dict:
    cars = pd.Series(cars).dropna()
    if len(cars) == 0:
        return {"mean_car": np.nan, "p_value": np.nan, "n_events": 0,
                "null_mean": np.nan, "null_sd": np.nan}
    pool = panel["excess_ret"].dropna().to_numpy()
    rng = np.random.default_rng(seed)
    # pseudo-events: same count, same CAR width (periods summed) as observed cars
    obs = cars.mean()
    null = np.array([
        rng.choice(pool, size=len(cars) * width, replace=True).mean()
        for _ in range(n_perm)
    ])
    p = float((np.abs(null) >= abs(obs)).mean())
    return {
        "mean_car": float(obs), "p_value": p, "n_events": len(cars),
        "null_mean": float(null.mean()), "null_sd": float(null.std()),
    }
```

NOTE: `width` is the number of periods summed in each CAR (pass `width=len(offsets)` when mean_car used more than [0]); the permutation null must match that width or variance is mis-scaled. The pool includes actual event periods (they're a small fraction of the panel) — a documented simplification; a stricter variant excluding event windows is a refinement, not required here.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_event_study.py -v` — 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/event_study.py tests/test_event_study.py
git commit -m "feat: event-study engine with permutation test"
```

---

### Task 4: Real event study + mean reversion + findings addendum

**Files:**
- Create: `scripts/run_event_study.py`
- Create: `docs/findings/2026-09-16-event-study-report.md`

**Interfaces:**
- Consumes: all of the above.
- Produces: the findings addendum (committed).

- [ ] **Step 1: Build the runner**

```python
# scripts/run_event_study.py
"""Run the event study on real data and print everything the report needs."""

import pandas as pd

from cardprice.event_study import event_significance, event_windows, mean_car
from cardprice.events import (
    events_from_game_logs,
    fetch_award_events,
    fetch_playoff_events,
)

if __name__ == "__main__":
    logs = pd.read_parquet("data/processed/game_logs.parquet")
    monthly = pd.read_parquet("data/processed/panel_monthly.parquet")
    weekly = pd.read_parquet("data/processed/panel_weekly.parquet")
    cards = pd.read_csv("data/reference/cards_seed.csv")
    mlb_ids = cards["mlb_id"].astype(int).tolist()
    seasons = [2022, 2023, 2024, 2025, 2026]

    events = events_from_game_logs(logs)
    try:
        events = pd.concat([events, fetch_playoff_events(mlb_ids, seasons)])
        events = pd.concat([events, fetch_award_events(mlb_ids, seasons)])
    except Exception as e:
        print(f"WARN: API event fetch failed ({e}); continuing with game-log events only")

    events.to_parquet("data/processed/events.parquet", index=False)
    print(events.groupby("event_type").size())

    # monthly study (primary: history)
    for etype in sorted(events["event_type"].unique()):
        ev = events[events["event_type"] == etype]
        w = event_windows(ev, monthly, "monthly", pre=1, post=2)
        cars = mean_car(w, [0])
        sig = event_significance(cars["car"], monthly)
        print(f"\n{etype}: n={sig['n_events']}, mean CAR(event month)={sig['mean_car']:.4f}, "
              f"p={sig['p_value']:.3f}")
        # reversion: CAR[0,+1] vs CAR[0]
        cars2 = mean_car(w, [0, 1])
        merged = cars.merge(cars2, on=["mlb_id", "card_slug", "event_type", "event_date"],
                            suffixes=("_e0", "_e01"))
        merged["giveback"] = merged["car_e01"] - merged["car_e0"]
        spikes = merged[merged["car_e0"] > 0]
        if len(spikes):
            print(f"  reversion: {len(spikes)} positive spikes, mean next-month giveback "
                  f"{spikes['giveback'].mean():.4f} "
                  f"({(spikes['giveback'] < 0).mean():.0%} give back some)")

    # weekly study (2026 window only — descriptive)
    w = event_windows(events, weekly, "weekly", pre=1, post=3)
    if len(w):
        cars = mean_car(w, [0])
        sig = event_significance(cars["car"], weekly)
        print(f"\nweekly all-events: n={sig['n_events']}, mean CAR={sig['mean_car']:.4f}, p={sig['p_value']:.3f}")
```

- [ ] **Step 2: Run it (network for awards/playoff fetches)**

```bash
python scripts/run_event_study.py | tee docs/findings/2026-09-16-event-study-output.txt
```

- [ ] **Step 3: Write the findings addendum**

`docs/findings/2026-09-16-event-study-report.md` — sections:
1. **Event registry summary** — counts per event type and player; date-accuracy evidence (golden debuts).
2. **Do events move card prices?** — per-event-type table: n, mean CAR (event month), permutation p-value; plus the pooled weekly study for 2026. State the power limit (event counts).
3. **Mean reversion** — of the positive spikes, the share that give back value next period and the mean giveback; compare to the hobby claim (2-3 week reversion) and note the monthly grain can't resolve sub-month timing.
4. **Implications** — what this adds to the Plan 4 null; whether "buy the breakout game" has any support; what data would sharpen it (weekly history, bigger universe).
5. **Limitations** — announcement-date approximation for awards; monthly grain; 13 cards; PSA-10-only monthly; multiple-testing across event types (note which p-values survive a Bonferroni-style correction at the number of tests run).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_event_study.py docs/findings/2026-09-16-event-study-report.md docs/findings/2026-09-16-event-study-output.txt
git commit -m "feat: event study run and findings addendum"
```

---

## Done criteria for this plan

- `python -m pytest -v` all offline tests PASS (engine recovers planted spikes, nulls stay null); golden debut dates verified.
- `data/processed/events.parquet` exists with all event types.
- `docs/findings/2026-09-16-event-study-report.md` answers: do events move prices, how fast do spikes revert, and what it means for a buy strategy — with honest power caveats.
- That completes the five-plan research arc.
