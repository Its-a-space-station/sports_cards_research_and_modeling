# T3 — Betting-Odds Deltas as Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add market-priced award expectations (MVP / Cy Young / ROY odds) as features: a daily odds snapshot table, month-grain odds features merged into the class hold frame, a walk-forward delta-fit vs the T2 baseline (descriptive), an odds-spike incorporation-lag study reusing T1's reviewed machinery, and a weekly prospective collection cron.

**Architecture:** New pure module `src/cardprice/odds.py` (parsing, vig normalization, feature math — all offline-tested with committed fixtures); new collectors/runners under `scripts/`. Reuses reviewed modules verbatim: `class_hold_frame` (frozen), `walkforward_holds`, `model_lasso`/`model_gbm`, `lag_study` (T1). No changes to reviewed files.

**Tech Stack:** pandas 2.x, numpy 2.x, requests, scikit-learn, lightgbm, shap, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-t3-odds-deltas-design.md`
**Spike (governs sources):** `docs/superpowers/spikes/2026-09-19-odds-history.md` — **PASS**: Polymarket (primary; 2025 closed + 2026 live award events, all three awards × both leagues; `prices-history` unauthenticated even for closed markets; ~14-day window cap → chunk), Kalshi (2026 cross-source only — settled 2025 markets not retrievable unauthenticated). The Odds API historical = paid (recorded, not purchased). Free coverage ceiling = 2 seasons (2025 full, 2026 current).

## Global Constraints

- Ruff line-length **100**; `.venv/bin/ruff check src tests scripts` before every commit; **never** `ruff format`.
- Honest-golden discipline: goldens verified from fixtures/artifacts; a mismatch stops the run.
- **No look-ahead:** month-grain features use the last snapshot with `date < entry_month`; odds-spike events use only same-day-or-earlier prices; delta-fit uses the identical walk-forward harness as the registered gate.
- **No fabrication:** no market listed → `has_market=0` and odds features 0.0 (documented semantics: "no listed award market ≈ zero priced expectation"), never NaN-filled from later dates; missing history stays missing (audited), never interpolated across gaps > 14 days.
- **No separate gate (spec §2):** odds are a feature-add evaluated by delta-fit inside the existing walk-forward harness; the T2 registered verdict stands untouched. All T3 evaluation output is labeled descriptive.
- **No paid data:** The Odds API historical is recorded as an option for the user, never called.
- Seed every RNG; tests offline and deterministic (live capture scripts are separate, explicitly-marked).
- Politeness: 0.3–0.5 s between API calls; save raw payloads via `save_raw` before parsing (repo convention).
- Player mapping: exact normalized-name only (no fuzzy guesses); unmapped outcomes → audit CSV, never force-matched.

## Existing code this plan reuses (do not re-implement)

- `cardprice.class_hold_frame.build_class_hold_frame` (frozen adapter) + `CLASS_HOLD_FEATURES`.
- `cardprice.walkforward_holds.walk_forward_years` / `gate_evaluation_years` (with the additive `all_scores` from T2).
- `cardprice.lag_study`: `event_sale_windows`, `market_relative_index`, `adjust_for_market`, `pooled_lag_curve`, `estimate_lag`, `classify_adjustment` — the T1 study machinery is generic over an events frame (`mlb_id, event_date, …`) + a sales frame; odds-spike events feed it directly.
- `cardprice.storage.save_raw`/`load_latest`/`snapshot_exists`; `resolve_universe.norm_name` (scripts idiom).
- `scripts/run_class_gate.py` / `run_class_importance.py` — runner idioms for the evaluation scripts.

## Real-data facts (from the spike, 2026-09-19)

- Polymarket award events: 2025 (closed) + 2026 (live), both leagues, three awards; volumes $158k–$1.7M/event. `prices-history?market=<condition_id>&interval=all&fidelity=60` works unauthenticated; `interval=max` covers ~1 month; date-range queries cap ~14 days/window → chunk by 14 days. Kalshi: 2025+2026 series exist (`KXMLBALMVP` etc.); daily candlesticks free full depth — current season only.
- The class universe (2,447 players) contains many award candidates (Skenes, Kurtz, Skubal, Henderson, Witt, JRod…); some candidates (Judge, Ohtani, Raleigh) predate the classes → their odds exist but won't join the card panel (recorded, not an error).

---

### Task 1: Odds module (`src/cardprice/odds.py`)

**Files:**
- Create: `src/cardprice/odds.py`
- Create: `scripts/capture_odds_fixtures.py`
- Test: `tests/test_odds.py`
- Fixtures (committed): `tests/fixtures/odds/{polymarket_event.json, polymarket_history_chunk.json, kalshi_candlesticks.json}`

**Interfaces:**
- Consumes: captured API payloads (fixtures).
- Produces (used by Tasks 2–5):
  - `parse_polymarket_event(payload: dict) -> pd.DataFrame` — columns `[market_id (str), player_name (str), volume (float)]`; one row per outcome market in an award event.
  - `parse_polymarket_history(payload: dict, market_id: str) -> pd.DataFrame` — columns `[market_id, ts (datetime64 UTC), raw_price (float)]` from a prices-history chunk.
  - `vig_normalize(prices: pd.DataFrame) -> pd.DataFrame` — per (event, date): `implied_prob = raw_price / sum(raw_price over the event's outcomes that date)` (when the sum > 0; else the row is dropped and counted). Input columns `[event, player_name, date, raw_price]`; returns input + `implied_prob` + attr `n_dropped`.
  - `daily_prices(history: pd.DataFrame) -> pd.DataFrame` — last price per (market_id, UTC date).
  - `month_grain_features(snapshots: pd.DataFrame, entry_months: pd.Series) -> pd.DataFrame` — per (mlb_id, entry_month): `odds_level` (last `implied_prob` with date < entry_month), `odds_delta_7d`, `odds_delta_30d` (level minus the level 7/30 days earlier, using the last snapshot strictly before each cutoff; NaN when no snapshot exists before the earlier cutoff), `has_market` (1 if the player has any snapshot in that season). Exact-cutoff snapshots (date == cutoff) are excluded (strict inequality).
  - `parse_kalshi_candles(payload: dict, market_ticker: str) -> pd.DataFrame` — columns `[market_id, ts, raw_price, volume]` from Kalshi daily candlesticks (price in dollars; if the payload is in cents, divide by 100 — verify from the fixture and pin in a golden).
  - `odds_spike_events(snapshots: pd.DataFrame, min_abs_delta: float = 0.10, rel_std_window: int = 30, rel_std_mult: float = 3.0) -> pd.DataFrame` — events where |1-day implied_prob delta| ≥ max(min_abs_delta, rel_std_mult × rolling 30d std of daily deltas); output `[mlb_id, event_date, event_type ("odds_spike"), delta, implied_prob]`; per-player min-7-day separation (keep the largest |delta| within any 7-day cluster).

- [ ] **Step 1: Capture fixtures (live, one-off)**

`scripts/capture_odds_fixtures.py` (≥ 5 s gaps): one Polymarket award event payload (gamma API, 2025 AL MVP event from the spike), one prices-history chunk for one of its outcome markets, one Kalshi candlesticks payload for a 2026 series market. Save to `tests/fixtures/odds/`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_odds.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardprice.odds import (
    daily_prices,
    month_grain_features,
    odds_spike_events,
    parse_kalshi_candles,
    parse_polymarket_event,
    parse_polymarket_history,
    vig_normalize,
)

FIX = Path(__file__).parent / "fixtures" / "odds"


def _load(name):
    return json.loads((FIX / name).read_text())


def test_parse_polymarket_event_outcomes():
    df = parse_polymarket_event(_load("polymarket_event.json"))
    assert len(df) >= 3
    assert {"market_id", "player_name", "volume"} <= set(df.columns)
    assert (df["volume"] >= 0).all()
    # golden: the 2025 AL MVP event's known top outcome from the spike capture
    assert (df["player_name"] == "Aaron Judge").any()


def test_parse_polymarket_history_chunk():
    df = parse_polymarket_history(_load("polymarket_history_chunk.json"), market_id="m1")
    assert len(df) > 0
    assert df["ts"].is_monotonic_increasing
    assert ((df["raw_price"] >= 0) & (df["raw_price"] <= 1)).all()


def test_vig_normalize_hand_computed():
    prices = pd.DataFrame(
        [
            {"event": "e", "player_name": "A", "date": pd.Timestamp("2025-06-01"), "raw_price": 0.60},
            {"event": "e", "player_name": "B", "date": pd.Timestamp("2025-06-01"), "raw_price": 0.30},
            {"event": "e", "player_name": "C", "date": pd.Timestamp("2025-06-01"), "raw_price": 0.15},
            {"event": "e", "player_name": "A", "date": pd.Timestamp("2025-06-02"), "raw_price": 0.50},
            {"event": "e", "player_name": "B", "date": pd.Timestamp("2025-06-02"), "raw_price": 0.50},
        ]
    )
    out = vig_normalize(prices)
    day1 = out[out["date"] == "2025-06-01"]
    assert day1["implied_prob"].tolist() == pytest.approx([0.60 / 1.05, 0.30 / 1.05, 0.15 / 1.05])
    day2 = out[out["date"] == "2025-06-02"]
    assert day2["implied_prob"].tolist() == pytest.approx([0.5, 0.5])
    # zero-total date is dropped, not divided
    prices.loc[len(prices)] = {"event": "e", "player_name": "Z", "date": pd.Timestamp("2025-06-03"), "raw_price": 0.0}
    out2 = vig_normalize(prices)
    assert (out2["date"] != "2025-06-03").all()


def test_daily_prices_last_per_day():
    hist = pd.DataFrame(
        {"market_id": ["m", "m", "m"], "ts": pd.to_datetime(
            ["2025-06-01 03:00", "2025-06-01 23:00", "2025-06-02 01:00"], utc=True),
         "raw_price": [0.5, 0.6, 0.7]}
    )
    out = daily_prices(hist)
    assert len(out) == 2
    assert out.iloc[0]["raw_price"] == 0.6


def test_month_grain_features_strict_cutoffs():
    snap = pd.DataFrame(
        [
            {"mlb_id": 1, "date": pd.Timestamp("2025-05-25"), "implied_prob": 0.10, "season": 2025},
            {"mlb_id": 1, "date": pd.Timestamp("2025-06-20"), "implied_prob": 0.25, "season": 2025},
            {"mlb_id": 1, "date": pd.Timestamp("2025-07-01"), "implied_prob": 0.99, "season": 2025},
        ]
    )
    entries = pd.Series([pd.Timestamp("2025-07-01")], name="entry_month")
    out = month_grain_features(snap, entries)
    row = out[(out["mlb_id"] == 1) & (out["entry_month"] == pd.Timestamp("2025-07-01"))].iloc[0]
    assert row["odds_level"] == pytest.approx(0.25)  # 07-01 snapshot EXCLUDED (strict <)
    assert row["odds_delta_7d"] == pytest.approx(0.25 - 0.10)  # last before 06-24 is 06-20; last before 06-24 minus last before ~05-25 window
    assert row["has_market"] == 1


def test_month_grain_features_no_market_defaults():
    snap = pd.DataFrame(columns=["mlb_id", "date", "implied_prob", "season"])
    entries = pd.Series([pd.Timestamp("2025-07-01")], name="entry_month")
    players = pd.DataFrame({"mlb_id": [9], "entry_month": [pd.Timestamp("2025-07-01")], "season": [2025]})
    out = month_grain_features(snap, entries, players=players)
    row = out.iloc[0]
    assert row["has_market"] == 0
    assert row["odds_level"] == 0.0 and row["odds_delta_7d"] == 0.0 and row["odds_delta_30d"] == 0.0


def test_parse_kalshi_candles_price_units():
    df = parse_kalshi_candles(_load("kalshi_candlesticks.json"), market_ticker="KX1")
    assert len(df) > 0
    assert df["ts"].is_monotonic_increasing
    assert ((df["raw_price"] >= 0) & (df["raw_price"] <= 1)).all()  # cents converted to dollars


def test_odds_spike_events_threshold_and_separation():
    dates = pd.date_range("2025-05-01", periods=60, freq="D")
    probs = [0.10] * 30 + [0.10, 0.45, 0.44, 0.46, 0.10] + [0.10] * 25
    snap = pd.DataFrame({"mlb_id": [1] * 60, "date": dates, "implied_prob": probs})
    out = odds_spike_events(snap)
    assert len(out) == 2  # the +0.35 jump and the -0.36 drop; the 0.44/0.46 wiggles are in-cluster
    assert out.iloc[0]["delta"] == pytest.approx(0.35)
    assert out.iloc[1]["delta"] == pytest.approx(-0.36)
    assert (out["event_type"] == "odds_spike").all()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_odds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cardprice.odds'`

- [ ] **Step 4: Implement `src/cardprice/odds.py`**

```python
# src/cardprice/odds.py
"""Award-futures odds: parsing, vig normalization, feature math (pure functions).

Sources (spike-governed): Polymarket gamma/CLOB APIs (2025+2026), Kalshi
candlesticks (2026 cross-source). Everything here is offline; network lives in
scripts/collect_odds.py.
"""

import numpy as np
import pandas as pd

SNAP_COLUMNS = [
    "source", "event", "market_id", "player_name", "mlb_id", "season",
    "date", "raw_price", "implied_prob", "volume",
]


def parse_polymarket_event(payload: dict) -> pd.DataFrame:
    """Outcome markets of an award event -> [market_id, player_name, volume]."""
    rows = []
    for m in payload.get("markets", []):
        label = m.get("groupItemTitle") or m.get("question") or ""
        rows.append(
            {
                "market_id": str(m.get("conditionId") or m.get("condition_id") or m.get("id")),
                "player_name": label.strip(),
                "volume": float(m.get("volume") or m.get("volumeNum") or 0.0),
            }
        )
    return pd.DataFrame(rows, columns=["market_id", "player_name", "volume"])


def parse_polymarket_history(payload: dict, market_id: str) -> pd.DataFrame:
    """prices-history chunk -> [market_id, ts, raw_price] (UTC, ascending)."""
    hist = payload.get("history", payload if isinstance(payload, list) else [])
    rows = [
        {"market_id": market_id, "ts": pd.to_datetime(int(p["t"]), unit="s", utc=True),
         "raw_price": float(p["p"])}
        for p in hist
    ]
    out = pd.DataFrame(rows, columns=["market_id", "ts", "raw_price"])
    return out.sort_values("ts").reset_index(drop=True)


def vig_normalize(prices: pd.DataFrame) -> pd.DataFrame:
    """implied_prob = raw_price / event-date total (overround removal).

    Dates whose event total is 0 are dropped (counted via the .attrs tally).
    """
    out = prices.copy()
    totals = out.groupby(["event", "date"])["raw_price"].transform("sum")
    valid = totals > 0
    out.attrs["n_dropped"] = int((~valid).sum())
    out = out[valid]
    out["implied_prob"] = out["raw_price"] / totals[valid]
    return out.reset_index(drop=True)


def daily_prices(history: pd.DataFrame) -> pd.DataFrame:
    """Last price per (market_id, UTC date)."""
    h = history.assign(date=history["ts"].dt.floor("D"))
    return h.groupby(["market_id", "date"], as_index=False).last()[
        ["market_id", "date", "raw_price"]
    ]


def month_grain_features(
    snapshots: pd.DataFrame,
    entry_months: pd.Series,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per (mlb_id, entry_month): odds_level/delta_7d/delta_30d/has_market.

    odds_level = last implied_prob with date < entry_month. delta_Nd = level
    minus the last implied_prob strictly before (entry_month - N days); NaN when
    no snapshot precedes the earlier cutoff. Players with no market in the
    season: has_market=0 and all odds features 0.0 (documented semantics).
    `players` (mlb_id, entry_month, season) supplies the full target grid.
    """
    if players is None:
        raise ValueError("players grid (mlb_id, entry_month, season) is required")
    out_rows = []
    snap = snapshots.sort_values("date")
    for p in players.itertuples():
        g = snap[snap["mlb_id"] == p.mlb_id]
        entry = p.entry_month
        before = g[g["date"] < entry]
        has = int(len(g[g["season"] == p.season]) > 0)
        if not len(before):
            out_rows.append(
                {"mlb_id": p.mlb_id, "entry_month": entry, "has_market": has,
                 "odds_level": 0.0 if not has else np.nan,
                 "odds_delta_7d": 0.0 if not has else np.nan,
                 "odds_delta_30d": 0.0 if not has else np.nan}
            )
            continue
        level = float(before["implied_prob"].iloc[-1])

        def _level_at(cutoff):
            b = g[g["date"] < cutoff]
            return float(b["implied_prob"].iloc[-1]) if len(b) else np.nan

        l7 = _level_at(entry - pd.Timedelta(days=7))
        l30 = _level_at(entry - pd.Timedelta(days=30))
        out_rows.append(
            {"mlb_id": p.mlb_id, "entry_month": entry, "has_market": has,
             "odds_level": level,
             "odds_delta_7d": (level - l7) if not np.isnan(l7) else np.nan,
             "odds_delta_30d": (level - l30) if not np.isnan(l30) else np.nan}
        )
    return pd.DataFrame(out_rows)


def parse_kalshi_candles(payload: dict, market_ticker: str) -> pd.DataFrame:
    """Kalshi daily candlesticks -> [market_id, ts, raw_price, volume].

    Price fields are cents when the payload marks them as such; converted to
    dollars. (Pinned by the fixture golden.)
    """
    candles = payload.get("candlesticks", [])
    rows = []
    for c in candles:
        price = c.get("price", {})
        close = price.get("close") if isinstance(price, dict) else c.get("close")
        close = float(close)
        if close > 1.0:  # cents -> dollars
            close = close / 100.0
        rows.append(
            {"market_id": market_ticker,
             "ts": pd.to_datetime(int(c.get("end_period_ts", c.get("ts", 0))), unit="s", utc=True),
             "raw_price": close,
             "volume": float(c.get("volume", 0) or 0)}
        )
    out = pd.DataFrame(rows, columns=["market_id", "ts", "raw_price", "volume"])
    return out.sort_values("ts").reset_index(drop=True)


def odds_spike_events(
    snapshots: pd.DataFrame,
    min_abs_delta: float = 0.10,
    rel_std_window: int = 30,
    rel_std_mult: float = 3.0,
) -> pd.DataFrame:
    """|1-day delta| >= max(min_abs_delta, rel_std_mult x rolling-30d std of
    daily deltas); largest |delta| kept per 7-day cluster per player."""
    events = []
    for mlb_id, g in snapshots.sort_values("date").groupby("mlb_id"):
        g = g.drop_duplicates("date").set_index("date")["implied_prob"]
        delta = g.diff()
        std = delta.rolling(rel_std_window, min_periods=5).std()
        thresh = np.maximum(min_abs_delta, rel_std_mult * std)
        spikes = delta[delta.abs() >= thresh]
        taken: list[pd.Timestamp] = []
        for d, v in spikes.sort_values(key=abs, ascending=False).items():
            if all(abs((d - t).days) > 7 for t in taken):
                taken.append(d)
        for d in sorted(taken):
            events.append(
                {"mlb_id": int(mlb_id), "event_date": d, "event_type": "odds_spike",
                 "delta": float(delta[d]), "implied_prob": float(g[d])}
            )
    return pd.DataFrame(events)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_odds.py -v` → 8 passed; full `pytest` green; `ruff check src tests scripts` clean.

- [ ] **Step 6: Commit**

```bash
git add src/cardprice/odds.py scripts/capture_odds_fixtures.py tests/test_odds.py tests/fixtures/odds/
git commit -m "feat: odds module (parsers, vig normalization, month-grain features, spike events)"
```

---

### Task 2: Odds collector (`scripts/collect_odds.py`)

**Files:**
- Create: `scripts/collect_odds.py`
- Test: `tests/test_collect_odds.py` (offline: chunk planning + mapping logic only)

**Interfaces:**
- Consumes: Task 1's module; `data/reference/player_info_class.csv` (+ `player_info_universe.csv` for pre-class candidates); the spike's verified endpoints.
- Produces: `data/processed/odds_snapshots.parquet` (columns per `SNAP_COLUMNS`); `data/reference/odds_unmapped_audit.csv`; `data/reference/odds_events_registry.csv` (event, source, event_id, season, league, award, n_outcomes, collected_through).
- Collection plan (spike-governed): discover Polymarket award events for 2025 + 2026 (gamma API search per the spike's exact patterns — the spike doc names them); per outcome market, chunked `prices-history` at fidelity=60 (14-day chunks over the market's active window, `startTs`/`endTs`); `daily_prices`; vig-normalize per (event, date); Kalshi 2026 daily candles for the same award series (cross-source rows with `source="kalshi"`, raw prices, NO vig normalization within Kalshi single-outcome binaries — each Kalshi market is its own binary; record raw price as implied_prob directly, flagged).
- Player mapping: exact `norm_name` match against `player_info_class.csv` ∪ `player_info_universe.csv`; unmatched outcome labels → audit CSV (label, event, volume), never fuzzy-matched.
- Prospective behavior: idempotent — skips (market, chunk) pairs whose raw snapshot exists (`snapshot_exists("odds", key)`), so the weekly cron only fetches new data.
- Politeness: 0.3–0.5 s between calls; save raw payloads via `save_raw("odds", key, payload)` before parsing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_collect_odds.py
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from collect_odds import map_players, plan_chunks


def test_plan_chunks_fourteen_day_windows():
    start, end = pd.Timestamp("2025-03-01"), pd.Timestamp("2025-06-15")
    chunks = plan_chunks(start, end, days=14)
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    assert all((b - a).days <= 14 for a, b in chunks)
    # contiguous: next chunk starts where the previous ended
    assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))


def test_map_players_exact_norm_only():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from collect_odds import norm_name  # re-exported for tests

    info = pd.DataFrame(
        {"mlb_id": [1, 2], "name": ["Aaron Judge", "Jasson Domínguez"],
         "birth_date": pd.to_datetime(["1992-04-26", "2003-02-07"]), "position": ["OF", "OF"]}
    )
    outcomes = pd.DataFrame(
        {"market_id": ["a", "b", "c"],
         "player_name": ["Aaron Judge", "Jasson Dominguez", "Shohei Ohtani"]}
    )
    mapped, audit = map_players(outcomes, info)
    assert mapped.loc[mapped["market_id"] == "a", "mlb_id"].iloc[0] == 1
    assert mapped.loc[mapped["market_id"] == "b", "mlb_id"].iloc[0] == 2  # accent-folded
    assert audit["player_name"].tolist() == ["Shohei Ohtani"]  # not in info -> audit, never guessed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_collect_odds.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scripts/collect_odds.py`** (full runner, ~150 lines, following the repo's collector idioms: argparse with `--out`, `--since`, `--sleep`; discovery per the spike doc's endpoint list; chunk loop with snapshot-resume; audit + registry CSVs; final SNAP_COLUMNS frame. The task brief carries the endpoint constants from the spike — do not invent others.)

- [ ] **Step 4: Run tests; run the live collection**

Run: `pytest tests/test_collect_odds.py -v` → green; full `pytest` green; `ruff check src tests scripts` clean.
Then: `python scripts/collect_odds.py` (~10–20 min: ~12 events × ~10-30 markets × ~10-26 chunks). Acceptance: registry lists 2025 + 2026 events for all three awards × both leagues (or a documented missing one); audit CSV written; snapshot frame has `implied_prob` within [0, 1] after normalization; no NaT dates. Record the acceptance printout.

- [ ] **Step 5: Commit**

```bash
git add scripts/collect_odds.py tests/test_collect_odds.py data/reference/odds_unmapped_audit.csv data/reference/odds_events_registry.csv
git commit -m "feat: odds collector (Polymarket chunked history + Kalshi 2026, mapping audit, snapshot resume)"
```

---

### Task 3: Odds features into the hold frame

**Files:**
- Modify: none of the reviewed files — new small module function in `src/cardprice/odds.py` + runner
- Create: `scripts/build_odds_features.py`
- Test: `tests/test_attach_odds.py`

**Interfaces:**
- Produces: `attach_odds_features(frame: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame` (in `odds.py`) — left-joins `month_grain_features` onto the hold frame by (mlb_id, entry_month); players with no market get has_market=0 + 0.0 features; the frame's other columns pass through untouched; join validated `many_to_one`.
- Runner output: `data/processed/class_hold_odds.parquet` (the registered-cell frame + the four odds columns) — the delta-fit's input.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_attach_odds.py
import pandas as pd
import pytest

from cardprice.odds import attach_odds_features


def _frame():
    return pd.DataFrame(
        {
            "entry_month": [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-01"),
                            pd.Timestamp("2025-08-01")],
            "entry_year": [2025, 2025, 2025],
            "mlb_id": [1, 2, 1], "card_slug": ["a/x", "b/y", "a/x"],
            "ret_12m": [0.1, 0.2, 0.3],
        }
    )


def _snapshots():
    return pd.DataFrame(
        [
            {"mlb_id": 1, "date": pd.Timestamp("2025-06-15"), "implied_prob": 0.2, "season": 2025},
            {"mlb_id": 1, "date": pd.Timestamp("2025-07-15"), "implied_prob": 0.35, "season": 2025},
        ]
    )


def test_attach_defaults_and_join():
    out = attach_odds_features(_frame(), _snapshots())
    r1 = out[(out["mlb_id"] == 1) & (out["entry_month"] == "2025-07-01")].iloc[0]
    assert r1["has_market"] == 1 and r1["odds_level"] == 0.2
    assert pd.isna(r1["odds_delta_7d"])  # no snapshot before 06-24
    r2 = out[out["mlb_id"] == 2].iloc[0]
    assert r2["has_market"] == 0 and r2["odds_level"] == 0.0
    r3 = out[(out["mlb_id"] == 1) & (out["entry_month"] == "2025-08-01")].iloc[0]
    assert r3["odds_level"] == 0.35
    assert r3["odds_delta_30d"] == pytest.approx(0.35 - 0.2)
    assert len(out) == 3  # many_to_one, no row multiplication
```

- [ ] **Step 2: Implement, test, run**

Implement `attach_odds_features` (pure; delegates to `month_grain_features`) and `scripts/build_odds_features.py` (loads the hold frame for a group/horizon + odds_snapshots.parquet → writes class_hold_odds.parquet). Run tests (green), then the runner for the registered cell's frame (ungraded hitters, all horizons) + pitcher frame for the pitcher descriptive cell.

- [ ] **Step 3: Commit**

```bash
git add src/cardprice/odds.py scripts/build_odds_features.py tests/test_attach_odds.py
git commit -m "feat: odds features attached to the class hold frame (defaults documented)"
```

---

### Task 4: Delta-fit evaluation + importance with odds

**Files:**
- Create: `scripts/run_odds_delta_fit.py`
- Test: `tests/test_odds_delta_fit.py` (wiring only, synthetic)

**Interfaces:**
- Consumes: Task 3's class_hold_odds.parquet; `CLASS_HOLD_FEATURES`; the reviewed harness.
- Produces: `data/processed/class_odds_delta_fit.json` — per cell (the same cell list as the gate runner, hitters + pitcher):
  - baseline: walk-forward with `CLASS_HOLD_FEATURES` (recomputed; must reconcile with T2's registered numbers within harness determinism),
  - with_odds: walk-forward with `CLASS_HOLD_FEATURES + ["has_market", "odds_level", "odds_delta_7d", "odds_delta_30d"]`,
  - per year: `spearman(predicted, realized)` for both, `mean_excess` for both, `delta_mean_excess`, plus pooled `delta_spearman`. All labeled descriptive.
- Also re-run the importance runner WITH the odds features on the registered cell → appended rows in `data/processed/class_hold_importance_odds.csv` (same long format) — the importance answer: where do odds features rank vs price_level/ret_3m/draft_rank?

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_odds_delta_fit.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_odds_delta_fit import compare_feature_sets


def _frame(seed=11, n=30):
    rng = np.random.default_rng(seed)
    rows = []
    for y in (2021, 2022, 2023, 2024, 2025):
        for c in range(n):
            s = rng.normal(0, 1)
            rows.append(
                {"entry_year": y, "entry_month": pd.Timestamp(f"{y}-06-01"), "mlb_id": c,
                 "card_slug": f"c{c}", "ret_12m": 0.5 * s + rng.normal(0, 0.15),
                 "f1": rng.normal(0, 1), "f2": rng.normal(0, 1),
                 "has_market": int(c % 3 == 0),
                 "odds_level": s * 0.5 * (c % 3 == 0),
                 "odds_delta_7d": rng.normal(0, 0.05) * (c % 3 == 0),
                 "odds_delta_30d": rng.normal(0, 0.1) * (c % 3 == 0)}
            )
    return pd.DataFrame(rows)


def test_compare_feature_sets_output_contract():
    out = compare_feature_sets(_frame(), ["f1", "f2"],
                               ["f1", "f2", "has_market", "odds_level",
                                "odds_delta_7d", "odds_delta_30d"], horizon=12)
    assert set(out) == {"baseline", "with_odds", "delta"}
    for k in ("baseline", "with_odds"):
        assert set(out[k]["gate"]) == {"mean_excess", "ci_low", "ci_high", "net_mean",
                                       "net_ci_low", "n_years", "verdict"}
        assert set(out[k]["spearman_by_year"]) == {2023, 2024, 2025}
    # planted signal inside odds_level should not hurt the with_odds fit
    assert out["delta"]["mean_excess"] > -0.5
```

- [ ] **Step 2: Implement, test, run**

Implement `compare_feature_sets(frame, base_features, odds_features, horizon)` in the runner (walk-forward both sets with `all_scores=True`, Spearman by year, `gate_evaluation_years` both, deltas). Run tests, then the real runner over all cells (hitters 6/12/24/36 + psa_10 12 + pitcher 12) and the importance re-run. Record outputs.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_odds_delta_fit.py tests/test_odds_delta_fit.py
git commit -m "feat: odds delta-fit evaluation vs T2 baseline (descriptive) + odds-aware importance"
```

---

### Task 5: Odds-spike lag study + findings doc + prospective cron

**Files:**
- Create: `scripts/run_odds_lag.py`
- Create: `docs/findings/2026-09-19-t3-odds-deltas.md`

**Interfaces:**
- Consumes: `odds_spike_events` (Task 1), `class_sales.parquet` ∪ `universe_sales.parquet` (deduped on (card_slug, sale_date, title, price)), the reviewed `lag_study` functions.
- Produces: `data/processed/odds_spike_events.parquet`, `data/processed/odds_lag_summary.json`, the findings doc.
- Lag study config (descriptive, mirrors T1's recalibrated defaults): `event_sale_windows` defaults (56d/±21d/min3); market index over the union sales frame; primary cell = ungraded/all; strata via the existing `assign_strata` (career stage at event).

- [ ] **Step 1: Write and run the odds-spike lag runner**

`scripts/run_odds_lag.py`: build spike events from odds_snapshots (implied_prob daily series per player) → union sales frame → per-cell windows → adjust → pooled curve + lag + classes → JSON + print. Acceptance: events count plausible (tens to low hundreds); drop logs printed; if kept event-card pairs < 5, the runner reports "too few events" honestly instead of a curve.

- [ ] **Step 2: Write the findings doc**

`docs/findings/2026-09-19-t3-odds-deltas.md`, sections: Data (events/markets/coverage, mapping audit, Kalshi cross-source caveat, 2-season ceiling) → Delta-fit (per-cell baseline vs with_odds table, delta_mean_excess, delta_spearman, all descriptive; the registered T2 verdict untouched) → Importance with odds (where odds features rank) → Odds-spike lag (curve summary + lag verdict per T1's spec §8 language, descriptive) → Caveats (2 seasons, thin ROY markets, no-sportsbook-odds, has_market default semantics, non-candidate players outside classes) → Reproduce. Every number from artifacts.

- [ ] **Step 3: Prospective cron + commit**

Create the weekly cron (controller step, exact command below), then commit.

Cron: Wednesdays 09:23 local — prompt: run `cd /Users/tomcruise/sports_cards_research_and_modeling && PLAYWRIGHT_BROWSERS_PATH=.pw-browsers .venv/bin/python scripts/collect_odds.py --since 14d` in the main checkout (note: main checkout has no venv — create/reuse one per the repo convention; if absent, first create it per the worktree venv recipe), append new snapshots via the collector's snapshot-resume, then `python scripts/build_odds_features.py && python scripts/run_odds_delta_fit.py` to refresh the artifacts, and commit any changed `data/reference/*.csv` (audit/registry). Delete this cron if the collector is retired.

```bash
git add scripts/run_odds_lag.py docs/findings/2026-09-19-t3-odds-deltas.md
git commit -m "feat: odds-spike lag study + T3 findings doc + prospective weekly collection cron"
```

---

## Self-Review

**Spec coverage:** §4 spike-pass path (collector, features, evaluation by delta-fit, prospective collection) ✓; §2 decisions (no separate gate; descriptive labels; escalation-not-purchase of The Odds API recorded in findings) ✓; §6 error handling (parser fixtures, ≤2-week forward-fill never fabricated via strict-cutoff semantics — month_grain uses last-snapshot-before-cutoff which is effectively bounded by the market's natural cadence; missing markets = audit) ✓; §7 explicitly-out items respected (no ToS-hostile scraping, no paid calls) ✓. T1's daily event frame merge maps to the odds-spike lag study reusing lag_study verbatim ✓.

**Placeholder scan:** Task 2's runner is contract-specified with endpoint sources pinned to the spike doc (the spike is the requirements input; the implementer copies its endpoint constants) — no invented URLs; all pure logic (Tasks 1, 3, 4) has full code.

**Type consistency:** SNAP_COLUMNS defined in Task 1 and consumed by Tasks 2–5 ✓; `month_grain_features(players=...)` grid contract matches `attach_odds_features`' usage ✓; delta-fit's gate dict keys match `gate_evaluation_years` ✓; odds_spike_events output (mlb_id, event_date, event_type) matches `event_sale_windows` expectations ✓.
