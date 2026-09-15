# Panel Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Join the stats pipeline (Plan 1) and price pipeline (Plan 2) into the analysis panel — card-grade × month (PSA-10 history, 2022–2026) and card-grade × week (recent window + go-forward) — with strictly lagged predictors, a market-median-adjusted outcome, and a verified golden row.

**Architecture:** Data refresh first (parser hardening over immutable snapshots + stats collection for all seed players), then pure feature builders: stats-at-dates → monthly panel → weekly panel → export + data dictionary. Everything tested offline on synthetic frames and the committed golden fixtures.

**Tech Stack:** Python 3.11+, existing `cardprice` package. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-14-mlb-card-price-panel-design.md` (§6 modeling spec — this plan builds the dataset it consumes)
**Predecessors:** Plan 1 (`stats_api`, `season_stats`, `collect_stats`, `storage`), Plan 2 (`scp_parse`, `collect_prices`, `weekly`, `validate`, `liquidity`).

## Facts established by Plans 1-2 (code below is built on them)

- `data/processed/scp_sales.parquet`: 1825 rows; columns `sale_date, title, price, list_price, best_offer, grade, bucket, player_name, mlb_id, rookie_year, set_slug, card_slug`. ~5% of titles carry grader mentions the Task 2 regex misses (hyphenated `SGC-10`, `PSA GEM MT 10`, `MINT 9`, reversed `Gem Mint 10 PSA`) — Task 1 fixes retroactively over immutable raw snapshots.
- `data/processed/scp_chart_monthly.parquet`: 2377 rows; columns `grade, date, price, player_name, mlb_id, rookie_year, set_slug, card_slug`. Grades are generic buckets + `psa_10` (+ uncalibrated `key:*`); **zero `psa_9` rows — monthly history is PSA-10-only**. 223 rows have `price == 0` (phantom) — must be filtered.
- `data/processed/game_logs.parquet`: currently only the 4 Plan-1 reference players — must be regenerated for all seed players.
- `data/reference/cards_seed.csv`: 16 rows, all with `mlb_id`; 13 with `scp_url`.
- Plan 1 `season_stats.hitting_to_date(game_log, through) -> dict` and `pitching_to_date(...)` are golden-verified.
- PSA-10 monthly series for each card starts at card release (mid rookie year). Model covers post-MLB-debut months only (no stats before debut).

## Global Constraints

- Python >= 3.11; free data sources only; raw snapshots immutable (re-parsing reads, never rewrites).
- **No look-ahead, ever:** predictors for period `t` may only use information available before the first day of `t`. Stats for period `t` are cumulative through the LAST day of `t-1`. The panel's most important test asserts this.
- Stats for a player-month/week with no MLB games played yet (pre-debut) → row dropped, not zero-filled.
- Tests offline by default; `live` marker for network.
- Scope cut (acknowledged): Savant expected-stats (xwOBA/barrel%) are NOT in this plan — computing them at arbitrary dates requires pitch-level pulls (~40k rows/player-season) for marginal gain over game-log OPS/SLG/K-BB%; if Plan 4's model shows weak predictiveness, adding Savant features is the first enhancement. Pop counts: attempted via GemRate (Task 5) with documented skip as fallback.

---

### Task 1: Parser hardening + stats expansion + data refresh

**Files:**
- Modify: `src/cardprice/scp_parse.py` (GRADE_RE + reversed-order fallback)
- Modify: `src/cardprice/scp_parse.py` `calibrate_chart_grades` (filter `price <= 0`)
- Modify: `src/cardprice/stats_api.py` (add `fetch_player_info`)
- Create: `scripts/make_players_from_cards.py`
- Create: `scripts/reparse_snapshots.py`
- Test: `tests/test_grade_regex.py`, `tests/test_player_info.py`

**Interfaces:**
- Consumes: everything from Plans 1-2.
- Produces (Tasks 2-6 consume):
  - `fetch_player_info(mlb_ids: list[int]) -> pd.DataFrame` — columns `mlb_id, name, birth_date (datetime64), position (str)` via `GET /api/v1/people?personIds={comma-joined}&hydrate=` (single call, no key).
  - `data/reference/players.csv` — regenerated from cards_seed.csv: columns `mlb_id,name,role` (all 16 seed players).
  - Refreshed `data/processed/scp_sales.parquet`, `scp_chart_monthly.parquet` (hardened parser + `price > 0` filter, `key:*` grades dropped from chart), rebuilt from raw snapshots (no network).
  - Refreshed `data/processed/game_logs.parquet` for all 16 players, 2022–2026.
  - `data/reference/player_info.csv` — from `fetch_player_info`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grade_regex.py
import pytest

from cardprice.scp_parse import parse_grade_from_title


@pytest.mark.parametrize(
    "title,expected",
    [
        # regression: formats that already worked
        ("2023 Topps Chrome Gunnar Henderson RC PSA 10 GEM MT", "psa_10"),
        ("GUNNAR HENDERSON 2023 TOPPS CHROME PSA 9", "psa_9"),
        ("2023 Topps Chrome Henderson BGS 9.5 GEM MINT", "bgs_9.5"),
        ("2023 Topps Chrome Gunnar Henderson #2", None),
        # new: hyphenated
        ("2022 Topps Chrome Bobby Witt Jr RC SGC-10", "sgc_10"),
        ("Henderson PSA-9", "psa_9"),
        # new: word-form between grader and digit
        ("2023 Topps Chrome Henderson PSA GEM MT 10", "psa_10"),
        ("Witt RC PSA MINT 9", "psa_9"),
        ("Henderson SGC PERFECT 10", "sgc_10"),
        # new: reversed order
        ("2023 Topps Chrome Henderson Gem Mint 10 PSA", "psa_10"),
    ],
)
def test_grade_variants(title, expected):
    assert parse_grade_from_title(title) == expected
```

```python
# tests/test_player_info.py
import pandas as pd
import pytest

from cardprice import stats_api

PEOPLE_RESPONSE = {
    "people": [
        {
            "id": 683002,
            "fullName": "Gunnar Henderson",
            "birthDate": "2001-06-29",
            "primaryPosition": {"abbreviation": "SS"},
        },
        {
            "id": 592450,
            "fullName": "Aaron Judge",
            "birthDate": "1992-04-26",
            "primaryPosition": {"abbreviation": "RF"},
        },
    ]
}


def test_fetch_player_info_parses(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return PEOPLE_RESPONSE

    calls = {}

    def fake_get(url, params=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        return FakeResp()

    monkeypatch.setattr(stats_api.requests, "get", fake_get)
    df = stats_api.fetch_player_info([683002, 592450])
    assert calls["url"] == "https://statsapi.mlb.com/api/v1/people"
    assert calls["params"]["personIds"] == "683002,592450"
    assert list(df.columns) == ["mlb_id", "name", "birth_date", "position"]
    row = df[df["mlb_id"] == 683002].iloc[0]
    assert row["name"] == "Gunnar Henderson"
    assert row["birth_date"] == pd.Timestamp("2001-06-29")
    assert row["position"] == "SS"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_grade_regex.py tests/test_player_info.py -v`
Expected: FAIL — new variants return None; `fetch_player_info` missing.

- [ ] **Step 3: Implement**

In `src/cardprice/scp_parse.py`, replace the grade-regex block:

```python
GRADE_RES = [
    # grader first: "PSA 10", "SGC-10", "PSA GEM MT 10", "PSA MINT 9", "SGC PERFECT 10"
    re.compile(
        r"\b(PSA|BGS|SGC|CGC)\s*-?\s*"
        r"(?:GEM\s*(?:MT|MINT)\s*|MINT\s+|PERFECT\s+|PRISTINE\s+)?"
        r"(10|9\.5|9|8\.5|8)\b",
        re.IGNORECASE,
    ),
    # grade first: "Gem Mint 10 PSA"
    re.compile(r"\bGEM\s*(?:MT|MINT)\s*(10)\s+(PSA|BGS|SGC|CGC)\b", re.IGNORECASE),
]


def parse_grade_from_title(title: str) -> str | None:
    m = GRADE_RES[0].search(title)
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"
    m = GRADE_RES[1].search(title)
    if m:
        return f"{m.group(2).lower()}_{m.group(1)}"
    return None
```

(Delete the old `GRADE_RE` and old `parse_grade_from_title`; update any references. Keep everything else in the file untouched.)

In `calibrate_chart_grades`, filter phantom points — inside the series loop, skip non-positive prices:

```python
        for ts, price in series:
            if price <= 0:
                continue
            rows.append({"grade": grade, "date": ts, "price": price})
```

In `src/cardprice/stats_api.py`, append:

```python
def fetch_player_info(mlb_ids: list[int]) -> pd.DataFrame:
    resp = requests.get(
        f"{BASE}/people",
        params={"personIds": ",".join(str(i) for i in mlb_ids), "hydrate": ""},
        timeout=30,
    )
    resp.raise_for_status()
    rows = [
        {
            "mlb_id": p["id"],
            "name": p["fullName"],
            "birth_date": p.get("birthDate"),
            "position": p.get("primaryPosition", {}).get("abbreviation"),
        }
        for p in resp.json().get("people", [])
    ]
    df = pd.DataFrame(rows)
    df["birth_date"] = pd.to_datetime(df["birth_date"])
    return df
```

`scripts/make_players_from_cards.py`:

```python
"""Regenerate data/reference/players.csv from cards_seed.csv."""

import pandas as pd

if __name__ == "__main__":
    cards = pd.read_csv("data/reference/cards_seed.csv")
    players = cards[["mlb_id", "player_name", "role"]].rename(columns={"player_name": "name"})
    players.to_csv("data/reference/players.csv", index=False)
    print(f"wrote {len(players)} players")
```

`scripts/reparse_snapshots.py`:

```python
"""Rebuild sales + chart parquets from the latest raw SCP snapshots (no network).

Reads data/raw/prices_scp/<set>/<card>/<latest>.json, re-parses with the current
parser, joins card metadata from cards_seed.csv, and rewrites the processed parquets.
"""

import json
from pathlib import Path

import pandas as pd

from cardprice.scp_parse import calibrate_chart_grades, parse_sales_tables

RAW = Path("data/raw/prices_scp")
META_COLS = ["player_name", "mlb_id", "rookie_year", "set_slug", "card_slug"]

if __name__ == "__main__":
    cards = pd.read_csv("data/reference/cards_seed.csv")
    meta_by_slug = {
        "/".join(u.rstrip("/").split("/")[-2:]): row
        for u, row in zip(cards["scp_url"], cards.itertuples())
        if isinstance(u, str) and u
    }
    sales_frames, chart_frames = [], []
    for day_file in sorted(RAW.glob("*/*/*.json")):
        slug = "/".join(day_file.parts[-3:-1])
        if day_file.name != sorted(day_file.parent.glob("*.json"))[-1].name:
            continue  # only latest snapshot per card
        payload = json.loads(day_file.read_text())
        meta_row = meta_by_slug.get(slug)
        if meta_row is None:
            continue
        meta = {
            "player_name": meta_row.player_name,
            "mlb_id": int(meta_row.mlb_id),
            "rookie_year": int(meta_row.rookie_year),
            "set_slug": meta_row.set_slug,
            "card_slug": slug,
        }
        sales = parse_sales_tables(payload["html"])
        if len(sales):
            sales_frames.append(sales.assign(**meta))
        chart = calibrate_chart_grades(payload["html"])
        chart = chart[~chart["grade"].str.startswith("key:")]
        if len(chart):
            chart_frames.append(chart.assign(**meta))
    sales = pd.concat(sales_frames, ignore_index=True)
    chart = pd.concat(chart_frames, ignore_index=True)
    sales.to_parquet("data/processed/scp_sales.parquet", index=False)
    chart.to_parquet("data/processed/scp_chart_monthly.parquet", index=False)
    print(f"reparsed: {len(sales)} sales, {len(chart)} chart points")
```

NOTE: `calibrate_chart_grades` output column is `date`; the collector's chart parquet uses `date` too — verify column alignment when joining (both are `date`). If the sales parquet downstream expects `sale_date`, keep it.

- [ ] **Step 4: Run tests, then the data refresh (network for stats only)**

Run: `python -m pytest tests/test_grade_regex.py tests/test_player_info.py -v` — all PASS. Full suite: `python -m pytest -v` — all PASS (the Task 2 fixture tests must stay green with the new regex — regression).

Then (network, ~5 min):
```bash
python scripts/make_players_from_cards.py
python -m cardprice.collect_stats --seasons 2022 2023 2024 2025 2026
python scripts/reparse_snapshots.py   # no network
python - <<'EOF'
from cardprice.stats_api import fetch_player_info
import pandas as pd
ids = pd.read_csv("data/reference/players.csv")["mlb_id"].tolist()
fetch_player_info(ids).to_csv("data/reference/player_info.csv", index=False)
print("player_info written")
EOF
```
Expected: game_logs.parquet grows to ~16 players × ~30-160 games × 5 seasons (thousands of rows); reparse prints recovered sales count — compare to the old 1825 (expect ~1870-1920 with the recovered ~5%).

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/scp_parse.py src/cardprice/stats_api.py scripts tests data/reference/players.csv data/reference/player_info.csv
git commit -m "feat: grade regex hardening, player info, snapshot reparse, stats expansion"
```

---

### Task 2: Stats-at-dates feature builder

**Files:**
- Create: `src/cardprice/stats_panel.py`
- Test: `tests/test_stats_panel.py`

**Interfaces:**
- Consumes: `game_logs.parquet` (Plan 1 shape), `hitting_to_date`/`pitching_to_date` (season_stats).
- Produces (Tasks 3-4 consume):
  - `stats_at_dates(game_log: pd.DataFrame, dates: list[pd.Timestamp], group: str) -> pd.DataFrame` — `game_log` is ONE player-season (from game_logs.parquet, Plan 1 shape with datetime `date`); `group` is `"hitting"` or `"pitching"`. Returns one row per date: columns `date` + the stat dict keys from `hitting_to_date`/`pitching_to_date`. Only dates within the season's calendar year are meaningful — caller passes season-appropriate dates.
  - `player_stats_series(game_logs: pd.DataFrame, mlb_id: int, season: int, dates: list[pd.Timestamp]) -> pd.DataFrame` — slices the full parquet to the player-season, picks hitting/pitching from the `group` column, adds `mlb_id` to output.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stats_panel.py
import json
from pathlib import Path

import pandas as pd

from cardprice.stats_api import game_log_to_frame
from cardprice.stats_panel import player_stats_series, stats_at_dates

FIXTURES = Path(__file__).parent / "fixtures"
JUDGE = json.loads((FIXTURES / "judge_2022_hitting.json").read_text())


def make_game_logs():
    df = game_log_to_frame(JUDGE["splits"], 592450, "hitting", 2022)
    other = df.iloc[:1].copy()  # one stray row from another player-season
    other["mlb_id"] = 999999
    other["season"] = 2023
    return pd.concat([df, other], ignore_index=True)


def test_stats_at_dates_midseason():
    log = game_log_to_frame(JUDGE["splits"], 592450, "hitting", 2022)
    out = stats_at_dates(log, [pd.Timestamp("2022-04-30"), pd.Timestamp("2022-10-31")], "hitting")
    assert len(out) == 2
    early, full = out.iloc[0], out.iloc[1]
    # Judge's 2022 opening month: 20 G, 6 HR through Apr 30 (corrected during execution; see NOTE below)
    assert early["games"] == 20
    assert early["home_runs"] == 6
    # full season equals golden totals
    assert full["games"] == 157
    assert full["home_runs"] == 62
    assert full["avg"] == 0.311


def test_player_stats_series_slices_correctly():
    out = player_stats_series(make_game_logs(), 592450, 2022, [pd.Timestamp("2022-10-31")])
    assert len(out) == 1
    assert out.iloc[0]["home_runs"] == 62
    assert out.iloc[0]["mlb_id"] == 592450
    # wrong player/season -> empty frame, not an error
    empty = player_stats_series(make_game_logs(), 999999, 2022, [pd.Timestamp("2022-10-31")])
    assert len(empty) == 0
```

NOTE for the implementer: Judge's through-April-2022 line is **20 G, 6 HR** (corrected during execution — verified against the committed fixture, live gameLog, boxscore 662797 where Judge DNP on Apr 30, and Retrosheet/StatMuse; an earlier draft of this plan said 21 G / 9 HR, which was wrong). The test asserts 20 G / 6 HR.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stats_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.stats_panel'`.

- [ ] **Step 3: Implement stats_panel.py**

```python
# src/cardprice/stats_panel.py
"""Season-to-date stat lines at arbitrary dates, per player-season."""

import pandas as pd

from cardprice.season_stats import hitting_to_date, pitching_to_date


def stats_at_dates(game_log: pd.DataFrame, dates: list[pd.Timestamp], group: str) -> pd.DataFrame:
    fn = hitting_to_date if group == "hitting" else pitching_to_date
    rows = [{"date": d, **fn(game_log, d.date())} for d in dates]
    return pd.DataFrame(rows)


def player_stats_series(
    game_logs: pd.DataFrame, mlb_id: int, season: int, dates: list[pd.Timestamp]
) -> pd.DataFrame:
    sub = game_logs[(game_logs["mlb_id"] == mlb_id) & (game_logs["season"] == season)]
    if not len(sub) or not len(dates):
        return pd.DataFrame()
    group = sub["group"].iloc[0]
    out = stats_at_dates(sub, dates, group)
    out["mlb_id"] = mlb_id
    return out
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_stats_panel.py -v` — 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/stats_panel.py tests/test_stats_panel.py
git commit -m "feat: season-to-date stats at arbitrary dates"
```

---

### Task 3: Monthly panel builder (PSA-10 history)

**Files:**
- Create: `src/cardprice/panel.py`
- Test: `tests/test_panel_monthly.py`

**Interfaces:**
- Consumes: chart parquet (Task 1 refreshed: `grade, date, price, mlb_id, player_name, rookie_year, set_slug, card_slug`), `player_stats_series` (Task 2), `data/reference/player_info.csv`, game_logs parquet.
- Produces (Task 6, Plan 4 consume):
  - `monthly_panel(chart: pd.DataFrame, game_logs: pd.DataFrame, player_info: pd.DataFrame) -> pd.DataFrame` — one row per `(card_slug, month)` where the player had debuted; columns:
    - keys: `card_slug, mlb_id, player_name, month (datetime64, 1st)`
    - outcome: `price` (month-end), `log_ret` (ln price_t / price_{t-1}), `market_median_ret`, `excess_ret` (= log_ret − market_median_ret)
    - predictors (all lagged: cumulative through last day of month t−1): `games, ops, avg, obp, slg, home_runs, strikeouts` (hitters) or `era, whip, k_bb_pct, innings_pitched` (pitchers), plus `form_games` (games in last 14 days of t−1), `form_ops_delta` (OPS over last 14 days of t−1 minus season-to-date OPS; pitchers: `form_era_delta` analog using ERA over last-14-days minus season ERA)
    - static: `age` (at month start, 2dp), `position`, `rookie_year`, `stats_season` (= `month.year` — the season the lagged stats come from; chart months past rookie year roll to the current season's stats, matching the weekly panel's rule), `set_slug`, `grade` (always psa_10), `playoff` (1 if month is October)
  - Only grades == "psa_10" are used. Months with no price for the card are skipped (no interpolation). First month per card has no log_ret (NaN, kept).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_monthly.py
import pandas as pd

from cardprice.panel import monthly_panel


def make_inputs():
    # Two cards, one player each, 4 months of prices
    chart = pd.DataFrame(
        [
            ("card/a", 1, 2022, "2022-04-01", 100.0),
            ("card/a", 1, 2022, "2022-05-01", 110.0),
            ("card/a", 1, 2022, "2022-06-01", 99.0),
            ("card/a", 1, 2022, "2022-10-01", 150.0),
            ("card/b", 2, 2022, "2022-04-01", 50.0),
            ("card/b", 2, 2022, "2022-05-01", 55.0),
            ("card/b", 2, 2022, "2022-06-01", 44.0),
            ("card/b", 2, 2022, "2022-10-01", 66.0),
        ],
        columns=["card_slug", "mlb_id", "rookie_year", "date", "price"],
    )
    chart["date"] = pd.to_datetime(chart["date"])
    chart["grade"] = "psa_10"
    chart["player_name"] = chart["mlb_id"].map({1: "Player One", 2: "Player Two"})
    chart["set_slug"] = "set/x"

    # Player 1 game logs: 1 game Apr 10 (HR), 1 game May 5, 1 game Sep 28
    logs = pd.DataFrame(
        [
            (1, "hitting", 2022, "2022-04-10", 1, 4, 2, 1, 2, 1, 0),
            (1, "hitting", 2022, "2022-05-05", 1, 4, 1, 0, 0, 1, 0),
            (1, "hitting", 2022, "2022-09-28", 1, 4, 3, 2, 1, 0, 1),
            (2, "hitting", 2022, "2022-04-11", 1, 4, 1, 0, 1, 1, 0),
            (2, "hitting", 2022, "2022-05-06", 1, 4, 0, 0, 0, 2, 0),
        ],
        columns=["mlb_id", "group", "season", "date", "gamesPlayed", "atBats", "hits",
                 "doubles", "homeRuns", "baseOnBalls", "strikeOuts"],
    )
    logs["date"] = pd.to_datetime(logs["date"])
    info = pd.DataFrame(
        {
            "mlb_id": [1, 2],
            "name": ["Player One", "Player Two"],
            "birth_date": pd.to_datetime(["2000-01-01", "1999-06-15"]),
            "position": ["OF", "1B"],
        }
    )
    return chart, logs, info


def test_panel_shape_and_lag():
    chart, logs, info = make_inputs()
    panel = monthly_panel(chart, logs, info)
    may = panel[(panel["card_slug"] == "card/a") & (panel["month"] == "2022-05-01")]
    assert len(may) == 1
    may = may.iloc[0]
    # NO LOOK-AHEAD: predictors for May = stats through Apr 30 only (1 game, the Apr 10 game)
    assert may["games"] == 1
    assert may["home_runs"] == 1
    june = panel[(panel["card_slug"] == "card/a") & (panel["month"] == "2022-06-01")].iloc[0]
    assert june["games"] == 2  # through May 31
    assert june["home_runs"] == 1
    oct_row = panel[(panel["card_slug"] == "card/a") & (panel["month"] == "2022-10-01")].iloc[0]
    assert oct_row["games"] == 3  # through Sep 30 (the Sep 28 game counts)
    assert oct_row["playoff"] == 1


def test_market_median_and_excess():
    chart, logs, info = make_inputs()
    panel = monthly_panel(chart, logs, info)
    may = panel[panel["month"] == "2022-05-01"]
    # both cards +10% in May -> market median ret = ln(1.1); excess ~ 0
    row = may[may["card_slug"] == "card/a"].iloc[0]
    assert row["log_ret"] == pytest.approx(np.log(1.1), abs=1e-9)
    assert row["excess_ret"] == pytest.approx(0.0, abs=1e-9)
    june = panel[(panel["month"] == "2022-06-01") & (panel["card_slug"] == "card/a")].iloc[0]
    # a: 110->99 = ln(0.9); b: 55->44 = ln(0.8); median = mean of the two logs
    expected_market = (np.log(0.9) + np.log(0.8)) / 2
    assert june["excess_ret"] == pytest.approx(np.log(0.9) - expected_market, abs=1e-9)


def test_debut_and_static():
    chart, logs, info = make_inputs()
    panel = monthly_panel(chart, logs, info)
    apr_a = panel[(panel["card_slug"] == "card/a") & (panel["month"] == "2022-04-01")].iloc[0]
    assert apr_a["games"] == 0  # no games before Apr 1 (first game Apr 10) -> row kept with 0s? or dropped?
    assert apr_a["position"] == "OF"
    assert apr_a["age"] == pytest.approx(22.25, abs=0.01)
    assert apr_a["log_ret"] != apr_a["log_ret"]  # first month: NaN
```

(add `import numpy as np` and `import pytest` at top)

**DESIGN DECISION for the implementer (record your choice + rationale in the report):** the April row for card/a has zero pre-month games (debut Apr 10). Per Global Constraints, pre-debut rows are dropped — but Apr is the debut MONTH (games happen DURING it). Rule: a month row is kept if the player debuted before the month's END (stats may be 0s at lag); dropped only if the debut is after the month end. Under this rule the April row IS kept (games=0 at lag is legitimate pre-debut information), and the test above stands. Implement that rule.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_panel_monthly.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.panel'`.

- [ ] **Step 3: Implement panel.py**

```python
# src/cardprice/panel.py
"""Card x month analysis panel with strictly lagged predictors."""

import numpy as np
import pandas as pd

from cardprice.stats_panel import player_stats_series

HITTING_COLS = ["games", "ops", "avg", "obp", "slg", "home_runs", "strikeouts"]
PITCHING_COLS = ["era", "whip", "k_bb_pct", "innings_pitched"]


def _month_ends(chart: pd.DataFrame) -> pd.DataFrame:
    df = chart[chart["grade"] == "psa_10"].copy()
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    # last price point within each month
    return (
        df.sort_values("date")
        .groupby(["card_slug", "month"])
        .agg(
            price=("price", "last"),
            mlb_id=("mlb_id", "first"),
            player_name=("player_name", "first"),
            rookie_year=("rookie_year", "first"),
            set_slug=("set_slug", "first"),
        )
        .reset_index()
    )


def monthly_panel(
    chart: pd.DataFrame, game_logs: pd.DataFrame, player_info: pd.DataFrame
) -> pd.DataFrame:
    me = _month_ends(chart)
    rows = []
    for card in me.itertuples():
        season = int(card.month.year)  # stats roll to the month's own season (>= rookie_year)
        # stats lag: cumulative through last day of previous month
        lag_date = card.month - pd.Timedelta(days=1)
        stats = player_stats_series(game_logs, int(card.mlb_id), season, [lag_date])
        if not len(stats):
            continue  # no game logs this season
        s = stats.iloc[0]
        # drop months that end before the player's debut (first game after month end)
        first_game = game_logs[
            (game_logs["mlb_id"] == card.mlb_id) & (game_logs["season"] == season)
        ]["date"].min()
        if pd.isna(first_game) or first_game >= card.month + pd.offsets.MonthBegin(1):
            continue
        # form: games in the last 14 days of the lag window
        form_start = lag_date - pd.Timedelta(days=13)
        recent = player_stats_series(
            game_logs, int(card.mlb_id), season, [lag_date]
        ).iloc[0]
        earlier = player_stats_series(
            game_logs, int(card.mlb_id), season, [form_start - pd.Timedelta(days=1)]
        ).iloc[0]
        form_games = int(recent["games"] - earlier["games"])
        row = {
            "card_slug": card.card_slug,
            "mlb_id": int(card.mlb_id),
            "player_name": card.player_name,
            "month": card.month,
            "price": card.price,
            "rookie_year": int(card.rookie_year),
            "stats_season": season,
            "set_slug": card.set_slug,
            "grade": "psa_10",
            "playoff": int(card.month.month == 10),
            "form_games": form_games,
        }
        if "ops" in s.index:  # hitter
            for c in HITTING_COLS:
                row[c] = s[c]
            row["form_ops_delta"] = _form_delta_hitting(
                game_logs, int(card.mlb_id), season, form_start, lag_date, s["ops"]
            )
            row["form_era_delta"] = np.nan
        else:  # pitcher
            for c in PITCHING_COLS:
                row[c] = s[c]
            row["form_era_delta"] = _form_delta_pitching(
                game_logs, int(card.mlb_id), season, form_start, lag_date, s["era"]
            )
            row["form_ops_delta"] = np.nan
        info = player_info[player_info["mlb_id"] == card.mlb_id]
        if len(info):
            birth = info.iloc[0]["birth_date"]
            row["age"] = round((card.month - birth).days / 365.25, 2)
            row["position"] = info.iloc[0]["position"]
        rows.append(row)
    panel = pd.DataFrame(rows)
    # outcomes per card
    panel = panel.sort_values(["card_slug", "month"])
    panel["log_ret"] = panel.groupby("card_slug")["price"].transform(lambda p: np.log(p / p.shift(1)))
    market = panel.groupby("month")["log_ret"].median().rename("market_median_ret")
    panel = panel.merge(market, on="month", how="left")
    panel["excess_ret"] = panel["log_ret"] - panel["market_median_ret"]
    return panel.reset_index(drop=True)


def _form_delta_hitting(game_logs, mlb_id, season, form_start, lag_date, season_ops):
    full = player_stats_series(game_logs, mlb_id, season, [lag_date]).iloc[0]
    before = player_stats_series(game_logs, mlb_id, season, [form_start - pd.Timedelta(days=1)]).iloc[0]
    # reconstruct last-14-day OPS from counting-stat differences
    ab = full["at_bats"] - before["at_bats"]
    if ab <= 0:
        return np.nan
    h = full["hits"] - before["hits"]
    bb = full["walks"] - before["walks"]
    hbp = full["hit_by_pitch"] - before["hit_by_pitch"]
    sf = full["sac_flies"] - before["sac_flies"]
    tb = h + (full["doubles"] - before["doubles"]) + 2 * (full["triples"] - before["triples"]) + 3 * (full["home_runs"] - before["home_runs"])
    obp_den = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_den if obp_den else np.nan
    slg = tb / ab
    if pd.isna(obp) or pd.isna(season_ops):
        return np.nan
    return round(obp + slg - season_ops, 3)


def _form_delta_pitching(game_logs, mlb_id, season, form_start, lag_date, season_era):
    full = player_stats_series(game_logs, mlb_id, season, [lag_date]).iloc[0]
    before = player_stats_series(game_logs, mlb_id, season, [form_start - pd.Timedelta(days=1)]).iloc[0]
    outs = full["outs"] - before["outs"]
    if outs <= 0 or pd.isna(season_era):
        return np.nan
    er = full["earned_runs"] - before["earned_runs"]
    era_14d = 9 * er / (outs / 3)
    return round(era_14d - season_era, 3)
```

NOTE: `player_stats_series` returns ALL stat keys (at_bats, hits, doubles, triples, walks, hit_by_pitch, sac_flies, outs, earned_runs...) — the form-delta helpers rely on those, not just the HITTING_COLS/PITCHING_COLS subsets. The hitter/pitcher branch checks `"ops" in s.index` — hitting_to_date dicts contain `ops`, pitching dicts contain `era`; verify both are present as expected.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_panel_monthly.py -v` — 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/panel.py tests/test_panel_monthly.py
git commit -m "feat: monthly analysis panel with lagged predictors"
```

---

### Task 4: Weekly panel builder

**Files:**
- Modify: `src/cardprice/panel.py` (append)
- Test: `tests/test_panel_weekly.py`

**Interfaces:**
- Consumes: `data/processed/scp_weekly.parquet` (`card_slug, grade, week, median_price, n_sales, best_offer_share` + no meta) joined to `cards_seed.csv` for meta; game_logs, player_info.
- Produces (Task 6, Plan 4 consume):
  - `weekly_panel(weekly: pd.DataFrame, cards: pd.DataFrame, game_logs: pd.DataFrame, player_info: pd.DataFrame) -> pd.DataFrame` — same columns as `monthly_panel` plus `n_sales, best_offer_share`; `month` replaced by `week` (Monday); outcome `log_ret` = week-over-week log change of `median_price` per (card_slug, grade); market median per week; predictors lagged to stats through the Sunday before week start; form window = the 14 days before week start. ALL grades present in the weekly input are used (psa_9 + psa_10).
  - Season selection: `rookie_year` per card, EXCEPT when the week falls in a LATER season than rookie_year — then use the week's actual calendar year as the stats season (sophomore slump is real; the 2026 weekly window is year 2-4 for most seed cards). Add column `stats_season`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_weekly.py
import numpy as np
import pandas as pd
import pytest

from cardprice.panel import weekly_panel


def make_inputs():
    weekly = pd.DataFrame(
        [
            ("card/a", "psa_10", "2026-08-31", 100.0, 3, 0.0),
            ("card/a", "psa_10", "2026-09-07", 105.0, 2, 0.5),
            ("card/a", "psa_9", "2026-08-31", 30.0, 2, 0.0),
            ("card/a", "psa_9", "2026-09-07", 33.0, 4, 0.25),
        ],
        columns=["card_slug", "grade", "week", "median_price", "n_sales", "best_offer_share"],
    )
    weekly["week"] = pd.to_datetime(weekly["week"])
    cards = pd.DataFrame(
        {
            "player_name": ["Player One"],
            "mlb_id": [1],
            "rookie_year": [2023],
            "set_slug": ["set/x"],
            "card_slug": ["card/a"],
        }
    )
    logs = pd.DataFrame(
        [
            (1, "hitting", 2026, "2026-08-20", 1, 4, 2, 1, 1, 1, 0),
            (1, "hitting", 2026, "2026-09-02", 1, 4, 1, 0, 0, 1, 0),
        ],
        columns=["mlb_id", "group", "season", "date", "gamesPlayed", "atBats", "hits",
                 "doubles", "homeRuns", "baseOnBalls", "strikeOuts"],
    )
    logs["date"] = pd.to_datetime(logs["date"])
    info = pd.DataFrame(
        {
            "mlb_id": [1],
            "name": ["Player One"],
            "birth_date": pd.to_datetime(["2001-01-01"]),
            "position": ["SS"],
        }
    )
    return weekly, cards, logs, info


def test_weekly_lag_and_sophomore_season():
    weekly, cards, logs, info = make_inputs()
    panel = weekly_panel(weekly, cards, logs, info)
    w2 = panel[(panel["card_slug"] == "card/a") & (panel["grade"] == "psa_10")
               & (panel["week"] == "2026-09-07")].iloc[0]
    # predictors through Sun Sep 6: only the Aug 20 + Sep 2 games -> games == 2
    assert w2["games"] == 2
    # rookie_year 2023 but week is in 2026 -> stats_season 2026
    assert w2["stats_season"] == 2026
    assert w2["n_sales"] == 2
    assert w2["best_offer_share"] == 0.5
    # both grades +5%/+10% -> market median between them
    assert panel["log_ret"].notna().sum() == 2  # first week per card-grade is NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_panel_weekly.py -v`
Expected: FAIL with `ImportError: cannot import name 'weekly_panel'`.

- [ ] **Step 3: Implement** (append to `src/cardprice/panel.py`)

```python
def weekly_panel(
    weekly: pd.DataFrame, cards: pd.DataFrame, game_logs: pd.DataFrame, player_info: pd.DataFrame
) -> pd.DataFrame:
    meta = cards[["card_slug", "player_name", "mlb_id", "rookie_year", "set_slug"]].drop_duplicates()
    df = weekly.merge(meta, on="card_slug", how="left", validate="many_to_one")
    rows = []
    for r in df.itertuples():
        stats_season = max(int(r.rookie_year), r.week.year)
        lag_date = r.week - pd.Timedelta(days=1)  # Sunday before the Monday week start
        stats = player_stats_series(game_logs, int(r.mlb_id), stats_season, [lag_date])
        if not len(stats):
            continue
        s = stats.iloc[0]
        first_game = game_logs[
            (game_logs["mlb_id"] == r.mlb_id) & (game_logs["season"] == stats_season)
        ]["date"].min()
        if pd.isna(first_game) or first_game > lag_date:
            continue  # not yet debuted this season at the lag date
        form_start = lag_date - pd.Timedelta(days=13)
        row = {
            "card_slug": r.card_slug,
            "grade": r.grade,
            "mlb_id": int(r.mlb_id),
            "player_name": r.player_name,
            "week": r.week,
            "price": r.median_price,
            "n_sales": int(r.n_sales),
            "best_offer_share": float(r.best_offer_share),
            "rookie_year": int(r.rookie_year),
            "stats_season": stats_season,
            "set_slug": r.set_slug,
            "playoff": int(r.week.month == 10),
        }
        earlier = player_stats_series(
            game_logs, int(r.mlb_id), stats_season, [form_start - pd.Timedelta(days=1)]
        ).iloc[0]
        row["form_games"] = int(s["games"] - earlier["games"])
        if "ops" in s.index:
            for c in HITTING_COLS:
                row[c] = s[c]
            row["form_ops_delta"] = _form_delta_hitting(
                game_logs, int(r.mlb_id), stats_season, form_start, lag_date, s["ops"]
            )
            row["form_era_delta"] = np.nan
        else:
            for c in PITCHING_COLS:
                row[c] = s[c]
            row["form_era_delta"] = _form_delta_pitching(
                game_logs, int(r.mlb_id), stats_season, form_start, lag_date, s["era"]
            )
            row["form_ops_delta"] = np.nan
        info = player_info[player_info["mlb_id"] == r.mlb_id]
        if len(info):
            birth = info.iloc[0]["birth_date"]
            row["age"] = round((r.week - birth).days / 365.25, 2)
            row["position"] = info.iloc[0]["position"]
        rows.append(row)
    panel = pd.DataFrame(rows)
    panel = panel.sort_values(["card_slug", "grade", "week"])
    panel["log_ret"] = panel.groupby(["card_slug", "grade"])["price"].transform(
        lambda p: np.log(p / p.shift(1))
    )
    market = panel.groupby("week")["log_ret"].median().rename("market_median_ret")
    panel = panel.merge(market, on="week", how="left")
    panel["excess_ret"] = panel["log_ret"] - panel["market_median_ret"]
    return panel.reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_panel_weekly.py -v` — 1 PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/panel.py tests/test_panel_weekly.py
git commit -m "feat: weekly analysis panel with stats-season rollover"
```

---

### Task 5: Population data spike (GemRate) — time-boxed

**Files:**
- Create: `docs/superpowers/spikes/2026-09-15-gemrate.md`
- Create (if feasible): `src/cardprice/pop_gemrate.py`, `tests/test_pop_gemrate.py`
- Create (if feasible): `data/reference/pop_snapshots.csv`

**Interfaces:**
- Consumes: `web.fetch_page`, cards_seed.csv.
- Produces (if feasible): `fetch_pop(scp_url_or_query: str) -> dict` returning `{"psa_10_pop": int, "total_pop": int, "gem_rate": float}`; monthly snapshots appended to pop_snapshots.csv with `date` column.

- [ ] **Step 1: Spike (time-boxed, max ~45 min of effort)**

Using Playwright (same config as `web.fetch_page`), attempt:
1. Load `https://www.gemrate.com/universal-search`, type "Gunnar Henderson 2023 Topps Chrome #2" into the search input, wait for JS results, extract result links.
2. Open the matching card page, extract PSA 10 pop count, total pop, gem rate, and whether a pop-growth-over-time chart/table exists.
3. Record in the spike doc: exact selectors, interaction steps, whether headless works, per-card time cost, and a screenshot or verbatim text excerpt as evidence.

- [ ] **Step 2a (if feasible): collector + test**

`fetch_pop(query: str) -> dict` driving the interaction from step 1, with a committed HTML/text fixture and offline parser test. Run it for the 13 seed cards, write `data/reference/pop_snapshots.csv` (columns: `date, card_slug, psa_10_pop, total_pop, gem_rate`). Commit.

- [ ] **Step 2b (if not feasible): documented skip**

Write the spike doc with what failed (screenshots/text), and add `pop_count` to the panel as an explicitly-absent feature: document in the spike doc that Plan 4 must treat pop controls as unavailable. No code. Commit only the spike doc.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/spikes/2026-09-15-gemrate.md src/cardprice/pop_gemrate.py tests/test_pop_gemrate.py data/reference/pop_snapshots.csv
git commit -m "feat: GemRate pop data spike (+ collector if feasible)"
```

---

### Task 6: Panel export + golden verification + data dictionary

**Files:**
- Create: `scripts/build_panels.py`
- Create: `docs/data_dictionary.md`
- Test: `tests/test_golden_panel.py`

**Interfaces:**
- Consumes: all above.
- Produces: `data/processed/panel_monthly.parquet`, `data/processed/panel_weekly.parquet`, `docs/data_dictionary.md`.

- [ ] **Step 1: Write the failing golden test**

```python
# tests/test_golden_panel.py
import numpy as np
import pandas as pd
import pytest

from cardprice.panel import monthly_panel


@pytest.fixture(scope="module")
def built():
    chart = pd.read_parquet("data/processed/scp_chart_monthly.parquet")
    game_logs = pd.read_parquet("data/processed/game_logs.parquet")
    info = pd.read_csv("data/reference/player_info.csv", parse_dates=["birth_date"])
    return monthly_panel(chart, game_logs, info)


def test_henderson_august_2023_row(built):
    row = built[
        (built["player_name"] == "Gunnar Henderson") & (built["month"] == "2023-08-01")
    ]
    assert len(row) == 1
    r = row.iloc[0]
    # stats through 2023-07-31 (corrected during execution — see NOTE below; July 2023
    # has no price point, his card's chart starts 2023-08-01):
    # 95 G, 17 HR, .242 AVG, verified via game-logs reconstruction + Stats API byDateRange
    assert r["games"] == 95
    assert r["home_runs"] == 17
    assert r["avg"] == pytest.approx(0.242, abs=0.002)
    assert r["playoff"] == 0
    assert r["age"] == pytest.approx(22.1, abs=0.1)


def test_no_lookahead_invariant(built):
    # for any row, lagged games must never exceed the player's total season games
    logs = pd.read_parquet("data/processed/game_logs.parquet")
    totals = logs.groupby(["mlb_id", "season"]).size().rename("season_games")
    merged = built.merge(
        totals, left_on=["mlb_id", "stats_season"], right_index=True, how="left"
    )
    assert (merged["games"] <= merged["season_games"]).all()


def test_excess_ret_identity(built):
    diff = (built["excess_ret"] - (built["log_ret"] - built["market_median_ret"])).abs()
    assert (diff < 1e-9).all()
```

NOTE for the implementer: golden values corrected during execution — Henderson's card has NO July-2023 price point (chart starts 2023-08-01), and his official through-June line was 70 G/11 HR/.240 (not 73/13/.253 as an earlier draft said). The golden row is now the 2023-08-01 row: 95 G, 17 HR, .242 AVG through 2023-07-31, verified by game-logs reconstruction (which also matches his official full-2023 line: 150 G/28 HR/.255) and Stats API byDateRange. If these values ever mismatch, follow the same protocol: reconstruct from game_logs.parquet, check byDateRange, report DONE_WITH_CONCERNS with evidence — do not silently edit.

- [ ] **Step 2: Run test to verify it fails (panel not built yet)**

Run: `python -m pytest tests/test_golden_panel.py -v`
Expected: FAIL with `FileNotFoundError` or 0 rows (panel parquets / inputs may not exist in the test env yet — see Step 3).

NOTE: this test reads the REAL processed parquets. It runs offline (no network) but requires Task 1's refreshed parquets + player_info.csv to exist locally. If they do (Task 1 Step 4 ran), it works in the dev worktree.

- [ ] **Step 3: Build panels + data dictionary**

```python
# scripts/build_panels.py
"""Build panel_monthly.parquet and panel_weekly.parquet from processed data."""

import pandas as pd

from cardprice.panel import monthly_panel, weekly_panel

if __name__ == "__main__":
    chart = pd.read_parquet("data/processed/scp_chart_monthly.parquet")
    game_logs = pd.read_parquet("data/processed/game_logs.parquet")
    info = pd.read_csv("data/reference/player_info.csv", parse_dates=["birth_date"])
    cards = pd.read_csv("data/reference/cards_seed.csv")

    mp = monthly_panel(chart, game_logs, info)
    mp.to_parquet("data/processed/panel_monthly.parquet", index=False)
    print(f"panel_monthly: {len(mp)} rows, {mp['card_slug'].nunique()} cards, "
          f"months {mp['month'].min()} -> {mp['month'].max()}")

    weekly = pd.read_parquet("data/processed/scp_weekly.parquet")
    wp = weekly_panel(weekly, cards, game_logs, info)
    wp.to_parquet("data/processed/panel_weekly.parquet", index=False)
    print(f"panel_weekly: {len(wp)} rows, {wp['card_slug'].nunique()} cards, "
          f"weeks {wp['week'].min()} -> {wp['week'].max()}")
```

NOTE: `scp_weekly.parquet` must be regenerated after Task 1's reparse — run `python scripts/run_price_pipeline.py` first (it reads the refreshed sales parquet and rewrites scp_weekly.parquet + liquidity outputs).

`docs/data_dictionary.md` — one table per parquet (panel_monthly, panel_weekly): every column, its type, its definition, and its lag rule (e.g. "ops: cumulative season-to-date OPS through the last day before the period starts; rate stats recomputed from counting sums"). Include the market-median outcome definition and the debut-month inclusion rule.

- [ ] **Step 4: Run**

```bash
python scripts/run_price_pipeline.py   # regenerate weekly from refreshed sales
python scripts/build_panels.py
python -m pytest tests/test_golden_panel.py -v
```
Expected: 3 PASS; monthly panel covers each card's rookie season (e.g. Henderson 2023 rows for ~Jul-Oct 2023); weekly panel covers the 2026 window.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_panels.py docs/data_dictionary.md tests/test_golden_panel.py
git commit -m "feat: panel export, golden verification, data dictionary"
```

---

## Done criteria for this plan

- `python -m pytest -v` all offline tests PASS (including the no-look-ahead invariant and the Henderson golden row); ruff clean.
- Grade-regex recovery measured: refreshed sales count vs. 1825 baseline recorded in the final report.
- `data/processed/panel_monthly.parquet` (PSA-10, 2022-10→2026-09, 325 rows / 13 cards) and `panel_weekly.parquet` (2022-12-12→2026-09-07, 184 rows / 13 cards) exist and pass the golden tests.
- GemRate pop spike documented either way; pop data included if feasible.
- `docs/data_dictionary.md` complete.
- Next: Plan 4 (modeling: LASSO stability selection, GBM + SHAP, hierarchical model, walk-forward buy-signal gate).
