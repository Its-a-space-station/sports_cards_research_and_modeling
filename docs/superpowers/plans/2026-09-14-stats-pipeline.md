# MLB Stats Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Python pipeline that collects MLB game logs (2022–2026) from the official MLB Stats API into immutable raw snapshots and reconstructs season-to-date cumulative stats at any date.

**Architecture:** Thin `requests` client over `statsapi.mlb.com` (no API key), raw JSON snapshots on disk (dated, immutable), parsers into pandas DataFrames, and pure aggregation functions that rebuild season-to-date lines. Later plans (prices, panel, modeling) consume the parquet output and the `season_stats` functions.

**Tech Stack:** Python 3.11+, requests, pandas, pyarrow, pytest, ruff. No database.

**Spec:** `docs/superpowers/specs/2026-09-14-mlb-card-price-panel-design.md`

## Global Constraints

- Python >= 3.11; free data sources only; no new paid dependencies.
- Raw snapshots in `data/raw/` are immutable: collectors may add new dated files but never edit or delete existing ones.
- The MLB Stats API needs no key; still be polite: `time.sleep(0.3)` between calls (collector default).
- Do NOT use FanGraphs (hard-blocks scripts) or bulk Baseball-Reference scraping (rate limits + ToS). MLB Stats API only for this plan.
- `data/raw/` and `data/processed/` are gitignored; test fixtures live in `tests/fixtures/` and ARE committed.
- Tests must run offline by default; anything hitting the real API is marked `@pytest.mark.live` and excluded by default.
- macOS, bash. Create the venv as `.venv` in the repo root.

## Roadmap (context for where this plan ends)

This is Plan 1 of 5. Later plans (each gets its own plan doc when this one lands):
2. Price pipeline (SportsCardsPro + 130point collectors, parsers, validate layer, card-universe selection).
3. Panel assembly (Savant metrics, GemRate pop snapshots, card×week panel, market index, lagged predictors).
4. Modeling (LASSO stability selection, GBM + SHAP, hierarchical model, walk-forward buy-signal gate).
5. Event study (event registry, abnormal-return windows, mean-reversion analysis).

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/cardprice/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/reference/players.csv`
- Create: `data/raw/.gitkeep`, `data/processed/.gitkeep`
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Consumes: nothing.
- Produces: installable `cardprice` package (`pip install -e ".[dev]"`), pytest with a `live` marker, `data/reference/players.csv` with columns `mlb_id,name,role` where role is `hitter` or `pitcher`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffold.py
import cardprice


def test_package_importable():
    assert cardprice.__version__ == "0.1.0"
```

```python
# src/cardprice/__init__.py will contain:
# __version__ = "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scaffold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice'` (package not installed yet).

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "cardprice"
version = "0.1.0"
description = "Within-season MLB performance vs graded rookie card prices"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "pyarrow>=15",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "ruff>=0.6",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
markers = [
    "live: hits the real MLB Stats API (network required)",
]
addopts = "-m 'not live'"

[tool.ruff]
line-length = 100
```

```
# .gitignore
.venv/
__pycache__/
*.egg-info/
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
```

```csv
# data/reference/players.csv
mlb_id,name,role
592450,Aaron Judge,hitter
683002,Gunnar Henderson,hitter
543037,Gerrit Cole,pitcher
605400,Aaron Nola,pitcher
```

```python
# src/cardprice/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
# (empty file)
```

Create `data/raw/.gitkeep` and `data/processed/.gitkeep` as empty files.

Then set up the environment:

```bash
cd /Users/tomcruise/sports_cards_research_and_modeling
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scaffold.py -v`
Expected: PASS. Also run `ruff check src tests` — expected: clean.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src tests data
git commit -m "feat: project scaffolding (pyproject, venv, players reference)"
```

---

### Task 2: Immutable raw snapshot storage

**Files:**
- Create: `src/cardprice/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing.
- Produces (used by Tasks 3 and 7, and by every later collector):
  - `save_raw(dataset: str, key: str, payload: dict, on: date | None = None) -> Path` — writes `data/raw/<dataset>/<key>/<YYYY-MM-DD>.json`; if a file for that date already exists with identical content it is left untouched; never modifies older dates.
  - `load_latest(dataset: str, key: str) -> dict` — returns the payload from the most recent dated file; raises `FileNotFoundError` if none exist.
  - `RAW_ROOT: Path` — module constant (`Path("data/raw")`); tests monkeypatch it to a tmp dir.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
import json
from datetime import date

import pytest

from cardprice import storage


@pytest.fixture
def raw_root(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)
    return tmp_path


def test_save_then_load_latest_roundtrip(raw_root):
    storage.save_raw("stats", "592450_hitting_2022", {"splits": [1, 2]}, on=date(2026, 9, 14))
    assert storage.load_latest("stats", "592450_hitting_2022") == {"splits": [1, 2]}


def test_save_is_idempotent_same_day(raw_root):
    p1 = storage.save_raw("stats", "k", {"a": 1}, on=date(2026, 9, 14))
    mtime = p1.stat().st_mtime_ns
    p2 = storage.save_raw("stats", "k", {"a": 1}, on=date(2026, 9, 14))
    assert p1 == p2 and p2.stat().st_mtime_ns == mtime  # not rewritten


def test_save_new_date_does_not_touch_old(raw_root):
    storage.save_raw("stats", "k", {"v": 1}, on=date(2026, 9, 13))
    storage.save_raw("stats", "k", {"v": 2}, on=date(2026, 9, 14))
    old = raw_root / "stats" / "k" / "2026-09-13.json"
    assert json.loads(old.read_text()) == {"v": 1}
    assert storage.load_latest("stats", "k") == {"v": 2}


def test_load_latest_missing_raises(raw_root):
    with pytest.raises(FileNotFoundError):
        storage.load_latest("stats", "nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.storage'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cardprice/storage.py
"""Immutable dated JSON snapshots. Collectors add files; nothing edits or deletes them."""

import json
from datetime import date
from pathlib import Path

RAW_ROOT = Path("data/raw")


def save_raw(dataset: str, key: str, payload: dict, on: date | None = None) -> Path:
    day = (on or date.today()).isoformat()
    path = RAW_ROOT / dataset / key / f"{day}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path.exists() and path.read_text() == text:
        return path
    path.write_text(text)
    return path


def load_latest(dataset: str, key: str) -> dict:
    files = sorted((RAW_ROOT / dataset / key).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no snapshots for {dataset}/{key}")
    return json.loads(files[-1].read_text())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_storage.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/storage.py tests/test_storage.py
git commit -m "feat: immutable dated raw snapshot storage"
```

---

### Task 3: MLB Stats API client

**Files:**
- Create: `src/cardprice/stats_api.py`
- Test: `tests/test_stats_api.py`

**Interfaces:**
- Consumes: nothing (pure HTTP + parsing). Does NOT write snapshots — that is the collector's job (Task 7).
- Produces (used by Tasks 6 and 7):
  - `fetch_game_log(mlb_id: int, group: str, season: int) -> list[dict]` — `group` is `"hitting"` or `"pitching"`; returns the API's per-game `splits` list, or `[]` if the player has no such log that season.
  - `game_log_to_frame(splits: list[dict], mlb_id: int, group: str, season: int) -> pd.DataFrame` — one row per game; columns: `mlb_id`, `group`, `season`, `date` (datetime64), plus every key in the split's `stat` dict; sorted by date ascending.

API shape (verified 2026-09-14): `GET https://statsapi.mlb.com/api/v1/people/{id}/stats?stats=gameLog&group=hitting&season=2022` returns `{"stats": [{"splits": [{"date": "2022-04-08", "game": {...}, "stat": {"gamesPlayed": 1, "atBats": 4, ...}}, ...]}]}`. `stats` is empty (`[]`) when no log exists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stats_api.py
import pandas as pd
import pytest

from cardprice import stats_api

SAMPLE_RESPONSE = {
    "stats": [
        {
            "splits": [
                {"date": "2022-04-10", "stat": {"gamesPlayed": 1, "atBats": 3, "hits": 2, "homeRuns": 1}},
                {"date": "2022-04-08", "stat": {"gamesPlayed": 1, "atBats": 4, "hits": 1, "homeRuns": 0}},
            ]
        }
    ]
}


def test_fetch_game_log_builds_url_and_returns_splits(monkeypatch):
    calls = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return SAMPLE_RESPONSE

    def fake_get(url, params=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        return FakeResp()

    monkeypatch.setattr(stats_api.requests, "get", fake_get)
    splits = stats_api.fetch_game_log(592450, "hitting", 2022)
    assert calls["url"] == "https://statsapi.mlb.com/api/v1/people/592450/stats"
    assert calls["params"] == {"stats": "gameLog", "group": "hitting", "season": 2022}
    assert len(splits) == 2


def test_fetch_game_log_empty_when_no_stats(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"stats": []}

    monkeypatch.setattr(stats_api.requests, "get", lambda *a, **k: FakeResp())
    assert stats_api.fetch_game_log(592450, "pitching", 2022) == []


def test_game_log_to_frame_flattens_and_sorts():
    splits = SAMPLE_RESPONSE["stats"][0]["splits"]
    df = stats_api.game_log_to_frame(splits, 592450, "hitting", 2022)
    assert list(df.columns)[:4] == ["mlb_id", "group", "season", "date"]
    assert df["date"].tolist() == [pd.Timestamp("2022-04-08"), pd.Timestamp("2022-04-10")]
    assert df.loc[df["date"] == pd.Timestamp("2022-04-10"), "homeRuns"].iloc[0] == 1
    assert (df["mlb_id"] == 592450).all()


@pytest.mark.live
def test_fetch_game_log_live_judge_2022():
    splits = stats_api.fetch_game_log(592450, "hitting", 2022)
    assert len(splits) == 157  # Judge played 157 games in 2022
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stats_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.stats_api'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cardprice/stats_api.py
"""Thin client for the official MLB Stats API (no key required)."""

import pandas as pd
import requests

BASE = "https://statsapi.mlb.com/api/v1"


def fetch_game_log(mlb_id: int, group: str, season: int) -> list[dict]:
    resp = requests.get(
        f"{BASE}/people/{mlb_id}/stats",
        params={"stats": "gameLog", "group": group, "season": season},
        timeout=30,
    )
    resp.raise_for_status()
    stats = resp.json().get("stats", [])
    if not stats:
        return []
    return stats[0].get("splits", [])


def game_log_to_frame(splits: list[dict], mlb_id: int, group: str, season: int) -> pd.DataFrame:
    rows = []
    for split in splits:
        row = {"mlb_id": mlb_id, "group": group, "season": season, "date": split["date"]}
        row.update(split["stat"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stats_api.py -v`
Expected: 3 PASS (live test excluded by default addopts). Then sanity-check the live one once:
Run: `python -m pytest tests/test_stats_api.py -v -m live`
Expected: 1 PASS (requires network).

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/stats_api.py tests/test_stats_api.py
git commit -m "feat: MLB Stats API game-log client"
```

---

### Task 4: Season-to-date hitting reconstruction

**Files:**
- Create: `src/cardprice/season_stats.py`
- Test: `tests/test_season_stats_hitting.py`

**Interfaces:**
- Consumes: DataFrames from `game_log_to_frame` (Task 3).
- Produces (used by Task 6 golden tests and by Plan 3's feature builder):
  - `hitting_to_date(game_log: pd.DataFrame, through: date) -> dict` with keys:
    `games, at_bats, hits, doubles, triples, home_runs, rbi, walks, strikeouts,
    hit_by_pitch, sac_flies, stolen_bases, avg, obp, slg, ops`
    Rate stats are recomputed from summed counting stats (never averaged from
    per-game rates). Rate stats are `None` when their denominator is 0.
    Column mapping from API fields: `gamesPlayed→games, atBats→at_bats, hits→hits,
    doubles→doubles, triples→triples, homeRuns→home_runs, rbi→rbi,
    baseOnBalls→walks, strikeOuts→strikeouts, hitByPitch→hit_by_pitch,
    sacFlies→sac_flies, stolenBases→stolen_bases`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_season_stats_hitting.py
from datetime import date

import pandas as pd

from cardprice.season_stats import hitting_to_date


def make_log():
    # Three games: Apr 8, Apr 10, Apr 12
    rows = [
        {"date": "2022-04-08", "gamesPlayed": 1, "atBats": 4, "hits": 1, "doubles": 0,
         "triples": 0, "homeRuns": 0, "rbi": 0, "baseOnBalls": 1, "strikeOuts": 2,
         "hitByPitch": 0, "sacFlies": 0, "stolenBases": 0},
        {"date": "2022-04-10", "gamesPlayed": 1, "atBats": 3, "hits": 2, "doubles": 1,
         "triples": 0, "homeRuns": 1, "rbi": 3, "baseOnBalls": 0, "strikeOuts": 1,
         "hitByPitch": 1, "sacFlies": 0, "stolenBases": 1},
        {"date": "2022-04-12", "gamesPlayed": 1, "atBats": 5, "hits": 3, "doubles": 0,
         "triples": 1, "homeRuns": 1, "rbi": 2, "baseOnBalls": 0, "strikeOuts": 0,
         "hitByPitch": 0, "sacFlies": 1, "stolenBases": 0},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_partial_season_excludes_later_games():
    out = hitting_to_date(make_log(), date(2022, 4, 10))
    assert out["games"] == 2
    assert out["at_bats"] == 7
    assert out["home_runs"] == 1
    assert out["rbi"] == 3
    assert out["stolen_bases"] == 1


def test_full_season_rate_stats():
    out = hitting_to_date(make_log(), date(2022, 4, 12))
    # totals: AB=12, H=6, 2B=1, 3B=1, HR=2, BB=1, HBP=1, SF=1
    assert out["avg"] == round(6 / 12, 3)            # .500
    assert out["obp"] == round(8 / 15, 3)            # (6+1+1)/(12+1+1+1) = .533
    assert out["slg"] == round(15 / 12, 3)           # TB = 6+1+2+6 = 15 -> 1.250
    assert out["ops"] == round(8 / 15 + 15 / 12, 3)  # 1.783


def test_empty_window_returns_zeroes_and_none_rates():
    out = hitting_to_date(make_log(), date(2022, 4, 1))
    assert out["games"] == 0
    assert out["avg"] is None
    assert out["ops"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_season_stats_hitting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.season_stats'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cardprice/season_stats.py
"""Rebuild season-to-date cumulative stats from game logs at any date.

Rate stats are always recomputed from summed counting stats, never averaged
from per-game rates.
"""

from datetime import date

import pandas as pd

HITTING_MAP = {
    "gamesPlayed": "games",
    "atBats": "at_bats",
    "hits": "hits",
    "doubles": "doubles",
    "triples": "triples",
    "homeRuns": "home_runs",
    "rbi": "rbi",
    "baseOnBalls": "walks",
    "strikeOuts": "strikeouts",
    "hitByPitch": "hit_by_pitch",
    "sacFlies": "sac_flies",
    "stolenBases": "stolen_bases",
}


def _safe_ratio(num: float, den: float) -> float | None:
    return round(num / den, 3) if den else None


def hitting_to_date(game_log: pd.DataFrame, through: date) -> dict:
    df = game_log[game_log["date"].dt.date <= through]
    out = {new: int(df[old].sum()) if old in df else 0 for old, new in HITTING_MAP.items()}
    ab, h, bb, hbp, sf = (out[k] for k in ("at_bats", "hits", "walks", "hit_by_pitch", "sac_flies"))
    total_bases = h + out["doubles"] + 2 * out["triples"] + 3 * out["home_runs"]
    out["avg"] = _safe_ratio(h, ab)
    out["obp"] = _safe_ratio(h + bb + hbp, ab + bb + hbp + sf)
    out["slg"] = _safe_ratio(total_bases, ab)
    out["ops"] = round(out["obp"] + out["slg"], 3) if out["obp"] is not None and out["slg"] is not None else None
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_season_stats_hitting.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/season_stats.py tests/test_season_stats_hitting.py
git commit -m "feat: season-to-date hitting reconstruction"
```

---

### Task 5: Season-to-date pitching reconstruction

**Files:**
- Modify: `src/cardprice/season_stats.py` (append)
- Test: `tests/test_season_stats_pitching.py`

**Interfaces:**
- Consumes: DataFrames from `game_log_to_frame` (Task 3, `group="pitching"`).
- Produces (used by Task 6 golden tests and Plan 3):
  - `innings_to_outs(ip) -> int` — MLB notation: `6.1` = 6⅓ innings = 19 outs; `6.2` = 20 outs; `6`/`6.0` = 18 outs. Accepts str, int, or float.
  - `pitching_to_date(game_log: pd.DataFrame, through: date) -> dict` with keys:
    `games, games_started, wins, losses, outs, innings_pitched, hits_allowed,
    earned_runs, walks, strikeouts, batters_faced, era, whip, k_bb_pct`
    `innings_pitched` = outs/3 rounded to 3 dp; `era` = 9·ER/IP; `whip` =
    (BB+H)/IP; `k_bb_pct` = (K−BB)/batters_faced. Rate stats `None` on zero
    denominator. Column mapping: `gamesPlayed→games, gamesStarted→games_started,
    wins→wins, losses→losses, hits→hits_allowed, earnedRuns→earned_runs,
    baseOnBalls→walks, strikeOuts→strikeouts, battersFaced→batters_faced`;
    `outs` comes from summing `innings_to_outs` over each game's `inningsPitched`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_season_stats_pitching.py
from datetime import date

import pandas as pd
import pytest

from cardprice.season_stats import innings_to_outs, pitching_to_date


@pytest.mark.parametrize(
    "ip,outs",
    [("6.0", 18), ("6.1", 19), ("6.2", 20), ("7", 21), (0.1, 1), ("0.2", 2)],
)
def test_innings_to_outs(ip, outs):
    assert innings_to_outs(ip) == outs


def make_log():
    rows = [
        {"date": "2018-04-02", "gamesPlayed": 1, "gamesStarted": 1, "wins": 1, "losses": 0,
         "inningsPitched": "7.0", "hits": 4, "earnedRuns": 1, "baseOnBalls": 1,
         "strikeOuts": 11, "battersFaced": 25},
        {"date": "2018-04-08", "gamesPlayed": 1, "gamesStarted": 1, "wins": 0, "losses": 1,
         "inningsPitched": "5.2", "hits": 6, "earnedRuns": 3, "baseOnBalls": 2,
         "strikeOuts": 7, "battersFaced": 23},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_pitching_full_window():
    out = pitching_to_date(make_log(), date(2018, 4, 30))
    assert out["games"] == 2
    assert out["wins"] == 1 and out["losses"] == 1
    assert out["outs"] == 21 + 17          # 7.0 IP + 5.2 IP = 38 outs
    assert out["innings_pitched"] == round(38 / 3, 3)
    assert out["era"] == round(9 * 4 / (38 / 3), 3)     # 2.842
    assert out["whip"] == round(13 / (38 / 3), 3)       # (2+4+6)/IP = 1.026
    assert out["k_bb_pct"] == round((18 - 3) / 48, 3)   # .313


def test_pitching_empty_window():
    out = pitching_to_date(make_log(), date(2018, 3, 1))
    assert out["games"] == 0
    assert out["era"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_season_stats_pitching.py -v`
Expected: FAIL with `ImportError: cannot import name 'innings_to_outs'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/cardprice/season_stats.py`:

```python
PITCHING_MAP = {
    "gamesPlayed": "games",
    "gamesStarted": "games_started",
    "wins": "wins",
    "losses": "losses",
    "hits": "hits_allowed",
    "earnedRuns": "earned_runs",
    "baseOnBalls": "walks",
    "strikeOuts": "strikeouts",
    "battersFaced": "batters_faced",
}


def innings_to_outs(ip) -> int:
    """MLB notation: 6.1 = 6 and 1/3 innings = 19 outs."""
    whole, _, frac = str(ip).partition(".")
    return int(whole) * 3 + (int(frac) if frac else 0)


def pitching_to_date(game_log: pd.DataFrame, through: date) -> dict:
    df = game_log[game_log["date"].dt.date <= through]
    out = {new: int(df[old].sum()) if old in df else 0 for old, new in PITCHING_MAP.items()}
    out["outs"] = int(df["inningsPitched"].map(innings_to_outs).sum()) if len(df) else 0
    ip = out["outs"] / 3
    out["innings_pitched"] = round(ip, 3)
    out["era"] = _safe_ratio(9 * out["earned_runs"], ip)
    out["whip"] = _safe_ratio(out["walks"] + out["hits_allowed"], ip)
    out["k_bb_pct"] = _safe_ratio(out["strikeouts"] - out["walks"], out["batters_faced"])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_season_stats_pitching.py -v`
Expected: 8 PASS. Also re-run `python -m pytest tests/test_season_stats_hitting.py -v` — still 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/season_stats.py tests/test_season_stats_pitching.py
git commit -m "feat: season-to-date pitching reconstruction with innings notation"
```

---

### Task 6: Golden fixtures — reconstruction matches official season totals

**Files:**
- Create: `scripts/fetch_fixtures.py`
- Create: `tests/fixtures/judge_2022_hitting.json`, `tests/fixtures/cole_2018_pitching.json` (generated, then committed)
- Test: `tests/test_golden.py`

**Interfaces:**
- Consumes: `fetch_game_log` (Task 3), `hitting_to_date`, `pitching_to_date` (Tasks 4–5).
- Produces: offline proof that reconstruction matches official totals; fixtures reused by later plans' integration tests.

Golden values (official season totals — assert exactly these):
- Aaron Judge 2022 hitting: 157 G, 570 AB, 177 H, 62 HR, 131 RBI, 111 BB, 175 SO, 16 SB, AVG .311, SLG .686.
- Gerrit Cole 2018 pitching: 32 G, 32 GS, 15 W, 5 L, 276 SO, 601 outs (200.1 IP), 64 ER, ERA 2.88.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_golden.py
import json
from datetime import date
from pathlib import Path

import pytest

from cardprice.season_stats import hitting_to_date, pitching_to_date
from cardprice.stats_api import game_log_to_frame

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name, group):
    payload = json.loads((FIXTURES / name).read_text())
    return game_log_to_frame(payload["splits"], payload["mlb_id"], group, payload["season"])


def test_judge_2022_official_totals():
    df = load_fixture("judge_2022_hitting.json", "hitting")
    out = hitting_to_date(df, date(2022, 10, 31))
    assert out["games"] == 157
    assert out["at_bats"] == 570
    assert out["hits"] == 177
    assert out["home_runs"] == 62
    assert out["rbi"] == 131
    assert out["walks"] == 111
    assert out["strikeouts"] == 175
    assert out["stolen_bases"] == 16
    assert out["avg"] == 0.311
    assert out["slg"] == 0.686


def test_cole_2018_official_totals():
    df = load_fixture("cole_2018_pitching.json", "pitching")
    out = pitching_to_date(df, date(2018, 10, 31))
    assert out["games"] == 32
    assert out["games_started"] == 32
    assert out["wins"] == 15
    assert out["losses"] == 5
    assert out["strikeouts"] == 276
    assert out["outs"] == 601
    assert out["earned_runs"] == 64
    assert out["era"] == pytest.approx(2.88, abs=0.005)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_golden.py -v`
Expected: FAIL with `FileNotFoundError` (fixtures don't exist yet).

- [ ] **Step 3: Write the fixture generator and generate fixtures**

```python
# scripts/fetch_fixtures.py
"""One-off: fetch golden game logs from the live API and commit them as fixtures.

Run from repo root with the venv active:
    python scripts/fetch_fixtures.py
"""

import json
from pathlib import Path

from cardprice.stats_api import fetch_game_log

OUT = Path("tests/fixtures")
CASES = [
    (592450, "hitting", 2022, "judge_2022_hitting.json"),
    (543037, "pitching", 2018, "cole_2018_pitching.json"),
]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for mlb_id, group, season, name in CASES:
        splits = fetch_game_log(mlb_id, group, season)
        assert splits, f"empty game log for {mlb_id}/{group}/{season}"
        (OUT / name).write_text(
            json.dumps({"mlb_id": mlb_id, "season": season, "splits": splits}, indent=2)
        )
        print(f"wrote {OUT / name} ({len(splits)} games)")
```

Run: `python scripts/fetch_fixtures.py`
Expected: prints `wrote tests/fixtures/judge_2022_hitting.json (157 games)` and `wrote tests/fixtures/cole_2018_pitching.json (32 games)`. If a count differs, stop and investigate before proceeding — do not edit the golden values to match the data.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_golden.py -v`
Expected: 2 PASS. If `avg`/`slg` mismatch by 0.001, check rounding (official values are 3-dp rounded — the reconstruction rounds identically, so they should match exactly; a real mismatch means a parsing bug).

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_fixtures.py tests/fixtures tests/test_golden.py
git commit -m "test: golden fixtures prove reconstruction matches official totals"
```

---

### Task 7: Batch collector CLI + parquet output

**Files:**
- Create: `src/cardprice/collect_stats.py`
- Test: `tests/test_collect_stats.py`

**Interfaces:**
- Consumes: `fetch_game_log`, `game_log_to_frame` (Task 3); `save_raw` (Task 2); `data/reference/players.csv` (Task 1).
- Produces:
  - `collect(players: pd.DataFrame, seasons: list[int], sleep_s: float = 0.3) -> pd.DataFrame` — for each player-season, fetches the game log for the group matching the player's `role` (`hitter`→`hitting`, `pitcher`→`pitching`), saves a raw snapshot under dataset `stats`, key `{mlb_id}_{group}_{season}`, and returns all game logs concatenated into one DataFrame.
  - CLI: `python -m cardprice.collect_stats --players data/reference/players.csv --seasons 2022 2023 2024 2025 2026 --out data/processed/game_logs.parquet`
  - Output parquet schema: columns from `game_log_to_frame` (`mlb_id, group, season, date, <stat fields...>`).
  - Plan 3's feature builder reads this parquet.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_stats.py
import pandas as pd
import pytest

from cardprice import collect_stats, stats_api, storage

SPLITS = [
    {"date": "2022-04-08", "stat": {"gamesPlayed": 1, "atBats": 4, "hits": 1}},
    {"date": "2022-04-10", "stat": {"gamesPlayed": 1, "atBats": 3, "hits": 2}},
]


@pytest.fixture
def players():
    return pd.DataFrame(
        [
            {"mlb_id": 1, "name": "Test Hitter", "role": "hitter"},
            {"mlb_id": 2, "name": "Test Pitcher", "role": "pitcher"},
        ]
    )


def test_collect_fetches_role_group_and_saves_raw(players, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)
    calls = []

    def fake_fetch(mlb_id, group, season):
        calls.append((mlb_id, group, season))
        return SPLITS

    monkeypatch.setattr(stats_api, "fetch_game_log", fake_fetch)
    monkeypatch.setattr(collect_stats, "fetch_game_log", fake_fetch)

    df = collect_stats.collect(players, [2022], sleep_s=0)
    assert (1, "hitting", 2022) in calls
    assert (2, "pitching", 2022) in calls
    assert len(calls) == 2  # role gates the group; no wasted calls
    assert len(df) == 4  # 2 players x 2 games
    assert storage.load_latest("stats", "1_hitting_2022") == {"splits": SPLITS}


def test_collect_handles_empty_log(players, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(collect_stats, "fetch_game_log", lambda *a: [])
    df = collect_stats.collect(players, [2022], sleep_s=0)
    assert len(df) == 0
    assert storage.load_latest("stats", "1_hitting_2022") == {"splits": []}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collect_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.collect_stats'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cardprice/collect_stats.py
"""Batch game-log collector: players.csv x seasons -> raw snapshots + parquet."""

import argparse
import time

import pandas as pd

from cardprice.stats_api import fetch_game_log, game_log_to_frame
from cardprice.storage import save_raw

ROLE_GROUP = {"hitter": "hitting", "pitcher": "pitching"}


def collect(players: pd.DataFrame, seasons: list[int], sleep_s: float = 0.3) -> pd.DataFrame:
    frames = []
    for player in players.itertuples():
        group = ROLE_GROUP[player.role]
        for season in seasons:
            splits = fetch_game_log(int(player.mlb_id), group, season)
            save_raw("stats", f"{player.mlb_id}_{group}_{season}", {"splits": splits})
            if splits:
                frames.append(game_log_to_frame(splits, int(player.mlb_id), group, season))
            time.sleep(sleep_s)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", default="data/reference/players.csv")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--out", default="data/processed/game_logs.parquet")
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    players = pd.read_csv(args.players)
    df = collect(players, args.seasons, sleep_s=args.sleep)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} game rows to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_collect_stats.py -v`
Expected: 2 PASS. Then the full offline suite:
Run: `python -m pytest -v`
Expected: all PASS (live test excluded).

Then do one real collection run (network):
```bash
python -m cardprice.collect_stats --seasons 2022 2023 2024 2025 2026
```
Expected: `wrote ~2000 game rows to data/processed/game_logs.parquet` (4 players × ~32–162 games × 5 seasons; Judge/Henderson hitting, Cole/Nola pitching). Verify snapshots exist: `ls data/raw/stats | head` should show entries like `592450_hitting_2022`.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/collect_stats.py tests/test_collect_stats.py
git commit -m "feat: batch collector CLI writing parquet game logs"
```

---

## Done criteria for this plan

- `python -m pytest -v` — all offline tests PASS; `ruff check src tests scripts` clean.
- `python -m pytest -m live -v` — live API test PASS (network required).
- `data/processed/game_logs.parquet` exists with game logs for the 4 reference players, 2022–2026.
- Golden tests prove season-to-date reconstruction matches official totals (Judge 2022, Cole 2018).
- Next: Plan 2 (price pipeline).
