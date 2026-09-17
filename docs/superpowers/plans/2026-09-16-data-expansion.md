# P6a: Data Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the data foundation — minor-league game logs (2011+, per level), MLB game logs back to 2015, a 39-player universe across 2015–2025 rookie classes, flagship + Bowman 1st card catalog, and price collection for the expanded universe with ungraded as the primary series.

**Architecture:** Thin additions to proven machinery: `stats_api.py` gains a `sport_id` parameter (backwards compatible); a new universe catalog + resolver script reuses the proven console-listing resolution; `collect_stats.py`/`collect_prices.py` run unchanged over bigger inputs; `liquidity.py` gains an ungraded series treatment.

**Tech Stack:** Python 3.11+, existing `cardprice` package. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-16-multiyear-minors-design.md`
**Predecessors:** Plans 1-5 merged. Pipelines verified: stats (golden-tested), prices (SCP via Playwright), resolver (no-guess rule, parallel exclusion).

## Facts established (verified live 2026-09-15/16 — code below is built on them)

- Minor-league game logs: `GET /api/v1/people/{id}/stats?stats=gameLog&group=hitting&season={y}&sportId={11|12|13|14}` — AAA=11, AA=12, A+=13, A=14. Verified: Trout 2011 AA = 91 G; Bryant 2014 AAA = 45 G; Henderson 2022 AAA = 65 G. Plural `sportIds=` returns PARTIAL data — never use it; one call per level.
- MLB gameLog covers 2015+: Bryant 2015 = 151 G (corrected during Task 3 execution from an erroneous 145 probe — snapshot splits=151, gamesPlayed sums to 151, matches his real 2015 season), Judge 2016 = 27 G. So game logs are uniform for everything ≥2015 MLB / ≥2011 minors; NO season-line fallback needed.
- SCP pre-2022 card chart history starts 2021-03 (Soto 2018 Update: all buckets incl. ungraded, 67 monthly points to 2026-09).
- Console listing pattern: `https://www.sportscardspro.com/console/{set_slug}?rookies-only=true&exclude-variants=true` (flagship sets); 2018 anchors verified live (`juan-soto-hmt55`, `ronald-acuna-jr-193`).
- Bowman 1st cards live in `baseball-cards-<year>-bowman-chrome` (and `-bowman-draft`) sets; they are NOT rookies-only flagged — resolution scans console listings for the player name across candidate years (rookie_year−6 … rookie_year), earliest year wins, autograph/parallel anchors excluded.
- The sales parser's `grade` column is None for ungraded listings; the chart bucket `used` maps to `ungraded` (Task 4 wires ungraded into liquidity/panel as a first-class label).

## Global Constraints

- Python >= 3.11; free sources only; raw snapshots immutable (new dated files only).
- Politeness: ≥5s between SCP fetches; ≥0.3s between MLB Stats API calls; never plural sportIds.
- Tests offline by default; `live` marker for network tests.
- Lint: `ruff check` / `ruff format --check` only; never repo-wide `ruff format` (it rewrites markdown-embedded code in plan docs).
- The no-guess rule for card URLs: unresolved stays empty; never a guessed or parallel/parallel-fallback URL.
- Worktree data note: `data/raw/` and `data/processed/` are gitignored. At merge time, copy any new parquets/snapshots back to the main checkout (they persist there).

---

### Task 1: Stats API — minor-league levels

**Files:**
- Modify: `src/cardprice/stats_api.py`
- Test: `tests/test_stats_api_minors.py`

**Interfaces:**
- Consumes: existing `fetch_game_log`.
- Produces (Tasks 3-4 consume):
  - `fetch_game_log(mlb_id: int, group: str, season: int, sport_id: int | None = None) -> list[dict]` — `sport_id=None` = MLB (unchanged behavior); otherwise adds `sportId` to params.
  - `MINOR_LEAGUE_LEVELS = {11: "aaa", 12: "aa", 13: "a_plus", 14: "a"}` (module constant).
  - `fetch_minor_league_logs(mlb_id: int, group: str, season: int) -> dict[str, list[dict]]` — one call per level, returns `{level_name: splits}` (levels with empty splits omitted); sleeps 0.3s between calls.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stats_api_minors.py
import pytest

from cardprice import stats_api


def test_fetch_game_log_sport_id_param(monkeypatch):
    calls = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"stats": [{"splits": [{"date": "2011-07-01", "stat": {"gamesPlayed": 1}}]}]}

    def fake_get(url, params=None, timeout=None):
        calls["params"] = params
        return FakeResp()

    monkeypatch.setattr(stats_api.requests, "get", fake_get)
    out = stats_api.fetch_game_log(545361, "hitting", 2011, sport_id=12)
    assert calls["params"]["sportId"] == 12
    assert len(out) == 1
    # None = MLB, param absent entirely (backwards compatible)
    stats_api.fetch_game_log(545361, "hitting", 2016)
    assert "sportId" not in calls["params"]


def test_fetch_minor_league_logs_omits_empty_levels(monkeypatch):
    responses = {11: [], 12: [{"date": "2011-07-01", "stat": {}}], 13: [], 14: []}

    def fake_fetch(mlb_id, group, season, sport_id=None):
        return responses[sport_id]

    monkeypatch.setattr(stats_api, "fetch_game_log", fake_fetch)
    out = stats_api.fetch_minor_league_logs(545361, "hitting", 2011)
    assert list(out) == ["aa"]


@pytest.mark.live
def test_minors_live_trout_2011_aa():
    splits = stats_api.fetch_game_log(545361, "hitting", 2011, sport_id=12)
    assert len(splits) == 91  # Trout's 2011 Arkansas (AA) season


@pytest.mark.live
def test_minors_live_henderson_2022_aaa():
    splits = stats_api.fetch_game_log(683002, "hitting", 2022, sport_id=11)
    assert len(splits) == 65  # Henderson's 2022 Norfolk (AAA) stint
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stats_api_minors.py -v`
Expected: FAIL — `fetch_game_log() got an unexpected keyword argument 'sport_id'` / no `fetch_minor_league_logs`.

- [ ] **Step 3: Implement**

In `src/cardprice/stats_api.py`:

```python
MINOR_LEAGUE_LEVELS = {11: "aaa", 12: "aa", 13: "a_plus", 14: "a"}


def fetch_game_log(mlb_id: int, group: str, season: int, sport_id: int | None = None) -> list[dict]:
    params = {"stats": "gameLog", "group": group, "season": season}
    if sport_id is not None:
        params["sportId"] = sport_id
    resp = requests.get(f"{BASE}/people/{mlb_id}/stats", params=params, timeout=30)
    resp.raise_for_status()
    stats = resp.json().get("stats", [])
    if not stats:
        return []
    return stats[0].get("splits", [])


def fetch_minor_league_logs(mlb_id: int, group: str, season: int) -> dict[str, list[dict]]:
    out = {}
    for sport_id, level in MINOR_LEAGUE_LEVELS.items():
        splits = fetch_game_log(mlb_id, group, season, sport_id=sport_id)
        if splits:
            out[level] = splits
        time.sleep(0.3)
    return out
```

(Keep the original function's body identical otherwise; `import time` goes at the top of the module, not inline.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_stats_api_minors.py -v` — 2 PASS (offline). Live: `python -m pytest tests/test_stats_api_minors.py -v -m live` — 2 PASS (network). Full suite stays green (existing fetch_game_log callers unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/stats_api.py tests/test_stats_api_minors.py
git commit -m "feat: minor-league game log support (sportId levels)"
```

---

### Task 2: Universe catalog + set-family resolver

**Files:**
- Create: `data/reference/players_universe.csv` (hand-written, committed)
- Create: `scripts/resolve_universe.py`
- Test: `tests/test_resolve_universe.py`

**Interfaces:**
- Consumes: `fetch_page` (web.py), MLB search endpoint `/api/v1/people/search?names=...` (verified working), resolver patterns from `scripts/resolve_seed_cards.py` (word-boundary parallel exclusion, no-guess policy — reuse by import where practical, do not break the seed resolver).
- Produces (Tasks 3-4 consume):
  - `data/reference/players_universe.csv` — columns `player_name,role,rookie_year`; 39 rows (below).
  - `data/reference/cards_universe.csv` — columns `player_name,mlb_id,rookie_year,role,card_type,set_slug,scp_url`; `card_type` ∈ {`flagship`, `bowman_1st`}; unresolved cells empty.
  - `resolve_player_id(name: str) -> int | None` and `resolve_player_cards(name, rookie_year) -> dict[str, str]` (keys `flagship` / `bowman_1st`, values URLs) in the script — unit-tested offline against the committed Henderson fixture + synthetic anchor HTML.

The universe (rookie_year = flagship RC year):

```csv
player_name,role,rookie_year
Kris Bryant,hitter,2015
Carlos Correa,hitter,2015
Francisco Lindor,hitter,2015
Corey Seager,hitter,2016
Trea Turner,hitter,2016
Aaron Judge,hitter,2017
Cody Bellinger,hitter,2017
Alex Bregman,hitter,2017
Juan Soto,hitter,2018
Ronald Acuna Jr.,hitter,2018
Walker Buehler,pitcher,2018
Fernando Tatis Jr.,hitter,2019
Pete Alonso,hitter,2019
Vladimir Guerrero Jr.,hitter,2019
Yordan Alvarez,hitter,2019
Kyle Lewis,hitter,2020
Randy Arozarena,hitter,2020
Devin Williams,pitcher,2020
Jonathan India,hitter,2021
Adolis Garcia,hitter,2021
Wander Franco,hitter,2021
Jarred Kelenic,hitter,2021
Bobby Witt Jr,hitter,2022
Julio Rodriguez,hitter,2022
Adley Rutschman,hitter,2022
Spencer Strider,pitcher,2022
Michael Harris II,hitter,2022
Gunnar Henderson,hitter,2023
Corbin Carroll,hitter,2023
Jordan Walker,hitter,2023
Anthony Volpe,hitter,2023
Jackson Chourio,hitter,2024
Paul Skenes,pitcher,2024
Wyatt Langford,hitter,2024
Jackson Merrill,hitter,2024
Nick Kurtz,hitter,2025
Jacob Wilson,hitter,2025
Roki Sasaki,pitcher,2025
Roman Anthony,hitter,2025
```

(Franco/Kelenic/Walker are deliberate bust variance — do not "fix" the list by dropping them.)

Resolution rules (in `resolve_universe.py`, documented in code):
1. **mlb_id:** `/api/v1/people/search?names={name}` → exact case/accent-insensitive fullName match; validate via non-empty `fetch_game_log(mlb_id, ROLE_GROUP[role], rookie_year)`. Name normalization handles "Acuna/Acuña", "Garcia/García", "Jr."/"II" suffixes (strip punctuation, fold case).
2. **flagship:** console listings `baseball-cards-{rookie_year}-topps-chrome` AND `-topps-chrome-update` with `?rookies-only=true&exclude-variants=true`; name-matched base anchor (`/game/<set>/...`), word-boundary parallel exclusion, no fallback to parallels; if both sets yield a match, prefer the plain `topps-chrome` one. **Fallback (added during execution, evidence-backed):** SCP's rookies-only tagging has gaps (4 universe players have clean base RCs the rookies-only flag hides — verified: Bryant 2015 #112, Correa 2015 Update #US174, Bellinger 2017 #79, A. Garcia 2021 Update #USC64). If the rookies-only listing yields no match on a rule-2 set, retry that SAME set's full console listing with `?exclude-variants=true` and identical name/parallel discipline; record which listing (rookies-only vs full) produced the match in the audit trail. The fallback never leaves the two rule-2 sets and never relaxes variant exclusion. (Known: 2022/2024/2025 update-set players — Rutschman, Strider, Skenes, Kurtz, Anthony — resolve from update sets; the spike confirmed this mapping.)
3. **bowman_1st:** for years `rookie_year-6 .. rookie_year`, console listing `baseball-cards-{y}-bowman-chrome` (plain, with `?exclude-variants=true`); name match, exclude anchors containing (word-boundary) `Autograph|Auto|Refractor|Shimmer|Gold|Orange|Purple|Blue|Green|Red|Black|Superfractor|Variation|Wave|Sparkle|Speckle|Mojo|Atomic|Lunar|Rookie of the Year Favorites|Talent Pipeline` (insert sets excluded — a Talent Pipeline anchor was caught mis-resolving as Wander Franco's 1st during execution); earliest matching year wins; if the anchor text contains "1st" prefer it within that year. Record ALL candidate (year, anchor, url) triples in the report for manual audit.
4. ≥5s before every uncached SCP fetch; console pages cached per set in-process; politeness identical to Plan 2's resolver.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolve_universe.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from resolve_universe import _name_matches, norm_name, pick_bowman_1st, pick_flagship


def test_norm_name_handles_accents_and_suffixes():
    assert norm_name("Ronald Acuna Jr.") == norm_name("Ronald Acuña Jr")
    assert norm_name("Michael Harris II") == norm_name("Michael Harris II")


def test_name_matches():
    assert _name_matches("Ronald Acuna Jr.", "Ronald Acuna Jr. #193")
    assert _name_matches("Michael Harris II", "Michael Harris II #251")
    assert not _name_matches("Bobby Witt Jr", "Bobby Witt III #99")
    assert not _name_matches("Nick Kurtz", "Nick Kurtzney #1")


def test_pick_flagship_prefers_base_over_parallel():
    anchors = [
        ("Aaron Judge [Refractor] #99", "/game/set/aaron-judge-refractor-99"),
        ("Aaron Judge #99", "/game/set/aaron-judge-99"),
    ]
    assert pick_flagship("Aaron Judge", anchors) == "/game/set/aaron-judge-99"
    assert pick_flagship("Aaron Judge", [anchors[0]]) is None  # never a parallel


def test_pick_bowman_1st_earliest_year_no_autos():
    candidates = [
        (2019, "Juan Soto #BCP52", "/game/baseball-cards-2019-bowman-chrome/juan-soto-bcp52"),
        (2016, "Juan Soto #BCP52", "/game/baseball-cards-2016-bowman-chrome/juan-soto-bcp52"),
        (2016, "Juan Soto [Autograph] #BCP52", "/game/x"),
    ]
    url, audit = pick_bowman_1st("Juan Soto", candidates)
    assert url == "/game/baseball-cards-2016-bowman-chrome/juan-soto-bcp52"
    assert len(audit) >= 2  # audit trail keeps candidates
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_resolve_universe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'resolve_universe'`.

- [ ] **Step 3: Implement resolve_universe.py**

Adapt the proven machinery from `scripts/resolve_seed_cards.py` (read it first): same `norm_name`/`pick_anchor` word-boundary discipline, extended to set families. Provide the four functions the tests import with exactly those signatures, plus `main()` that: reads `players_universe.csv` → resolves mlb_id → resolves flagship + bowman_1st → writes `cards_universe.csv` → prints a resolution summary and the full audit trail for bowman_1st picks.

- [ ] **Step 4: Run tests, then the resolver (network, ~15-25 min of polite fetching)**

Run: `python -m pytest tests/test_resolve_universe.py -v` — 4 PASS.
Then: `PLAYWRIGHT_BROWSERS_PATH=.pw-browsers python scripts/resolve_universe.py`
Expected: mlb_id 39/39 (validate via game logs); flagship ≥30/39 (the 5 known update-set players — Rutschman, Strider, Skenes, Kurtz, Anthony — resolve from update sets; 2026-era gaps may leave a few more); bowman_1st ≥25/39 (pitchers and international free agents have Bowman 1st; some older players may only have Bowman Draft). Investigate any flagship miss before accepting; record bowman audit trail in your report.

- [ ] **Step 5: Commit**

```bash
git add data/reference/players_universe.csv data/reference/cards_universe.csv scripts/resolve_universe.py tests/test_resolve_universe.py
git commit -m "feat: 39-player universe catalog + set-family resolver"
```

---

### Task 3: Stats collection for the universe (minors + MLB backfill)

**Files:**
- Create: `scripts/collect_universe_stats.py`
- Test: `tests/test_collect_universe_stats.py`

**Interfaces:**
- Consumes: `fetch_game_log`, `fetch_minor_league_logs`, `MINOR_LEAGUE_LEVELS` (Task 1); `cards_universe.csv` (Task 2 — the player table is derived from it by deduping `player_name,mlb_id,role`; `players_universe.csv` has NO mlb_id, so it cannot drive collection); `save_raw`, `game_log_to_frame` (Plan 1).
- Produces (P6b consumes):
  - `collect_universe_stats(players: pd.DataFrame, mlb_seasons, minor_seasons, sleep_s=0.3) -> pd.DataFrame` — per player: MLB game logs for `mlb_seasons` (2015–2026), minors game logs per level for `minor_seasons` (2011–2026); raw snapshots under dataset `stats`, keys `{mlb_id}_{group}_{season}` (MLB, unchanged — dedupes against Plan 1/3 snapshots via storage idempotency) and `{mlb_id}_{group}_{season}_{level}` (minors); returns one concatenated DataFrame with a `level` column (`"mlb"` / `"aaa"` / …).
  - CLI writes `data/processed/game_logs_universe.parquet`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_universe_stats.py
import sys
from pathlib import Path

import pandas as pd

from cardprice import stats_api, storage

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from collect_universe_stats import collect_universe_stats


def test_collect_writes_levels_and_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)

    def fake_game_log(mlb_id, group, season, sport_id=None):
        return [{"date": f"{season}-07-01", "stat": {"gamesPlayed": 1, "homeRuns": 2}}]

    def fake_minors(mlb_id, group, season):
        return {"aaa": [{"date": f"{season}-06-01", "stat": {"gamesPlayed": 1, "homeRuns": 1}}]} if season == 2021 else {}

    monkeypatch.setattr(stats_api, "fetch_game_log", fake_game_log)
    monkeypatch.setattr(stats_api, "fetch_minor_league_logs", fake_minors)
    import collect_universe_stats as collect_mod  # aliased: bare `import` would shadow the function above
    monkeypatch.setattr(collect_mod, "fetch_game_log", fake_game_log)
    monkeypatch.setattr(collect_mod, "fetch_minor_league_logs", fake_minors)

    players = pd.DataFrame([{"mlb_id": 1, "name": "Test Player", "role": "hitter"}])
    df = collect_universe_stats(players, mlb_seasons=[2021, 2022], minor_seasons=[2021], sleep_s=0)
    assert set(df["level"]) == {"mlb", "aaa"}
    assert len(df) == 3
    assert storage.load_latest("stats", "1_hitting_2021")["splits"]
    assert storage.load_latest("stats", "1_hitting_2021_aaa")["splits"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collect_universe_stats.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement collect_universe_stats.py**

```python
# scripts/collect_universe_stats.py
"""Collect MLB (2015+) + minor-league (2011+, per level) game logs for the universe."""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.stats_api import (  # noqa: E402
    fetch_game_log,
    fetch_minor_league_logs,
    game_log_to_frame,
)
from cardprice.storage import save_raw  # noqa: E402

ROLE_GROUP = {"hitter": "hitting", "pitcher": "pitching"}


def collect_universe_stats(
    players: pd.DataFrame,
    mlb_seasons: list[int],
    minor_seasons: list[int],
    sleep_s: float = 0.3,
) -> pd.DataFrame:
    frames = []
    for player in players.itertuples():
        group = ROLE_GROUP[player.role]
        for season in mlb_seasons:
            splits = fetch_game_log(int(player.mlb_id), group, season)
            save_raw("stats", f"{player.mlb_id}_{group}_{season}", {"splits": splits})
            if splits:
                frames.append(
                    game_log_to_frame(splits, int(player.mlb_id), group, season).assign(level="mlb")
                )
            time.sleep(sleep_s)
        for season in minor_seasons:
            logs = fetch_minor_league_logs(int(player.mlb_id), group, season)
            for level, splits in logs.items():
                save_raw("stats", f"{player.mlb_id}_{group}_{season}_{level}", {"splits": splits})
                frames.append(
                    game_log_to_frame(splits, int(player.mlb_id), group, season).assign(level=level)
                )
            time.sleep(sleep_s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", default="data/reference/cards_universe.csv")
    parser.add_argument("--out", default="data/processed/game_logs_universe.parquet")
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    cards = pd.read_csv(args.players)
    players = (
        cards[["player_name", "mlb_id", "role"]]
        .dropna(subset=["mlb_id"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df = collect_universe_stats(
        players,
        mlb_seasons=list(range(2015, 2027)),
        minor_seasons=list(range(2011, 2027)),
        sleep_s=args.sleep,
    )
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} game rows ({df['mlb_id'].nunique()} players) to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then the real collection (network, ~15-25 min)**

Run: `python -m pytest tests/test_collect_universe_stats.py -v` — 1 PASS.
Then: `python scripts/collect_universe_stats.py` — expect roughly 25-40k game rows across 39 players (actual: 35,152; the 12-20k estimate was low). Verify goldens inline after the run: Henderson 2022 AAA = 65 rows, Trout not in universe (skip), Soto 2017 A = 23 rows, Bryant 2015 MLB = 151 rows (corrected from 145 — see Facts). Report row counts per level.

- [ ] **Step 5: Commit**

```bash
git add scripts/collect_universe_stats.py tests/test_collect_universe_stats.py
git commit -m "feat: universe stats collection (minors + MLB backfill)"
```

---

### Task 4: Price collection for the expanded universe (ungraded-primary)

**Files:**
- Modify: `src/cardprice/weekly.py`, `src/cardprice/liquidity.py` (ungraded as first-class label)
- Test: `tests/test_ungraded_series.py`

**Interfaces:**
- Consumes: cards_universe.csv (Task 2), existing collectors.
- Produces (Task 5, P6b consume):
  - Sales/chart frames where the ungraded series appears as `grade == "ungraded"` (not None): `weekly_price_series` and `liquidity_report` treat it as a label. Chart parser already emits `ungraded` (calibration labels come from the price-summary header text — "Ungraded" → `ungraded`; verified in the 2022 Topps Chrome snapshots). Sales parser emits None for raw titles — add a normalization step: `normalize_grade(grade) -> str` mapping None/NaN → "ungraded", applied at the weekly/liquidity layer (do NOT touch scp_parse's contract — downstream relabeling).
  - `PANEL_SERIES = ("ungraded", "psa_9", "psa_10")` replacing `PANEL_GRADES` in liquidity.py.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ungraded_series.py
import pandas as pd

from cardprice.liquidity import liquidity_report
from cardprice.weekly import weekly_price_series


def make_sales():
    rows = [
        ("card/a", None, "2026-08-31", 5.0, False),
        ("card/a", None, "2026-09-02", 7.0, False),
        ("card/a", None, "2026-09-05", 6.0, False),
        ("card/a", "psa_10", "2026-08-31", 40.0, False),
        ("card/a", "psa_10", "2026-09-02", 50.0, False),
    ]
    df = pd.DataFrame(rows, columns=["card_slug", "grade", "sale_date", "price", "best_offer"])
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["title"] = "Test Card"  # contamination_audit (inside liquidity_report) requires it
    return df


def test_ungraded_is_first_class_series():
    out = weekly_price_series(make_sales(), min_sales=2)
    assert set(out["grade"]) == {"ungraded", "psa_10"}
    raw = out[out["grade"] == "ungraded"].iloc[0]
    assert raw["median_price"] == 6.0 and raw["n_sales"] == 3


def test_liquidity_report_includes_ungraded():
    sales = make_sales()
    chart = pd.DataFrame(
        {
            "card_slug": ["card/a"] * 20,
            "grade": ["ungraded"] * 20,
            "date": pd.date_range("2025-01-01", periods=20, freq="MS"),
            "price": [5.0] * 20,
        }
    )
    rep = liquidity_report(sales, chart)
    assert "ungraded" in set(rep["grade"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ungraded_series.py -v`
Expected: FAIL — weekly drops null grades today (`assert set(...) == {"psa_10"}` path), liquidity filters to psa_9/psa_10 only.

- [ ] **Step 3: Implement**

In `src/cardprice/weekly.py`:

```python
def normalize_grade(grade) -> str:
    if grade is None or (isinstance(grade, float) and pd.isna(grade)) or grade == "":
        return "ungraded"
    return str(grade)


def weekly_price_series(sales: pd.DataFrame, min_sales: int = 2) -> pd.DataFrame:
    df = sales.copy()
    df["grade"] = df["grade"].map(normalize_grade)
    df["week"] = df["sale_date"].dt.to_period("W-SUN").dt.start_time
    grouped = (
        df.groupby(["card_slug", "grade", "week"])
        .agg(
            median_price=("price", "median"),
            n_sales=("price", "size"),
            best_offer_share=("best_offer", "mean"),
        )
        .reset_index()
    )
    return grouped[grouped["n_sales"] >= min_sales].reset_index(drop=True)
```

(The `.notna()` filter is gone — replaced by normalization.)

In `src/cardprice/liquidity.py`: replace `PANEL_GRADES = ("psa_9", "psa_10")` with `PANEL_SERIES = ("ungraded", "psa_9", "psa_10")` and use `sales["grade"].map(normalize_grade)` (import from weekly) before the `isin(PANEL_SERIES)` filter. Keep the old name as an alias (`PANEL_GRADES = PANEL_SERIES`) so Plan 3-era imports don't break; note it as deprecated in a comment.

- [ ] **Step 4: Run tests, then the collection (network, ~10-15 min)**

Run: `python -m pytest tests/test_ungraded_series.py -v` — 2 PASS; full suite green (watch for Plan 2/3 tests that assumed null-grade dropping — if any break, they encode the OLD contract; update them per the new contract and note it).
Then:
```bash
PLAYWRIGHT_BROWSERS_PATH=.pw-browsers python -m cardprice.collect_prices --cards data/reference/cards_universe.csv --sales-out data/processed/universe_sales.parquet --chart-out data/processed/universe_chart_monthly.parquet
```
NOTE: `collect_prices` reads `player_name,mlb_id,rookie_year,set_slug,scp_url` — cards_universe.csv has those plus `card_type`/`role`; verify the collector tolerates extra columns (it should — it reads named columns via itertuples; if `card_type` needs carrying into output meta, add it to META_COLS and note the change). Expect ~70-90 card pages, ≥80% OK. Then `python scripts/reparse_snapshots.py`-equivalent isn't needed (fresh snapshots); do NOT run run_price_pipeline.py — Task 5 does that against the universe parquets.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/weekly.py src/cardprice/liquidity.py tests/test_ungraded_series.py src/cardprice/collect_prices.py
git commit -m "feat: ungraded as first-class series; universe price collection"
```

---

### Task 5: Universe liquidity report + final card selection

**Files:**
- Create: `scripts/run_universe_liquidity.py`
- Create: `data/reference/cards_modeling.csv` (generated, committed)
- Test: `tests/test_run_universe_liquidity.py`

**Interfaces:**
- Consumes: universe parquets (Task 4), liquidity.py (PANEL_SERIES).
- Produces (P6b consumes): `data/reference/cards_modeling.csv` — the cards that pass liquidity, one row per (card, series), columns `card_slug, player_name, mlb_id, rookie_year, card_type, grade, sales_per_week, n_chart_points, chart_first, chart_last`. Rule: `sales_per_week >= 0.3` (recent window) OR `n_chart_points >= 36` (≥3 years of monthly history). Documented in code.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_universe_liquidity.py
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_universe_liquidity import select_modeling_cards


def test_selection_rule():
    report = pd.DataFrame(
        [
            # deep history, thin recent sales -> kept (chart rule)
            {"card_slug": "card/old", "grade": "ungraded", "sales_per_week": 0.1, "n_chart_points": 60},
            # thin history, active recent sales -> kept (sales rule)
            {"card_slug": "card/new", "grade": "psa_10", "sales_per_week": 1.5, "n_chart_points": 10},
            # neither -> dropped
            {"card_slug": "card/thin", "grade": "ungraded", "sales_per_week": 0.1, "n_chart_points": 10},
        ]
    )
    out = select_modeling_cards(report)
    assert set(out["card_slug"]) == {"card/old", "card/new"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_universe_liquidity.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement run_universe_liquidity.py**

```python
# scripts/run_universe_liquidity.py
"""Liquidity report over the universe parquets + modeling-card selection."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.liquidity import liquidity_report  # noqa: E402
from cardprice.weekly import normalize_grade  # noqa: E402

MIN_SALES_PER_WEEK = 0.3
MIN_CHART_POINTS = 36


def select_modeling_cards(report: pd.DataFrame) -> pd.DataFrame:
    mask = (report["sales_per_week"] >= MIN_SALES_PER_WEEK) | (
        report["n_chart_points"] >= MIN_CHART_POINTS
    )
    return report[mask].reset_index(drop=True)


if __name__ == "__main__":
    sales = pd.read_parquet("data/processed/universe_sales.parquet")
    chart = pd.read_parquet("data/processed/universe_chart_monthly.parquet")
    # drop uncalibrated chart keys (e.g. `key:cib`) — same filter reparse_snapshots.py applies
    chart = chart[~chart["grade"].astype(str).str.startswith("key:")]
    sales["grade"] = sales["grade"].map(normalize_grade)
    report = liquidity_report(sales, chart)
    report.to_csv("data/processed/universe_liquidity.csv", index=False)
    selected = select_modeling_cards(report)
    selected.to_csv("data/reference/cards_modeling.csv", index=False)
    print(f"universe liquidity: {len(report)} card-series rows; selected {len(selected)}")
    print(selected.groupby(["card_type" if "card_type" in selected else "grade"]).size())
```

(The report lacks `card_type`/`player_name` — liquidity_report doesn't join meta. Derive `card_slug` from cards_universe.csv's `scp_url` with `collect_prices.card_slug()`, then left-join `player_name, mlb_id, rookie_year, card_type` onto the report before writing cards_modeling.csv. universe_liquidity.csv can stay report-only.)

- [ ] **Step 4: Run it + review the selection**

Run: `python scripts/run_universe_liquidity.py` — then EYEBALL `cards_modeling.csv`: does the selected universe cover all 4 decades of rookie classes, both card types, both raw and graded? Note the coverage table (class year × card_type × grade counts) in your report. If a whole class or card type is missing, flag it — that's a data problem to surface, not to patch silently.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_universe_liquidity.py tests/test_run_universe_liquidity.py data/reference/cards_modeling.csv
git commit -m "feat: universe liquidity report and modeling card selection"
```

---

## Done criteria for this plan

- `python -m pytest -v` all offline tests PASS; ruff clean; live suite PASS (`-m live`: 2 new minors tests + the 4 existing live tests = 6).
- `data/processed/game_logs_universe.parquet` — MLB 2015-2026 + minors 2011-2026, 39 players, level-tagged; golden counts verified (Henderson AAA 65, Bryant MLB 2015 151, Soto A 2017 23).
- `data/processed/universe_sales.parquet` + `universe_chart_monthly.parquet` — ≥60 card pages collected; ungraded series first-class.
- `data/reference/cards_modeling.csv` — the selected modeling universe with a documented coverage table.
- Misses documented honestly (unresolved cards, thin classes) in the final report, not patched.
- Next: P6b (multi-year panel builder: career-to-date features, career_stage, hold-return outcomes).
