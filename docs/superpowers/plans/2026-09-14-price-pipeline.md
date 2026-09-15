# Price Pipeline (SportsCardsPro) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested pipeline that collects graded-rookie card sales and price history from SportsCardsPro (SCP) into immutable snapshots, parses them into sales/monthly/weekly series, validates data quality, and produces a liquidity report that selects the modeling universe.

**Architecture:** Playwright-based browser fetcher (Cloudflare-tolerant, verified in spike) → raw HTML snapshots via Plan 1's `storage.save_raw` → BeautifulSoup parsers (sales tables, attributes, price summary, chart JSON) → parquet outputs → weekly series builder + validation + liquidity report. Pure parsers tested offline against committed HTML fixtures.

**Tech Stack:** Python 3.11+, Plan 1 codebase (`cardprice` package), + new deps: `playwright`, `beautifulsoup4`. No database.

**Spec:** `docs/superpowers/specs/2026-09-14-mlb-card-price-panel-design.md`
**Predecessor:** Plan 1 (`docs/superpowers/plans/2026-09-14-stats-pipeline.md`) — merged; provides `storage.save_raw/load_latest`, `stats_api.fetch_game_log/game_log_to_frame`, players reference.

## Verified spike findings (2026-09-14 — these are facts, code below is built on them)

- Plain `curl`/`requests` to sportscardspro.com → Cloudflare 403 challenge. **Playwright headless Chromium passes SCP's challenge automatically** (wait ~8s, title becomes the real page title). 130point.com's challenge does NOT auto-resolve headless → **130point is dropped from this plan** (see Deviations).
- Card page URL: `https://www.sportscardspro.com/game/<set-slug>/<player>-<card-number>` e.g. `baseball-cards-2023-topps-chrome/gunnar-henderson-2`.
- Card page HTML (~650 KB) contains:
  - `table#price_data` — per-grade current-value summary; header cells like `Ungraded, Grade 7, Grade 8, Grade 9, Grade 9.5, PSA 10, ...`; prices as `$34.50`.
  - Multiple sold-listing tables, each inside `div` whose class starts with `completed-auctions-` (e.g. `completed-auctions-used`, `-graded`, `-grade-seventeen`); table header cells: `Sale Date`, `TW`, `Title`, `Price`, empty; ≤30 most recent rows per grade bucket; date format `2026-09-14`; price cell text either `$7.50` or dual `$35.00$40.00` (list price + accepted Best Offer — semantics to be pinned by the Task 1 spike doc).
  - `table#attribute` — rows like `Is Rookie Card: Yes`.
  - Pop-by-grade current-value table (last table; rows `PSA 10 | $34.50`, `BGS 10 | $45.00`, …).
  - Inline JS: `VGPC.chart_data = {"<key>":[[<ms-epoch>,<price-cents>],...], ...}` — monthly price history per grade-bucket key, back to ~tracking start (2019+). Key→grade mapping is NOT self-describing; resolve dynamically (Task 3).
- Grade per sale row comes from **parsing the listing title** (`PSA 10`, `BGS 9.5`, …), not from which table it sits in — the div-class buckets are PriceCharting-internal and not grade-labeled.

## Global Constraints

- Python >= 3.11; free data sources only; no paid APIs.
- Raw snapshots immutable (Plan 1 `storage` contract); HTML stored as `{"url": ..., "html": ...}` payload under dataset `prices_scp`, key = card page slug.
- Politeness: minimum **5 seconds** between SCP page fetches; max 3 challenge retries then raise.
- Playwright browsers installed project-local: `PLAYWRIGHT_BROWSERS_PATH=.pw-browsers` (gitignored).
- Tests offline by default (`@pytest.mark.live` for anything fetching SCP).
- Deviations from spec (acknowledged): 130point collector dropped (Cloudflare interactive challenge blocks headless; SCP sufficient for launch); auction-vs-BIN split unavailable per-sale from SCP — we capture a `best_offer` flag from dual-price cells instead; historical seasons use **monthly** chart series (SCP keeps only ~30 recent sales per grade); weekly transaction series accumulates going forward.

---

### Task 1: Browser fetch infrastructure + SCP structure spike + seed catalog

**Files:**
- Modify: `pyproject.toml` (add deps), `.gitignore` (add `.pw-browsers/`)
- Create: `src/cardprice/web.py`
- Test: `tests/test_web.py`
- Create: `data/reference/cards_seed.csv`
- Create: `scripts/resolve_seed_cards.py`
- Create: `docs/superpowers/spikes/2026-09-14-scp-structure.md`
- Create: `tests/fixtures/scp/henderson_2023_topps_chrome.html` (saved by spike, committed)

**Interfaces:**
- Produces (Tasks 2-4 consume):
  - `fetch_page(url: str, wait_ms: int = 8000, retries: int = 3) -> str` (web.py)
  - `ChallengeError(RuntimeError)` (web.py)
  - `is_challenge_title(title: str) -> bool` (web.py, pure, unit-tested)
  - `data/reference/cards_seed.csv` columns: `player_name,role,rookie_year,set_slug,mlb_id,scp_url` (last two filled by resolver script)
  - Spike doc answers, with verbatim HTML excerpts: (a) dual-price cell semantics (which span is list vs accepted price); (b) chart_data key list for the fixture page; (c) category listing URL scheme; (d) confirmed challenge wait/retry behavior.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web.py
import pytest

from cardprice import web


def test_is_challenge_title_detects_cloudflare():
    assert web.is_challenge_title("Just a moment...")
    assert web.is_challenge_title("Performing security verification")
    assert not web.is_challenge_title(
        "Gunnar Henderson #2 Prices [Rookie] | 2023 Topps Chrome | Baseball Cards"
    )


@pytest.mark.live
def test_fetch_page_real_card():
    html = web.fetch_page(
        "https://www.sportscardspro.com/game/baseball-cards-2023-topps-chrome/gunnar-henderson-2"
    )
    assert "VGPC.chart_data" in html
    assert "completed-auctions-" in html
    assert 'id="price_data"' in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.web'` (and playwright not installed yet).

- [ ] **Step 3: Implement web.py + deps**

Add to `pyproject.toml` dependencies: `"playwright>=1.45"`, `"beautifulsoup4>=4.12"`. Add `.pw-browsers/` to `.gitignore`.

```python
# src/cardprice/web.py
"""Browser-based page fetching via Playwright (tolerates Cloudflare auto-challenges)."""

import time

from playwright.sync_api import sync_playwright

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class ChallengeError(RuntimeError):
    """Bot challenge did not resolve after retries."""


def is_challenge_title(title: str) -> bool:
    t = title.lower()
    return "just a moment" in t or "security verification" in t


def fetch_page(url: str, wait_ms: int = 8000, retries: int = 3) -> str:
    """Return fully-rendered HTML for url. Raises ChallengeError if the
    bot challenge never resolves."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            ctx = browser.new_context(
                user_agent=USER_AGENT, viewport={"width": 1400, "height": 900}
            )
            page = ctx.new_page()
            for _ in range(retries):
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(wait_ms)
                if not is_challenge_title(page.title()):
                    return page.content()
                time.sleep(5)
            raise ChallengeError(url)
        finally:
            browser.close()
```

Install:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
PLAYWRIGHT_BROWSERS_PATH=.pw-browsers playwright install chromium
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_web.py -v` (offline: 1 pass, 1 deselected)
Run: `python -m pytest tests/test_web.py -v -m live` (network: 1 pass)

- [ ] **Step 5: Seed catalog + resolver + spike doc**

```csv
# data/reference/cards_seed.csv (as written before resolution; resolver fills mlb_id, scp_url)
player_name,role,rookie_year,set_slug,mlb_id,scp_url
Bobby Witt Jr,hitter,2022,baseball-cards-2022-topps-chrome,,
Julio Rodriguez,hitter,2022,baseball-cards-2022-topps-chrome,,
Adley Rutschman,hitter,2022,baseball-cards-2022-topps-chrome,,
Spencer Strider,pitcher,2022,baseball-cards-2022-topps-chrome,,
Gunnar Henderson,hitter,2023,baseball-cards-2023-topps-chrome,,
Corbin Carroll,hitter,2023,baseball-cards-2023-topps-chrome,,
Jordan Walker,hitter,2023,baseball-cards-2023-topps-chrome,,
Anthony Volpe,hitter,2023,baseball-cards-2023-topps-chrome,,
Jackson Chourio,hitter,2024,baseball-cards-2024-topps-chrome,,
Paul Skenes,pitcher,2024,baseball-cards-2024-topps-chrome,,
Wyatt Langford,hitter,2024,baseball-cards-2024-topps-chrome,,
Jackson Merrill,hitter,2024,baseball-cards-2024-topps-chrome,,
Nick Kurtz,hitter,2025,baseball-cards-2025-topps-chrome,,
Jacob Wilson,hitter,2025,baseball-cards-2025-topps-chrome,,
Roki Sasaki,pitcher,2025,baseball-cards-2025-topps-chrome,,
Roman Anthony,hitter,2025,baseball-cards-2025-topps-chrome,,
```

`scripts/resolve_seed_cards.py` — for each row:
1. **mlb_id:** GET `https://statsapi.mlb.com/api/v1/sports/1/players?season={rookie_year}` (one call per unique season, cache in-process), exact case-insensitive match on `fullName`; then validate `fetch_game_log(mlb_id, ROLE_GROUP[role], rookie_year)` returns non-empty (reuse `cardprice.stats_api`). Sleep 0.3s between API calls.
2. **scp_url:** fetch the SCP category listing page `https://www.sportscardspro.com/category/{set_slug}` via `fetch_page`, find the `<a>` whose text contains the player's last name + first name and whose href starts with `/game/{set_slug}/`, prefer the href whose anchor text does NOT contain parallel keywords (`Refractor`, `Autograph`, `Gold`, `Orange`, `Purple`, `Blue`, `Green`, `Red`, `Superfractor`, `Variation`); build absolute URL. If the category page 404s or no match, try SCP search `https://www.sportscardspro.com/search?q={player}+{year}+Topps+Chrome` with the same anchor rules. Sleep ≥5s between SCP fetches. If still unresolved, leave `scp_url` empty and print a warning — do not guess.
3. Rewrite `data/reference/cards_seed.csv` with filled columns; print a summary table.

Run it (network): `python scripts/resolve_seed_cards.py` — expect ≥14 of 16 rows resolved; investigate any unresolved.

Spike: fetch the resolved Henderson URL with `fetch_page`, save to `tests/fixtures/scp/henderson_2023_topps_chrome.html`. Write `docs/superpowers/spikes/2026-09-14-scp-structure.md` answering (a)-(d) from the Interfaces block, each with a short verbatim HTML excerpt as evidence. For (a), find a dual-price row in the fixture and document which child element holds the accepted price (inspect the `td` inner HTML — spans/classes). For (b), list the `VGPC.chart_data` keys and each key's last `[ts, cents]` pair. For (c), record the working category URL.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/cardprice/web.py tests/test_web.py data/reference/cards_seed.csv scripts/resolve_seed_cards.py docs/superpowers/spikes tests/fixtures/scp
git commit -m "feat: playwright fetch infra, seed card catalog, SCP structure spike"
```

---

### Task 2: SCP sales-table parser

**Files:**
- Create: `src/cardprice/scp_parse.py`
- Test: `tests/test_scp_parse.py`

**Interfaces:**
- Consumes: `tests/fixtures/scp/henderson_2023_topps_chrome.html` (Task 1); spike doc dual-price semantics.
- Produces (Tasks 3-7 consume):
  - `parse_grade_from_title(title: str) -> str | None` — returns lowercase like `"psa_10"`, `"bgs_9.5"`, `"sgc_9"`; `None` if no grader+grade pattern. Regex: `\b(PSA|BGS|SGC|CGC)\s*(10|9\.5|9|8\.5|8)\b` case-insensitive; also map `GEM MT|GEM MINT` + PSA → psa_10 only when "PSA" also present (GEM MT alone is ambiguous — return None).
  - `parse_sales_tables(html: str) -> pd.DataFrame` — columns: `sale_date (datetime64), title (str), price (float), list_price (float | NA), best_offer (bool), grade (str | None), bucket (str)`. One row per sold listing across all `div[class^="completed-auctions-"]`; `bucket` = the div's class suffix.
  - `parse_attributes(html: str) -> dict` — from `table#attribute`: `{"is_rookie_card": bool, ...}` (keys snake_cased from row labels).
  - `parse_price_summary(html: str) -> dict[str, float]` — `table#price_data` header grade labels → current value dollars.
  - `parse_chart_data(html: str) -> dict[str, list[tuple[pd.Timestamp, float]]]` — from `VGPC.chart_data`; ms-epoch→`pd.Timestamp` (UTC, date precision), cents→dollars.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scp_parse.py
from pathlib import Path

import pandas as pd
import pytest

from cardprice import scp_parse

FIXTURE = (
    Path(__file__).parent / "fixtures" / "scp" / "henderson_2023_topps_chrome.html"
).read_text()


def test_parse_grade_from_title():
    assert scp_parse.parse_grade_from_title("2023 Topps Chrome Gunnar Henderson RC PSA 10 GEM MT") == "psa_10"
    assert scp_parse.parse_grade_from_title("GUNNAR HENDERSON 2023 TOPPS CHROME PSA 9") == "psa_9"
    assert scp_parse.parse_grade_from_title("2023 Topps Chrome Henderson BGS 9.5 GEM MINT") == "bgs_9.5"
    assert scp_parse.parse_grade_from_title("2023 Topps Chrome Gunnar Henderson #2") is None
    assert scp_parse.parse_grade_from_title("Henderson RC GEM MT") is None  # ambiguous without grader


def test_parse_sales_tables_rows():
    df = scp_parse.parse_sales_tables(FIXTURE)
    assert len(df) >= 100                      # fixture has ~9 grade buckets x <=30 rows
    assert set(df.columns) == {"sale_date", "title", "price", "list_price", "best_offer", "grade", "bucket"}
    assert df["sale_date"].notna().all()
    assert (df["price"] > 0).all()
    grades_seen = {"psa_10", "psa_9", "psa_8.5", "psa_8", "bgs_10", "bgs_9.5", "bgs_9", "bgs_8.5",
                   "sgc_10", "sgc_9", "sgc_8.5", "sgc_8", "cgc_10", "cgc_9", "cgc_9.5", "cgc_8.5", None}
    assert df["grade"].isin(grades_seen).all()
    # every row title mentions Henderson (SCP pre-filters to this card)
    assert df["title"].str.lower().str.contains("henderson").all()


def test_parse_sales_tables_best_offer():
    df = scp_parse.parse_sales_tables(FIXTURE)
    bo = df[df["best_offer"]]
    assert len(bo) >= 1                        # fixture contains a dual-price row
    assert (bo["list_price"] > bo["price"]).all()   # accepted < list
    # the spike-documented row: list $40.00, accepted $35.00
    assert ((bo["list_price"] == 40.00) & (bo["price"] == 35.00)).any()


def test_parse_attributes():
    attrs = scp_parse.parse_attributes(FIXTURE)
    assert attrs["is_rookie_card"] is True


def test_parse_price_summary():
    summary = scp_parse.parse_price_summary(FIXTURE)
    assert summary["PSA 10"] == pytest.approx(34.50)
    assert summary["Ungraded"] == pytest.approx(1.63)


def test_parse_chart_data():
    chart = scp_parse.parse_chart_data(FIXTURE)
    assert len(chart) >= 3
    for series in chart.values():
        ts, price = series[-1]
        assert isinstance(ts, pd.Timestamp)
        assert price > 0
        assert series == sorted(series, key=lambda p: p[0])  # chronological
```

NOTE for the implementer: `test_parse_sales_tables_best_offer` assumes the spike confirms list>accepted and that the documented fixture row is (list 40.00, accepted 35.00). If the spike documented different exact values, use the spike's values in the test and explain in your report. If the fixture has NO dual-price row, drop this test and note it.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scp_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.scp_parse'`.

- [ ] **Step 3: Implement scp_parse.py**

```python
# src/cardprice/scp_parse.py
"""Parse SportsCardsPro card pages: sales tables, attributes, price summary, chart data."""

import json
import re

import pandas as pd
from bs4 import BeautifulSoup

GRADE_RE = re.compile(r"\b(PSA|BGS|SGC|CGC)\s*(10|9\.5|9|8\.5|8)\b", re.IGNORECASE)
PRICE_RE = re.compile(r"\$([\d,]+(?:\.\d{2})?)")
CHART_RE = re.compile(r"VGPC\.chart_data\s*=\s*(\{.*?\})\s*;", re.DOTALL)


def parse_grade_from_title(title: str) -> str | None:
    m = GRADE_RE.search(title)
    if not m:
        return None
    return f"{m.group(1).lower()}_{m.group(2)}"


def _parse_price_cell(text: str) -> tuple[float, float | None, bool]:
    """Return (price_paid, list_price, best_offer). Single amount => (amount, None, False).
    Dual amount => per spike doc: first is list (struck through), second is accepted."""
    amounts = [float(a.replace(",", "")) for a in PRICE_RE.findall(text)]
    if not amounts:
        raise ValueError(f"no price in cell: {text!r}")
    if len(amounts) == 1:
        return amounts[0], None, False
    return amounts[-1], amounts[0], True


def parse_sales_tables(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for div in soup.find_all("div", class_=re.compile(r"^completed-auctions-")):
        bucket = div["class"][0].replace("completed-auctions-", "")
        table = div.find("table")
        if not table:
            continue
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            date_text = tds[0].get_text(strip=True)
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
                continue  # header or non-sale row
            title = tds[2].get_text(strip=True)
            price, list_price, best_offer = _parse_price_cell(tds[3].get_text(strip=True))
            rows.append(
                {
                    "sale_date": date_text,
                    "title": title,
                    "price": price,
                    "list_price": list_price,
                    "best_offer": best_offer,
                    "grade": parse_grade_from_title(title),
                    "bucket": bucket,
                }
            )
    df = pd.DataFrame(rows)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df


def parse_attributes(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="attribute")
    out = {}
    if table:
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) == 2:
                key = tds[0].get_text(strip=True).rstrip(":").lower().replace(" ", "_")
                val = tds[1].get_text(strip=True)
                out[key] = val.lower() == "yes" if val.lower() in ("yes", "no") else val
    return out


def parse_price_summary(html: str) -> dict[str, float]:
    """table#price_data: pair the header row's grade labels with the first data
    row's value cells positionally. Value cells may hold two amounts
    ('$34.50$0.00' = price + week-change); take the first."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="price_data")
    out = {}
    if table:
        rows = table.find_all("tr")
        if len(rows) >= 2:
            labels = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
            values = [c.get_text(strip=True) for c in rows[1].find_all(["th", "td"])]
            for label, val in zip(labels, values):
                amounts = PRICE_RE.findall(val)
                if label and amounts:
                    out[label] = float(amounts[0].replace(",", ""))
    return out


def parse_chart_data(html: str) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    m = CHART_RE.search(html)
    if not m:
        return {}
    raw = json.loads(m.group(1))
    out = {}
    for key, points in raw.items():
        series = [
            (pd.Timestamp(ts, unit="ms", tz="UTC").tz_convert(None).normalize(), cents / 100)
            for ts, cents in points
        ]
        out[key] = sorted(series, key=lambda p: p[0])
    return out
```

NOTE: `table#price_data`'s first data row can have fewer cells than the header row (a trailing `+` expander cell) — positional `zip` truncates safely because grade columns lead. The test's exact-value assertions (PSA 10 = 34.50, Ungraded = 1.63) are the guard; if they fail, inspect the fixture's table structure, fix the pairing, and document the actual structure in your report.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_scp_parse.py -v` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/scp_parse.py tests/test_scp_parse.py
git commit -m "feat: SCP card page parsers (sales, attributes, prices, chart)"
```

---

### Task 3: Chart-series grade calibration

**Files:**
- Modify: `src/cardprice/scp_parse.py` (append)
- Test: `tests/test_scp_chart_calibrate.py`

**Interfaces:**
- Consumes: `parse_chart_data`, `parse_price_summary` (Task 2).
- Produces (Task 4, Plan 3 consume):
  - `calibrate_chart_grades(html: str) -> pd.DataFrame` — columns: `grade (str), date (datetime64), price (float)`. Maps each chart key to a grade label by matching the key's final price to the `parse_price_summary` values (exact cent match); keys that match no summary value are kept with `grade = "key:<keyname>"`. Grade label normalization: lowercase, spaces→underscore (`"PSA 10"` → `"psa_10"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scp_chart_calibrate.py
from pathlib import Path

import pytest

from cardprice.scp_parse import calibrate_chart_grades, parse_price_summary

FIXTURE = (
    Path(__file__).parent / "fixtures" / "scp" / "henderson_2023_topps_chrome.html"
).read_text()


def test_calibrated_grades_cover_psa10():
    df = calibrate_chart_grades(FIXTURE)
    grades = set(df["grade"])
    assert "psa_10" in grades
    psa10 = df[df["grade"] == "psa_10"].sort_values("date")
    assert len(psa10) >= 24  # ~2+ years of monthly points
    # final chart point equals the current price summary value
    assert psa10.iloc[-1]["price"] == pytest.approx(parse_price_summary(FIXTURE)["PSA 10"])


def test_history_reaches_back():
    df = calibrate_chart_grades(FIXTURE)
    psa10 = df[df["grade"] == "psa_10"]
    assert psa10["date"].min() <= pd.Timestamp("2024-06-01")  # card tracked since 2023
```

(add `import pandas as pd` at top)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scp_chart_calibrate.py -v`
Expected: FAIL with `ImportError: cannot import name 'calibrate_chart_grades'`.

- [ ] **Step 3: Implement**

Append to `src/cardprice/scp_parse.py`:

```python
def calibrate_chart_grades(html: str) -> pd.DataFrame:
    """Long-form monthly price history with grade labels, resolved by matching
    each chart key's latest price to the price-summary table."""
    summary = {v: k for k, v in parse_price_summary(html).items()}  # price -> label
    rows = []
    for key, series in parse_chart_data(html).items():
        if not series:
            continue
        label = summary.get(series[-1][1])
        grade = label.lower().replace(" ", "_") if label else f"key:{key}"
        for ts, price in series:
            rows.append({"grade": grade, "date": ts, "price": price})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_scp_chart_calibrate.py -v` — 2 PASS. If the exact-cent match fails for psa_10 (e.g. summary shows a value chart never hits), inspect the fixture and document why; acceptable fallback: match within $0.01, then nearest-within-$0.50 — implement only if needed and explain in the report.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/scp_parse.py tests/test_scp_chart_calibrate.py
git commit -m "feat: chart-series grade calibration against price summary"
```

---

### Task 4: SCP collector CLI

**Files:**
- Create: `src/cardprice/collect_prices.py`
- Test: `tests/test_collect_prices.py`

**Interfaces:**
- Consumes: `fetch_page` (web.py); `save_raw` (storage.py); `parse_sales_tables`, `calibrate_chart_grades`, `parse_attributes` (scp_parse.py); `data/reference/cards_seed.csv`.
- Produces:
  - `collect_cards(cards: pd.DataFrame, sleep_s: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]` — per card: fetch `scp_url`, `save_raw("prices_scp", slug, {"url": scp_url, "html": html})`, parse sales + calibrated chart; returns `(sales_df, chart_df)` concatenated, each with added columns `player_name, mlb_id, rookie_year, set_slug, card_slug`. Skips rows with empty `scp_url`.
  - `card_slug(url: str) -> str` — last two path segments joined with `/` (e.g. `baseball-cards-2023-topps-chrome/gunnar-henderson-2`).
  - CLI: `python -m cardprice.collect_prices --cards data/reference/cards_seed.csv --sales-out data/processed/scp_sales.parquet --chart-out data/processed/scp_chart_monthly.parquet`
  - On `ChallengeError` for one card: log warning, continue with next card (do not abort the batch).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_prices.py
import pandas as pd
import pytest

from cardprice import collect_prices, storage

MINI_HTML = """
<html><head><title>x</title></head><body>
<table id="attribute"><tr><td>Is Rookie Card:</td><td>Yes</td></tr></table>
<table id="price_data"><tr><th>PSA 10</th></tr><tr><td>$34.50</td></tr></table>
<div class="completed-auctions-graded"><table>
<tr><th>Sale Date</th><th>TW</th><th>Title</th><th>Price</th><th></th></tr>
<tr><td>2026-09-05</td><td></td><td>2023 Topps Chrome Gunnar Henderson RC PSA 10</td><td>$45.99</td><td></td></tr>
</table></div>
<script>VGPC.chart_data = {"graded":[[1756600000000,4599],[1759200000000,3450]]};</script>
</body></html>
"""


@pytest.fixture
def cards():
    return pd.DataFrame(
        [
            {
                "player_name": "Gunnar Henderson",
                "role": "hitter",
                "rookie_year": 2023,
                "set_slug": "baseball-cards-2023-topps-chrome",
                "mlb_id": 683002,
                "scp_url": "https://www.sportscardspro.com/game/baseball-cards-2023-topps-chrome/gunnar-henderson-2",
            },
            {
                "player_name": "Unresolved Player",
                "role": "hitter",
                "rookie_year": 2025,
                "set_slug": "baseball-cards-2025-topps-chrome",
                "mlb_id": 999999,
                "scp_url": "",
            },
        ]
    )


def test_card_slug():
    assert (
        collect_prices.card_slug("https://www.sportscardspro.com/game/baseball-cards-2023-topps-chrome/gunnar-henderson-2")
        == "baseball-cards-2023-topps-chrome/gunnar-henderson-2"
    )


def test_collect_cards_happy_path_and_skip(cards, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(collect_prices, "fetch_page", lambda url: MINI_HTML)
    sales, chart = collect_prices.collect_cards(cards, sleep_s=0)
    assert len(sales) == 1
    assert sales.iloc[0]["grade"] == "psa_10"
    assert sales.iloc[0]["mlb_id"] == 683002
    assert len(chart) == 2
    assert storage.load_latest(
        "prices_scp", "baseball-cards-2023-topps-chrome/gunnar-henderson-2"
    )["url"] == cards.iloc[0]["scp_url"]


def test_collect_cards_challenge_continues(cards, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)

    def flaky(url):
        raise collect_prices.ChallengeError(url)

    monkeypatch.setattr(collect_prices, "fetch_page", flaky)
    sales, chart = collect_prices.collect_cards(cards, sleep_s=0)
    assert len(sales) == 0 and len(chart) == 0  # no exception propagated
```

(The second test uses `collect_prices.ChallengeError` — re-export it in collect_prices via `from cardprice.web import ChallengeError, fetch_page`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collect_prices.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.collect_prices'`.

- [ ] **Step 3: Implement collect_prices.py**

```python
# src/cardprice/collect_prices.py
"""Batch SCP collector: cards_seed.csv -> raw HTML snapshots + parsed parquets."""

import argparse
import time

import pandas as pd

from cardprice.scp_parse import calibrate_chart_grades, parse_attributes, parse_sales_tables
from cardprice.storage import save_raw
from cardprice.web import ChallengeError, fetch_page

META_COLS = ["player_name", "mlb_id", "rookie_year", "set_slug", "card_slug"]


def card_slug(url: str) -> str:
    parts = url.rstrip("/").split("/")
    return "/".join(parts[-2:])


def collect_cards(cards: pd.DataFrame, sleep_s: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales_frames, chart_frames = [], []
    for card in cards.itertuples():
        if not card.scp_url:
            print(f"SKIP {card.player_name}: no scp_url")
            continue
        slug = card_slug(card.scp_url)
        try:
            html = fetch_page(card.scp_url)
        except ChallengeError as e:
            print(f"WARN challenge unresolved for {slug}: {e}")
            continue
        save_raw("prices_scp", slug, {"url": card.scp_url, "html": html})
        meta = {
            "player_name": card.player_name,
            "mlb_id": int(card.mlb_id),
            "rookie_year": int(card.rookie_year),
            "set_slug": card.set_slug,
            "card_slug": slug,
        }
        sales = parse_sales_tables(html)
        if len(sales):
            sales = sales.assign(**meta)
            sales_frames.append(sales)
        chart = calibrate_chart_grades(html)
        if len(chart):
            chart = chart.assign(**meta)
            chart_frames.append(chart)
        attrs = parse_attributes(html)
        print(f"OK {slug}: {len(sales)} sales, {len(chart)} chart points, rookie={attrs.get('is_rookie_card')}")
        time.sleep(sleep_s)
    return (
        pd.concat(sales_frames, ignore_index=True) if sales_frames else pd.DataFrame(),
        pd.concat(chart_frames, ignore_index=True) if chart_frames else pd.DataFrame(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="data/reference/cards_seed.csv")
    parser.add_argument("--sales-out", default="data/processed/scp_sales.parquet")
    parser.add_argument("--chart-out", default="data/processed/scp_chart_monthly.parquet")
    parser.add_argument("--sleep", type=float, default=5.0)
    args = parser.parse_args()

    cards = pd.read_csv(args.cards)
    sales, chart = collect_cards(cards, sleep_s=args.sleep)
    sales.to_parquet(args.sales_out, index=False)
    chart.to_parquet(args.chart_out, index=False)
    print(f"wrote {len(sales)} sales to {args.sales_out}; {len(chart)} chart points to {args.chart_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then real collection**

Run: `python -m pytest tests/test_collect_prices.py -v` — 3 PASS.
Then (network, ~2-3 min at 5s/card):
```bash
python -m cardprice.collect_prices
```
Expected: ≥14 cards OK, sales rows in the hundreds, chart points in the thousands. If a card 404s or challenges out, note it in the report.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/collect_prices.py tests/test_collect_prices.py
git commit -m "feat: SCP batch collector CLI writing sales + chart parquets"
```

---

### Task 5: Weekly price series builder

**Files:**
- Create: `src/cardprice/weekly.py`
- Test: `tests/test_weekly.py`

**Interfaces:**
- Consumes: sales DataFrame from `collect_cards` (Task 4) — real data only accumulates going forward; tests use synthetic frames.
- Produces (Plan 3 panel consumes):
  - `weekly_price_series(sales: pd.DataFrame, min_sales: int = 2) -> pd.DataFrame` — input columns `card_slug, grade, sale_date, price, best_offer`; output columns: `card_slug, grade, week (datetime64, Monday), median_price (float), n_sales (int), best_offer_share (float)`. Weeks with `< min_sales` sales are omitted (missing, not interpolated). Only rows with non-null `grade` are used.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_weekly.py
import pandas as pd

from cardprice.weekly import weekly_price_series


def make_sales():
    rows = [
        # week of Mon 2026-08-31: 3 sales
        ("card/a", "psa_10", "2026-08-31", 40.0, False),
        ("card/a", "psa_10", "2026-09-02", 50.0, True),
        ("card/a", "psa_10", "2026-09-05", 60.0, False),
        # week of Mon 2026-09-07: 1 sale -> omitted
        ("card/a", "psa_10", "2026-09-08", 70.0, False),
        # different grade same week
        ("card/a", "psa_9", "2026-09-01", 10.0, False),
        ("card/a", "psa_9", "2026-09-03", 20.0, False),
        # ungraded (grade None) -> excluded
        ("card/a", None, "2026-09-01", 3.0, False),
    ]
    df = pd.DataFrame(rows, columns=["card_slug", "grade", "sale_date", "price", "best_offer"])
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df


def test_weekly_aggregation_and_min_sales():
    out = weekly_price_series(make_sales())
    psa10 = out[out["grade"] == "psa_10"].sort_values("week")
    assert len(psa10) == 1  # the 1-sale week is omitted
    row = psa10.iloc[0]
    assert row["week"] == pd.Timestamp("2026-08-31")
    assert row["median_price"] == 50.0
    assert row["n_sales"] == 3
    assert row["best_offer_share"] == 1 / 3


def test_grades_separate_and_ungraded_excluded():
    out = weekly_price_series(make_sales())
    assert set(out["grade"]) == {"psa_10", "psa_9"}
    psa9 = out[out["grade"] == "psa_9"].iloc[0]
    assert psa9["median_price"] == 15.0 and psa9["n_sales"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_weekly.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.weekly'`.

- [ ] **Step 3: Implement weekly.py**

```python
# src/cardprice/weekly.py
"""Card-grade-week price series from individual sales."""

import pandas as pd


def weekly_price_series(sales: pd.DataFrame, min_sales: int = 2) -> pd.DataFrame:
    df = sales[sales["grade"].notna()].copy()
    df["week"] = df["sale_date"].dt.to_period("W-SUN").dt.start_time  # Mondays
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

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_weekly.py -v` — 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/weekly.py tests/test_weekly.py
git commit -m "feat: weekly card-grade price series builder"
```

---

### Task 6: Validation layer

**Files:**
- Create: `src/cardprice/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: sales frames (Task 4 shape), weekly frames (Task 5 shape).
- Produces (Plan 3 consumes; CLI in Task 7):
  - `quarantine_outliers(sales: pd.DataFrame, iqr_mult: float = 4.0) -> pd.DataFrame` — returns input with added bool column `outlier`; a sale is an outlier if its price falls outside `[Q1 − iqr_mult·IQR, Q3 + iqr_mult·IQR]` within its `(card_slug, grade, calendar month)` group AND the group has ≥5 sales (small groups never flag; IQR = 0 never flags). Quarantine = flag, never delete (spec §7). NOTE: this is the robust analog of the spec's ">4σ within card-grade-month" rule — a σ-based test can never flag a single extreme point because the point itself inflates the group std; the IQR fence is the correct implementation of the spec's intent.
  - `contamination_audit(sales: pd.DataFrame) -> pd.DataFrame` — returns the subset of rows whose title matches `(?i)\b(lot of|reprint|rp\b|proxy|custom card|case break|digital)\b`, with added column `flag_reason`.
  - `stale_series(weekly: pd.DataFrame, max_gap_weeks: int = 3, as_of: pd.Timestamp | None = None) -> list[str]` — card_slugs whose latest week is more than `max_gap_weeks` before `as_of` (default: today).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate.py
import pandas as pd

from cardprice.validate import contamination_audit, quarantine_outliers, stale_series


def test_quarantine_outliers_flags_extreme_only_in_big_groups():
    prices = [40.0, 41.0, 42.0, 43.0, 44.0, 3000.0]  # 6 sales, one wild
    df = pd.DataFrame(
        {
            "card_slug": "card/a",
            "grade": "psa_10",
            "sale_date": pd.to_datetime(["2026-09-0%d" % d for d in range(1, 7)]),
            "price": prices,
        }
    )
    out = quarantine_outliers(df)
    assert out["outlier"].sum() == 1
    assert out.loc[out["price"] == 3000.0, "outlier"].iloc[0]
    # small group (4 sales incl. extreme) -> never flags
    small = df.iloc[:4].copy()
    small.loc[small.index[-1], "price"] = 900.0
    assert not quarantine_outliers(small)["outlier"].any()


def test_contamination_audit():
    df = pd.DataFrame(
        {
            "title": [
                "2023 Topps Chrome Henderson PSA 10",
                "Henderson PSA 10 lot of 3",
                "2023 Topps Chrome Henderson REPRINT psa 10",
            ],
            "price": [40.0, 90.0, 12.0],
        }
    )
    out = contamination_audit(df)
    assert len(out) == 2
    assert "lot" in out.iloc[0]["flag_reason"]


def test_stale_series():
    weekly = pd.DataFrame(
        {
            "card_slug": ["card/fresh", "card/stale"],
            "week": [pd.Timestamp("2026-09-07"), pd.Timestamp("2026-08-03")],
        }
    )
    stale = stale_series(weekly, max_gap_weeks=3, as_of=pd.Timestamp("2026-09-14"))
    assert stale == ["card/stale"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.validate'`.

- [ ] **Step 3: Implement validate.py**

```python
# src/cardprice/validate.py
"""Post-ingest data-quality checks. Quarantine flags, never deletes."""

import re

import pandas as pd

CONTAMINATION_RE = re.compile(
    r"\b(?:lot of|reprint|rp|proxy|custom card|case break|digital)\b", re.IGNORECASE
)


def quarantine_outliers(sales: pd.DataFrame, iqr_mult: float = 4.0) -> pd.DataFrame:
    df = sales.copy()
    df["outlier"] = False
    month = df["sale_date"].dt.to_period("M")
    for _, idx in df.groupby([df["card_slug"], df["grade"], month]).groups.items():
        group = df.loc[idx, "price"]
        if len(group) < 5:
            continue
        q1, q3 = group.quantile(0.25), group.quantile(0.75)
        iqr = q3 - q1
        if not iqr or pd.isna(iqr):
            continue
        lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        df.loc[idx, "outlier"] = (group < lo) | (group > hi)
    return df


def contamination_audit(sales: pd.DataFrame) -> pd.DataFrame:
    mask = sales["title"].str.contains(CONTAMINATION_RE, na=False)
    out = sales[mask].copy()
    out["flag_reason"] = out["title"].str.extract(
        "(" + CONTAMINATION_RE.pattern + ")", expand=False
    )
    return out


def stale_series(
    weekly: pd.DataFrame, max_gap_weeks: int = 3, as_of: pd.Timestamp | None = None
) -> list[str]:
    as_of = as_of or pd.Timestamp.today().normalize()
    latest = weekly.groupby("card_slug")["week"].max()
    gap = (as_of - latest).dt.days / 7
    return sorted(latest[gap > max_gap_weeks].index.tolist())
```

NOTE: `flag_reason` uses `str.extract("(" + CONTAMINATION_RE.pattern + ")")` — the regex body uses a non-capturing group so exactly one capture group exists; pandas fills NaN for non-matches (there are none in the filtered subset).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_validate.py -v` — 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/validate.py tests/test_validate.py
git commit -m "feat: validation layer (outliers, contamination, stale series)"
```

---

### Task 7: Liquidity report + universe recommendation

**Files:**
- Create: `src/cardprice/liquidity.py`
- Create: `scripts/run_price_pipeline.py`
- Test: `tests/test_liquidity.py`

**Interfaces:**
- Consumes: `data/processed/scp_sales.parquet`, `data/processed/scp_chart_monthly.parquet` (Task 4); `weekly_price_series` (Task 5); all of validate.py (Task 6).
- Produces:
  - `liquidity_report(sales: pd.DataFrame, chart: pd.DataFrame) -> pd.DataFrame` — one row per `(card_slug, grade)` for grades psa_9/psa_10 with columns: `n_sales, first_sale, last_sale, sales_span_days, sales_per_week, n_chart_points, chart_first, chart_last, outlier_share, contamination_count`. Sorted by `sales_per_week` descending.
  - `recommend_universe(report: pd.DataFrame, min_sales_per_week: float = 0.5, min_chart_points: int = 18) -> pd.DataFrame` — rows passing both thresholds.
  - `scripts/run_price_pipeline.py` — runs: read parquets → validate (print quarantine + contamination + stale summaries) → weekly series → write `data/processed/scp_weekly.parquet` → liquidity report → write `data/processed/liquidity_report.csv` and print the recommended universe.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_liquidity.py
import pandas as pd

from cardprice.liquidity import liquidity_report, recommend_universe


def make_inputs():
    sales = pd.DataFrame(
        {
            "card_slug": ["card/hot"] * 10 + ["card/cold"] * 2,
            "grade": ["psa_10"] * 10 + ["psa_10"] * 2,
            "sale_date": pd.to_datetime(
                ["2026-08-%02d" % d for d in range(1, 11)] + ["2026-09-01", "2026-09-10"]
            ),
            "price": [40.0] * 10 + [5.0, 6.0],
            "best_offer": [False] * 12,
            "title": ["Henderson PSA 10"] * 12,
        }
    )
    chart = pd.DataFrame(
        {
            "card_slug": ["card/hot"] * 24 + ["card/cold"] * 6,
            "grade": ["psa_10"] * 30,
            "date": pd.to_datetime(
                ["2024-%02d-01" % m for m in range(1, 13)] + ["2025-%02d-01" % m for m in range(1, 13)]
                + ["2026-0%d-01" % m for m in range(1, 7)]
            ),
            "price": [30.0] * 30,
        }
    )
    return sales, chart


def test_liquidity_report_metrics():
    sales, chart = make_inputs()
    rep = liquidity_report(sales, chart)
    hot = rep[rep["card_slug"] == "card/hot"].iloc[0]
    cold = rep[rep["card_slug"] == "card/cold"].iloc[0]
    assert hot["n_sales"] == 10
    assert hot["sales_per_week"] > cold["sales_per_week"]
    assert hot["n_chart_points"] == 24
    assert rep.iloc[0]["card_slug"] == "card/hot"  # sorted desc


def test_recommend_universe_thresholds():
    sales, chart = make_inputs()
    rep = liquidity_report(sales, chart)
    # hot: 10 sales / ~1.3 weeks ≈ 7.8/wk; cold: 2 sales / ~1.3 weeks ≈ 1.6/wk
    rec = recommend_universe(rep, min_sales_per_week=5.0, min_chart_points=18)
    assert rec["card_slug"].tolist() == ["card/hot"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_liquidity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cardprice.liquidity'`.

- [ ] **Step 3: Implement liquidity.py + runner**

```python
# src/cardprice/liquidity.py
"""Liquidity metrics per card-grade; recommends the modeling universe."""

import pandas as pd

from cardprice.validate import contamination_audit, quarantine_outliers

PANEL_GRADES = ("psa_9", "psa_10")


def liquidity_report(sales: pd.DataFrame, chart: pd.DataFrame) -> pd.DataFrame:
    sales = sales[sales["grade"].isin(PANEL_GRADES)]
    flagged = quarantine_outliers(sales)
    contaminated = contamination_audit(sales)
    rows = []
    for (slug, grade), grp in flagged.groupby(["card_slug", "grade"]):
        span = max((grp["sale_date"].max() - grp["sale_date"].min()).days, 1)
        ch = chart[(chart["card_slug"] == slug) & (chart["grade"] == grade)]
        contam = contaminated[
            (contaminated["card_slug"] == slug) & (contaminated["grade"] == grade)
        ]
        rows.append(
            {
                "card_slug": slug,
                "grade": grade,
                "n_sales": len(grp),
                "first_sale": grp["sale_date"].min(),
                "last_sale": grp["sale_date"].max(),
                "sales_span_days": span,
                "sales_per_week": len(grp) / (span / 7),
                "n_chart_points": len(ch),
                "chart_first": ch["date"].min() if len(ch) else pd.NaT,
                "chart_last": ch["date"].max() if len(ch) else pd.NaT,
                "outlier_share": grp["outlier"].mean(),
                "contamination_count": len(contam),
            }
        )
    return pd.DataFrame(rows).sort_values("sales_per_week", ascending=False).reset_index(drop=True)


def recommend_universe(
    report: pd.DataFrame, min_sales_per_week: float = 0.5, min_chart_points: int = 18
) -> pd.DataFrame:
    mask = (report["sales_per_week"] >= min_sales_per_week) & (
        report["n_chart_points"] >= min_chart_points
    )
    return report[mask].reset_index(drop=True)
```

```python
# scripts/run_price_pipeline.py
"""End-to-end: parquets -> validation summary -> weekly series -> liquidity report."""

import pandas as pd

from cardprice.liquidity import liquidity_report, recommend_universe
from cardprice.validate import contamination_audit, quarantine_outliers, stale_series
from cardprice.weekly import weekly_price_series

if __name__ == "__main__":
    sales = pd.read_parquet("data/processed/scp_sales.parquet")
    chart = pd.read_parquet("data/processed/scp_chart_monthly.parquet")

    flagged = quarantine_outliers(sales)
    print(f"sales: {len(sales)} rows, {int(flagged['outlier'].sum())} outlier-flagged")
    contam = contamination_audit(sales)
    print(f"contamination audit: {len(contam)} suspicious rows")
    if len(contam):
        print(contam[["card_slug", "title", "price"]].to_string(index=False))

    weekly = weekly_price_series(flagged[~flagged["outlier"]])
    weekly.to_parquet("data/processed/scp_weekly.parquet", index=False)
    print(f"weekly series: {len(weekly)} card-grade-weeks")

    stale = stale_series(weekly)
    print(f"stale series (>3 weeks no sales): {len(stale)}")
    for slug in stale:
        print(f"  STALE {slug}")

    report = liquidity_report(sales, chart)
    report.to_csv("data/processed/liquidity_report.csv", index=False)
    rec = recommend_universe(report)
    print(f"\nrecommended universe ({len(rec)} card-grades):")
    print(rec[["card_slug", "grade", "sales_per_week", "n_chart_points"]].to_string(index=False))
```

- [ ] **Step 4: Run tests, then real run**

Run: `python -m pytest tests/test_liquidity.py -v` — 2 PASS. Full suite: `python -m pytest -v` — all PASS (live deselected).
Then: `python scripts/run_price_pipeline.py` — inspect the printed summary; paste the key numbers (sales count, weekly rows, recommended universe size) into your report.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/liquidity.py scripts/run_price_pipeline.py tests/test_liquidity.py
git commit -m "feat: liquidity report and universe recommendation"
```

---

## Done criteria for this plan

- `python -m pytest -v` all offline tests PASS; `ruff check src tests scripts` clean.
- `python -m pytest -m live -v` PASS (network).
- `data/processed/scp_sales.parquet`, `scp_chart_monthly.parquet`, `scp_weekly.parquet`, `liquidity_report.csv` exist, built from ≥14 seed cards.
- Spike doc `docs/superpowers/spikes/2026-09-14-scp-structure.md` records dual-price semantics, chart keys, category URL scheme.
- Liquidity report recommends a concrete universe (or honestly reports the seed set is too illiquid and why).
- Next: Plan 3 (panel assembly: Savant metrics, GemRate pops, card×week/month panel, market index, lagged predictors).
