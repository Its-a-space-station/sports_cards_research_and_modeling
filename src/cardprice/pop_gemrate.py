# src/cardprice/pop_gemrate.py
"""GemRate population data via the interactive universal search.

Flow per card (see docs/superpowers/spikes/2026-09-15-gemrate.md):
1. Load https://www.gemrate.com/universal-search, type a query into `#search`;
   the page POSTs `/universal-search-query` and gets JSON results back.
2. Pick the base-card result via its structured `parsed_description`.
3. Open `/card/<gemrate_id>`; the page fetches `/card-details` JSON (with an
   `X-Card-Details-Token` header minted into the page) holding per-grader pops.
4. `psa_10_pop` = population_data[grader=="psa"].grades.psa_10;
   `total_pop` = total_population; `gem_rate` = gems / total_population.

Card pages sit behind a Cloudflare challenge that resolves headless with a
~10 s wait (same retry pattern as `web.fetch_page`).
"""

import argparse
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from cardprice.collect_prices import card_slug
from cardprice.storage import save_raw
from cardprice.web import USER_AGENT, ChallengeError, is_challenge_title

SEARCH_URL = "https://www.gemrate.com/universal-search"
CARD_URL = "https://www.gemrate.com/card/{}"
VIEWPORT = {"width": 1400, "height": 900}

CSV_COLS = ["date", "card_slug", "psa_10_pop", "total_pop", "gem_rate"]


class PopLookupError(RuntimeError):
    """Search returned no matching card, or the card-details JSON never arrived."""


def set_query_from_slug(set_slug: str) -> tuple[str, str]:
    """'baseball-cards-2023-topps-chrome' -> ('2023', 'Topps Chrome')."""
    parts = set_slug.split("-")
    return parts[2], " ".join(parts[3:]).title()


def card_number_from_scp_url(url: str) -> str:
    """'.../gunnar-henderson-2' -> '2'; '.../paul-skenes-usc88' -> 'USC88'."""
    return url.rstrip("/").split("/")[-1].split("-")[-1].upper()


def build_query(player_name: str, set_slug: str, scp_url: str) -> str:
    year, set_words = set_query_from_slug(set_slug)
    return f"{player_name} {year} {set_words} #{card_number_from_scp_url(scp_url)}"


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def _num_exact(want: str, cand: str | None) -> bool:
    return str(cand or "").upper() == str(want).upper()


def _num_digits(want: str, cand: str | None) -> bool:
    """Digits-only comparison: GemRate sometimes drops the 'USC' prefix that
    SCP keeps (e.g. 2025 Topps Chrome Update Nick Kurtz is 'Base 178' on
    GemRate but 'usc178' in the SCP slug)."""
    want_d = re.sub(r"\D", "", str(want))
    return want_d != "" and want_d == re.sub(r"\D", "", str(cand or ""))


# Extra set-descriptor tokens that mark a DIFFERENT product line rather than
# set-name drift. A loose (token-subset) set match is rejected when the
# candidate's set name adds any of these — otherwise an entry like
# 'Topps Chrome Update Refractors' or 'Topps Chrome Logofractor Edition'
# (parallel: null) silently passes as the base card.
BLOCKED_SET_TOKENS = frozenset(
    {
        "refractor",
        "refractors",
        "fractor",
        "logofractor",
        "xfractor",
        "edition",
        "stars",
        "cosmic",
        "silver",
        "pack",
        "sapphire",
        "platinum",
        "anniversary",
    }
)


def pick_search_result(
    results: list[dict],
    *,
    name: str,
    year: str,
    set_name: str,
    card_number: str,
) -> dict | None:
    """Pick the right card entry: exact year/name, ranked card-number/set match.

    Parallel preference order: "Base" first, then "SP" — a few cards exist only
    as short prints in GemRate's taxonomy (e.g. 2022 Topps Chrome #221 Witt and
    #222 Rodriguez have no Base entry; PSA numbers the SP as the base card).
    Set-name match is exact first, then a token-subset fallback (GemRate set
    naming can drift from SCP's); the loose pass rejects candidates whose set
    name adds product-line tokens (BLOCKED_SET_TOKENS, e.g. 'refractors',
    'logofractor', 'edition'). Card-number match is exact first, then
    digits-only ('USC178' vs '178'). Prefers universal matches, then highest pop.
    """

    def matches(r: dict, parallels: set[str], strict_set: bool, num_ok) -> bool:
        pd_ = r.get("parsed_description") or {}
        if str(pd_.get("year") or "") != str(year):
            return False
        if _norm(pd_.get("name")) != _norm(name):
            return False
        if not num_ok(card_number, pd_.get("card_number")):
            return False
        if _norm(pd_.get("parallel") or "Base") not in parallels:
            return False
        cand_tokens = _norm(pd_.get("set_name")).split()
        want_tokens = _norm(set_name).split()
        if strict_set:
            return cand_tokens == want_tokens
        if not all(tok in cand_tokens for tok in want_tokens):
            return False
        extra = set(cand_tokens) - set(want_tokens)
        return not (extra & BLOCKED_SET_TOKENS)

    for parallels in ({"base"}, {"sp"}):
        for strict in (True, False):
            for num_ok in (_num_exact, _num_digits):
                cands = [r for r in results if matches(r, parallels, strict, num_ok)]
                if cands:
                    cands.sort(
                        key=lambda r: (
                            not r.get("is_universal_match"),
                            -(r.get("total_population") or 0),
                        )
                    )
                    return cands[0]
    return None


def parse_card_details(data: dict) -> dict:
    """Extract pop figures from the /card-details JSON payload."""
    total_pop = int(data.get("total_population") or 0)
    gems = int(data.get("total_gems_or_greater") or 0)
    psa_row = next((r for r in data.get("population_data", []) if r.get("grader") == "psa"), None)
    psa_10_pop = psa_total_pop = None
    if psa_row:
        # psa_10 absent from grades != 0 graded PSA 10s: keep NULL, don't invent a 0
        psa_10_val = (psa_row.get("grades") or {}).get("psa_10")
        psa_10_pop = int(psa_10_val) if psa_10_val is not None else None
        psa_total_val = psa_row.get("card_total_grades")
        psa_total_pop = int(psa_total_val) if psa_total_val is not None else None
    return {
        "psa_10_pop": psa_10_pop,
        "psa_total_pop": psa_total_pop,
        "total_pop": total_pop,
        "gem_rate": round(gems / total_pop, 4) if total_pop else None,
        "description": data.get("description"),
        "gemrate_id": data.get("gemrate_id"),
        "data_last_updated": data.get("data_last_updated"),
    }


def _new_page():
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=USER_AGENT, viewport=VIEWPORT)
    return pw, browser, ctx.new_page()


def _wait_real_page(page, url: str, wait_ms: int = 10000, retries: int = 3) -> None:
    for _ in range(retries):
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(wait_ms)
        if not is_challenge_title(page.title()):
            return
        time.sleep(5)
    raise ChallengeError(url)


def _capture_json(page, url_fragment: str, action, polls: int = 30) -> dict:
    """Arm a JSON-response listener, run `action` (which triggers the request),
    and return the first captured 200 payload.

    The listener is armed *before* the action: card pages fire `/card-details`
    during initial load, so arming after `goto` would miss it. Polling (vs
    `expect_response`) lets a Cloudflare retry re-trigger the request without
    racing a one-shot waiter.
    """
    captured: dict = {}
    seen: list[str] = []

    def on_response(resp):
        if url_fragment not in resp.url:
            return
        seen.append(str(resp.status))
        if resp.status == 200 and "data" not in captured:
            try:
                captured["data"] = resp.json()
            except (PlaywrightError, ValueError):  # listener must never raise
                pass

    page.on("response", on_response)
    try:
        action()
        for _ in range(polls):
            if "data" in captured:
                return captured["data"]
            page.wait_for_timeout(500)
        raise PopLookupError(
            f"no 200 JSON response containing {url_fragment!r} (statuses seen: {seen or 'none'})"
        )
    finally:
        page.remove_listener("response", on_response)


def _search_results(page, query: str) -> dict:
    def do_search():
        box = page.locator("#search")
        box.click()
        box.fill("")
        box.press_sequentially(query, delay=50)

    _wait_real_page(page, SEARCH_URL, wait_ms=5000)
    for attempt in range(2):  # GemRate debounces the POST; a retry re-types the query
        try:
            return _capture_json(page, "universal-search-query", do_search)
        except PopLookupError:
            if attempt == 1:
                raise
            time.sleep(5)
    return {}  # unreachable


def _fetch_search_page(page, query: str, page_num: int) -> dict | None:
    """Page 2+ of search results via in-page fetch (same endpoint the widget
    uses, inherits session cookies). Page 1 comes from typing into #search."""
    return page.evaluate(
        """async ([q, pg]) => {
            const r = await fetch('/universal-search-query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: q, page: pg}),
            });
            return r.ok ? await r.json() : null;
        }""",
        [query, page_num],
    )


def _search_all_pages(
    page, query: str, first_payload: dict, max_pages: int = 3, sleep_s: float = 5.0
) -> list[dict]:
    results = list(first_payload.get("results", []))
    has_more = bool(first_payload.get("has_more"))
    page_num = 1
    while has_more and page_num < max_pages:
        page_num += 1
        time.sleep(sleep_s)  # politeness: space out paginated POSTs
        payload = _fetch_search_page(page, query, page_num)
        if not payload:
            break
        results.extend(payload.get("results", []))
        has_more = bool(payload.get("has_more"))
    return results


def _fetch_pop_on_page(page, query: str, match: dict | None) -> dict:
    first = _search_results(page, query)
    results = first.get("results", [])
    hit = pick_search_result(results, **match) if match else (results[0] if results else None)
    if hit is None and match and first.get("has_more"):
        results = _search_all_pages(page, query, first)
        hit = pick_search_result(results, **match)
    if not hit:
        raise PopLookupError(f"no matching card for {query!r}")
    time.sleep(5)  # politeness: search capture returns fast; hop to card page stays >=5 s
    details = _capture_json(
        page, "/card-details", lambda: _wait_real_page(page, CARD_URL.format(hit["gemrate_id"]))
    )
    pop = parse_card_details(details)
    pop["parallel"] = (hit.get("parsed_description") or {}).get("parallel")
    pop["query"] = query
    pop["gemrate_url"] = hit.get("gemrate_url")
    return pop


def fetch_pop(query: str, match: dict | None = None, page=None) -> dict:
    """Pop data for one card. Returns psa_10_pop, total_pop, gem_rate (+ context).

    `match` (name/year/set_name/card_number) selects the correct base card from
    search results; without it the first result is used. Pass a shared `page`
    to reuse one browser session (keeps Cloudflare clearance across cards).
    """
    if page is not None:
        return _fetch_pop_on_page(page, query, match)
    pw, browser, own_page = _new_page()
    try:
        return _fetch_pop_on_page(own_page, query, match)
    finally:
        browser.close()
        pw.stop()


def collect_pops(cards: pd.DataFrame, sleep_s: float = 6.0, on: date | None = None) -> pd.DataFrame:
    """Fetch pop snapshots for every resolved card (non-empty scp_url)."""
    day = (on or date.today()).isoformat()  # noqa: DTZ011  # local calendar date is intended
    rows = []
    pw, browser, page = _new_page()
    try:
        for card in cards.itertuples():
            if not isinstance(card.scp_url, str) or not card.scp_url:
                print(f"SKIP {card.player_name}: no scp_url")
                continue
            slug = card_slug(card.scp_url)
            year, set_words = set_query_from_slug(card.set_slug)
            match = {
                "name": card.player_name,
                "year": year,
                "set_name": set_words,
                "card_number": card_number_from_scp_url(card.scp_url),
            }
            query = build_query(card.player_name, card.set_slug, card.scp_url)
            row = {"date": day, "card_slug": slug}
            try:
                pop = fetch_pop(query, match=match, page=page)
            except (PopLookupError, ChallengeError, PlaywrightError, ValueError) as e:
                print(f"WARN {slug}: {type(e).__name__}: {e}")
                row.update({"psa_10_pop": None, "total_pop": None, "gem_rate": None})
            else:
                save_raw("pop_gemrate", slug, {"query": query, **pop}, on=on)
                row.update(
                    {
                        "psa_10_pop": pop["psa_10_pop"],
                        "total_pop": pop["total_pop"],
                        "gem_rate": pop["gem_rate"],
                    }
                )
                print(
                    f"OK {slug}: {pop.get('description')} psa10={pop['psa_10_pop']} "
                    f"total={pop['total_pop']} gem_rate={pop['gem_rate']}"
                )
            rows.append(row)
            time.sleep(sleep_s)
    finally:
        if browser is not None:
            browser.close()
        if pw is not None:
            pw.stop()
    df = pd.DataFrame(rows, columns=CSV_COLS)
    return df.astype({"psa_10_pop": "Int64", "total_pop": "Int64"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="data/reference/cards_seed.csv")
    parser.add_argument("--out", default="data/reference/pop_snapshots.csv")
    parser.add_argument("--sleep", type=float, default=6.0)
    args = parser.parse_args()

    cards = pd.read_csv(args.cards)
    new = collect_pops(cards, sleep_s=args.sleep)
    out = Path(args.out)
    if out.exists() and len(new):
        old = pd.read_csv(out)
        # A same-day failed (all-NaN) rerun row must not clobber an existing
        # good row for that card — drop it instead.
        fail_mask = new[["psa_10_pop", "total_pop", "gem_rate"]].isna().all(axis=1)
        old_keys = set(zip(old["date"], old["card_slug"], strict=True))
        has_old = pd.Series(
            [k in old_keys for k in zip(new["date"], new["card_slug"], strict=True)],
            index=new.index,
        )
        new = new[~(fail_mask & has_old)]
        # same-day reruns are idempotent per card: drop only replaced (date, card_slug) rows
        old = old.merge(
            new[["date", "card_slug"]], on=["date", "card_slug"], how="left", indicator=True
        )
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df = df.astype({"psa_10_pop": "Int64", "total_pop": "Int64"})
    df.to_csv(args.out, index=False)
    print(f"wrote {len(new)} snapshots ({len(df)} total rows) to {args.out}")


if __name__ == "__main__":
    main()
