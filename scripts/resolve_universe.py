# scripts/resolve_universe.py
"""Resolve the 39-player universe to mlb_ids + SCP card URLs (network required).

Reads data/reference/players_universe.csv (hand-written, 39 rows), resolves each
player's mlb_id and the SCP URLs of two card families, and writes
data/reference/cards_universe.csv (long format: one row per player x card_type;
unresolved cells stay EMPTY — no-guess rule: a card URL is only ever taken from
a real fetched SCP listing anchor, never constructed or guessed).

Resolution rules:
1. mlb_id: MLB Stats API people/search?names={name}; exact fullName match after
   normalization (strip punctuation, fold case, fold accents — "Acuna/Acuña",
   "Garcia/García"; suffixes "Jr."/"II"/"III" stay distinctive), validated by a
   non-empty rookie-year game log fetch_game_log(mlb_id, ROLE_GROUP[role],
   rookie_year). Sleeps >=0.3s between MLB API calls.
2. flagship: SCP console listings baseball-cards-{rookie_year}-topps-chrome AND
   ...-topps-chrome-update with ?rookies-only=true&exclude-variants=true;
   name-matched base anchor under /game/<set>/ with word-boundary parallel
   exclusion (PARALLEL_RE, shared with resolve_seed_cards); no fallback to
   parallels; the plain topps-chrome set wins when both match. Known update-set
   players: Rutschman, Strider, Skenes, Kurtz, Anthony.
3. bowman_1st: console listings baseball-cards-{y}-bowman-chrome for y in
   rookie_year-6..rookie_year (?exclude-variants=true), scanned ascending and
   stopping at the first year with a valid match (earliest matching year wins);
   name-matched anchors with word-boundary exclusion of parallel/insert keywords
   (BOWMAN_EXCLUDE_RE); within the winning year an anchor whose text contains
   "1st" is preferred. Every name-matched (year, anchor, url) triple is kept in
   an audit trail printed at the end of the run for manual review.

Console pages are cached per set in-process; >=5s of sleep precedes every
uncached SCP fetch — politeness identical to scripts/resolve_seed_cards.py.

Run from the repo root with the venv active:
    PLAYWRIGHT_BROWSERS_PATH=.pw-browsers python scripts/resolve_universe.py
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from resolve_seed_cards import PARALLEL_RE, SCP_BASE, SCP_SLEEP_S, norm_name

from cardprice.collect_stats import ROLE_GROUP
from cardprice.stats_api import BASE, fetch_game_log
from cardprice.web import ChallengeError, fetch_page

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "data" / "reference" / "players_universe.csv"
CARDS = ROOT / "data" / "reference" / "cards_universe.csv"

MLB_SLEEP_S = 0.3
BOWMAN_LOOKBACK_YEARS = 6
CARD_TYPES = ("flagship", "bowman_1st")

BOWMAN_EXCLUDE_RE = re.compile(
    r"\b(?:Autograph|Auto|Refractor|Shimmer|Gold|Orange|Purple|Blue|Green|Red|Black"
    r"|Superfractor|Variation|Wave|Sparkle|Speckle|Mojo|Atomic|Lunar"
    r"|Rookie of the Year Favorites)\b",
    re.IGNORECASE,
)
FIRST_RE = re.compile(r"\b1st\b", re.IGNORECASE)

_pages_cache: dict[str, str] = {}


def _name_matches(name: str, text: str) -> bool:
    """Whole-word match of the full normalized name (suffixes included) inside
    normalized anchor text — "Nick Kurtz" does not match "Nick Kurtzney #1" and
    "Bobby Witt Jr" does not match "Bobby Witt III #99"."""
    tokens = norm_name(name).split()
    pattern = r"\b" + r"\s+".join(re.escape(t) for t in tokens) + r"\b"
    return re.search(pattern, norm_name(text)) is not None


def pick_flagship(name: str, anchors: list[tuple[str, str]]) -> str | None:
    """Href of the name-matched base flagship anchor from (text, href) pairs, or
    None when only parallels match (never a parallel — no guessing)."""
    base = [
        (text, href)
        for text, href in anchors
        if _name_matches(name, text) and not PARALLEL_RE.search(text)
    ]
    return base[0][1] if base else None


def pick_bowman_1st(
    name: str, candidates: list[tuple[int, str, str]]
) -> tuple[str | None, list[dict]]:
    """Pick the Bowman 1st URL from (year, anchor_text, href) candidates.

    Earliest matching year wins; within that year an anchor whose text contains
    "1st" is preferred; parallel/insert keywords exclude. Returns (url | None,
    audit) where audit keeps every candidate with its decision."""
    audit = []
    valid = []
    for year, text, href in sorted(candidates, key=lambda c: c[0]):
        entry = {"year": year, "anchor": text, "url": href}
        if not _name_matches(name, text):
            entry["decision"] = "skip:name-mismatch"
        elif BOWMAN_EXCLUDE_RE.search(text):
            entry["decision"] = "skip:parallel-keyword"
        else:
            entry["decision"] = "candidate"
            valid.append((year, text, href))
        audit.append(entry)
    if not valid:
        return None, audit
    best_year = valid[0][0]
    best = [(t, h) for y, t, h in valid if y == best_year]
    first = [b for b in best if FIRST_RE.search(b[0])]
    url = (first[0] if first else best[0])[1]
    for entry in audit:
        if entry["url"] == url and entry["decision"] == "candidate":
            entry["decision"] = "chosen"
            break
    return url, audit


def _search_people(name: str) -> list[dict]:
    resp = requests.get(f"{BASE}/people/search", params={"names": name}, timeout=30)
    resp.raise_for_status()
    time.sleep(MLB_SLEEP_S)
    data = resp.json()
    return data.get("people", []) if isinstance(data, dict) else data


def resolve_player_id(
    name: str, role: str | None = None, rookie_year: int | None = None
) -> int | None:
    """Exact normalized fullName match from people/search; when role and
    rookie_year are given, the match is validated by a non-empty rookie-year
    game log. None when no validated match exists."""
    target = norm_name(name)
    for person in _search_people(name):
        if norm_name(str(person.get("fullName", ""))) != target:
            continue
        mlb_id = int(person["id"])
        if role is None or rookie_year is None:
            return mlb_id
        splits = fetch_game_log(mlb_id, ROLE_GROUP[role], rookie_year)
        time.sleep(MLB_SLEEP_S)
        if splits:
            return mlb_id
        print(f"WARN {name}: empty {rookie_year} {ROLE_GROUP[role]} game log for {mlb_id}")
    print(f"WARN {name}: no validated mlb_id")
    return None


def fetch_scp(url: str) -> str:
    time.sleep(SCP_SLEEP_S)  # >=5s before every uncached SCP fetch
    return fetch_page(url)


def _console_page(cache_key: str, url: str) -> str:
    if cache_key not in _pages_cache:
        _pages_cache[cache_key] = fetch_scp(url)
    return _pages_cache[cache_key]


def _listing_anchors(html: str, set_slug: str) -> list[tuple[str, str]]:
    """(anchor_text, href) for every /game/<set_slug>/ link on a console page."""
    soup = BeautifulSoup(html, "html.parser")
    prefix = f"/game/{set_slug}/"
    return [
        (a.get_text(strip=True), a["href"])
        for a in soup.find_all("a", href=True)
        if a["href"].startswith(prefix)
    ]


def _resolve_flagship(name: str, rookie_year: int) -> tuple[str, str]:
    """(scp_url, set_slug); ('', '') when unresolved. Plain topps-chrome is
    preferred over topps-chrome-update when both yield a base match."""
    found: dict[str, tuple[str, str]] = {}
    for family in ("topps-chrome", "topps-chrome-update"):
        slug = f"baseball-cards-{rookie_year}-{family}"
        url = f"{SCP_BASE}/console/{slug}?rookies-only=true&exclude-variants=true"
        href = pick_flagship(name, _listing_anchors(_console_page(slug, url), slug))
        if href:
            found[family] = (SCP_BASE + href, slug)
    for family in ("topps-chrome", "topps-chrome-update"):
        if family in found:
            return found[family]
    return "", ""


def _resolve_bowman_1st(name: str, rookie_year: int) -> tuple[str, str, dict]:
    """(scp_url, set_slug, audit). Scans rookie_year-6..rookie_year ascending and
    stops at the first year with a valid match (earliest matching year wins).
    audit = {'scanned_years': [...], 'candidates': [pick_bowman_1st audit]}."""
    candidates_audit: list[dict] = []
    scanned: list[int] = []
    for year in range(rookie_year - BOWMAN_LOOKBACK_YEARS, rookie_year + 1):
        scanned.append(year)
        slug = f"baseball-cards-{year}-bowman-chrome"
        html = _console_page(slug, f"{SCP_BASE}/console/{slug}?exclude-variants=true")
        candidates = [
            (year, text, href)
            for text, href in _listing_anchors(html, slug)
            if _name_matches(name, text)
        ]
        href, year_audit = pick_bowman_1st(name, candidates)
        candidates_audit.extend(year_audit)
        if href:
            audit = {"scanned_years": scanned, "candidates": candidates_audit}
            return SCP_BASE + href, slug, audit
    return "", "", {"scanned_years": scanned, "candidates": candidates_audit}


def resolve_player_card_details(name: str, rookie_year: int) -> dict[str, tuple[str, str, dict]]:
    """card_type -> (scp_url, set_slug, audit) for both card families."""
    url, slug = _resolve_flagship(name, rookie_year)
    b_url, b_slug, audit = _resolve_bowman_1st(name, rookie_year)
    return {"flagship": (url, slug, {}), "bowman_1st": (b_url, b_slug, audit)}


def resolve_player_cards(name: str, rookie_year: int) -> dict[str, str]:
    """{'flagship': url, 'bowman_1st': url}; '' when unresolved (no guessing)."""
    details = resolve_player_card_details(name, rookie_year)
    return {card_type: url for card_type, (url, _slug, _audit) in details.items()}


def main() -> None:
    players = pd.read_csv(UNIVERSE, dtype=str).fillna("")
    rows: list[dict] = []
    audits: dict[str, dict] = {}
    resolved = {"mlb_id": 0, "flagship": 0, "bowman_1st": 0}
    for p in players.itertuples():
        name, role, year = str(p.player_name), str(p.role), int(p.rookie_year)
        mlb_id = resolve_player_id(name, role, year)
        try:
            details = resolve_player_card_details(name, year)
        except ChallengeError as e:
            print(f"WARN {name}: SCP challenge unresolved ({e}); card cells left empty")
            details = {ct: ("", "", {}) for ct in CARD_TYPES}
        audits[name] = details["bowman_1st"][2]
        id_str = str(mlb_id) if mlb_id is not None else ""
        for card_type in CARD_TYPES:
            url, slug, _audit = details[card_type]
            rows.append(
                {
                    "player_name": name,
                    "mlb_id": id_str,
                    "rookie_year": str(year),
                    "role": role,
                    "card_type": card_type,
                    "set_slug": slug,
                    "scp_url": url,
                }
            )
            resolved[card_type] += bool(url)
        resolved["mlb_id"] += mlb_id is not None
        print(
            f"{name}: mlb_id={id_str or '-'} "
            f"flagship={details['flagship'][0] or '-'} "
            f"bowman_1st={details['bowman_1st'][0] or '-'}"
        )
        pd.DataFrame(rows).to_csv(CARDS, index=False)  # incremental: survive a mid-run crash

    total = len(players)
    print(
        f"\nresolved mlb_id {resolved['mlb_id']}/{total}, "
        f"flagship {resolved['flagship']}/{total}, "
        f"bowman_1st {resolved['bowman_1st']}/{total} -> {CARDS}"
    )
    print("\n=== BOWMAN_1ST AUDIT TRAIL (all name-matched candidates considered) ===")
    for name in players["player_name"]:
        audit = audits.get(name) or {}
        scanned = audit.get("scanned_years") or []
        span = f"{scanned[0]}-{scanned[-1]}" if scanned else "-"
        print(f"\n{name} (years scanned: {span}):")
        entries = audit.get("candidates") or []
        if not entries:
            print("  (no name-matched Bowman Chrome anchors in scanned years)")
        for e in entries:
            print(f"  {e['year']} [{e['decision']}] {e['anchor']} -> {e['url']}")


if __name__ == "__main__":
    main()
