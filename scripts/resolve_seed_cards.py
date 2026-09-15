# scripts/resolve_seed_cards.py
"""Fill mlb_id + scp_url in data/reference/cards_seed.csv (network required).

mlb_id: MLB Stats API season player directory (one call per unique season,
cached in-process), exact match on fullName (case/diacritics/periods
normalized), validated by a non-empty rookie-year game log. Sleeps 0.3s
between API calls.
scp_url: SCP console listing per set with ?rookies-only=true&exclude-variants=
true (complete rookie base-card list; cached in-process), anchor whose text
contains the player's first + last name and whose href starts with
/game/<set_slug>/, keeping only anchors without parallel keywords (no parallel
fallback — unresolved stays empty). Falls back to SCP search
(challenge-blocked headless as of 2026-09-15 — expected to warn).
Sleeps >=5s between SCP fetches. Unresolved cells stay empty.

Run from repo root with the venv active:
    python scripts/resolve_seed_cards.py
"""

import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from cardprice.collect_stats import ROLE_GROUP
from cardprice.stats_api import BASE, fetch_game_log
from cardprice.web import ChallengeError, fetch_page

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "reference" / "cards_seed.csv"
SCP_BASE = "https://www.sportscardspro.com"
SCP_SLEEP_S = 5.0
PARALLEL_RE = re.compile(
    r"\b(?:Refractor|Autograph|Gold|Orange|Purple|Blue|Green|Red|Superfractor|Variation)\b",
    re.IGNORECASE,
)
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

_players_cache: dict[int, list[dict]] = {}
_pages_cache: dict[str, str] = {}


def season_players(season: int) -> list[dict]:
    if season not in _players_cache:
        resp = requests.get(f"{BASE}/sports/1/players", params={"season": season}, timeout=30)
        resp.raise_for_status()
        _players_cache[season] = resp.json().get("people", [])
        time.sleep(0.3)
    return _players_cache[season]


def norm_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).replace(".", "").lower()


def resolve_mlb_id(name: str, role: str, season: int) -> int | None:
    target = norm_name(name)
    for person in season_players(season):
        if norm_name(person.get("fullName", "")) != target:
            continue
        mlb_id = int(person["id"])
        time.sleep(0.3)
        if fetch_game_log(mlb_id, ROLE_GROUP[role], season):
            return mlb_id
        print(f"WARN {name}: empty {season} {ROLE_GROUP[role]} game log for {mlb_id}")
    print(f"WARN {name}: no validated mlb_id for {season}")
    return None


def name_parts(name: str) -> tuple[str, str]:
    parts = norm_name(name).split()
    while len(parts) > 2 and parts[-1] in NAME_SUFFIXES:
        parts.pop()
    return parts[0], parts[-1]


def pick_anchor(html: str, name: str, set_slug: str) -> str | None:
    """Href of the player's base card, or None. Parallel-only matches return
    None (no guessing): parallels are filtered by word-boundary keyword match
    on the anchor text."""
    soup = BeautifulSoup(html, "html.parser")
    first, last = name_parts(name)
    matches = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith(f"/game/{set_slug}/"):
            continue
        text = a.get_text(strip=True).lower()
        if first in text and last in text:
            matches.append((a.get_text(strip=True), href))
    base = [(t, h) for t, h in matches if not PARALLEL_RE.search(t)]
    if not base:
        return None
    return base[0][1]


def fetch_scp(url: str) -> str:
    time.sleep(SCP_SLEEP_S)
    return fetch_page(url)


def resolve_scp_url(name: str, set_slug: str, rookie_year: int) -> str:
    if set_slug not in _pages_cache:
        _pages_cache[set_slug] = fetch_scp(
            f"{SCP_BASE}/console/{set_slug}?rookies-only=true&exclude-variants=true"
        )
    href = pick_anchor(_pages_cache[set_slug], name, set_slug)
    if href is None:
        query = "+".join(name.split()) + f"+{rookie_year}+Topps+Chrome"
        try:
            search_html = fetch_scp(f"{SCP_BASE}/search-products?q={query}")
        except ChallengeError as e:
            print(f"WARN {name}: SCP search challenge unresolved ({e})")
            search_html = ""
        href = pick_anchor(search_html, name, set_slug) if search_html else None
    if href is None:
        print(f"WARN {name}: no SCP card URL resolved")
        return ""
    return SCP_BASE + href


def main() -> None:
    cards = pd.read_csv(SEED, dtype=str).fillna("")
    for i, card in cards.iterrows():
        mlb_id = resolve_mlb_id(card["player_name"], card["role"], int(card["rookie_year"]))
        cards.loc[i, "mlb_id"] = str(mlb_id) if mlb_id is not None else ""
        cards.loc[i, "scp_url"] = resolve_scp_url(
            card["player_name"], card["set_slug"], int(card["rookie_year"])
        )
        print(f"{card['player_name']}: mlb_id={mlb_id} scp_url={cards.loc[i, 'scp_url']}")
    cards.to_csv(SEED, index=False)
    resolved = int((cards["mlb_id"] != "").sum()), int((cards["scp_url"] != "").sum())
    print(f"\nresolved {resolved[0]}/{len(cards)} mlb_id, {resolved[1]}/{len(cards)} scp_url -> {SEED}")


if __name__ == "__main__":
    main()
