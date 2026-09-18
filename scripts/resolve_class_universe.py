# scripts/resolve_class_universe.py
"""Class-universe assembly: checklist players -> mlb_id + base-card URLs.

Incremental CSV flush per player (resolve_universe idiom); mlb_id via MLB
people/search + yearByYear-hydrate pro-stats validation (any level) with a
curated nickname map and an MiLB player-directory fallback; base cards from
Task 1's saved base-page snapshots (parsed with checklist.parse_set_page —
real anchors, no-guess).
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from resolve_universe import _name_matches, norm_name

from cardprice.checklist import parse_set_page
from cardprice.stats_api import BASE, fetch_player_info
from cardprice.storage import RAW_ROOT, load_latest, save_raw

OUT_CARDS = Path("data/reference/cards_class_universe.csv")
OUT_AUDIT = Path("data/reference/class_universe_audit.csv")
OUT_INFO = Path("data/reference/player_info_class.csv")
CARD_TYPE = "bowman_1st_base"
DIR_SPORT_IDS = (11, 12, 13, 14, 15, 16, 17)  # aaa, aa, a_plus, a + rookie/DSL levels
DIR_SEASONS = range(2014, 2027)  # 2014..2026 inclusive
SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}
# Curated search strings for known nickname/initial/hyphen forms (plan
# correction 4): folded form -> folded form. The map proposes the search
# string; the normal search+validation path still decides acceptance.
NICKNAME_MAP = {
    "joe wendle": "joseph wendle",  # MLB IF Joe Wendle
    "giovanny urshela": "gio urshela",  # MLB 3B
    "ozhaino albies": "ozzie albies",  # MLB 2B (full name Ozhaino)
    "jose adolis garcia": "adolis garcia",  # MLB OF
    "jacob faria": "jake faria",  # MLB RHP
    "pete crow armstrong": "pete crow-armstrong",  # MLB OF (hyphenated)
    "christian encarnacion strand": "christian encarnacion-strand",  # MLB 1B
    "justyn henry malloy": "justyn-henry malloy",  # MLB OF
    "hoy jun park": "hoy park",  # MLB IF (probe-verify at runtime)
    "hao yu lee": "hao-yu lee",  # Giants IF (probe-verify)
    "won bin cho": "won-bin cho",  # probe-verify
    "nikau pouaka grego": "nikau pouaka-grego",  # probe-verify
    "t j white": "tj white",  # probe-verify (SCP initials form "T. J. White")
    "c j kayfus": "cj kayfus",  # probe-verify (SCP initials form "C. J. Kayfus")
}
# yearByYear without sportId is MLB-only for many players; careers that never
# reached AAA (measured: Gamboa/Smith/Gibson) have splits only at 12-17 —
# iterate hydrate variants in this order, short-circuit on first non-empty split
_HYDRATE_SPORT_IDS: tuple[int | None, ...] = (None, 14, 13, 12, 16, 11, 17, 15)


def players_from_checklists(checklists: pd.DataFrame) -> pd.DataFrame:
    """One row per unique player: earliest class year; family at that year (tie -> chrome)."""
    rows = []
    for _, g in checklists.groupby(checklists["name"].map(norm_name)):
        # year asc, then family asc ("chrome" < "draft") so a same-year tie picks chrome
        g = g.sort_values(["year", "family"], ascending=[True, True])
        first = g.iloc[0]
        rows.append(
            {
                "player_name": first["name"], "class_year": int(first["year"]),
                "family": first["family"], "auto_card_url": first["card_url"],
            }
        )
    return pd.DataFrame(rows).sort_values(["class_year", "player_name"]).reset_index(drop=True)


def resolve_class_player_id(
    name: str, class_year: int, sleep_s: float = 0.3, directory: dict | None = None
) -> tuple[int | None, str]:
    """Three-stage id resolution. Never returns an unvalidated id.

    1. people/search exact normalized match, each candidate validated by
       _has_pro_stats: exactly one validated -> "ok"; two or more ->
       "ambiguous:multiple validated" (never a silent API-order pick). If none
       validate and the folded name is in NICKNAME_MAP, the search stage is
       retried once with the curated search string (same full validation path).
    2. Directory fallback (search found no candidate, or none validated):
       accent-folded match against the snapshot-cached MiLB player directories
       — roster presence on a level-season IS the affiliation evidence
       -> "ok:directory".
    3. Suffix rule (inside 2, bidirectional): candidate = query tokens + one
       trailing token in {jr, sr, ii, iii, iv, v}, or query minus its own
       trailing suffix token = candidate.

    (Correction history in the plan doc; round 4 2026-09-18: multi-validated =
    ambiguous, resolution_source provenance, bidirectional suffix, NICKNAME_MAP.)"""
    exact, validated = _search_validated_ids(name, sleep_s)
    if not validated:
        alt = NICKNAME_MAP.get(norm_name(name))
        if alt is not None:
            exact, validated = _search_validated_ids(alt, sleep_s)
    if len(validated) == 1:
        return validated[0], "ok"
    if len(validated) > 1:
        ids = ", ".join(str(i) for i in sorted(validated))
        return None, f"ambiguous:multiple validated ({ids})"
    if directory is None:
        directory = load_player_directory(sleep_s)
    dir_id, dir_status = _directory_lookup(directory, name, class_year)
    if dir_status is not None:
        return dir_id, dir_status
    if not exact:
        return None, "no_exact_match"
    if len(exact) > 1:  # validated is empty here, so every exact candidate failed
        statuses = [f"candidate {int(c['id'])} without pro stats" for c in exact]
        return None, "ambiguous:" + "; ".join(statuses)
    return None, "no_pro_stats"


def _search_validated_ids(name: str, sleep_s: float) -> tuple[list[dict], list[int]]:
    """people/search exact normalized matches + _has_pro_stats validation
    -> (exact candidates, ids that validated), both in API order."""
    resp = requests.get(f"{BASE}/people/search", params={"names": name}, timeout=30)
    resp.raise_for_status()
    time.sleep(sleep_s)
    people = resp.json().get("people", [])
    exact = [p for p in people if norm_name(p.get("fullName", "")) == norm_name(name)]
    validated = [int(c["id"]) for c in exact if _has_pro_stats(int(c["id"]), sleep_s)]
    return exact, validated


def audit_multi_validated(players: pd.DataFrame, sleep_s: float = 0.3) -> pd.DataFrame:
    """Re-validation audit (correction 4): for each already-mapped player, re-run
    the search stage and report rows whose exact-candidate set now yields >= 2
    validated ids. Pure query function — never writes."""
    rows = []
    for p in players.itertuples():
        _, validated = _search_validated_ids(p.player_name, sleep_s)
        if len(validated) >= 2:
            rows.append(
                {
                    "player_name": p.player_name, "current_mlb_id": int(p.mlb_id),
                    "validated_ids": ",".join(str(i) for i in sorted(validated)),
                }
            )
    return pd.DataFrame(rows, columns=["player_name", "current_mlb_id", "validated_ids"])


def _pro_stat_splits(mlb_id: int, sleep_s: float) -> list[dict]:
    """yearByYear splits from the first hydrate variant (see _HYDRATE_SPORT_IDS)
    that has any non-empty split; [] when every variant is empty. Same call
    sequence and short-circuit as _has_pro_stats."""
    for sport_id in _HYDRATE_SPORT_IDS:
        hydrate = "stats(group=[hitting,pitching],type=yearByYear"
        hydrate += f",sportId={sport_id})" if sport_id is not None else ")"
        resp = requests.get(f"{BASE}/people/{mlb_id}", params={"hydrate": hydrate}, timeout=30)
        resp.raise_for_status()
        time.sleep(sleep_s)
        people = resp.json().get("people", [])
        if not people:
            continue
        splits = [s for g in people[0].get("stats", []) for s in g.get("splits", [])]
        if splits:
            return splits
    return []


def _has_pro_stats(mlb_id: int, sleep_s: float) -> bool:
    """GET /people/{id}?hydrate=stats(group=[hitting,pitching],type=yearByYear[,sportId=<sid>])
    for sid in [none, 14, 13, 12, 16, 11, 17, 15] — True iff any variant has a
    non-empty split; short-circuits on the first. (Third correction 2026-09-18:
    sportId=11-only missed careers that topped out below AAA; without sportId
    the hydrate is MLB-only for many players.)"""
    return bool(_pro_stat_splits(mlb_id, sleep_s))


def disambiguate_by_career_timing(
    class_year: int, validated_ids: list[int], sleep_s: float = 0.3
) -> tuple[int | None, str]:
    """Career-timing disambiguation for multi-validated resolutions: a 1st
    Bowman auto is issued within a few years of a player's first pro season, so
    keep candidates with class_year - 6 <= career_start <= class_year + 3
    (career_start = earliest season across their yearByYear splits). Exactly
    one survivor -> (id, "ok:timing"); zero or >= 2 -> (None,
    "ambiguous:timing"). Candidates with no splits at all are eliminated."""
    survivors = []
    for mlb_id in validated_ids:
        seasons = {int(s["season"]) for s in _pro_stat_splits(mlb_id, sleep_s) if s.get("season")}
        if seasons and class_year - 6 <= min(seasons) <= class_year + 3:
            survivors.append(mlb_id)
    if len(survivors) == 1:
        return survivors[0], "ok:timing"
    return None, "ambiguous:timing"


def load_player_directory(sleep_s: float = 0.3) -> dict[tuple[int, int], list[dict]]:
    """MiLB player directories {(sport_id, season): [{"id", "fullName"}]} for
    sport_id 11-17 x seasons 2014-2026 (91 cells; sport-15 cells may be empty in
    recent years — tolerated as gaps, still snapshotted). Snapshot-cached
    (save_raw/load_latest — one live call per missing cell); fetched once per
    run, shared across players."""
    directory: dict[tuple[int, int], list[dict]] = {}
    for sport_id in DIR_SPORT_IDS:
        for season in DIR_SEASONS:
            key = f"{sport_id}_{season}"
            try:
                payload = load_latest("stats_directory", key)
            except FileNotFoundError:
                resp = requests.get(
                    f"{BASE}/sports/{sport_id}/players", params={"season": season}, timeout=60
                )
                resp.raise_for_status()
                payload = resp.json()
                save_raw("stats_directory", key, payload)
                time.sleep(sleep_s)
            directory[(sport_id, season)] = [
                {"id": int(p["id"]), "fullName": p.get("fullName", "")}
                for p in payload.get("people", [])
            ]
    return directory


def _fold_tokens(name: str) -> list[str]:
    """Accent-folded, punctuation-free lowercase name tokens (norm_name basis)."""
    return ["".join(ch for ch in tok if ch.isalnum()) for tok in norm_name(name).split()]


def _directory_lookup(
    directory: dict[tuple[int, int], list[dict]], name: str, class_year: int
) -> tuple[int | None, str | None]:
    """-> (mlb_id, "ok:directory") | (None, "ambiguous:...") | (None, None).

    Accent-folded exact fullName match, else the suffix rule — bidirectional:
    candidate = query tokens + one trailing suffix token, or (when the query
    itself ends in a suffix token) query minus that token = candidate. Hits in
    several (level, season) cells -> the cell nearest class_year wins; >1
    distinct id in that cell -> ambiguous."""
    query = _fold_tokens(name)
    queries = [query]
    if query and query[-1] in SUFFIX_TOKENS:
        queries.append(query[:-1])  # bidirectional: the query carries the suffix
    exact_hits, suffix_hits = [], []
    for (sport_id, season), people in directory.items():
        for p in people:
            cand = _fold_tokens(p["fullName"])
            if any(cand == q for q in queries):
                exact_hits.append((sport_id, season, p["id"]))
            elif (len(cand) == len(query) + 1 and cand[:-1] == query
                  and cand[-1] in SUFFIX_TOKENS):
                suffix_hits.append((sport_id, season, p["id"]))
    hits = exact_hits or suffix_hits
    if not hits:
        return None, None
    by_cell: dict[tuple[int, int], set[int]] = {}
    for sport_id, season, mlb_id in hits:
        by_cell.setdefault((sport_id, season), set()).add(mlb_id)
    cell = min(by_cell, key=lambda c: (abs(c[1] - class_year), c[1], c[0]))
    ids = sorted(by_cell[cell])
    if len(ids) > 1:
        dupes = "; ".join(f"candidate {i} at sport {cell[0]} season {cell[1]}" for i in ids)
        return None, "ambiguous:directory duplicate " + dupes
    return ids[0], "ok:directory"


def _slug_from_card_url(card_url: str) -> str:
    """Set slug from a /game/<slug>/ card URL; '' when the URL has no /game/ path."""
    if "/game/" not in card_url:
        return ""
    return card_url.split("/game/")[1].split("/")[0]


def resolve_base_cards(
    players: pd.DataFrame, base_rows_by_year: dict[int, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each player to their base 1st Bowman (BCP/BDC) row: exact name match,
    year in [class_year, class_year+3], earliest year wins (tie -> checklist family)."""
    resolved, unresolved = [], []
    for p in players.itertuples():
        cands = []
        for year, rows in base_rows_by_year.items():
            if not (p.class_year <= year <= p.class_year + 3):
                continue
            for r in rows.itertuples():
                if _name_matches(p.player_name, r.name):
                    fam_bonus = 0 if (("draft" in r.card_url) == (p.family == "draft")) else 1
                    cands.append((year, fam_bonus, r))
        if not cands:
            unresolved.append({"player_name": p.player_name, "class_year": p.class_year,
                               "reason": "no_base_card"})
            continue
        cands.sort(key=lambda c: (c[0], c[1]))
        year, _, r = cands[0]
        resolved.append(
            {
                "player_name": p.player_name, "set_slug": _slug_from_card_url(r.card_url),
                "scp_url": r.card_url, "base_number": r.number, "base_year": year,
            }
        )
    return pd.DataFrame(resolved), pd.DataFrame(unresolved)


def _load_base_rows() -> dict[int, pd.DataFrame]:
    """Parse Task 1's saved base-page snapshots (chrome base + draft base)."""
    by_year: dict[int, list[pd.DataFrame]] = {}
    root = RAW_ROOT / "scp_setpages"
    for d in sorted(root.iterdir()):
        if not d.is_dir() or ("prospects" not in d.name and "draft-chrome" not in d.name):
            continue
        if "autograph" in d.name:
            continue
        year = int(d.name.split("-")[2])  # baseball-cards-<year>-...
        try:
            payload = load_latest("scp_setpages", d.name)
        except FileNotFoundError:
            continue
        for html in payload.get("pages", [payload["html"]]):
            by_year.setdefault(year, []).append(parse_set_page(html))
    return {year: pd.concat(frames, ignore_index=True) for year, frames in by_year.items()}


def _fetch_player_info_batched(mlb_ids: list[int], sleep_s: float = 0.3) -> pd.DataFrame:
    """fetch_player_info in <=100-id chunks (statsapi 414s URI Too Long on long
    personIds lists — measured at ~800 ids) and concat."""
    frames = []
    for i in range(0, len(mlb_ids), 100):
        frames.append(fetch_player_info(mlb_ids[i : i + 100]))
        time.sleep(sleep_s)
    if not frames:
        return pd.DataFrame(columns=["mlb_id", "name", "birth_date", "position"])
    return pd.concat(frames, ignore_index=True)


def _universe_row(p, mlb_id: int | None, status: str) -> dict:
    """One cards_class_universe row; resolution_source records the provenance of
    the id ("search" = stats-validated, "directory" = roster-presence)."""
    return {
        "player_name": p.player_name, "mlb_id": mlb_id, "class_year": p.class_year,
        "family": p.family, "role": None, "card_type": CARD_TYPE,
        "set_slug": None, "scp_url": None, "auto_card_url": p.auto_card_url,
        "resolution_source": ("search" if status == "ok"
                              else "directory" if status == "ok:directory" else ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checklists", default="data/reference/class_checklists.parquet")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    players = players_from_checklists(pd.read_parquet(args.checklists))
    print("unique players:", len(players))
    base_rows = _load_base_rows()
    print("base years loaded:", sorted(base_rows))
    directory = load_player_directory(args.sleep)
    print("player directory cells:", len(directory))

    out_path = OUT_CARDS
    done = set()
    if out_path.exists():  # resume: skip players already in the CSV
        existing = pd.read_csv(out_path)
        done = set(existing["player_name"])
        if "resolution_source" not in existing.columns:
            # pre-round-4 CSV: add the column (provenance unknown) so per-player
            # appends stay column-aligned; the final rewrite carries it through
            existing["resolution_source"] = ""
            existing.to_csv(out_path, index=False)
        print("resuming;", len(done), "already resolved")

    audits = []
    for p in players.itertuples():
        if p.player_name in done:
            continue
        mlb_id, status = resolve_class_player_id(p.player_name, p.class_year, args.sleep,
                                                 directory=directory)
        row = _universe_row(p, mlb_id, status)
        if not status.startswith("ok"):  # "ok" and "ok:directory" are both resolutions
            audits.append(pd.DataFrame([{"player_name": p.player_name, "class_year": p.class_year,
                                         "reason": status}]))
        pd.DataFrame([row]).to_csv(
            out_path, mode="a", header=not out_path.exists(), index=False
        )
        if (len(done) + 1) % 100 == 0:
            print("resolved", len(done) + 1, "of", len(players))
        done.add(p.player_name)

    cards = pd.read_csv(out_path)
    ok = cards.dropna(subset=["mlb_id"]).copy()
    ok["mlb_id"] = ok["mlb_id"].astype(int)
    info = _fetch_player_info_batched(sorted(ok["mlb_id"].unique()), args.sleep)
    info.to_csv(OUT_INFO, index=False)
    role_of = dict(
        zip(info["mlb_id"], info["position"].map(lambda pos: "pitcher" if pos == "P" else "hitter"))
    )
    ok["role"] = ok["mlb_id"].map(role_of)

    resolved, unresolved = resolve_base_cards(ok, base_rows)
    if len(resolved):
        ok = ok.merge(resolved[["player_name", "set_slug", "scp_url"]], on="player_name",
                      how="left", suffixes=(None, "_r"))
        ok["set_slug"] = ok["set_slug_r"].fillna(ok["set_slug"])
        ok["scp_url"] = ok["scp_url_r"].fillna(ok["scp_url"])
        ok = ok.drop(columns=["set_slug_r", "scp_url_r"])
    ok.to_csv(out_path, index=False)
    audits.append(unresolved.assign(reason="no_base_card"))
    # audits holds DataFrames only (pd.concat rejects dicts); guard the empty case
    audit_df = pd.concat(audits) if audits else pd.DataFrame(
        columns=["player_name", "class_year", "reason"]
    )
    audit_df.to_csv(OUT_AUDIT, index=False)
    print("cards:", len(ok), "of", len(players),
          "| mlb_id mapping rate:", round(len(ok) / len(players), 3),
          "| audit rows:", len(audit_df),
          "| base-card hit rate:", round(ok["scp_url"].notna().mean(), 3))


if __name__ == "__main__":
    main()
