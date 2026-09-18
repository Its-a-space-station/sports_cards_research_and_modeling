# scripts/resolve_class_universe.py
"""Class-universe assembly: checklist players -> mlb_id + base-card URLs.

Incremental CSV flush per player (resolve_universe idiom); mlb_id via MLB
people/search + yearByYear-hydrate pro-stats validation (any level); base
cards from Task 1's saved base-page snapshots (parsed with
checklist.parse_set_page — real anchors, no-guess).
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
from cardprice.storage import RAW_ROOT, load_latest

OUT_CARDS = Path("data/reference/cards_class_universe.csv")
OUT_AUDIT = Path("data/reference/class_universe_audit.csv")
OUT_INFO = Path("data/reference/player_info_class.csv")
CARD_TYPE = "bowman_1st_base"


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
    name: str, class_year: int, sleep_s: float = 0.3
) -> tuple[int | None, str]:
    """people/search exact normalized match + affiliation validation via one
    yearByYear hydrate call per candidate. Never returns an unvalidated id.

    (Corrected 2026-09-18: the original fetch_game_log validation was MLB-level
    only, failing every prospect whose MLB debut fell outside the class window —
    measured 27% mapping, caught by the <70% stop-trigger.)"""
    resp = requests.get(f"{BASE}/people/search", params={"names": name}, timeout=30)
    resp.raise_for_status()
    time.sleep(sleep_s)
    people = resp.json().get("people", [])
    exact = [p for p in people if norm_name(p.get("fullName", "")) == norm_name(name)]
    if not exact:
        return None, "no_exact_match"
    statuses = []
    for cand in exact:
        mlb_id = int(cand["id"])
        if _has_pro_stats(mlb_id, sleep_s):
            return mlb_id, "ok"
        statuses.append(f"candidate {mlb_id} without pro stats")
    if len(exact) > 1:
        return None, "ambiguous:" + "; ".join(statuses)
    return None, "no_pro_stats"


def _has_pro_stats(mlb_id: int, sleep_s: float) -> bool:
    """One call: GET /people/{id}?hydrate=stats(group=[hitting,pitching],type=yearByYear)
    — True iff any non-empty pro-stats split exists (any level, any year)."""
    resp = requests.get(
        f"{BASE}/people/{mlb_id}",
        params={"hydrate": "stats(group=[hitting,pitching],type=yearByYear)"},
        timeout=30,
    )
    resp.raise_for_status()
    time.sleep(sleep_s)
    people = resp.json().get("people", [])
    if not people:
        return False
    for stat_group in people[0].get("stats", []):
        if stat_group.get("splits"):
            return True
    return False


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checklists", default="data/reference/class_checklists.parquet")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    players = players_from_checklists(pd.read_parquet(args.checklists))
    print("unique players:", len(players))
    base_rows = _load_base_rows()
    print("base years loaded:", sorted(base_rows))

    out_path = OUT_CARDS
    done = set()
    if out_path.exists():  # resume: skip players already in the CSV
        done = set(pd.read_csv(out_path)["player_name"])
        print("resuming;", len(done), "already resolved")

    audits = []
    for p in players.itertuples():
        if p.player_name in done:
            continue
        mlb_id, status = resolve_class_player_id(p.player_name, p.class_year, args.sleep)
        row = {
            "player_name": p.player_name, "mlb_id": mlb_id, "class_year": p.class_year,
            "family": p.family, "role": None, "card_type": CARD_TYPE,
            "set_slug": None, "scp_url": None, "auto_card_url": p.auto_card_url,
        }
        if status != "ok":
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
    info = fetch_player_info(sorted(ok["mlb_id"].unique()))
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
