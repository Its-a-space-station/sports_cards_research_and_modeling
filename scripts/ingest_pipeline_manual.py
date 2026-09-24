# scripts/ingest_pipeline_manual.py
"""Ingest manually-captured MLB Pipeline Top-100 pages into expectations.parquet.

Sources: HTML pages saved by the user under "MLB stats info/" (mlb.com is
bot-blocked for our fetches but public for a human browser). Per-year parsers
for the three server-rendered formats found in the captures; unmatched names
are kept with mlb_id = <NA> and reported (never fuzzy-matched).

Years 2021+ use the React year-page format: the saved HTML embeds an Apollo
GraphQL cache that holds the full Top-100 list (the server-rendered table
shows only 5 rows, but the cache payload has all 100). Rank and mlb_id come
straight from RankedPlayerEntity entries and their Person refs, so no name
resolution is needed for those years.

Provenance: snapshots captured 2026-09-24. Archived 2021-2025 year pages were
verified to show preseason list state (no same-year draftees among list
members), so as_of = <season>-04-01 is defensible; the 2026 page state could
not be independently verified and is treated the same way.
"""

import html
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from resolve_universe import norm_name  # noqa: E402

SOURCE_DIR = Path("MLB stats info")
OUT = Path("data/processed/expectations.parquet")
AUDIT = Path("data/reference/pipeline_manual_audit.csv")

FILES = {
    2015: "2015 Top 100 MLB Prospects list.html",
    2016: "2016 Top 100 MLB Prospects list.html",
    2018: "2018 Top 100 MLB Prospects list.html",
    **{y: f"{y} Top 100 Baseball Prospects _ MiLB.com_2.html" for y in range(2021, 2027)},
}
APOLLO_YEARS = set(range(2021, 2027))
TEAM_FIX = {"PHi": "PHI", "MN": "MIN", "PH": "PHI"}


def _load(path: Path) -> str:
    return BeautifulSoup(
        open(path, encoding="utf-8", errors="replace"), "html.parser"
    ).get_text("\n", strip=True)


def parse_2015(text: str) -> pd.DataFrame:
    """'N. Name, POS, TEAM' numbered rows (also tolerates '100 .' and a
    missing comma between position and team, e.g. '1B PIT')."""
    rows = []
    for line in text.split("\n"):
        m = re.match(r"(\d{1,3})\s*\.\s*(.+?),\s*([A-Z0-9/ ]+?)(?:,\s*|\s+)([A-Z]{2,3})$", line)
        if m:
            rows.append((int(m.group(1)), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()))
    return pd.DataFrame(rows, columns=["rank", "player_name", "position", "team"])


def parse_linewalk(text: str, first_marker: str) -> pd.DataFrame:
    """'Name, POS, TEAM' rows in rank order; tolerates split names."""
    seg = text[text.find(first_marker):]
    lines = [l for l in seg.split("\n") if l.strip()]
    rows, cur = [], None
    for line in lines:
        m = re.match(r"^(.+?),\s*([A-Z0-9/ ]+),\s*([A-Za-z]{2,3})$", line)
        if m:
            name = ((cur or "") + m.group(1)).strip(" .")
            rows.append((re.sub(r"\s+", " ", name), m.group(2).strip(), m.group(3).strip()))
            cur = None
        elif not re.match(r"^[•\[]", line) and "Season-end" not in line:
            cur = ((cur or "") + line.strip()).strip()
        if re.search(r"Season-end", line):
            break
    df = pd.DataFrame(rows, columns=["player_name", "position", "team"])
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def parse_2018(text: str) -> pd.DataFrame:
    """Split-line 'Name\n, POS, TEAM' rows in rank order."""
    seg = text[text.find("Shohei Ohtani"):]
    rows = re.findall(r"([A-ZÀ-Ý][A-Za-zÀ-ÿ'. -]{2,40})\n,\s*([A-Z0-9/ ]+),\s*([A-Za-z]{2,3})\n", seg)
    df = pd.DataFrame([(n.strip(), p.strip(), t.strip()) for n, p, t in rows],
                      columns=["player_name", "position", "team"])
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def parse_apollo(path: Path) -> pd.DataFrame:
    """React year-page (2021+): parse the embedded Apollo cache payload.

    RankedPlayerEntity entries carry rank + a Person ref; normalized Person
    entries carry useName/useLastName/primaryPosition. Returns rows with
    mlb_id already resolved."""
    dec = html.unescape(open(path, encoding="utf-8", errors="replace").read())
    persons = {}
    for m in re.finditer(r'"Person:(\d+)":\{"__typename":"Person",', dec):
        win = dec[m.start():m.start() + 1500]
        first = re.search(r'"useName":"([^"]*)"', win)
        last = re.search(r'"useLastName":"([^"]*)"', win)
        pos = re.search(r'"primaryPosition":\{"__typename":"Position","abbreviation":"([^"]*)"', win)
        persons[int(m.group(1))] = (
            f"{first.group(1)} {last.group(1)}".strip() if first and last else None,
            pos.group(1) if pos else None,
        )
    rows = []
    for m in re.finditer(r'"__typename":"RankedPlayerEntity","rank":(\d+),"playerEntity":\{', dec):
        win = dec[m.end():m.end() + 3000]
        ref = re.search(r'"player":\{"__ref":"Person:(\d+)"\}', win)
        pos = re.search(r'"position":"([^"]*)"', win)
        if not ref:
            continue
        pid = int(ref.group(1))
        name, ppos = persons.get(pid, (None, None))
        rows.append((int(m.group(1)), name, pos.group(1) if pos else ppos, pid))
    df = pd.DataFrame(rows, columns=["rank", "player_name", "position", "mlb_id"])
    df["team"] = pd.NA
    return df


PARSERS = {
    2015: parse_2015,
    2016: lambda t: parse_linewalk(t, "Core\ny\nSeager"),
    2018: parse_2018,
}


def _clean(df: pd.DataFrame, season: int) -> pd.DataFrame:
    df = df.copy()
    df["season"] = season
    df["team"] = df["team"].map(lambda t: TEAM_FIX.get(t, t))
    df["player_name"] = df["player_name"].str.replace(r"\s+", " ", regex=True).str.strip(" .")
    cols = ["season", "rank", "player_name", "position", "team"]
    if "mlb_id" in df.columns:
        cols.append("mlb_id")
    return df[cols]


def _map_ids(df: pd.DataFrame, id_of: dict) -> pd.DataFrame:
    df = df.copy()
    if "mlb_id" not in df.columns:
        df["mlb_id"] = pd.NA
    df["mlb_id"] = df["mlb_id"].astype("Int64")
    missing = df["mlb_id"].isna()
    df.loc[missing, "mlb_id"] = df.loc[missing, "player_name"].map(
        lambda n: id_of.get(norm_name(n), pd.NA)
    )
    df["mlb_id"] = df["mlb_id"].astype("Int64")
    return df


def main() -> None:
    frames = []
    for season, fname in FILES.items():
        if season in APOLLO_YEARS:
            df = _clean(parse_apollo(SOURCE_DIR / fname), season)
        else:
            text = _load(SOURCE_DIR / fname)
            df = _clean(PARSERS[season](text), season)
        frames.append(df)
        warn = "  <-- WARN: incomplete capture" if len(df) < 95 else ""
        print(f"{season}: {len(df)} rows (ranks {df['rank'].min()}-{df['rank'].max()}){warn}")
    parsed = pd.concat(frames, ignore_index=True)

    info = pd.read_csv("data/reference/player_info_class.csv")
    cards = pd.read_csv("data/reference/cards_class_universe.csv")
    id_of = {norm_name(n): int(i) for n, i in zip(info["name"], info["mlb_id"])}
    for n, i in zip(cards["player_name"], cards["mlb_id"]):
        if pd.notna(i):
            id_of.setdefault(norm_name(n), int(i))
    parsed = _map_ids(parsed, id_of)
    unmatched = parsed[parsed["mlb_id"].isna()]
    print(f"mapped {parsed['mlb_id'].notna().sum()} of {len(parsed)}; unmatched: {len(unmatched)}")
    if len(unmatched):
        unmatched.to_csv(AUDIT, index=False)
        print("audit:", AUDIT)

    universe_ids = set(id_of.values())
    apollo = parsed[parsed["season"].isin(APOLLO_YEARS)]
    in_universe = apollo["mlb_id"].isin(universe_ids)
    print("apollo rows with mlb_id in card universe:",
          in_universe.groupby(apollo["season"]).sum().to_dict())

    out = parsed.assign(
        source="pipeline", fv=pd.Series(pd.NA, dtype="Float64"),
        as_of=parsed["season"].map(lambda y: pd.Timestamp(f"{y}-04-01")),
    )[["player_name", "mlb_id", "season", "source", "rank", "fv", "as_of"]]
    out["rank"] = out["rank"].astype("Int64")

    if OUT.exists():
        old = pd.read_parquet(OUT)
        kept = old[old["source"] != "pipeline"]
        combined = pd.concat([kept, out], ignore_index=True)
    else:
        combined = out
    combined.to_parquet(OUT, index=False)
    print(f"wrote {len(out)} pipeline rows; expectations.parquet now {len(combined)} rows")
    print("pipeline rows per season:", out.groupby("season").size().to_dict())


if __name__ == "__main__":
    main()
