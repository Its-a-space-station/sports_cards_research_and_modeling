# scripts/collect_class_stats.py
"""Resumable class-universe game-log collection (MLB 2015-2026 + minors 2014-2026).

Resume contract: a player/season/level whose snapshot already exists is LOADED
from the snapshot (no fetch, no sleep); only misses hit the API. Accumulated
frame flushes to the output parquet every flush_every players.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cardprice.stats_api import (
    MINOR_LEAGUE_LEVELS,
    fetch_game_log,
    game_log_to_frame,
)
from cardprice.storage import load_latest, save_raw, snapshot_exists

ROLE_GROUP = {"hitter": "hitting", "pitcher": "pitching"}

MINOR_LEAGUE_LEVELS_INV = {v: k for k, v in MINOR_LEAGUE_LEVELS.items()}


def _splits(mlb_id: int, group: str, season: int, level: str | None, sleep_s: float) -> list[dict]:
    key = f"{mlb_id}_{group}_{season}" + (f"_{level}" if level else "")
    if snapshot_exists("stats", key):
        return load_latest("stats", key).get("splits", [])
    if level is None:
        splits = fetch_game_log(mlb_id, group, season)
    else:
        splits = fetch_game_log(mlb_id, group, season, MINOR_LEAGUE_LEVELS_INV[level])
    save_raw("stats", key, {"splits": splits})
    time.sleep(sleep_s)
    return splits


def collect_class_stats(
    players: pd.DataFrame,
    mlb_seasons: list[int],
    minor_seasons: list[int],
    out: str,
    flush_every: int = 25,
    sleep_s: float = 0.3,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for done, p in enumerate(players.itertuples(), start=1):
        group = ROLE_GROUP[p.role]
        for season in mlb_seasons:
            splits = _splits(int(p.mlb_id), group, season, None, sleep_s)
            if splits:
                frames.append(
                    game_log_to_frame(splits, int(p.mlb_id), group, season).assign(level="mlb")
                )
        for season in minor_seasons:
            for level in ("aaa", "aa", "a_plus", "a"):
                splits = _splits(int(p.mlb_id), group, season, level, sleep_s)
                if splits:
                    frames.append(
                        game_log_to_frame(splits, int(p.mlb_id), group, season).assign(level=level)
                    )
        if done % flush_every == 0 and frames:
            pd.concat(frames).to_parquet(out, index=False)
            print(f"flushed after {done} players ({sum(len(f) for f in frames)} rows)")
    result = pd.concat(frames) if frames else pd.DataFrame()
    result.to_parquet(out, index=False)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="data/reference/cards_class_universe.csv")
    ap.add_argument("--out", default="data/processed/game_logs_class.parquet")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--flush-every", type=int, default=25)
    args = ap.parse_args()
    cards = pd.read_csv(args.cards).dropna(subset=["mlb_id"])
    players = cards[["player_name", "mlb_id", "role"]].drop_duplicates("mlb_id")
    print("players:", len(players))
    df = collect_class_stats(
        players, list(range(2015, 2027)), list(range(2014, 2027)),
        args.out, flush_every=args.flush_every, sleep_s=args.sleep,
    )
    print("wrote", args.out, len(df), "rows")


if __name__ == "__main__":
    main()
