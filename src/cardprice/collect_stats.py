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
