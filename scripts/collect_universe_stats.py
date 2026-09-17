# scripts/collect_universe_stats.py
"""Collect MLB (2015+) + minor-league (2011+, per level) game logs for the universe."""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.stats_api import (
    fetch_game_log,
    fetch_minor_league_logs,
    game_log_to_frame,
)
from cardprice.storage import save_raw

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
