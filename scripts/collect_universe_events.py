# scripts/collect_universe_events.py
"""Universe event registry (debut + award_win) and player info for the 39 players."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.events import EVENT_COLUMNS, events_from_game_logs, fetch_award_events
from cardprice.stats_api import fetch_player_info


def build_universe_events(game_logs: pd.DataFrame, awards: pd.DataFrame) -> pd.DataFrame:
    mlb_logs = game_logs[game_logs["level"] == "mlb"].copy()
    # events_from_game_logs needs the counting-stat columns to collapse
    # doubleheaders; debut derivation does not use them, so fill missing with 0.
    for col in ("homeRuns", "hits", "strikeOuts"):
        if col not in mlb_logs.columns:
            mlb_logs[col] = 0
    debuts = events_from_game_logs(mlb_logs)[lambda d: d["event_type"] == "debut"]
    out = pd.concat([debuts, awards], ignore_index=True)
    return out[EVENT_COLUMNS].sort_values(["mlb_id", "event_date"]).reset_index(drop=True)


def main() -> None:
    cards = pd.read_csv("data/reference/cards_universe.csv")
    mlb_ids = sorted(cards["mlb_id"].dropna().astype(int).unique())
    print(f"{len(mlb_ids)} players")

    info = fetch_player_info(mlb_ids)
    info.to_csv("data/reference/player_info_universe.csv", index=False)

    game_logs = pd.read_parquet("data/processed/game_logs_universe.parquet")
    awards = fetch_award_events(mlb_ids, list(range(2015, 2027)))
    events = build_universe_events(game_logs, awards)
    events.to_parquet("data/processed/events_universe.parquet", index=False)
    print(events.groupby("event_type").size())


if __name__ == "__main__":
    main()
