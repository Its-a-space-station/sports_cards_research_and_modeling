# scripts/collect_class_events.py
"""Class-universe events: debuts (first MLB-level game) + award wins.

Awards are cheap: events.fetch_award_events iterates awards x seasons (not
players), so one call covers the whole class universe.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cardprice.events import EVENT_COLUMNS, fetch_award_events

GAME_LOGS = Path("data/processed/game_logs_class.parquet")
EVENTS_OUT = Path("data/processed/events_class.parquet")


def build_events(game_logs: pd.DataFrame, award_events: pd.DataFrame) -> pd.DataFrame:
    """Debut rows (first MLB-level game per player) + award rows.

    ``award_events`` is kept as-is: the upstream fetch is already restricted to
    class players. Columns match ``events_universe.parquet`` (EVENT_COLUMNS).
    """
    mlb = game_logs[game_logs["level"] == "mlb"]
    debuts = (
        mlb.groupby("mlb_id")["date"]
        .min()
        .reset_index()
        .rename(columns={"date": "event_date"})
        .assign(event_type="debut", details="first game in dataset")
    )
    out = pd.concat([debuts, award_events], ignore_index=True)
    return out[EVENT_COLUMNS].sort_values(["mlb_id", "event_date"]).reset_index(drop=True)


def main() -> None:
    if not GAME_LOGS.exists():
        sys.exit(f"missing {GAME_LOGS} — run scripts/collect_class_stats.py first")
    logs = pd.read_parquet(GAME_LOGS)
    mlb_ids = sorted(logs["mlb_id"].unique())
    print(f"{len(mlb_ids)} players with game logs")
    awards = fetch_award_events(mlb_ids, list(range(2015, 2027)))
    events = build_events(logs, awards)
    events.to_parquet(EVENTS_OUT, index=False)
    print("events:", len(events), events["event_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
