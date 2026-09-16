"""Run the event study on real data and print everything the report needs."""

import pandas as pd

from cardprice.event_study import event_significance, event_windows, mean_car
from cardprice.events import (
    events_from_game_logs,
    fetch_award_events,
    fetch_playoff_events,
)

if __name__ == "__main__":
    logs = pd.read_parquet("data/processed/game_logs.parquet")
    monthly = pd.read_parquet("data/processed/panel_monthly.parquet")
    weekly = pd.read_parquet("data/processed/panel_weekly.parquet")
    cards = pd.read_csv("data/reference/cards_seed.csv")
    mlb_ids = cards["mlb_id"].astype(int).tolist()
    seasons = [2022, 2023, 2024, 2025, 2026]

    events = events_from_game_logs(logs)
    try:
        events = pd.concat([events, fetch_playoff_events(mlb_ids, seasons)])
        events = pd.concat([events, fetch_award_events(mlb_ids, seasons)])
    except Exception as e:  # noqa: BLE001 — one-shot runner: any API failure degrades to offline events
        print(f"WARN: API event fetch failed ({e}); continuing with game-log events only")

    events.to_parquet("data/processed/events.parquet", index=False)
    print(events.groupby("event_type").size())

    # monthly study (primary: history)
    for etype in sorted(events["event_type"].unique()):
        ev = events[events["event_type"] == etype]
        w = event_windows(ev, monthly, "monthly", pre=1, post=2)
        if not len(w):  # no panel rows within the window of any event of this type
            print(f"\n{etype}: n=0 (no panel rows within [-1, +2] months of any event)")
            continue
        cars = mean_car(w, [0])
        sig = event_significance(cars["car"], monthly)
        print(
            f"\n{etype}: n={sig['n_events']}, mean CAR(event month)={sig['mean_car']:.4f}, "
            f"p={sig['p_value']:.3f}"
        )
        # reversion: CAR[0,+1] vs CAR[0]
        cars2 = mean_car(w, [0, 1])
        merged = cars.merge(
            cars2,
            on=["mlb_id", "card_slug", "event_type", "event_date"],
            suffixes=("_e0", "_e01"),
        )
        merged["giveback"] = merged["car_e01"] - merged["car_e0"]
        spikes = merged[merged["car_e0"] > 0]
        if len(spikes):
            print(
                f"  reversion: {len(spikes)} positive spikes, mean next-month giveback "
                f"{spikes['giveback'].mean():.4f} "
                f"({(spikes['giveback'] < 0).mean():.0%} give back some)"
            )

    # weekly study (2026 window only — descriptive)
    w = event_windows(events, weekly, "weekly", pre=1, post=3)
    if len(w):
        cars = mean_car(w, [0])
        sig = event_significance(cars["car"], weekly)
        print(
            f"\nweekly all-events: n={sig['n_events']}, mean CAR={sig['mean_car']:.4f}, "
            f"p={sig['p_value']:.3f}"
        )
