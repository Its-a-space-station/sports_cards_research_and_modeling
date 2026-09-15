"""Build panel_monthly.parquet and panel_weekly.parquet from processed data."""

import pandas as pd

from cardprice.collect_prices import card_slug
from cardprice.panel import monthly_panel, weekly_panel

if __name__ == "__main__":
    chart = pd.read_parquet("data/processed/scp_chart_monthly.parquet")
    game_logs = pd.read_parquet("data/processed/game_logs.parquet")
    info = pd.read_csv("data/reference/player_info.csv", parse_dates=["birth_date"])
    cards = pd.read_csv("data/reference/cards_seed.csv")
    # cards_seed.csv carries scp_url, not card_slug; derive it like the collectors do
    cards["card_slug"] = cards["scp_url"].map(
        lambda u: card_slug(u) if isinstance(u, str) and u else None
    )
    cards = cards.dropna(subset=["card_slug"])  # 3 unresolved cards have no SCP page yet

    mp = monthly_panel(chart, game_logs, info)
    mp.to_parquet("data/processed/panel_monthly.parquet", index=False)
    print(
        f"panel_monthly: {len(mp)} rows, {mp['card_slug'].nunique()} cards, "
        f"months {mp['month'].min()} -> {mp['month'].max()}"
    )

    weekly = pd.read_parquet("data/processed/scp_weekly.parquet")
    wp = weekly_panel(weekly, cards, game_logs, info)
    wp.to_parquet("data/processed/panel_weekly.parquet", index=False)
    print(
        f"panel_weekly: {len(wp)} rows, {wp['card_slug'].nunique()} cards, "
        f"weeks {wp['week'].min()} -> {wp['week'].max()}"
    )
