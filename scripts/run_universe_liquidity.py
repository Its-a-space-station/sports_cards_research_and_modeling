# scripts/run_universe_liquidity.py
"""Liquidity report over the universe parquets + modeling-card selection."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.collect_prices import card_slug
from cardprice.liquidity import liquidity_report
from cardprice.weekly import normalize_grade

MIN_SALES_PER_WEEK = 0.3
MIN_CHART_POINTS = 36

MODELING_COLUMNS = [
    "card_slug",
    "player_name",
    "mlb_id",
    "rookie_year",
    "card_type",
    "grade",
    "sales_per_week",
    "n_chart_points",
    "chart_first",
    "chart_last",
]


def select_modeling_cards(report: pd.DataFrame) -> pd.DataFrame:
    # Keep a (card, series) row if it is liquid on EITHER axis:
    #   - sales_per_week >= 0.3 over the observed sales window (active recent market), OR
    #   - n_chart_points >= 36 (>= ~3 years of monthly chart history).
    mask = (report["sales_per_week"] >= MIN_SALES_PER_WEEK) | (
        report["n_chart_points"] >= MIN_CHART_POINTS
    )
    return report[mask].reset_index(drop=True)


if __name__ == "__main__":
    sales = pd.read_parquet("data/processed/universe_sales.parquet")
    chart = pd.read_parquet("data/processed/universe_chart_monthly.parquet")
    # drop uncalibrated chart keys (e.g. `key:cib`) — same filter reparse_snapshots.py applies
    chart = chart[~chart["grade"].astype(str).str.startswith("key:")]
    sales["grade"] = sales["grade"].map(normalize_grade)
    report = liquidity_report(sales, chart)
    report.to_csv("data/processed/universe_liquidity.csv", index=False)
    # liquidity_report carries no player meta — derive card_slug from the universe
    # catalog's scp_url and left-join the modeling columns on.
    meta = pd.read_csv("data/reference/cards_universe.csv")
    meta["card_slug"] = meta["scp_url"].map(
        lambda u: card_slug(u) if isinstance(u, str) and u else ""
    )
    meta = meta[["card_slug", "player_name", "mlb_id", "rookie_year", "card_type"]]
    report = report.merge(meta, on="card_slug", how="left")
    selected = select_modeling_cards(report)
    selected[MODELING_COLUMNS].to_csv("data/reference/cards_modeling.csv", index=False)
    print(f"universe liquidity: {len(report)} card-series rows; selected {len(selected)}")
    print(selected.groupby(["card_type" if "card_type" in selected else "grade"]).size())
