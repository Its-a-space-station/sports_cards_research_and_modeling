"""Rebuild sales + chart parquets from the latest raw SCP snapshots (no network).

Reads data/raw/prices_scp/<set>/<card>/<latest>.json, re-parses with the current
parser, joins card metadata from cards_seed.csv, and rewrites the processed parquets.
"""

import json
from pathlib import Path

import pandas as pd

from cardprice.scp_parse import calibrate_chart_grades, parse_sales_tables

RAW = Path("data/raw/prices_scp")
META_COLS = ["player_name", "mlb_id", "rookie_year", "set_slug", "card_slug"]

if __name__ == "__main__":
    cards = pd.read_csv("data/reference/cards_seed.csv")
    meta_by_slug = {
        "/".join(u.rstrip("/").split("/")[-2:]): row
        for u, row in zip(cards["scp_url"], cards.itertuples())
        if isinstance(u, str) and u
    }
    sales_frames, chart_frames = [], []
    for day_file in sorted(RAW.glob("*/*/*.json")):
        slug = "/".join(day_file.parts[-3:-1])
        if day_file.name != max(day_file.parent.glob("*.json")).name:
            continue  # only latest snapshot per card
        payload = json.loads(day_file.read_text())
        meta_row = meta_by_slug.get(slug)
        if meta_row is None:
            continue
        meta = {
            "player_name": meta_row.player_name,
            "mlb_id": int(meta_row.mlb_id),
            "rookie_year": int(meta_row.rookie_year),
            "set_slug": meta_row.set_slug,
            "card_slug": slug,
        }
        sales = parse_sales_tables(payload["html"])
        if len(sales):
            sales_frames.append(sales.assign(**meta))
        chart = calibrate_chart_grades(payload["html"])
        chart = chart[~chart["grade"].str.startswith("key:")]
        if len(chart):
            chart_frames.append(chart.assign(**meta))
    sales = pd.concat(sales_frames, ignore_index=True)
    chart = pd.concat(chart_frames, ignore_index=True)
    sales.to_parquet("data/processed/scp_sales.parquet", index=False)
    chart.to_parquet("data/processed/scp_chart_monthly.parquet", index=False)
    print(f"reparsed: {len(sales)} sales, {len(chart)} chart points")
