# src/cardprice/collect_prices.py
"""Batch SCP collector: cards_seed.csv -> raw HTML snapshots + parsed parquets."""

import argparse
import time

import pandas as pd

from cardprice.scp_parse import calibrate_chart_grades, parse_attributes, parse_sales_tables
from cardprice.storage import save_raw
from cardprice.web import ChallengeError, fetch_page

META_COLS = ["player_name", "mlb_id", "rookie_year", "set_slug", "card_slug"]

# Markers of a real SCP card page; a soft-404 page passes the challenge-title
# check but has neither, so don't snapshot it.
PAGE_MARKERS = ("completed-auctions-", "price_data")


def card_slug(url: str) -> str:
    parts = url.rstrip("/").split("/")
    return "/".join(parts[-2:])


def collect_cards(cards: pd.DataFrame, sleep_s: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales_frames, chart_frames = [], []
    for card in cards.itertuples():
        # pandas reads empty CSV cells as NaN (truthy) — treat non-str/"" as empty
        if not isinstance(card.scp_url, str) or not card.scp_url:
            print(f"SKIP {card.player_name}: no scp_url")
            continue
        slug = card_slug(card.scp_url)
        try:
            html = fetch_page(card.scp_url)
        except ChallengeError as e:
            print(f"WARN challenge unresolved for {slug}: {e}")
            continue
        if not any(marker in html for marker in PAGE_MARKERS):
            print(f"WARN {slug}: no card-page markers (soft-404?); not snapshotting")
            continue
        save_raw("prices_scp", slug, {"url": card.scp_url, "html": html})
        meta = {
            "player_name": card.player_name,
            "mlb_id": int(card.mlb_id),
            "rookie_year": int(card.rookie_year),
            "set_slug": card.set_slug,
            "card_slug": slug,
        }
        sales = parse_sales_tables(html)
        if len(sales):
            sales = sales.assign(**meta)
            sales_frames.append(sales)
        chart = calibrate_chart_grades(html)
        if len(chart):
            chart = chart.assign(**meta)
            chart_frames.append(chart)
        attrs = parse_attributes(html)
        print(
            f"OK {slug}: {len(sales)} sales, {len(chart)} chart points, rookie={attrs.get('is_rookie_card')}"
        )
        time.sleep(sleep_s)
    return (
        pd.concat(sales_frames, ignore_index=True) if sales_frames else pd.DataFrame(),
        pd.concat(chart_frames, ignore_index=True) if chart_frames else pd.DataFrame(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="data/reference/cards_seed.csv")
    parser.add_argument("--sales-out", default="data/processed/scp_sales.parquet")
    parser.add_argument("--chart-out", default="data/processed/scp_chart_monthly.parquet")
    parser.add_argument("--sleep", type=float, default=5.0)
    args = parser.parse_args()

    cards = pd.read_csv(args.cards)
    sales, chart = collect_cards(cards, sleep_s=args.sleep)
    sales.to_parquet(args.sales_out, index=False)
    chart.to_parquet(args.chart_out, index=False)
    print(
        f"wrote {len(sales)} sales to {args.sales_out}; {len(chart)} chart points to {args.chart_out}"
    )


if __name__ == "__main__":
    main()
