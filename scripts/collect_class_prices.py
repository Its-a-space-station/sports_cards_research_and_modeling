# scripts/collect_class_prices.py
"""Resumable class price collection: one part-file pair per card, merged at the end.

Part files (part_root/sales|chart/<set>__<card>.parquet) are the done-markers —
written even when a frame is empty. Failures (challenge, soft-404, parse errors)
write no parts, are recorded, and retry on the next run.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cardprice.collect_prices import card_slug
from cardprice.scp_parse import calibrate_chart_grades, parse_sales_tables
from cardprice.storage import save_raw
from cardprice.web import ChallengeError, fetch_page

PAGE_MARKERS = ("completed-auctions-", "price_data")
META_COLS = ["player_name", "mlb_id", "rookie_year", "set_slug", "card_slug", "card_type"]


def _safe(slug: str) -> str:
    return slug.replace("/", "__")


def collect_class_prices(cards: pd.DataFrame, part_root: str, sleep_s: float = 5.0) -> dict:
    root = Path(part_root)
    (root / "sales").mkdir(parents=True, exist_ok=True)
    (root / "chart").mkdir(parents=True, exist_ok=True)
    done, skipped, failed = 0, 0, []
    for card in cards.itertuples():
        if not isinstance(card.scp_url, str) or not card.scp_url:
            failed.append((getattr(card, "player_name", "?"), "no_scp_url"))
            continue
        slug = card_slug(card.scp_url)
        sales_part = root / "sales" / f"{_safe(slug)}.parquet"
        chart_part = root / "chart" / f"{_safe(slug)}.parquet"
        if sales_part.exists() and chart_part.exists():
            skipped += 1
            continue
        try:
            html = fetch_page(card.scp_url)
        except ChallengeError:
            failed.append((slug, "challenge"))
            continue
        if not any(m in html for m in PAGE_MARKERS):
            failed.append((slug, "soft_404"))
            continue
        save_raw("prices_scp", slug, {"url": card.scp_url, "html": html})
        meta = {
            "player_name": card.player_name, "mlb_id": int(card.mlb_id),
            "rookie_year": int(card.class_year), "set_slug": card.set_slug,
            "card_slug": slug, "card_type": card.card_type,
        }
        try:
            sales = parse_sales_tables(html)
            chart = calibrate_chart_grades(html)
        except Exception as e:  # noqa: BLE001  # poison card: record, retry; don't block the run
            failed.append((slug, f"parse:{type(e).__name__}"))
            continue
        for df in (sales, chart):
            for k, v in meta.items():
                df[k] = v
        sales.to_parquet(sales_part, index=False)
        chart.to_parquet(chart_part, index=False)
        done += 1
        if done % 50 == 0:
            print(f"{done} cards collected, {len(failed)} failed")
        time.sleep(sleep_s)
    return {"done": done, "skipped": skipped, "failed": failed}


def merge_parts(part_root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(part_root)

    def _cat(kind: str) -> pd.DataFrame:
        files = sorted((root / kind).glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    return _cat("sales"), _cat("chart")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="data/reference/cards_class_universe.csv")
    ap.add_argument("--part-root", default="data/processed/class_price_parts")
    ap.add_argument("--sales-out", default="data/processed/class_sales.parquet")
    ap.add_argument("--chart-out", default="data/processed/class_chart_monthly.parquet")
    ap.add_argument("--failures-out", default="data/reference/class_price_failures.csv")
    ap.add_argument("--sleep", type=float, default=5.0)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    if not args.merge_only:
        cards = pd.read_csv(args.cards).dropna(subset=["mlb_id"])
        res = collect_class_prices(cards, args.part_root, sleep_s=args.sleep)
        print(res)
        if res["failed"]:
            pd.DataFrame(res["failed"], columns=["card", "reason"]).to_csv(
                args.failures_out, index=False
            )
    sales, chart = merge_parts(args.part_root)
    sales.to_parquet(args.sales_out, index=False)
    chart.to_parquet(args.chart_out, index=False)
    print("wrote", args.sales_out, len(sales), "|", args.chart_out, len(chart))


if __name__ == "__main__":
    main()
