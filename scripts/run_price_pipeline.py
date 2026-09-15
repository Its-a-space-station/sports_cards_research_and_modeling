"""End-to-end: parquets -> validation summary -> weekly series -> liquidity report."""

import pandas as pd

from cardprice.liquidity import liquidity_report, recommend_universe
from cardprice.validate import contamination_audit, quarantine_outliers, stale_series
from cardprice.weekly import weekly_price_series

if __name__ == "__main__":
    sales = pd.read_parquet("data/processed/scp_sales.parquet")
    chart = pd.read_parquet("data/processed/scp_chart_monthly.parquet")

    flagged = quarantine_outliers(sales)
    print(f"sales: {len(sales)} rows, {int(flagged['outlier'].sum())} outlier-flagged")
    contam = contamination_audit(sales)
    print(f"contamination audit: {len(contam)} suspicious rows")
    if len(contam):
        print(contam[["card_slug", "title", "price"]].to_string(index=False))

    weekly = weekly_price_series(flagged[~flagged["outlier"]])
    weekly.to_parquet("data/processed/scp_weekly.parquet", index=False)
    print(f"weekly series: {len(weekly)} card-grade-weeks")

    stale = stale_series(weekly)
    print(f"stale series (>3 weeks no sales): {len(stale)}")
    for slug in stale:
        print(f"  STALE {slug}")

    report = liquidity_report(sales, chart)
    report.to_csv("data/processed/liquidity_report.csv", index=False)
    rec = recommend_universe(report)
    print(f"\nrecommended universe ({len(rec)} card-grades):")
    print(rec[["card_slug", "grade", "sales_per_week", "n_chart_points"]].to_string(index=False))
