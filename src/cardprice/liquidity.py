"""Liquidity metrics per card-grade; recommends the modeling universe."""

import pandas as pd

from cardprice.validate import contamination_audit, quarantine_outliers
from cardprice.weekly import normalize_grade

PANEL_SERIES = ("ungraded", "psa_9", "psa_10")
# Deprecated alias kept for Plan 3-era imports; use PANEL_SERIES.
PANEL_GRADES = PANEL_SERIES


def liquidity_report(sales: pd.DataFrame, chart: pd.DataFrame) -> pd.DataFrame:
    sales = sales.copy()
    sales["grade"] = sales["grade"].map(normalize_grade)
    sales = sales[sales["grade"].isin(PANEL_SERIES)]
    flagged = quarantine_outliers(sales)
    contaminated = contamination_audit(sales)
    rows = []
    for (slug, grade), grp in flagged.groupby(["card_slug", "grade"]):
        span = max((grp["sale_date"].max() - grp["sale_date"].min()).days, 1)
        ch = chart[(chart["card_slug"] == slug) & (chart["grade"] == grade)]
        contam = contaminated[
            (contaminated["card_slug"] == slug) & (contaminated["grade"] == grade)
        ]
        rows.append(
            {
                "card_slug": slug,
                "grade": grade,
                "n_sales": len(grp),
                "first_sale": grp["sale_date"].min(),
                "last_sale": grp["sale_date"].max(),
                "sales_span_days": span,
                "sales_per_week": len(grp) / (span / 7),
                "n_chart_points": len(ch),
                "chart_first": ch["date"].min() if len(ch) else pd.NaT,
                "chart_last": ch["date"].max() if len(ch) else pd.NaT,
                "outlier_share": grp["outlier"].mean(),
                "contamination_count": len(contam),
            }
        )
    return pd.DataFrame(rows).sort_values("sales_per_week", ascending=False).reset_index(drop=True)


def recommend_universe(
    report: pd.DataFrame, min_sales_per_week: float = 0.5, min_chart_points: int = 18
) -> pd.DataFrame:
    mask = (report["sales_per_week"] >= min_sales_per_week) & (
        report["n_chart_points"] >= min_chart_points
    )
    return report[mask].reset_index(drop=True)
