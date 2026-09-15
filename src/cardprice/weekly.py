"""Card-grade-week price series from individual sales."""

import pandas as pd


def weekly_price_series(sales: pd.DataFrame, min_sales: int = 2) -> pd.DataFrame:
    df = sales[sales["grade"].notna()].copy()
    df["week"] = df["sale_date"].dt.to_period("W-SUN").dt.start_time  # Mondays
    grouped = (
        df.groupby(["card_slug", "grade", "week"])
        .agg(
            median_price=("price", "median"),
            n_sales=("price", "size"),
            best_offer_share=("best_offer", "mean"),
        )
        .reset_index()
    )
    return grouped[grouped["n_sales"] >= min_sales].reset_index(drop=True)
