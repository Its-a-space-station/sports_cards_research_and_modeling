# tests/test_ungraded_series.py
import pandas as pd

from cardprice.liquidity import liquidity_report
from cardprice.weekly import weekly_price_series


def make_sales():
    rows = [
        ("card/a", None, "2026-08-31", 5.0, False),
        ("card/a", None, "2026-09-02", 7.0, False),
        ("card/a", None, "2026-09-05", 6.0, False),
        ("card/a", "psa_10", "2026-08-31", 40.0, False),
        ("card/a", "psa_10", "2026-09-02", 50.0, False),
    ]
    df = pd.DataFrame(rows, columns=["card_slug", "grade", "sale_date", "price", "best_offer"])
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["title"] = "Test Card"  # contamination_audit (inside liquidity_report) requires it
    return df


def test_ungraded_is_first_class_series():
    out = weekly_price_series(make_sales(), min_sales=2)
    assert set(out["grade"]) == {"ungraded", "psa_10"}
    raw = out[out["grade"] == "ungraded"].iloc[0]
    assert raw["median_price"] == 6.0 and raw["n_sales"] == 3


def test_liquidity_report_includes_ungraded():
    sales = make_sales()
    chart = pd.DataFrame(
        {
            "card_slug": ["card/a"] * 20,
            "grade": ["ungraded"] * 20,
            "date": pd.date_range("2025-01-01", periods=20, freq="MS"),
            "price": [5.0] * 20,
        }
    )
    rep = liquidity_report(sales, chart)
    assert "ungraded" in set(rep["grade"])
