import pandas as pd

from cardprice.liquidity import liquidity_report, recommend_universe


def make_inputs():
    sales = pd.DataFrame(
        {
            "card_slug": ["card/hot"] * 10 + ["card/cold"] * 2,
            "grade": ["psa_10"] * 10 + ["psa_10"] * 2,
            "sale_date": pd.to_datetime(
                [f"2026-08-{d:02d}" for d in range(1, 11)] + ["2026-09-01", "2026-09-10"]
            ),
            "price": [40.0] * 10 + [5.0, 6.0],
            "best_offer": [False] * 12,
            "title": ["Henderson PSA 10"] * 12,
        }
    )
    chart = pd.DataFrame(
        {
            "card_slug": ["card/hot"] * 24 + ["card/cold"] * 6,
            "grade": ["psa_10"] * 30,
            "date": pd.to_datetime(
                [f"2024-{m:02d}-01" for m in range(1, 13)]
                + [f"2025-{m:02d}-01" for m in range(1, 13)]
                + [f"2026-0{m}-01" for m in range(1, 7)]
            ),
            "price": [30.0] * 30,
        }
    )
    return sales, chart


def test_liquidity_report_metrics():
    sales, chart = make_inputs()
    rep = liquidity_report(sales, chart)
    hot = rep[rep["card_slug"] == "card/hot"].iloc[0]
    cold = rep[rep["card_slug"] == "card/cold"].iloc[0]
    assert hot["n_sales"] == 10
    assert hot["sales_per_week"] > cold["sales_per_week"]
    assert hot["n_chart_points"] == 24
    assert rep.iloc[0]["card_slug"] == "card/hot"  # sorted desc


def test_recommend_universe_thresholds():
    sales, chart = make_inputs()
    rep = liquidity_report(sales, chart)
    # hot: 10 sales / ~1.3 weeks ≈ 7.8/wk; cold: 2 sales / ~1.3 weeks ≈ 1.6/wk
    rec = recommend_universe(rep, min_sales_per_week=5.0, min_chart_points=18)
    assert rec["card_slug"].tolist() == ["card/hot"]
