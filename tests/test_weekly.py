import pandas as pd

from cardprice.weekly import weekly_price_series


def make_sales():
    rows = [
        # week of Mon 2026-08-31: 3 sales
        ("card/a", "psa_10", "2026-08-31", 40.0, False),
        ("card/a", "psa_10", "2026-09-02", 50.0, True),
        ("card/a", "psa_10", "2026-09-05", 60.0, False),
        # week of Mon 2026-09-07: 1 sale -> omitted
        ("card/a", "psa_10", "2026-09-08", 70.0, False),
        # different grade same week
        ("card/a", "psa_9", "2026-09-01", 10.0, False),
        ("card/a", "psa_9", "2026-09-03", 20.0, False),
        # ungraded (grade None) -> excluded
        ("card/a", None, "2026-09-01", 3.0, False),
    ]
    df = pd.DataFrame(rows, columns=["card_slug", "grade", "sale_date", "price", "best_offer"])
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df


def test_weekly_aggregation_and_min_sales():
    out = weekly_price_series(make_sales())
    psa10 = out[out["grade"] == "psa_10"].sort_values("week")
    assert len(psa10) == 1  # the 1-sale week is omitted
    row = psa10.iloc[0]
    assert row["week"] == pd.Timestamp("2026-08-31")
    assert row["median_price"] == 50.0
    assert row["n_sales"] == 3
    assert row["best_offer_share"] == 1 / 3


def test_grades_separate_and_ungraded_excluded():
    out = weekly_price_series(make_sales())
    assert set(out["grade"]) == {"psa_10", "psa_9"}
    psa9 = out[out["grade"] == "psa_9"].iloc[0]
    assert psa9["median_price"] == 15.0 and psa9["n_sales"] == 2
