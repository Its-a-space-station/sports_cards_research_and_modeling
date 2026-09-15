import pandas as pd

from cardprice.validate import contamination_audit, quarantine_outliers, stale_series


def test_quarantine_outliers_flags_extreme_only_in_big_groups():
    prices = [40.0, 41.0, 42.0, 43.0, 44.0, 3000.0]  # 6 sales, one wild
    df = pd.DataFrame(
        {
            "card_slug": "card/a",
            "grade": "psa_10",
            "sale_date": pd.to_datetime([f"2026-09-0{d}" for d in range(1, 7)]),
            "price": prices,
        }
    )
    out = quarantine_outliers(df)
    assert out["outlier"].sum() == 1
    assert out.loc[out["price"] == 3000.0, "outlier"].iloc[0]
    # small group (4 sales incl. extreme) -> never flags
    small = df.iloc[:4].copy()
    small.loc[small.index[-1], "price"] = 900.0
    assert not quarantine_outliers(small)["outlier"].any()


def test_contamination_audit():
    df = pd.DataFrame(
        {
            "title": [
                "2023 Topps Chrome Henderson PSA 10",
                "Henderson PSA 10 lot of 3",
                "2023 Topps Chrome Henderson REPRINT psa 10",
            ],
            "price": [40.0, 90.0, 12.0],
        }
    )
    out = contamination_audit(df)
    assert len(out) == 2
    assert "lot" in out.iloc[0]["flag_reason"]


def test_stale_series():
    weekly = pd.DataFrame(
        {
            "card_slug": ["card/fresh", "card/stale"],
            "week": [pd.Timestamp("2026-09-07"), pd.Timestamp("2026-08-03")],
        }
    )
    stale = stale_series(weekly, max_gap_weeks=3, as_of=pd.Timestamp("2026-09-14"))
    assert stale == ["card/stale"]
