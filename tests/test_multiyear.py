import numpy as np
import pandas as pd

from cardprice.multiyear import HORIZONS, hold_returns, series_month_ends, trailing_features


def make_chart(
    slug="set/card", grade="ungraded", start="2021-01", months=66, price0=10.0, growth=0.01
):
    dates = pd.date_range(start, periods=months, freq="MS")
    n = len(dates)
    return pd.DataFrame(
        {
            "card_slug": slug,
            "grade": grade,
            "date": dates,
            "price": [price0 * np.exp(growth * i) for i in range(n)],
            "mlb_id": 1,
            "player_name": "Test Player",
            "rookie_year": 2021,
            "card_type": "flagship",
        }
    )


def test_series_month_ends_filters_and_aggregates():
    chart = pd.concat(
        [
            make_chart(grade="ungraded"),
            make_chart(grade="psa_10"),
            make_chart(grade="grade_9"),  # grader-agnostic label: excluded
            make_chart(grade="key:cib"),  # uncalibrated: excluded
        ]
    )
    me = series_month_ends(chart)
    assert set(me["grade"]) == {"ungraded", "psa_10"}
    assert len(me) == 2 * 66
    assert {"card_type", "mlb_id", "rookie_year"} <= set(me.columns)


def test_hold_returns_exact_and_truncated():
    me = series_month_ends(make_chart(months=30))  # 2021-01 .. 2023-06
    out = hold_returns(me)
    # math check needs 2021-04 + 36m = 2024-04 present: use a 40-month chart
    me_long = series_month_ends(make_chart(months=40))  # 2021-01 .. 2024-04
    out_long = hold_returns(me_long)
    row = out_long[out_long["entry_month"] == pd.Timestamp("2021-04-01")].iloc[0]
    # growth=0.01 log-linear: annualized return == 0.12 at every horizon
    for h in HORIZONS:
        assert abs(row[f"ret_{h}m"] - 0.12) < 1e-9
    # truncation: last month-end is 2023-06; 12m needs <= 2022-06
    last12 = out[out["ret_12m"].notna()]["entry_month"].max()
    assert last12 == pd.Timestamp("2022-06-01")
    assert out[out["entry_month"] > pd.Timestamp("2022-06-01")]["ret_12m"].isna().all()
    # entries before 2021-04 excluded
    assert out["entry_month"].min() == pd.Timestamp("2021-04-01")


def test_hold_returns_gap_month_is_nan_not_fabricated():
    chart = make_chart(months=30)
    chart = chart[chart["date"] != pd.Timestamp("2021-10-01")]  # punch a hole
    out = hold_returns(series_month_ends(chart))
    row = out[out["entry_month"] == pd.Timestamp("2021-04-01")].iloc[0]
    assert pd.isna(row["ret_6m"])  # 2021-04 + 6m = 2021-10 = the hole
    assert not pd.isna(row["ret_12m"])


def test_trailing_features():
    me = series_month_ends(make_chart(months=30))
    tf = trailing_features(me)
    row = tf[tf["month"] == pd.Timestamp("2021-04-01")].iloc[0]
    assert abs(row["ret_3m"] - 0.03) < 1e-9  # 3 months of 0.01 growth
    # first 3 months have insufficient history
    early = tf[tf["month"] <= pd.Timestamp("2021-03-01")]
    assert early["price_level"].isna().all() and early["ret_3m"].isna().all()
