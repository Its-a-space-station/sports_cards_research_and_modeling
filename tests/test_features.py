import numpy as np
import pandas as pd

from cardprice.features import build_modeling_frame, hitter_matrix, standardize


def make_monthly():
    rows = []
    for i, month in enumerate(pd.date_range("2023-04-01", periods=6, freq="MS")):
        rows.append(
            {
                "card_slug": "card/a",
                "mlb_id": 1,
                "player_name": "Player One",
                "month": month,
                "price": 100.0 * (1.01**i),
                "log_ret": 0.01 if i else np.nan,
                "months_since_prev": np.nan if i == 0 else (1 if i != 4 else 3),
                "market_median_ret": 0.005 if i else np.nan,
                "excess_ret": 0.005 if i else np.nan,
                "games": 20 + i,
                "ops": 0.7 + 0.01 * i,
                "avg": 0.28,
                "obp": 0.35,
                "slg": 0.45,
                "home_runs": float(i),
                "strikeouts": 15.0,
                "form_games": 4,
                "form_ops_delta": 0.05,
                "form_era_delta": np.nan,
                "age": 22.5,
                "position": "SS",
                "rookie_year": 2023,
                "stats_season": 2023,
                "set_slug": "set/x",
                "grade": "psa_10",
                "playoff": 0,
            }
        )
    return pd.DataFrame(rows)


def test_build_monthly_frame_filters_horizon_and_adds_columns():
    frame = build_modeling_frame(make_monthly(), "monthly")
    # first row: NaN outcome dropped; i==4 row: months_since_prev=3 dropped
    assert len(frame) == 4
    assert (frame["position_group"] == "IF").all()
    assert (frame["years_since_rookie"] == 0).all()


def test_hitter_matrix_shapes_and_standardization():
    frame = build_modeling_frame(make_monthly(), "monthly")
    X, y, groups = hitter_matrix(frame)
    assert list(X.columns) == [
        "ops",
        "slg",
        "home_runs",
        "strikeouts",
        "games",
        "form_games",
        "form_ops_delta",
        "age",
        "playoff",
        "years_since_rookie",
    ]
    assert len(X) == len(y) == len(groups) == 4
    # standardized: ~0 mean, ~1 sd (population sd)
    assert abs(X["ops"].mean()) < 1e-9
    assert abs(X["ops"].std(ddof=0) - 1) < 1e-9
    assert list(groups.columns) == ["mlb_id", "position_group", "stats_season"]


def test_standardize_handles_zero_variance():
    X = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
    Z, params = standardize(X)
    assert (Z["a"] == 0).all()  # zero-variance column maps to 0, sd recorded
    assert params.loc["a", "sd"] == 0
    assert abs(Z["b"].mean()) < 1e-9
