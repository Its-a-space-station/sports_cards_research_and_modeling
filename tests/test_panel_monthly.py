import numpy as np
import pandas as pd
import pytest

from cardprice.panel import monthly_panel


def make_inputs():
    # Two cards, one player each, 4 months of prices
    chart = pd.DataFrame(
        [
            ("card/a", 1, 2022, "2022-04-01", 100.0),
            ("card/a", 1, 2022, "2022-05-01", 110.0),
            ("card/a", 1, 2022, "2022-06-01", 99.0),
            ("card/a", 1, 2022, "2022-10-01", 150.0),
            ("card/b", 2, 2022, "2022-04-01", 50.0),
            ("card/b", 2, 2022, "2022-05-01", 55.0),
            ("card/b", 2, 2022, "2022-06-01", 44.0),
            ("card/b", 2, 2022, "2022-10-01", 66.0),
        ],
        columns=["card_slug", "mlb_id", "rookie_year", "date", "price"],
    )
    chart["date"] = pd.to_datetime(chart["date"])
    chart["grade"] = "psa_10"
    chart["player_name"] = chart["mlb_id"].map({1: "Player One", 2: "Player Two"})
    chart["set_slug"] = "set/x"

    # Player 1 game logs: 1 game Apr 10 (HR), 1 game May 5, 1 game Sep 28
    logs = pd.DataFrame(
        [
            (1, "hitting", 2022, "2022-04-10", 1, 4, 2, 1, 1, 1, 0),
            (1, "hitting", 2022, "2022-05-05", 1, 4, 1, 0, 0, 1, 0),
            (1, "hitting", 2022, "2022-09-28", 1, 4, 3, 2, 1, 0, 1),
            (2, "hitting", 2022, "2022-04-11", 1, 4, 1, 0, 1, 1, 0),
            (2, "hitting", 2022, "2022-05-06", 1, 4, 0, 0, 0, 2, 0),
        ],
        columns=[
            "mlb_id",
            "group",
            "season",
            "date",
            "gamesPlayed",
            "atBats",
            "hits",
            "doubles",
            "homeRuns",
            "baseOnBalls",
            "strikeOuts",
        ],
    )
    logs["date"] = pd.to_datetime(logs["date"])
    info = pd.DataFrame(
        {
            "mlb_id": [1, 2],
            "name": ["Player One", "Player Two"],
            "birth_date": pd.to_datetime(["2000-01-01", "1999-06-15"]),
            "position": ["OF", "1B"],
        }
    )
    return chart, logs, info


def test_panel_shape_and_lag():
    chart, logs, info = make_inputs()
    panel = monthly_panel(chart, logs, info)
    may = panel[(panel["card_slug"] == "card/a") & (panel["month"] == "2022-05-01")]
    assert len(may) == 1
    may = may.iloc[0]
    # NO LOOK-AHEAD: predictors for May = stats through Apr 30 only (1 game, the Apr 10 game)
    assert may["games"] == 1
    assert may["home_runs"] == 1
    june = panel[(panel["card_slug"] == "card/a") & (panel["month"] == "2022-06-01")].iloc[0]
    assert june["games"] == 2  # through May 31
    assert june["home_runs"] == 1
    oct_row = panel[(panel["card_slug"] == "card/a") & (panel["month"] == "2022-10-01")].iloc[0]
    assert oct_row["games"] == 3  # through Sep 30 (the Sep 28 game counts)
    assert oct_row["playoff"] == 1


def test_market_median_and_excess():
    chart, logs, info = make_inputs()
    panel = monthly_panel(chart, logs, info)
    may = panel[panel["month"] == "2022-05-01"]
    # both cards +10% in May -> market median ret = ln(1.1); excess ~ 0
    row = may[may["card_slug"] == "card/a"].iloc[0]
    assert row["log_ret"] == pytest.approx(np.log(1.1), abs=1e-9)
    assert row["excess_ret"] == pytest.approx(0.0, abs=1e-9)
    june = panel[(panel["month"] == "2022-06-01") & (panel["card_slug"] == "card/a")].iloc[0]
    # a: 110->99 = ln(0.9); b: 55->44 = ln(0.8); median = mean of the two logs
    expected_market = (np.log(0.9) + np.log(0.8)) / 2
    assert june["excess_ret"] == pytest.approx(np.log(0.9) - expected_market, abs=1e-9)


def test_debut_and_static():
    chart, logs, info = make_inputs()
    panel = monthly_panel(chart, logs, info)
    apr_a = panel[(panel["card_slug"] == "card/a") & (panel["month"] == "2022-04-01")].iloc[0]
    assert apr_a["games"] == 0  # debut-month row kept with 0 stats at lag
    assert apr_a["position"] == "OF"
    assert apr_a["age"] == pytest.approx(22.25, abs=0.01)
    assert apr_a["log_ret"] != apr_a["log_ret"]  # first month: NaN
