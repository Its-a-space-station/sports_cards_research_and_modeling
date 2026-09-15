import pandas as pd

from cardprice.panel import weekly_panel


def make_inputs():
    weekly = pd.DataFrame(
        [
            ("card/a", "psa_10", "2026-08-31", 100.0, 3, 0.0),
            ("card/a", "psa_10", "2026-09-07", 105.0, 2, 0.5),
            ("card/a", "psa_9", "2026-08-31", 30.0, 2, 0.0),
            ("card/a", "psa_9", "2026-09-07", 33.0, 4, 0.25),
        ],
        columns=["card_slug", "grade", "week", "median_price", "n_sales", "best_offer_share"],
    )
    weekly["week"] = pd.to_datetime(weekly["week"])
    cards = pd.DataFrame(
        {
            "player_name": ["Player One"],
            "mlb_id": [1],
            "rookie_year": [2023],
            "set_slug": ["set/x"],
            "card_slug": ["card/a"],
        }
    )
    logs = pd.DataFrame(
        [
            (1, "hitting", 2026, "2026-08-20", 1, 4, 2, 1, 1, 1, 0),
            (1, "hitting", 2026, "2026-09-02", 1, 4, 1, 0, 0, 1, 0),
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
            "mlb_id": [1],
            "name": ["Player One"],
            "birth_date": pd.to_datetime(["2001-01-01"]),
            "position": ["SS"],
        }
    )
    return weekly, cards, logs, info


def test_weekly_lag_and_sophomore_season():
    weekly, cards, logs, info = make_inputs()
    panel = weekly_panel(weekly, cards, logs, info)
    w2 = panel[
        (panel["card_slug"] == "card/a")
        & (panel["grade"] == "psa_10")
        & (panel["week"] == "2026-09-07")
    ].iloc[0]
    # predictors through Sun Sep 6: only the Aug 20 + Sep 2 games -> games == 2
    assert w2["games"] == 2
    # rookie_year 2023 but week is in 2026 -> stats_season 2026
    assert w2["stats_season"] == 2026
    assert w2["n_sales"] == 2
    assert w2["best_offer_share"] == 0.5
    # both grades +5%/+10% -> market median between them
    assert panel["log_ret"].notna().sum() == 2  # first week per card-grade is NaN


def test_days_since_prev():
    weekly, cards, logs, info = make_inputs()
    panel = weekly_panel(weekly, cards, logs, info)
    for grade in ["psa_10", "psa_9"]:
        g = panel[(panel["card_slug"] == "card/a") & (panel["grade"] == grade)].sort_values("week")
        assert pd.isna(g.iloc[0]["days_since_prev"])  # first observed week: NA
        assert g.iloc[1]["days_since_prev"] == 7  # 2026-08-31 -> 2026-09-07
