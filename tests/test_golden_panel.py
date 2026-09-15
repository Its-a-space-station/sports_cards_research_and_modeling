import pandas as pd
import pytest

from cardprice.panel import monthly_panel


@pytest.fixture(scope="module")
def built():
    chart = pd.read_parquet("data/processed/scp_chart_monthly.parquet")
    game_logs = pd.read_parquet("data/processed/game_logs.parquet")
    info = pd.read_csv("data/reference/player_info.csv", parse_dates=["birth_date"])
    return monthly_panel(chart, game_logs, info)


def test_henderson_august_2023_row(built):
    # CORRECTED from plan 3 (which asserted a 2023-07-01 row with 73 G / 13 HR / .253).
    # Two findings, verified 2026-09-15:
    # 1. Henderson's 2023 Topps Chrome #2 chart has no July 2023 price point
    #    (first scp_chart_monthly date is 2023-08), so the first panel row is
    #    2023-08-01, lagged through 2023-07-31.
    # 2. The plan's stat line is wrong. Official MLB Stats API byDateRange
    #    (people/683002/stats?stats=byDateRange&group=hitting&season=2023&
    #     startDate=2023-03-30&endDate=2023-07-31) gives 95 G, 17 HR,
    #    79 H / 327 AB = .242 AVG; game_logs.parquet reconstructs the identical
    #    line (full-season check: 150 G / 28 HR / .255 matches his official 2023).
    row = built[(built["player_name"] == "Gunnar Henderson") & (built["month"] == "2023-08-01")]
    assert len(row) == 1
    r = row.iloc[0]
    # stats through 2023-07-31 (lag = day before month start)
    assert r["games"] == 95
    assert r["home_runs"] == 17
    assert r["avg"] == pytest.approx(0.242, abs=0.002)
    assert r["playoff"] == 0
    assert r["age"] == pytest.approx(22.09, abs=0.05)  # born 2001-06-29


def test_henderson_june_cumulative_reconstruction():
    # the plan's "through June 2023" claim, corrected: 70 G / 11 HR / .240, not
    # 73 / 13 / .253. Verified against Stats API byDateRange 2023-03-30..2023-06-30.
    logs = pd.read_parquet("data/processed/game_logs.parquet")
    h = logs[
        (logs["mlb_id"] == 683002)
        & (logs["season"] == 2023)
        & (logs["group"] == "hitting")
        & (logs["date"] <= "2023-06-30")
    ]
    assert h["gamesPlayed"].sum() == 70
    assert h["homeRuns"].sum() == 11
    assert h["hits"].sum() / h["atBats"].sum() == pytest.approx(0.240, abs=0.002)


def test_no_lookahead_invariant(built):
    # for any row, lagged games must never exceed the player's total season games
    logs = pd.read_parquet("data/processed/game_logs.parquet")
    totals = logs.groupby(["mlb_id", "season"]).size().rename("season_games")
    merged = built.merge(totals, left_on=["mlb_id", "stats_season"], right_index=True, how="left")
    assert (merged["games"] <= merged["season_games"]).all()


def test_excess_ret_identity(built):
    # first month per card has NaN log_ret (no prior price), hence NaN excess_ret;
    # those rows are legitimately NaN (Task 3 note) and excluded from the identity
    diff = (built["excess_ret"] - (built["log_ret"] - built["market_median_ret"])).abs()
    assert (diff.dropna() < 1e-9).all()
    assert (built["excess_ret"].isna() == built["log_ret"].isna()).all()
