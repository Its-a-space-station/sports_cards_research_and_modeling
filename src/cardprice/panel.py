"""Card x month analysis panel with strictly lagged predictors."""

import numpy as np
import pandas as pd

from cardprice.stats_panel import player_stats_series

HITTING_COLS = ["games", "ops", "avg", "obp", "slg", "home_runs", "strikeouts"]
PITCHING_COLS = ["era", "whip", "k_bb_pct", "innings_pitched"]


def _month_ends(chart: pd.DataFrame) -> pd.DataFrame:
    df = chart[chart["grade"] == "psa_10"].copy()
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    # last price point within each month
    return (
        df.sort_values("date")
        .groupby(["card_slug", "month"])
        .agg(
            price=("price", "last"),
            mlb_id=("mlb_id", "first"),
            player_name=("player_name", "first"),
            rookie_year=("rookie_year", "first"),
            set_slug=("set_slug", "first"),
        )
        .reset_index()
    )


def monthly_panel(
    chart: pd.DataFrame, game_logs: pd.DataFrame, player_info: pd.DataFrame
) -> pd.DataFrame:
    me = _month_ends(chart)
    rows = []
    for card in me.itertuples():
        season = int(card.month.year)  # stats roll to the month's own season (>= rookie_year)
        # stats lag: cumulative through last day of previous month
        lag_date = card.month - pd.Timedelta(days=1)
        stats = player_stats_series(game_logs, int(card.mlb_id), season, [lag_date])
        if not len(stats):
            continue  # no game logs this season
        s = stats.iloc[0]
        # drop months that end before the player's debut (first game after month end)
        first_game = game_logs[
            (game_logs["mlb_id"] == card.mlb_id) & (game_logs["season"] == season)
        ]["date"].min()
        if pd.isna(first_game) or first_game >= card.month + pd.offsets.MonthBegin(1):
            continue
        # form: games in the last 14 days of the lag window
        form_start = lag_date - pd.Timedelta(days=13)
        recent = player_stats_series(game_logs, int(card.mlb_id), season, [lag_date]).iloc[0]
        earlier = player_stats_series(
            game_logs, int(card.mlb_id), season, [form_start - pd.Timedelta(days=1)]
        ).iloc[0]
        form_games = int(recent["games"] - earlier["games"])
        row = {
            "card_slug": card.card_slug,
            "mlb_id": int(card.mlb_id),
            "player_name": card.player_name,
            "month": card.month,
            "price": card.price,
            "rookie_year": int(card.rookie_year),
            "stats_season": season,
            "set_slug": card.set_slug,
            "grade": "psa_10",
            "playoff": int(card.month.month == 10),
            "form_games": form_games,
        }
        if "ops" in s.index:  # hitter
            for c in HITTING_COLS:
                row[c] = s[c]
            row["form_ops_delta"] = _form_delta_hitting(
                game_logs, int(card.mlb_id), season, form_start, lag_date, s["ops"]
            )
            row["form_era_delta"] = np.nan
        else:  # pitcher
            for c in PITCHING_COLS:
                row[c] = s[c]
            row["games"] = s["games"]  # appearances (gamesPlayed), same lag rule as hitters
            row["form_era_delta"] = _form_delta_pitching(
                game_logs, int(card.mlb_id), season, form_start, lag_date, s["era"]
            )
            row["form_ops_delta"] = np.nan
        info = player_info[player_info["mlb_id"] == card.mlb_id]
        if len(info):
            birth = info.iloc[0]["birth_date"]
            row["age"] = round((card.month - birth).days / 365.25, 2)
            row["position"] = info.iloc[0]["position"]
        rows.append(row)
    panel = pd.DataFrame(rows)
    # outcomes per card
    panel = panel.sort_values(["card_slug", "month"])
    panel["log_ret"] = panel.groupby("card_slug")["price"].transform(
        lambda p: np.log(p / p.shift(1))
    )
    market = panel.groupby("month")["log_ret"].median().rename("market_median_ret")
    panel = panel.merge(market, on="month", how="left")
    panel["excess_ret"] = panel["log_ret"] - panel["market_median_ret"]
    return panel.reset_index(drop=True)


def _form_delta_hitting(game_logs, mlb_id, season, form_start, lag_date, season_ops):
    full = player_stats_series(game_logs, mlb_id, season, [lag_date]).iloc[0]
    before = player_stats_series(
        game_logs, mlb_id, season, [form_start - pd.Timedelta(days=1)]
    ).iloc[0]
    # reconstruct last-14-day OPS from counting-stat differences
    ab = full["at_bats"] - before["at_bats"]
    if ab <= 0:
        return np.nan
    h = full["hits"] - before["hits"]
    bb = full["walks"] - before["walks"]
    hbp = full["hit_by_pitch"] - before["hit_by_pitch"]
    sf = full["sac_flies"] - before["sac_flies"]
    tb = (
        h
        + (full["doubles"] - before["doubles"])
        + 2 * (full["triples"] - before["triples"])
        + 3 * (full["home_runs"] - before["home_runs"])
    )
    obp_den = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_den if obp_den else np.nan
    slg = tb / ab
    if pd.isna(obp) or pd.isna(season_ops):
        return np.nan
    return round(obp + slg - season_ops, 3)


def _form_delta_pitching(game_logs, mlb_id, season, form_start, lag_date, season_era):
    full = player_stats_series(game_logs, mlb_id, season, [lag_date]).iloc[0]
    before = player_stats_series(
        game_logs, mlb_id, season, [form_start - pd.Timedelta(days=1)]
    ).iloc[0]
    outs = full["outs"] - before["outs"]
    if outs <= 0 or pd.isna(season_era):
        return np.nan
    er = full["earned_runs"] - before["earned_runs"]
    era_14d = 9 * er / (outs / 3)
    return round(era_14d - season_era, 3)


def weekly_panel(
    weekly: pd.DataFrame, cards: pd.DataFrame, game_logs: pd.DataFrame, player_info: pd.DataFrame
) -> pd.DataFrame:
    meta = cards[
        ["card_slug", "player_name", "mlb_id", "rookie_year", "set_slug"]
    ].drop_duplicates()
    df = weekly.merge(meta, on="card_slug", how="left", validate="many_to_one")
    rows = []
    for r in df.itertuples():
        stats_season = max(int(r.rookie_year), r.week.year)
        lag_date = r.week - pd.Timedelta(days=1)  # Sunday before the Monday week start
        stats = player_stats_series(game_logs, int(r.mlb_id), stats_season, [lag_date])
        if not len(stats):
            continue
        s = stats.iloc[0]
        first_game = game_logs[
            (game_logs["mlb_id"] == r.mlb_id) & (game_logs["season"] == stats_season)
        ]["date"].min()
        if pd.isna(first_game) or first_game > lag_date:
            continue  # not yet debuted this season at the lag date
        form_start = lag_date - pd.Timedelta(days=13)
        row = {
            "card_slug": r.card_slug,
            "grade": r.grade,
            "mlb_id": int(r.mlb_id),
            "player_name": r.player_name,
            "week": r.week,
            "price": r.median_price,
            "n_sales": int(r.n_sales),
            "best_offer_share": float(r.best_offer_share),
            "rookie_year": int(r.rookie_year),
            "stats_season": stats_season,
            "set_slug": r.set_slug,
            "playoff": int(r.week.month == 10),
        }
        earlier = player_stats_series(
            game_logs, int(r.mlb_id), stats_season, [form_start - pd.Timedelta(days=1)]
        ).iloc[0]
        row["form_games"] = int(s["games"] - earlier["games"])
        if "ops" in s.index:
            for c in HITTING_COLS:
                row[c] = s[c]
            row["form_ops_delta"] = _form_delta_hitting(
                game_logs, int(r.mlb_id), stats_season, form_start, lag_date, s["ops"]
            )
            row["form_era_delta"] = np.nan
        else:
            for c in PITCHING_COLS:
                row[c] = s[c]
            row["games"] = s["games"]  # appearances (gamesPlayed), same lag rule as hitters
            row["form_era_delta"] = _form_delta_pitching(
                game_logs, int(r.mlb_id), stats_season, form_start, lag_date, s["era"]
            )
            row["form_ops_delta"] = np.nan
        info = player_info[player_info["mlb_id"] == r.mlb_id]
        if len(info):
            birth = info.iloc[0]["birth_date"]
            row["age"] = round((r.week - birth).days / 365.25, 2)
            row["position"] = info.iloc[0]["position"]
        rows.append(row)
    panel = pd.DataFrame(rows)
    panel = panel.sort_values(["card_slug", "grade", "week"])
    panel["log_ret"] = panel.groupby(["card_slug", "grade"])["price"].transform(
        lambda p: np.log(p / p.shift(1))
    )
    market = panel.groupby("week")["log_ret"].median().rename("market_median_ret")
    panel = panel.merge(market, on="week", how="left")
    panel["excess_ret"] = panel["log_ret"] - panel["market_median_ret"]
    return panel.reset_index(drop=True)
