"""Season-to-date stat lines at arbitrary dates, per player-season."""

import pandas as pd

from cardprice.season_stats import hitting_to_date, pitching_to_date


def stats_at_dates(game_log: pd.DataFrame, dates: list[pd.Timestamp], group: str) -> pd.DataFrame:
    fn = hitting_to_date if group == "hitting" else pitching_to_date
    rows = [{"date": d, **fn(game_log, d.date())} for d in dates]
    return pd.DataFrame(rows)


def player_stats_series(
    game_logs: pd.DataFrame, mlb_id: int, season: int, dates: list[pd.Timestamp]
) -> pd.DataFrame:
    sub = game_logs[(game_logs["mlb_id"] == mlb_id) & (game_logs["season"] == season)]
    if not len(sub) or not len(dates):
        return pd.DataFrame()
    group = sub["group"].iloc[0]
    out = stats_at_dates(sub, dates, group)
    out["mlb_id"] = mlb_id
    return out
