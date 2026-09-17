# src/cardprice/career.py
"""Career-to-date features at an entry date: MLB line, minors pedigree, career stage, awards.

All functions take an as_of/entry_month and use only data through that date —
the no-look-ahead contract lives here.
"""

from datetime import date

import pandas as pd

from cardprice.season_stats import hitting_to_date, pitching_to_date

LEVEL_RANK = {"a": 1, "a_plus": 2, "aa": 3, "aaa": 4}
STAGES = ("prospect", "rookie_year", "sophomore", "established")


def season_year(month: pd.Timestamp) -> int:
    """Baseball-season year for a calendar month: Nov/Dec roll to next season."""
    return month.year + 1 if month.month >= 11 else month.year


def _group_of(logs: pd.DataFrame) -> str:
    return logs["group"].iloc[0]


def career_to_date(game_logs_mlb: pd.DataFrame, mlb_id: int, as_of: date) -> dict:
    sub = game_logs_mlb[game_logs_mlb["mlb_id"] == mlb_id]
    if not len(sub):
        return {}
    fn = hitting_to_date if _group_of(sub) == "hitting" else pitching_to_date
    return fn(sub, as_of)


def minors_pedigree(game_logs_minors: pd.DataFrame, mlb_id: int, as_of: date) -> dict:
    sub = game_logs_minors[game_logs_minors["mlb_id"] == mlb_id]
    sub = sub[sub["date"].dt.date <= as_of]
    out = {"max_level": None, "max_level_rank": 0, "rate_at_max_level": None, "minor_games": 0}
    if not len(sub):
        return out
    out["minor_games"] = int(sub["gamesPlayed"].sum())
    top = max(sub["level"].unique(), key=lambda lv: LEVEL_RANK[lv])
    out["max_level"] = top
    out["max_level_rank"] = LEVEL_RANK[top]
    at_top = sub[sub["level"] == top]
    fn = hitting_to_date if _group_of(at_top) == "hitting" else pitching_to_date
    line = fn(at_top, as_of)
    out["rate_at_max_level"] = line.get("ops") if fn is hitting_to_date else line.get("era")
    return out


def career_stage(game_logs_mlb: pd.DataFrame, mlb_id: int, entry_month: pd.Timestamp) -> str:
    sub = game_logs_mlb[game_logs_mlb["mlb_id"] == mlb_id]
    before = sub[sub["date"] < entry_month]
    if not len(before):
        return "prospect"
    debut_season = int(sub["season"].min())
    idx = season_year(entry_month) - debut_season
    # any game before entry implies debut_season <= season_year(entry_month), so idx >= 0
    return "rookie_year" if idx == 0 else "sophomore" if idx == 1 else "established"


def awards_to_date(events: pd.DataFrame, mlb_id: int, as_of: date) -> int:
    wins = events[
        (events["mlb_id"] == mlb_id)
        & (events["event_type"] == "award_win")
        & (events["event_date"].dt.date <= as_of)
    ]
    return len(wins)
