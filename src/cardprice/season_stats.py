# src/cardprice/season_stats.py
"""Rebuild season-to-date cumulative stats from game logs at any date.

Rate stats are always recomputed from summed counting stats, never averaged
from per-game rates.
"""

from datetime import date

import pandas as pd

HITTING_MAP = {
    "gamesPlayed": "games",
    "atBats": "at_bats",
    "hits": "hits",
    "doubles": "doubles",
    "triples": "triples",
    "homeRuns": "home_runs",
    "rbi": "rbi",
    "baseOnBalls": "walks",
    "strikeOuts": "strikeouts",
    "hitByPitch": "hit_by_pitch",
    "sacFlies": "sac_flies",
    "stolenBases": "stolen_bases",
}


def _safe_ratio(num: float, den: float) -> float | None:
    return round(num / den, 3) if den else None


def hitting_to_date(game_log: pd.DataFrame, through: date) -> dict:
    df = game_log[game_log["date"].dt.date <= through]
    out = {new: int(df[old].sum()) if old in df else 0 for old, new in HITTING_MAP.items()}
    ab, h, bb, hbp, sf = (out[k] for k in ("at_bats", "hits", "walks", "hit_by_pitch", "sac_flies"))
    total_bases = h + out["doubles"] + 2 * out["triples"] + 3 * out["home_runs"]
    out["avg"] = _safe_ratio(h, ab)
    out["obp"] = _safe_ratio(h + bb + hbp, ab + bb + hbp + sf)
    out["slg"] = _safe_ratio(total_bases, ab)
    out["ops"] = (
        round(out["obp"] + out["slg"], 3)
        if out["obp"] is not None and out["slg"] is not None
        else None
    )
    return out


PITCHING_MAP = {
    "gamesPlayed": "games",
    "gamesStarted": "games_started",
    "wins": "wins",
    "losses": "losses",
    "hits": "hits_allowed",
    "earnedRuns": "earned_runs",
    "baseOnBalls": "walks",
    "strikeOuts": "strikeouts",
    "battersFaced": "batters_faced",
}


def innings_to_outs(ip) -> int:
    """MLB notation: 6.1 = 6 and 1/3 innings = 19 outs."""
    whole, _, frac = str(ip).partition(".")
    return int(whole) * 3 + (int(frac) if frac else 0)


def pitching_to_date(game_log: pd.DataFrame, through: date) -> dict:
    df = game_log[game_log["date"].dt.date <= through]
    out = {new: int(df[old].sum()) if old in df else 0 for old, new in PITCHING_MAP.items()}
    out["outs"] = int(df["inningsPitched"].map(innings_to_outs).sum()) if len(df) else 0
    ip = out["outs"] / 3
    out["innings_pitched"] = round(ip, 3)
    out["era"] = _safe_ratio(9 * out["earned_runs"], ip)
    out["whip"] = _safe_ratio(out["walks"] + out["hits_allowed"], ip)
    out["k_bb_pct"] = _safe_ratio(out["strikeouts"] - out["walks"], out["batters_faced"])
    return out
