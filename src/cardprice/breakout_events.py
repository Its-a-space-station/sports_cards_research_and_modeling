"""Breakout-game detection: statistically extreme single-game performances.

A breakout is a game whose score sits z >= z_min above the player's trailing
baseline at the same level (hitters: TB+BB+SB composite; pitchers: Bill James
game score, starts only). Baselines are strictly trailing (no look-ahead).
"""

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "mlb_id", "event_date", "event_type", "group", "level", "score", "z", "baseline_n",
]


def _v(x) -> float:
    return 0.0 if pd.isna(x) else float(x)


def hitter_game_score(row: pd.Series) -> float:
    """Total bases + walks + stolen bases; NaN components count as zero."""
    return _v(row.get("totalBases")) + _v(row.get("baseOnBalls")) + _v(row.get("stolenBases"))


def pitcher_game_score(row: pd.Series) -> float:
    """Bill James game score: 50 + outs + 2*(IP completed after 4th) + K
    - 2*H - 4*ER - 2*unearned - BB. `runs` is runs allowed (pitching rows)."""
    outs, k = _v(row.get("outs")), _v(row.get("strikeOuts"))
    h, er = _v(row.get("hits")), _v(row.get("earnedRuns"))
    r, bb = _v(row.get("runs")), _v(row.get("baseOnBalls"))
    bonus = 2.0 * max(0, int((outs - 12) // 3))
    return 50.0 + outs + bonus + k - 2.0 * h - 4.0 * er - 2.0 * max(0.0, r - er) - bb


def detect_breakouts(
    game_logs: pd.DataFrame,
    z_min: float = 2.5,
    baseline_games: int = 30,
    min_pa: int = 20,
    min_gs: int = 5,
) -> pd.DataFrame:
    """Detect breakout games per (mlb_id, level).

    Baseline = up to `baseline_games` trailing games at the same level
    (hitters: games with PA > 0; pitchers: starts only). Hitter baseline is
    usable when it totals >= `min_pa` plate appearances and >= 5 games;
    pitcher baseline needs >= `min_gs` starts. Baselines with sd == 0 are
    skipped (z undefined).
    """
    logs = game_logs.sort_values(["mlb_id", "level", "date"])
    rows: list[dict] = []
    for (mlb_id, level), sub in logs.groupby(["mlb_id", "level"], sort=False):
        group = sub["group"].iloc[0]
        if group == "hitting":
            scores = sub.apply(hitter_game_score, axis=1).to_numpy(dtype=float)
            usable = (sub["plateAppearances"].fillna(0) > 0).to_numpy()
            pa = sub["plateAppearances"].fillna(0).to_numpy(dtype=float)
            event_ok = np.ones(len(sub), dtype=bool)
        else:
            starts = (sub["gamesStarted"].fillna(0) == 1).to_numpy()
            scores = (
                sub.apply(pitcher_game_score, axis=1).where(starts).to_numpy(dtype=float)
            )
            usable = starts
            pa = np.zeros(len(sub))
            event_ok = starts
        dates = sub["date"].to_numpy()
        for i in range(len(sub)):
            if not event_ok[i] or np.isnan(scores[i]):
                continue
            base, pa_sum, gs_n = [], 0.0, 0
            j = i - 1
            while j >= 0 and len(base) < baseline_games:
                if usable[j] and not np.isnan(scores[j]):
                    base.append(scores[j])
                    pa_sum += pa[j]
                    gs_n += 1
                j -= 1
            base = np.asarray(base)
            if group == "hitting":
                if pa_sum < min_pa or len(base) < 5:
                    continue
            elif gs_n < min_gs:
                continue
            if len(base) < 2 or base.std(ddof=1) == 0:
                continue
            z = (scores[i] - base.mean()) / base.std(ddof=1)
            if z >= z_min:
                rows.append(
                    {
                        "mlb_id": int(mlb_id), "event_date": dates[i],
                        "event_type": "breakout", "group": group, "level": level,
                        "score": float(scores[i]), "z": float(z), "baseline_n": len(base),
                    }
                )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def merge_debuts(breakouts: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Append dated MLB debuts as a second event class (pure info shocks)."""
    debuts = events[events["event_type"] == "debut"].copy()
    for col in ("group", "level", "score", "z", "baseline_n"):
        debuts[col] = np.nan
    out = pd.concat([breakouts, debuts[EVENT_COLUMNS]], ignore_index=True)
    return out.sort_values(["mlb_id", "event_date"]).reset_index(drop=True)


def dedupe_overlaps(events: pd.DataFrame, window_days: int = 14) -> pd.DataFrame:
    """Per player: keep the first event, drop later events within +/-window_days
    of any kept event (their windows would double-count the same repricing)."""
    keep_idx = []
    for _, sub in events.sort_values("event_date").groupby("mlb_id", sort=False):
        kept_dates: list[pd.Timestamp] = []
        for idx, row in sub.iterrows():
            d = row["event_date"]
            if all(abs((d - k).days) > window_days for k in kept_dates):
                keep_idx.append(idx)
                kept_dates.append(d)
    return events.loc[keep_idx].sort_values(["mlb_id", "event_date"]).reset_index(drop=True)
