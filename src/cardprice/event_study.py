"""Event study: abnormal card returns around discrete performance events."""

import numpy as np
import pandas as pd


def _event_period(event_date: pd.Timestamp, grain: str) -> pd.Timestamp:
    if grain == "monthly":
        return event_date.to_period("M").start_time
    return event_date.to_period("W-SUN").start_time


def event_windows(events, panel, grain, pre, post) -> pd.DataFrame:
    """Build event windows; offsets walk the panel's period index positionally, so
    adjacent offsets may not be calendar-adjacent across offseason gaps (monthly)
    or sparse weeks (weekly)."""
    period_col = "month" if grain == "monthly" else "week"
    horizon_col = "months_since_prev" if grain == "monthly" else "days_since_prev"
    horizon_max = 2 if grain == "monthly" else 21
    periods = sorted(panel[period_col].unique())
    rows = []
    for e in events.itertuples():
        center = _event_period(e.event_date, grain)
        if center not in periods:
            continue
        center_idx = periods.index(center)
        for offset in range(-pre, post + 1):
            idx = center_idx + offset
            if idx < 0 or idx >= len(periods):
                continue
            period = periods[idx]
            prow = panel[(panel["mlb_id"] == e.mlb_id) & (panel[period_col] == period)]
            for r in prow.itertuples():
                hz = getattr(r, horizon_col)
                rows.append(
                    {
                        "mlb_id": e.mlb_id,
                        "card_slug": r.card_slug,
                        "event_type": e.event_type,
                        "event_date": e.event_date,
                        "period_offset": offset,
                        "excess_ret": r.excess_ret,
                        "horizon_ok": bool(pd.isna(hz) or hz <= horizon_max),
                    }
                )
    return pd.DataFrame(rows)


def mean_car(windows: pd.DataFrame, offsets: list[int]) -> pd.DataFrame:
    w = windows[windows["period_offset"].isin(offsets) & windows["horizon_ok"]]
    return (
        w.groupby(["mlb_id", "card_slug", "event_type", "event_date"])["excess_ret"]
        .sum()
        .rename("car")
        .reset_index()
    )


def event_significance(cars, panel, n_perm=5000, seed=42, width: int = 1) -> dict:
    cars = pd.Series(cars).dropna()
    if len(cars) == 0:
        return {
            "mean_car": np.nan,
            "p_value": np.nan,
            "n_events": 0,
            "null_mean": np.nan,
            "null_sd": np.nan,
        }
    pool = panel["excess_ret"].dropna().to_numpy()
    rng = np.random.default_rng(seed)
    # pseudo-events: same count, same CAR width (periods summed) as observed cars
    obs = cars.mean()
    null = np.array(
        [rng.choice(pool, size=len(cars) * width, replace=True).mean() for _ in range(n_perm)]
    )
    p = float((np.abs(null) >= abs(obs)).mean())
    return {
        "mean_car": float(obs),
        "p_value": p,
        "n_events": len(cars),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
    }
