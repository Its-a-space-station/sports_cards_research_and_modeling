"""Multi-year hold panel pieces: month-end series, hold-return outcomes, trailing features.

Ungraded is the primary series; psa_10 the robustness check. psa_9 is excluded:
its chart label (grade_9) is grader-agnostic and never joins sales psa_9.
"""

import numpy as np
import pandas as pd

HORIZONS = (6, 12, 24, 36)
PANEL_SERIES = ("ungraded", "psa_10")
ENTRY_FLOOR = pd.Timestamp("2021-04-01")  # 3 trailing months exist from the 2021-03 chart floor
META_COLS = ["mlb_id", "player_name", "rookie_year", "card_type"]


def series_month_ends(chart: pd.DataFrame) -> pd.DataFrame:
    df = chart[chart["grade"].isin(PANEL_SERIES)].copy()
    df = df[~df["grade"].astype(str).str.startswith("key:")]
    # duplicate (slug, grade, date) prints exist on some psa_10 pages (chart
    # calibration artifacts) — collapse by median so the month-end pick is
    # order-independent
    df = df.groupby(["card_slug", "grade", "date"], as_index=False).agg(
        price=("price", "median"),
        mlb_id=("mlb_id", "first"),
        player_name=("player_name", "first"),
        rookie_year=("rookie_year", "first"),
        card_type=("card_type", "first"),
    )
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    return (
        df.sort_values("date")
        .groupby(["card_slug", "grade", "month"])
        .agg(price=("price", "last"), **{c: (c, "first") for c in META_COLS})
        .reset_index()
    )


def _month_idx(s: pd.Series) -> pd.Series:
    return s.dt.year * 12 + s.dt.month


def hold_returns(me: pd.DataFrame, horizons: tuple = HORIZONS) -> pd.DataFrame:
    rows = []
    for (slug, grade), grp in me.groupby(["card_slug", "grade"]):
        price_by_idx = {_month_idx(pd.Series([r.month])).iloc[0]: r.price for r in grp.itertuples()}
        for r in grp.itertuples():
            if r.month < ENTRY_FLOOR:
                continue
            t = _month_idx(pd.Series([r.month])).iloc[0]
            row = {
                "card_slug": slug,
                "grade": grade,
                "entry_month": r.month,
                "entry_price": r.price,
            }
            for h in horizons:
                p2 = price_by_idx.get(t + h)
                row[f"ret_{h}m"] = np.log(p2 / r.price) / (h / 12) if p2 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def trailing_features(me: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (slug, grade), grp in me.groupby(["card_slug", "grade"]):
        grp = grp.sort_values("month")
        price_by_idx = {_month_idx(pd.Series([r.month])).iloc[0]: r.price for r in grp.itertuples()}
        for r in grp.itertuples():
            t = _month_idx(pd.Series([r.month])).iloc[0]
            prior = [price_by_idx.get(t - k) for k in (1, 2, 3)]
            known = [p for p in prior if p]
            level = np.log(np.median(known)) if len(known) == 3 else np.nan
            p3 = price_by_idx.get(t - 3)
            rows.append(
                {
                    "card_slug": slug,
                    "grade": grade,
                    "month": r.month,
                    "price_level": level,
                    "ret_3m": np.log(r.price / p3) if p3 else np.nan,
                }
            )
    return pd.DataFrame(rows)
