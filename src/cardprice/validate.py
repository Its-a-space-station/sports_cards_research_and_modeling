"""Post-ingest data-quality checks. Quarantine flags, never deletes."""

import re

import pandas as pd

CONTAMINATION_RE = re.compile(
    r"\b(?:lot of|reprint|rp|proxy|custom card|case break|digital)\b", re.IGNORECASE
)


def quarantine_outliers(sales: pd.DataFrame, iqr_mult: float = 4.0) -> pd.DataFrame:
    df = sales.copy()
    df["outlier"] = False
    month = df["sale_date"].dt.to_period("M")
    for idx in df.groupby([df["card_slug"], df["grade"], month]).groups.values():
        group = df.loc[idx, "price"]
        if len(group) < 5:
            continue
        q1, q3 = group.quantile(0.25), group.quantile(0.75)
        iqr = q3 - q1
        if not iqr or pd.isna(iqr):
            continue
        lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        df.loc[idx, "outlier"] = (group < lo) | (group > hi)
    return df


def contamination_audit(sales: pd.DataFrame) -> pd.DataFrame:
    mask = sales["title"].str.contains(CONTAMINATION_RE, na=False)
    out = sales[mask].copy()
    out["flag_reason"] = out["title"].str.extract(
        "(" + CONTAMINATION_RE.pattern + ")", expand=False
    )
    return out


def stale_series(
    weekly: pd.DataFrame, max_gap_weeks: int = 3, as_of: pd.Timestamp | None = None
) -> list[str]:
    as_of = as_of or pd.Timestamp.today().normalize()
    latest = weekly.groupby("card_slug")["week"].max()
    gap = (as_of - latest).dt.days / 7
    return sorted(latest[gap > max_gap_weeks].index.tolist())
