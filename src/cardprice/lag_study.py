"""Price incorporation lag: sale-level event study at daily grain.

Each sale is normalized by its card's EVENT-ANCHORED baseline (median of
same-grade-class sales strictly before the event) so post-event repricing is
visible instead of absorbed by a moving baseline. Pooled curves + cluster
bootstrap live in part 2 (Tasks 3-4).
"""

import pandas as pd

GRADE_CLASSES = {"ungraded": None, "psa_10": "psa_10"}

WINDOW_COLUMNS = [
    "event_id", "mlb_id", "card_slug", "event_date", "sale_date", "t_days", "rel_price",
]


def _grade_mask(sales: pd.DataFrame, grade_class: str) -> pd.Series:
    grade = GRADE_CLASSES[grade_class]
    return sales["grade"].isna() if grade is None else (sales["grade"] == grade)


def event_sale_windows(
    sales: pd.DataFrame,
    events: pd.DataFrame,
    grade_class: str = "ungraded",
    baseline_days: int = 28,
    window_days: int = 14,
    min_baseline: int = 3,
    min_window: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Align a card's sales around each event; normalize by pre-event baseline.

    Returns (windows, drop_log); drop_log counts event-card pairs dropped for
    too few baseline sales / too few window sales, and kept pairs.
    """
    s = sales[_grade_mask(sales, grade_class) & sales["price"].notna()]
    rows: list[dict] = []
    drop_log = {"dropped_baseline": 0, "dropped_window": 0, "kept": 0}
    by_card = {c: g.sort_values("sale_date") for c, g in s.groupby("card_slug")}
    cards_of = s.groupby("mlb_id")["card_slug"].unique()
    for event_id, e in enumerate(events.itertuples()):
        for card in cards_of.get(e.mlb_id, []):
            g = by_card.get(card)
            if g is None:
                continue
            d = g["sale_date"]
            pre = g[(d >= e.event_date - pd.Timedelta(days=baseline_days)) & (d < e.event_date)]
            if len(pre) < min_baseline:
                drop_log["dropped_baseline"] += 1
                continue
            baseline = float(pre["price"].median())
            win = g[(d >= e.event_date - pd.Timedelta(days=window_days))
                    & (d <= e.event_date + pd.Timedelta(days=window_days))]
            if len(win) < min_window:
                drop_log["dropped_window"] += 1
                continue
            drop_log["kept"] += 1
            t = (win["sale_date"] - e.event_date).dt.total_seconds() / 86400.0
            for sale_date, t_days, price in zip(win["sale_date"], t, win["price"]):
                rows.append(
                    {
                        "event_id": event_id, "mlb_id": e.mlb_id, "card_slug": card,
                        "event_date": e.event_date, "sale_date": sale_date,
                        "t_days": float(t_days), "rel_price": float(price) / baseline,
                    }
                )
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS), drop_log
