import numpy as np
import pandas as pd
import pytest

from cardprice.lag_study import event_sale_windows


def _sales(rows):
    return pd.DataFrame(rows, columns=["sale_date", "price", "grade", "mlb_id", "card_slug"])


def _events():
    return pd.DataFrame(
        {"mlb_id": [1], "event_date": pd.to_datetime(["2024-06-15"]),
         "event_type": ["breakout"], "group": ["hitting"], "level": ["mlb"],
         "score": [12.0], "z": [3.0], "baseline_n": [30]}
    )


def _card_sales(card="a/x", base_price=100.0, n_base=5, n_pre_in_window=3, post=()):
    """Baseline sales at days -20..-16 (inside 28d baseline, OUTSIDE the ±14d
    window), in-window pre-event sales at days -3,-2,-1, then post-event sales
    from day +1."""
    rows = []
    for i in range(n_base):
        rows.append((pd.Timestamp("2024-05-26") + pd.Timedelta(days=i), base_price, None, 1, card))
    for i in range(n_pre_in_window):
        rows.append((pd.Timestamp("2024-06-12") + pd.Timedelta(days=i), base_price, None, 1, card))
    for j, price in enumerate(post):
        rows.append((pd.Timestamp("2024-06-16") + pd.Timedelta(days=j), price, None, 1, card))
    return rows


def test_rel_price_normalized_by_pre_event_median():
    sales = _sales(_card_sales(base_price=100.0, post=[130.0, 132.0, 128.0, 131.0, 129.0]))
    windows, drop_log = event_sale_windows(sales, _events())
    assert drop_log["kept"] == 1
    post = windows[windows["t_days"] > 0]
    assert np.isclose(post["rel_price"].mean(), 1.30, atol=0.02)
    pre = windows[windows["t_days"] < 0]
    assert np.isclose(pre["rel_price"].median(), 1.0)


def test_baseline_never_uses_post_event_sales():
    # baseline sales at 100 (some inside the ±14d window); post-event sales at
    # 200. If the baseline leaked post-event sales, rel would be pulled below 2.
    rows = [(pd.Timestamp("2024-06-01"), 100.0, None, 1, "a/x"),
            (pd.Timestamp("2024-06-05"), 100.0, None, 1, "a/x"),
            (pd.Timestamp("2024-06-10"), 100.0, None, 1, "a/x")]
    rows += [(pd.Timestamp("2024-06-16") + pd.Timedelta(days=j), 200.0, None, 1, "a/x")
             for j in range(5)]
    windows, _ = event_sale_windows(_sales(rows), _events())
    post = windows[windows["t_days"] > 0]
    assert (post["rel_price"] == 2.0).all()


def test_min_baseline_sales_rule():
    # only 2 pre-event sales total -> baseline too thin (in-window pre sales
    # would count toward the baseline too, so none are planted here)
    sales = _sales(_card_sales(n_base=2, n_pre_in_window=0, post=[130.0] * 5))
    windows, drop_log = event_sale_windows(sales, _events())
    assert drop_log["kept"] == 0 and drop_log["dropped_baseline"] == 1
    assert len(windows) == 0


def test_min_window_sales_rule():
    # 5 baseline sales but nothing inside the ±14d window -> dropped_window
    sales = _sales(_card_sales(n_base=5, n_pre_in_window=0, post=[]))
    _windows, drop_log = event_sale_windows(sales, _events(), min_window=6)
    assert drop_log["kept"] == 0 and drop_log["dropped_window"] == 1


def test_grade_class_filters():
    rows = _card_sales(base_price=100.0, post=[130.0] * 5)
    rows = [(d, p, "psa_10", m, c) for (d, p, g, m, c) in rows]  # all graded
    _windows, drop_log = event_sale_windows(_sales(rows), _events(), grade_class="ungraded")
    assert drop_log["kept"] == 0
    windows10, drop_log10 = event_sale_windows(_sales(rows), _events(), grade_class="psa_10")
    assert drop_log10["kept"] == 1
    assert (windows10["rel_price"] > 0).all()


def test_window_bounds():
    # sales at exactly -14d and +14d are inside; +15d is outside
    rows = _card_sales(n_base=5, n_pre_in_window=0, post=[130.0] * 3)
    rows.append((pd.Timestamp("2024-06-01"), 99.0, None, 1, "a/x"))   # exactly -14d
    rows.append((pd.Timestamp("2024-06-29"), 130.0, None, 1, "a/x"))   # exactly +14d
    rows.append((pd.Timestamp("2024-06-30"), 130.0, None, 1, "a/x"))   # +15d -> outside
    windows, drop_log = event_sale_windows(_sales(rows), _events())
    assert drop_log["kept"] == 1
    assert windows["t_days"].min() == pytest.approx(-14.0)
    assert windows["t_days"].max() == pytest.approx(14.0)
