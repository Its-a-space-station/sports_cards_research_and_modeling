import numpy as np
import pandas as pd
import pytest

from cardprice.lag_study import adjust_for_market, event_sale_windows, market_relative_index


def _sales(rows):
    return pd.DataFrame(rows, columns=["sale_date", "price", "grade", "mlb_id", "card_slug"])


def _events():
    return pd.DataFrame(
        {"mlb_id": [1], "event_date": pd.to_datetime(["2024-06-15"]),
         "event_type": ["breakout"], "group": ["hitting"], "level": ["mlb"],
         "score": [12.0], "z": [3.0], "baseline_n": [30]}
    )


def _card_sales(card="a/x", base_price=100.0, n_base=5, n_pre_in_window=3, post=()):
    """Baseline sales at days -20..-16 (inside the 56d baseline; these also land
    INSIDE the ±21d default window — only the planted pre sales at -3,-2,-1 are
    guaranteed in-window), then post-event sales from day +1."""
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
    # 5 baseline sales land inside the ±21d default window, but 5 < explicit
    # min_window=6 -> dropped_window (window is not empty)
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
    # explicit window_days=14 is intentional: the calibrated default is ±21d,
    # but the bounds logic is window-agnostic; ±14d are inside, +15d is outside
    rows = _card_sales(n_base=5, n_pre_in_window=0, post=[130.0] * 3)
    rows.append((pd.Timestamp("2024-06-01"), 99.0, None, 1, "a/x"))   # exactly -14d
    rows.append((pd.Timestamp("2024-06-29"), 130.0, None, 1, "a/x"))   # exactly +14d
    rows.append((pd.Timestamp("2024-06-30"), 130.0, None, 1, "a/x"))   # +15d -> outside
    windows, drop_log = event_sale_windows(_sales(rows), _events(), window_days=14)
    assert drop_log["kept"] == 1
    assert windows["t_days"].min() == pytest.approx(-14.0)
    assert windows["t_days"].max() == pytest.approx(14.0)


def _market_sales():
    rows = []
    # card A: baseline 100 (days -30..-26 of the probe day), then 130 on probe days
    for i in range(5):
        rows.append((pd.Timestamp("2024-05-20") + pd.Timedelta(days=i), 100.0, None, 1, "a/x"))
    # card B: baseline 50, flat (no event)
    for i in range(5):
        rows.append((pd.Timestamp("2024-05-20") + pd.Timedelta(days=i), 50.0, None, 2, "b/y"))
    # card C: baseline 200, flat
    for i in range(5):
        rows.append((pd.Timestamp("2024-05-20") + pd.Timedelta(days=i), 200.0, None, 3, "c/z"))
    probe = pd.Timestamp("2024-06-15")
    rows.append((probe, 130.0, None, 1, "a/x"))
    rows.append((probe, 50.0, None, 2, "b/y"))
    rows.append((probe, 200.0, None, 3, "c/z"))
    return _sales(rows), probe


def test_market_index_hand_computed():
    sales, probe = _market_sales()
    mkt = market_relative_index(sales)
    # at probe: A rel = 130/100 = 1.3, B = 1.0, C = 1.0 -> median = 1.0
    assert mkt.loc[probe] == 1.0
    probe2 = probe + pd.Timedelta(days=1)
    # only card A trades next day: < min_cards contributing -> the day never
    # enters as data; interpolation (limit 3, from probe's 1.0) fills it
    sales2 = pd.concat([sales, _sales([(probe2, 130.0, None, 1, "a/x")])])
    mkt2 = market_relative_index(sales2)
    assert mkt2.loc[probe2] == 1.0


def test_adjust_for_market_divides():
    sales, probe = _market_sales()
    mkt = market_relative_index(sales)
    # pretend a window sale of card A at probe with rel_price 1.3
    windows = pd.DataFrame(
        {"event_id": [0], "mlb_id": [1], "card_slug": ["a/x"],
         "event_date": [probe], "sale_date": [probe], "t_days": [0.0], "rel_price": [1.3]}
    )
    out, n_unadj = adjust_for_market(windows, mkt)
    assert out["rel_adj"].iloc[0] == 1.3 / 1.0
    assert n_unadj == 0


def test_adjust_for_market_missing_index_is_nan_not_fabricated():
    windows = pd.DataFrame(
        {"event_id": [0], "mlb_id": [1], "card_slug": ["a/x"],
         "event_date": [pd.Timestamp("2024-06-15")],
         "sale_date": [pd.Timestamp("2019-01-01")],  # far outside any market coverage
         "t_days": [0.0], "rel_price": [1.3]}
    )
    sales, _ = _market_sales()
    mkt = market_relative_index(sales)
    out, n_unadj = adjust_for_market(windows, mkt)
    assert pd.isna(out["rel_adj"].iloc[0])
    assert n_unadj == 1
