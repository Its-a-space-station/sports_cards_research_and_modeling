import numpy as np
import pandas as pd
import pytest

from cardprice.lag_study import (
    adjust_for_market,
    assign_strata,
    classify_adjustment,
    estimate_lag,
    event_sale_windows,
    fit_half_life,
    market_relative_index,
    pooled_lag_curve,
)


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


def _planted_windows(event_id, baseline=1.0, jump=0.3, jump_day=2.0, n_per_day=3, seed=0):
    """Synthetic adjusted windows: rel_adj ~ N(1, .01) pre, ramps to 1+jump at jump_day."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(-14, 15):
        level = baseline if t < jump_day else baseline + jump
        for _ in range(n_per_day):
            rows.append({"event_id": event_id, "t_days": t + rng.uniform(-0.3, 0.3),
                         "rel_adj": level + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_pooled_lag_curve_bins_and_cis():
    windows = pd.concat([_planted_windows(0, seed=1), _planted_windows(1, seed=2)])
    curve = pooled_lag_curve(windows, n_boot=200, seed=42)
    row = curve[curve["t_bin"] == 5].iloc[0]
    assert row["median"] > 1.2
    assert row["ci_lo"] > 1.0
    pre = curve[curve["t_bin"] == -5].iloc[0]
    assert pre["ci_lo"] <= 1.0 <= pre["ci_hi"]


def test_estimate_lag_recovers_planted_two_day_lag():
    windows = pd.concat([_planted_windows(e, jump_day=2.0, seed=e) for e in range(6)])
    curve = pooled_lag_curve(windows, n_boot=200, seed=42)
    res = estimate_lag(curve)
    assert res["lag_days"] is not None and res["lag_days"] <= 3
    assert res["pre_ok"]
    assert res["pre_cover_share"] >= 0.8


def test_estimate_lag_pre_ok_false_when_pre_shifted():
    # jump planted at day -5 -> 10 of 14 pre bins are elevated -> coverage share
    # collapses -> pre_ok must be False (catches a broken baseline)
    windows = pd.concat([_planted_windows(e, jump_day=-5.0, seed=e) for e in range(6)])
    curve = pooled_lag_curve(windows, n_boot=200, seed=42)
    res = estimate_lag(curve)
    assert not res["pre_ok"]
    assert res["pre_cover_share"] < 0.8


def test_estimate_lag_none_when_no_adjustment():
    windows = pd.concat([_planted_windows(e, jump=0.0, seed=e) for e in range(6)])
    curve = pooled_lag_curve(windows, n_boot=200, seed=42)
    res = estimate_lag(curve)
    assert res["lag_days"] is None


def test_fit_half_life_planted():
    windows = pd.concat([_planted_windows(e, jump=0.3, jump_day=0.0, seed=e) for e in range(6)])
    curve = pooled_lag_curve(windows, n_boot=200, seed=42)
    res = fit_half_life(curve)
    assert res["half_life_days"] <= 1.0  # jump is immediate at t=0
    assert 0.2 < res["amplitude"] < 0.4


def test_fit_half_life_no_move_is_nan():
    windows = pd.concat([_planted_windows(e, jump=0.0, seed=e) for e in range(3)])
    curve = pooled_lag_curve(windows, n_boot=100, seed=42)
    res = fit_half_life(curve)
    assert np.isnan(res["half_life_days"])


def test_classify_adjustment_fast_late_flat():
    fast = _planted_windows(0, jump_day=1.0, seed=1)
    slow = _planted_windows(1, jump_day=9.0, seed=2)
    flat = _planted_windows(2, jump=0.0, seed=3)
    out = classify_adjustment(pd.concat([fast, slow, flat])).set_index("event_id")
    assert out.loc[0, "cls"] == "fast"
    assert out.loc[1, "cls"] in {"late", "intermediate"}
    assert out.loc[2, "cls"] == "no_adjustment"


def test_classify_adjustment_insufficient_segment():
    rows = [{"event_id": 0, "t_days": 1.0, "rel_adj": 1.3}]  # one sale total
    out = classify_adjustment(pd.DataFrame(rows))
    assert out.iloc[0]["cls"] == "insufficient"


def test_assign_strata_uses_career_stage_at_event_date():
    logs = pd.DataFrame(
        {"mlb_id": [1, 1], "group": ["hitting", "hitting"], "season": [2023, 2023],
         "date": pd.to_datetime(["2023-04-01", "2023-04-02"]), "level": ["mlb", "mlb"]}
    )
    events = pd.DataFrame(
        {"mlb_id": [1, 1, 2],
         "event_date": pd.to_datetime(["2022-06-01", "2023-06-01", "2024-06-01"])}
    )
    strata = assign_strata(events, logs)
    assert list(strata) == ["prospect", "prospect", "prospect"]
    # mlb_id 2 has no MLB logs at all -> prospect; pre-debut 2022 event -> prospect;
    # 2023-06 is rookie_year (debut season 2023) -> prospect stratum
    events2 = pd.DataFrame({"mlb_id": [1], "event_date": pd.to_datetime(["2026-06-01"])})
    assert list(assign_strata(events2, logs)) == ["established"]
