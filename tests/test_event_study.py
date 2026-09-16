import numpy as np
import pandas as pd

from cardprice.event_study import event_significance, event_windows, mean_car


def make_panel_and_events(seed=7, spike=0.0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-03-01", periods=8, freq="MS")
    rows = []
    for m in months:
        for c in range(6):
            rows.append(
                {
                    "mlb_id": c,
                    "card_slug": f"card/{c}",
                    "month": m,
                    "excess_ret": rng.normal(0, 0.05),
                    "months_since_prev": 1,
                }
            )
    panel = pd.DataFrame(rows)
    events = pd.DataFrame(
        {
            "mlb_id": [0, 1, 2],
            "event_date": pd.to_datetime(["2023-06-10"] * 3),
            "event_type": "three_hr_game",
            "details": "3 HR",
        }
    )
    if spike:
        for _, e in events.iterrows():
            mask = (panel["mlb_id"] == e["mlb_id"]) & (panel["month"] == "2023-06-01")
            panel.loc[mask, "excess_ret"] += spike
    return panel, events


def test_event_windows_offsets():
    panel, events = make_panel_and_events()
    w = event_windows(events, panel, "monthly", pre=1, post=2)
    assert set(w["period_offset"]) == {-1, 0, 1, 2}
    assert (w["n"].count() if "n" in w else len(w)) == 12  # 3 events x 4 offsets
    assert w["horizon_ok"].all()


def test_mean_car_with_planted_spike():
    # seed 3, not the brief's default 7: at seed 7 the mlb_id=2 July draw is
    # -0.126, so its [0,1] CAR lands at 0.182 and the 0.30 threshold fails
    # deterministically. Seed-only adjustment per the brief's allowance.
    panel, events = make_panel_and_events(seed=3, spike=0.40)
    w = event_windows(events, panel, "monthly", pre=1, post=1)
    cars = mean_car(w, [0, 1])
    assert (
        cars["car"] > 0.30
    ).all()  # planted 0.40 minus small noise (fixed seed makes this deterministic)


def test_significance_detects_spike_and_null():
    panel, events = make_panel_and_events(seed=11, spike=0.0)
    w = event_windows(events, panel, "monthly", pre=0, post=0)
    cars = mean_car(w, [0])
    res = event_significance(cars["car"], panel, n_perm=2000, seed=42)
    assert res["p_value"] > 0.05  # no planted effect -> null not rejected

    panel2, events2 = make_panel_and_events(seed=11, spike=0.40)
    w2 = event_windows(events2, panel2, "monthly", pre=0, post=0)
    cars2 = mean_car(w2, [0])
    res2 = event_significance(cars2["car"], panel2, n_perm=2000, seed=42)
    assert res2["p_value"] <= 0.05  # strong planted effect detected
