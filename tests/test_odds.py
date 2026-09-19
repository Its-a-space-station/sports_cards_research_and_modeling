import json
from pathlib import Path

import pandas as pd
import pytest

from cardprice.odds import (
    daily_prices,
    month_grain_features,
    odds_spike_events,
    parse_kalshi_candles,
    parse_polymarket_event,
    parse_polymarket_history,
    vig_normalize,
)

FIX = Path(__file__).parent / "fixtures" / "odds"


def _load(name):
    return json.loads((FIX / name).read_text())


def test_parse_polymarket_event_outcomes():
    df = parse_polymarket_event(_load("polymarket_event.json"))
    assert len(df) >= 3
    assert {"market_id", "player_name", "volume"} <= set(df.columns)
    assert (df["volume"] >= 0).all()
    # golden: the 2025 AL MVP event's known top outcome from the spike capture
    assert (df["player_name"] == "Aaron Judge").any()


def test_parse_polymarket_history_chunk():
    df = parse_polymarket_history(_load("polymarket_history_chunk.json"), market_id="m1")
    assert len(df) > 0
    assert df["ts"].is_monotonic_increasing
    assert ((df["raw_price"] >= 0) & (df["raw_price"] <= 1)).all()


def test_vig_normalize_hand_computed():
    prices = pd.DataFrame(
        [
            {"event": "e", "player_name": "A", "date": pd.Timestamp("2025-06-01"), "raw_price": 0.60},
            {"event": "e", "player_name": "B", "date": pd.Timestamp("2025-06-01"), "raw_price": 0.30},
            {"event": "e", "player_name": "C", "date": pd.Timestamp("2025-06-01"), "raw_price": 0.15},
            {"event": "e", "player_name": "A", "date": pd.Timestamp("2025-06-02"), "raw_price": 0.50},
            {"event": "e", "player_name": "B", "date": pd.Timestamp("2025-06-02"), "raw_price": 0.50},
        ]
    )
    out = vig_normalize(prices)
    day1 = out[out["date"] == "2025-06-01"]
    assert day1["implied_prob"].tolist() == pytest.approx([0.60 / 1.05, 0.30 / 1.05, 0.15 / 1.05])
    day2 = out[out["date"] == "2025-06-02"]
    assert day2["implied_prob"].tolist() == pytest.approx([0.5, 0.5])
    # zero-total date is dropped, not divided
    prices.loc[len(prices)] = {"event": "e", "player_name": "Z", "date": pd.Timestamp("2025-06-03"), "raw_price": 0.0}
    out2 = vig_normalize(prices)
    assert (out2["date"] != "2025-06-03").all()


def test_daily_prices_last_per_day():
    hist = pd.DataFrame(
        {"market_id": ["m", "m", "m"], "ts": pd.to_datetime(
            ["2025-06-01 03:00", "2025-06-01 23:00", "2025-06-02 01:00"], utc=True),
         "raw_price": [0.5, 0.6, 0.7]}
    )
    out = daily_prices(hist)
    assert len(out) == 2
    assert out.iloc[0]["raw_price"] == 0.6


def test_month_grain_features_strict_cutoffs():
    snap = pd.DataFrame(
        [
            {"mlb_id": 1, "date": pd.Timestamp("2025-05-25"), "implied_prob": 0.10, "season": 2025},
            {"mlb_id": 1, "date": pd.Timestamp("2025-06-20"), "implied_prob": 0.25, "season": 2025},
            {"mlb_id": 1, "date": pd.Timestamp("2025-07-01"), "implied_prob": 0.99, "season": 2025},
        ]
    )
    entries = pd.Series([pd.Timestamp("2025-07-01")], name="entry_month")
    out = month_grain_features(snap, entries)
    row = out[(out["mlb_id"] == 1) & (out["entry_month"] == pd.Timestamp("2025-07-01"))].iloc[0]
    assert row["odds_level"] == pytest.approx(0.25)  # 07-01 snapshot EXCLUDED (strict <)
    assert row["odds_delta_7d"] == pytest.approx(0.25 - 0.10)  # last before 06-24 is 06-20; last before 06-24 minus last before ~05-25 window
    assert row["has_market"] == 1


def test_month_grain_features_no_market_defaults():
    snap = pd.DataFrame(columns=["mlb_id", "date", "implied_prob", "season"])
    entries = pd.Series([pd.Timestamp("2025-07-01")], name="entry_month")
    players = pd.DataFrame({"mlb_id": [9], "entry_month": [pd.Timestamp("2025-07-01")], "season": [2025]})
    out = month_grain_features(snap, entries, players=players)
    row = out.iloc[0]
    assert row["has_market"] == 0
    assert row["odds_level"] == 0.0 and row["odds_delta_7d"] == 0.0 and row["odds_delta_30d"] == 0.0


def test_parse_kalshi_candles_price_units():
    df = parse_kalshi_candles(_load("kalshi_candlesticks.json"), market_ticker="KX1")
    assert len(df) > 0
    assert df["ts"].is_monotonic_increasing
    assert ((df["raw_price"] >= 0) & (df["raw_price"] <= 1)).all()  # cents converted to dollars


def test_odds_spike_events_threshold_and_separation():
    dates = pd.date_range("2025-05-01", periods=60, freq="D")
    probs = [0.10] * 30 + [0.10, 0.45, 0.44, 0.46, 0.10] + [0.10] * 25
    snap = pd.DataFrame({"mlb_id": [1] * 60, "date": dates, "implied_prob": probs})
    out = odds_spike_events(snap)
    assert len(out) == 2  # the +0.35 jump and the -0.36 drop; the 0.44/0.46 wiggles are in-cluster
    assert out.iloc[0]["delta"] == pytest.approx(0.35)
    assert out.iloc[1]["delta"] == pytest.approx(-0.36)
    assert (out["event_type"] == "odds_spike").all()
