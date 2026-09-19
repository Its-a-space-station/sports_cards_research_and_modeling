# tests/test_attach_odds.py
import pandas as pd
import pytest

from cardprice.odds import attach_odds_features


def _frame():
    return pd.DataFrame(
        {
            "entry_month": [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-01"),
                            pd.Timestamp("2025-08-01")],
            "entry_year": [2025, 2025, 2025],
            "mlb_id": [1, 2, 1], "card_slug": ["a/x", "b/y", "a/x"],
            "ret_12m": [0.1, 0.2, 0.3],
        }
    )


def _snapshots():
    return pd.DataFrame(
        [
            {"mlb_id": 1, "date": pd.Timestamp("2025-06-15"), "implied_prob": 0.2, "season": 2025},
            {"mlb_id": 1, "date": pd.Timestamp("2025-07-15"), "implied_prob": 0.35, "season": 2025},
        ]
    )


def test_attach_defaults_and_join():
    out = attach_odds_features(_frame(), _snapshots())
    r1 = out[(out["mlb_id"] == 1) & (out["entry_month"] == "2025-07-01")].iloc[0]
    assert r1["has_market"] == 1 and r1["odds_level"] == 0.2
    assert pd.isna(r1["odds_delta_7d"])  # no snapshot before 06-24
    r2 = out[out["mlb_id"] == 2].iloc[0]
    assert r2["has_market"] == 0 and r2["odds_level"] == 0.0
    r3 = out[(out["mlb_id"] == 1) & (out["entry_month"] == "2025-08-01")].iloc[0]
    assert r3["odds_level"] == 0.35
    assert r3["odds_delta_30d"] == pytest.approx(0.35 - 0.2)
    assert len(out) == 3  # many_to_one, no row multiplication


def test_attach_duplicate_player_month_no_row_multiplication():
    # two cards of the same player in the same entry month (the real hold frame
    # is per card_slug): the feature grid must dedupe so many_to_one holds
    frame = pd.concat([_frame().iloc[[0]], _frame().iloc[[0]]], ignore_index=True)
    frame["card_slug"] = ["a/x", "a/y"]
    out = attach_odds_features(frame, _snapshots())
    assert len(out) == 2
    assert out["odds_level"].tolist() == [0.2, 0.2]
    assert out["card_slug"].tolist() == ["a/x", "a/y"]  # passthrough untouched
