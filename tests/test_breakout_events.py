import numpy as np
import pandas as pd

from cardprice.breakout_events import (
    dedupe_overlaps,
    detect_breakouts,
    hitter_game_score,
    merge_debuts,
    pitcher_game_score,
)


def _hitting_row(date, tb, bb=0, sb=0, pa=4, mlb_id=1, level="mlb"):
    return {
        "mlb_id": mlb_id, "group": "hitting", "season": 2024, "date": pd.Timestamp(date),
        "level": level, "totalBases": tb, "baseOnBalls": bb, "stolenBases": sb,
        "plateAppearances": pa,
    }


def _pitching_row(date, outs, k, h, er, r, bb, gs=1, mlb_id=2, level="mlb"):
    return {
        "mlb_id": mlb_id, "group": "pitching", "season": 2024, "date": pd.Timestamp(date),
        "level": level, "outs": outs, "strikeOuts": k, "hits": h, "earnedRuns": er,
        "runs": r, "baseOnBalls": bb, "gamesStarted": gs,
    }


def test_hitter_game_score_sums_counting_stats():
    row = pd.Series({"totalBases": 7, "baseOnBalls": 2, "stolenBases": 1})
    assert hitter_game_score(row) == 10.0


def test_hitter_game_score_nan_is_zero():
    row = pd.Series({"totalBases": 4, "baseOnBalls": np.nan, "stolenBases": np.nan})
    assert hitter_game_score(row) == 4.0


def test_pitcher_game_score_bill_james_golden():
    # 7 IP (21 outs), 9 K, 4 H, 1 BB, 2 R (2 ER).
    # 50 + 21 outs + 2*3 (innings completed after the 4th: (21-12)//3) + 9 K
    # - 2*4 H - 4*2 ER - 2*0 unearned - 1 BB = 50+21+6+9-8-8-0-1 = 69
    row = pd.Series({"outs": 21, "strikeOuts": 9, "hits": 4, "earnedRuns": 2,
                     "runs": 2, "baseOnBalls": 1})
    assert pitcher_game_score(row) == 69.0


def test_pitcher_game_score_unearned_runs_cost_two():
    # same as above but one run unearned: -4*1 ER -2*1 unearned = -6 instead of -8 -> 71
    row = pd.Series({"outs": 21, "strikeOuts": 9, "hits": 4, "earnedRuns": 1,
                     "runs": 2, "baseOnBalls": 1})
    assert pitcher_game_score(row) == 71.0


def test_detect_breakouts_finds_planted_z():
    rows = []
    # 30 quiet games alternating tb 1,2 (mean 1.5, sd ~0.51), then a 14-TB monster
    dates = pd.date_range("2024-04-01", periods=31, freq="D")
    for i, d in enumerate(dates[:-1]):
        rows.append(_hitting_row(d, tb=1 + i % 2))
    rows.append(_hitting_row(dates[-1], tb=14))
    out = detect_breakouts(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0]["event_date"] == dates[-1]
    assert out.iloc[0]["z"] >= 2.5
    assert out.iloc[0]["group"] == "hitting"
    assert out.iloc[0]["baseline_n"] == 30


def test_detect_breakouts_respects_min_baseline():
    # only 10 prior games: detection must still work (baseline = trailing UP TO
    # 30 games); alternating 1,2 keeps baseline sd > 0
    rows = [
        _hitting_row(d, tb=1 + i % 2)
        for i, d in enumerate(pd.date_range("2024-04-01", periods=10, freq="D"))
    ]
    rows.append(_hitting_row(pd.Timestamp("2024-04-11"), tb=12))
    out = detect_breakouts(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0]["baseline_n"] == 10


def test_detect_breakouts_skips_relief_appearances_as_events_and_baselines():
    rows = []
    dates = pd.date_range("2024-04-01", periods=8, freq="7D")
    for d in dates[:-1]:  # 7 mediocre starts
        rows.append(_pitching_row(d, outs=15, k=3, h=6, er=3, r=3, bb=2))
    # a relief gem must NOT be an event (gamesStarted=0)
    rows.append(_pitching_row(dates[-1], outs=9, k=8, h=0, er=0, r=0, bb=0, gs=0))
    out = detect_breakouts(pd.DataFrame(rows))
    assert len(out) == 0


def test_detect_breakouts_pitcher_start_detected():
    rows = []
    dates = pd.date_range("2024-04-01", periods=8, freq="7D")
    # 7 mediocre starts alternating K=3/4 (game scores 44/45, sd > 0)
    for i, d in enumerate(dates[:-1]):
        rows.append(_pitching_row(d, outs=15, k=3 + i % 2, h=6, er=3, r=3, bb=2))
    # monster: 50 + 27 outs + 2*5 ((27-12)//3) + 14 K - 2*2 H - 0 - 0 - 1 BB = 96
    rows.append(_pitching_row(dates[-1], outs=27, k=14, h=2, er=0, r=0, bb=1))
    out = detect_breakouts(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0]["group"] == "pitching"
    assert out.iloc[0]["z"] >= 2.5


def test_detect_breakouts_constant_baseline_skipped():
    # sd == 0 baseline must not produce events (division guard)
    rows = [_hitting_row(d, tb=2) for d in pd.date_range("2024-04-01", periods=31, freq="D")]
    out = detect_breakouts(pd.DataFrame(rows))
    assert len(out) == 0


def test_merge_debuts_appends_debut_rows():
    breakouts = pd.DataFrame(
        {"mlb_id": [1], "event_date": pd.to_datetime(["2024-05-01"]),
         "event_type": ["breakout"], "group": ["hitting"], "level": ["mlb"],
         "score": [12.0], "z": [3.1], "baseline_n": [30]}
    )
    events = pd.DataFrame(
        {"mlb_id": [1, 2], "event_date": pd.to_datetime(["2023-04-02", "2024-03-30"]),
         "event_type": ["debut", "award_win"], "details": ["x", "y"]}
    )
    out = merge_debuts(breakouts, events)
    assert len(out) == 2
    debut = out[out["event_type"] == "debut"].iloc[0]
    assert debut["mlb_id"] == 1 and pd.isna(debut["z"])
    # award_win rows are NOT events for this study
    assert "award_win" not in set(out["event_type"])


def test_dedupe_overlaps_keeps_first_drops_within_window():
    events = pd.DataFrame(
        {"mlb_id": [1, 1, 1, 2],
         "event_date": pd.to_datetime(
             ["2024-05-01", "2024-05-10", "2024-06-01", "2024-05-02"]),
         "event_type": ["breakout"] * 4}
    )
    out = dedupe_overlaps(events, window_days=14)
    kept = out.sort_values(["mlb_id", "event_date"])
    assert list(kept["event_date"]) == [
        pd.Timestamp("2024-05-01"),  # 05-10 is +9d -> dropped
        pd.Timestamp("2024-06-01"),  # +31d from kept 05-01 -> kept
        pd.Timestamp("2024-05-02"),  # different player
    ]
