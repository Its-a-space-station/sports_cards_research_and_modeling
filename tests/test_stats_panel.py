import json
from pathlib import Path

import pandas as pd

from cardprice.stats_api import game_log_to_frame
from cardprice.stats_panel import player_stats_series, stats_at_dates

FIXTURES = Path(__file__).parent / "fixtures"
JUDGE = json.loads((FIXTURES / "judge_2022_hitting.json").read_text())


def make_game_logs():
    df = game_log_to_frame(JUDGE["splits"], 592450, "hitting", 2022)
    other = df.iloc[:1].copy()  # one stray row from another player-season
    other["mlb_id"] = 999999
    other["season"] = 2023
    return pd.concat([df, other], ignore_index=True)


def test_stats_at_dates_midseason():
    log = game_log_to_frame(JUDGE["splits"], 592450, "hitting", 2022)
    out = stats_at_dates(log, [pd.Timestamp("2022-04-30"), pd.Timestamp("2022-10-31")], "hitting")
    assert len(out) == 2
    early, full = out.iloc[0], out.iloc[1]
    # Judge's official April 2022 line: 20 G, 6 HR (games Apr 8-29; DNP Apr 30 @ KCR).
    # Verified vs MLB Stats API gameLog + boxscore (gamePk 662797) and Retrosheet via StatMuse.
    assert early["games"] == 20
    assert early["home_runs"] == 6
    # full season equals golden totals
    assert full["games"] == 157
    assert full["home_runs"] == 62
    assert full["avg"] == 0.311


def test_player_stats_series_slices_correctly():
    out = player_stats_series(make_game_logs(), 592450, 2022, [pd.Timestamp("2022-10-31")])
    assert len(out) == 1
    assert out.iloc[0]["home_runs"] == 62
    assert out.iloc[0]["mlb_id"] == 592450
    # wrong player/season -> empty frame, not an error
    empty = player_stats_series(make_game_logs(), 999999, 2022, [pd.Timestamp("2022-10-31")])
    assert len(empty) == 0
