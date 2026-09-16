from pathlib import Path

import pandas as pd
import pytest

from cardprice.events import events_from_game_logs


def make_logs():
    rows = [
        # player 1: debut Apr 10 2022; 3-HR game Jun 5; 4-hit game Jul 1 (no HR);
        #           doubleheader Aug 10 with 2 HR + 1 HR (no single-game 3 -> no event)
        (1, "hitting", 2022, "2022-04-10", {"homeRuns": 0, "hits": 1, "strikeOuts": 1}),
        (1, "hitting", 2022, "2022-06-05", {"homeRuns": 3, "hits": 3, "strikeOuts": 0}),
        (1, "hitting", 2022, "2022-07-01", {"homeRuns": 0, "hits": 4, "strikeOuts": 1}),
        (1, "hitting", 2022, "2022-08-10", {"homeRuns": 2, "hits": 3, "strikeOuts": 0}),
        (1, "hitting", 2022, "2022-08-10", {"homeRuns": 1, "hits": 1, "strikeOuts": 1}),
        # player 2 (pitcher): debut May 1; 10-K game Jun 15
        (2, "pitching", 2022, "2022-05-01", {"homeRuns": 0, "hits": 0, "strikeOuts": 5}),
        (2, "pitching", 2022, "2022-06-15", {"homeRuns": 0, "hits": 0, "strikeOuts": 10}),
    ]
    df = pd.DataFrame(
        [(m, g, s, d, *st.values()) for m, g, s, d, st in rows],
        columns=["mlb_id", "group", "season", "date", "homeRuns", "hits", "strikeOuts"],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_registry_rules():
    ev = events_from_game_logs(make_logs())
    p1 = ev[ev["mlb_id"] == 1].sort_values("event_date")
    assert p1["event_type"].tolist() == ["debut", "three_hr_game", "four_hit_game"]
    assert p1["event_date"].tolist() == list(
        pd.to_datetime(["2022-04-10", "2022-06-05", "2022-07-01"])
    )
    p2 = ev[ev["mlb_id"] == 2]
    assert set(p2["event_type"]) == {"debut", "ten_k_game"}


def test_no_event_from_doubleheader_split():
    ev = events_from_game_logs(make_logs())
    # 2 HR + 1 HR across a doubleheader is NOT a three-HR game
    assert not (ev["event_date"] == pd.Timestamp("2022-08-10")).any()


@pytest.mark.skipif(
    not Path("data/processed/game_logs.parquet").exists(),
    reason="game_logs.parquet not available",
)
def test_golden_real_debuts():
    logs = pd.read_parquet("data/processed/game_logs.parquet")
    ev = events_from_game_logs(logs)
    henderson = ev[(ev["mlb_id"] == 683002) & (ev["event_type"] == "debut")].iloc[0]
    assert henderson["event_date"] == pd.Timestamp("2022-08-31")  # Henderson's MLB debut
    skenes = ev[(ev["mlb_id"] == 694973) & (ev["event_type"] == "debut")].iloc[0]
    assert skenes["event_date"] == pd.Timestamp("2024-05-11")  # Skenes' MLB debut
