# tests/test_collect_class_events.py
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from collect_class_events import build_events


def test_build_events_debuts_and_awards():
    logs = pd.DataFrame(
        {
            "mlb_id": [1, 1, 1, 2],
            "level": ["aaa", "mlb", "mlb", "mlb"],
            "date": pd.to_datetime(["2019-05-01", "2019-06-25", "2019-07-01", "2023-04-02"]),
        }
    )
    awards = pd.DataFrame(
        {
            "mlb_id": [1, 3],
            "event_date": pd.to_datetime(["2023-11-16", "2023-11-16"]),
            "event_type": ["award_win", "award_win"],
            "details": ["NL Rookie of the Year 2023"] * 2,
        }
    )
    out = build_events(logs, awards)
    debuts = out[out["event_type"] == "debut"]
    assert len(debuts) == 2
    # first MLB game, not the earlier AAA game
    assert debuts[debuts["mlb_id"] == 1]["event_date"].iloc[0] == pd.Timestamp("2019-06-25")
    wins = out[out["event_type"] == "award_win"]
    assert len(wins) == 2  # both kept (award frame is already class-filtered upstream)
    # events_universe.parquet column convention
    assert list(out.columns) == ["mlb_id", "event_date", "event_type", "details"]
