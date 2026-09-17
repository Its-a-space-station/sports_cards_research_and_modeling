# tests/test_collect_universe_events.py
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from collect_universe_events import build_universe_events


def test_debut_from_mlb_level_only():
    game_logs = pd.DataFrame(
        [
            {"mlb_id": 1, "group": "hitting", "season": 2017, "date": "2017-05-01", "level": "a"},
            {"mlb_id": 1, "group": "hitting", "season": 2019, "date": "2019-04-01", "level": "mlb"},
            {"mlb_id": 1, "group": "hitting", "season": 2019, "date": "2019-04-20", "level": "mlb"},
        ]
    )
    game_logs["date"] = pd.to_datetime(game_logs["date"])
    awards = pd.DataFrame(
        [
            {
                "mlb_id": 1,
                "event_date": pd.Timestamp("2019-11-11"),
                "event_type": "award_win",
                "details": "AL ROY 2019",
            }
        ]
    )
    out = build_universe_events(game_logs, awards)
    debut = out[out["event_type"] == "debut"].iloc[0]
    assert debut["event_date"] == pd.Timestamp("2019-04-01")  # NOT the 2017 minor-league game
    assert set(out["event_type"]) == {"debut", "award_win"}
    assert list(out.columns) == ["mlb_id", "event_date", "event_type", "details"]
