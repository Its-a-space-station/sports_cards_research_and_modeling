# tests/test_collect_universe_stats.py
import sys
from pathlib import Path

import pandas as pd

from cardprice import stats_api, storage

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from collect_universe_stats import collect_universe_stats


def test_collect_writes_levels_and_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)

    def fake_game_log(mlb_id, group, season, sport_id=None):
        return [{"date": f"{season}-07-01", "stat": {"gamesPlayed": 1, "homeRuns": 2}}]

    def fake_minors(mlb_id, group, season):
        return (
            {"aaa": [{"date": f"{season}-06-01", "stat": {"gamesPlayed": 1, "homeRuns": 1}}]}
            if season == 2021
            else {}
        )

    monkeypatch.setattr(stats_api, "fetch_game_log", fake_game_log)
    monkeypatch.setattr(stats_api, "fetch_minor_league_logs", fake_minors)
    import collect_universe_stats as collect_mod  # aliased: bare `import` would shadow the function above

    monkeypatch.setattr(collect_mod, "fetch_game_log", fake_game_log)
    monkeypatch.setattr(collect_mod, "fetch_minor_league_logs", fake_minors)

    players = pd.DataFrame([{"mlb_id": 1, "name": "Test Player", "role": "hitter"}])
    df = collect_universe_stats(players, mlb_seasons=[2021, 2022], minor_seasons=[2021], sleep_s=0)
    assert set(df["level"]) == {"mlb", "aaa"}
    assert len(df) == 3
    assert storage.load_latest("stats", "1_hitting_2021")["splits"]
    assert storage.load_latest("stats", "1_hitting_2021_aaa")["splits"]
