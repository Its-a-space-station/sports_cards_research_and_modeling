# tests/test_collect_odds.py
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from collect_odds import map_players, plan_chunks


def test_plan_chunks_fourteen_day_windows():
    start, end = pd.Timestamp("2025-03-01"), pd.Timestamp("2025-06-15")
    chunks = plan_chunks(start, end, days=14)
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    assert all((b - a).days <= 14 for a, b in chunks)
    # contiguous: next chunk starts where the previous ended
    assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))


def test_map_players_exact_norm_only():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from collect_odds import norm_name  # re-exported for tests

    assert norm_name("Jasson Domínguez") == norm_name("Jasson Dominguez")  # ruff F401: use it
    info = pd.DataFrame(
        {"mlb_id": [1, 2], "name": ["Aaron Judge", "Jasson Domínguez"],
         "birth_date": pd.to_datetime(["1992-04-26", "2003-02-07"]), "position": ["OF", "OF"]}
    )
    outcomes = pd.DataFrame(
        {"market_id": ["a", "b", "c"],
         "player_name": ["Aaron Judge", "Jasson Dominguez", "Shohei Ohtani"]}
    )
    mapped, audit = map_players(outcomes, info)
    assert mapped.loc[mapped["market_id"] == "a", "mlb_id"].iloc[0] == 1
    assert mapped.loc[mapped["market_id"] == "b", "mlb_id"].iloc[0] == 2  # accent-folded
    assert audit["player_name"].tolist() == ["Shohei Ohtani"]  # not in info -> audit, never guessed
