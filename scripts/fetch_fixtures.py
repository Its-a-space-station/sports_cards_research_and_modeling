"""One-off: fetch golden game logs from the live API and commit them as fixtures.

Run from repo root with the venv active:
    python scripts/fetch_fixtures.py
"""

import json
from pathlib import Path

from cardprice.stats_api import fetch_game_log

OUT = Path("tests/fixtures")
CASES = [
    (592450, "hitting", 2022, "judge_2022_hitting.json"),
    (543037, "pitching", 2018, "cole_2018_pitching.json"),
]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for mlb_id, group, season, name in CASES:
        splits = fetch_game_log(mlb_id, group, season)
        assert splits, f"empty game log for {mlb_id}/{group}/{season}"
        (OUT / name).write_text(
            json.dumps({"mlb_id": mlb_id, "season": season, "splits": splits}, indent=2)
        )
        print(f"wrote {OUT / name} ({len(splits)} games)")
