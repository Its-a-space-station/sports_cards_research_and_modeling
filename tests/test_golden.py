import json
from datetime import date
from pathlib import Path

import pytest

from cardprice.season_stats import hitting_to_date, pitching_to_date
from cardprice.stats_api import game_log_to_frame

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name, group):
    payload = json.loads((FIXTURES / name).read_text())
    return game_log_to_frame(payload["splits"], payload["mlb_id"], group, payload["season"])


def test_judge_2022_official_totals():
    df = load_fixture("judge_2022_hitting.json", "hitting")
    out = hitting_to_date(df, date(2022, 10, 31))
    assert out["games"] == 157
    assert out["at_bats"] == 570
    assert out["hits"] == 177
    assert out["home_runs"] == 62
    assert out["rbi"] == 131
    assert out["walks"] == 111
    assert out["strikeouts"] == 175
    assert out["stolen_bases"] == 16
    assert out["avg"] == 0.311
    assert out["slg"] == 0.686


def test_cole_2018_official_totals():
    df = load_fixture("cole_2018_pitching.json", "pitching")
    out = pitching_to_date(df, date(2018, 10, 31))
    assert out["games"] == 32
    assert out["games_started"] == 32
    assert out["wins"] == 15
    assert out["losses"] == 5
    assert out["strikeouts"] == 276
    assert out["outs"] == 601
    assert out["earned_runs"] == 64
    assert out["era"] == pytest.approx(2.88, abs=0.005)
