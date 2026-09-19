# tests/test_run_class_gate.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_class_gate import run_cells

from cardprice.class_hold_frame import build_class_hold_frame


def _synth_panel(n_cards=40, seed=5, signal_strength=0.5):
    """5 entry years x n_cards; ret_12m = signal_strength * surprise_ops + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in (2021, 2022, 2023, 2024, 2025):
        for c in range(n_cards):
            surprise = rng.normal(0, 1)
            rows.append(
                {
                    "card_slug": f"set/p-{c}", "grade": "ungraded",
                    "month": pd.Timestamp(f"{year}-06-01"), "entry_price": 10.0,
                    "ret_12m": signal_strength * surprise + rng.normal(0, 0.2),
                    "ret_6m": 0.05, "ret_24m": 0.2, "ret_36m": 0.3,
                    "price_level": rng.normal(2, 0.1), "ret_3m": rng.normal(0, 0.05),
                    "market_ret_3m": rng.normal(0, 0.02), "mlb_id": c,
                    "player_name": f"P{c}", "rookie_year": 2020,
                    "card_type": "bowman_1st_base", "career_stage": "prospect",
                    "awards_to_date": 0, "age": 22.0, "position": "SS",
                    "max_level": "aa", "max_level_rank": 3, "rate_at_max_level": 0.75,
                    "minor_games": 200, "games": 100, "ops": 0.75, "home_runs": 10,
                    "era": None, "k_bb_pct": None, "innings_pitched": None, "whip": None,
                    "surprise_ops": surprise, "surprise_era": None,
                    "marcel_rate": 0.7, "pace_rate": 0.75, "draft_rank": 5.0,
                    "prospect_rank": pd.NA, "rank_change": pd.NA, "fg_draft_fv": pd.NA,
                    "price_visible_breakout": True,
                }
            )
    return pd.DataFrame(rows)


def _events_info():
    events = pd.DataFrame({"mlb_id": [0], "event_date": [pd.Timestamp("2022-01-01")],
                           "event_type": ["debut"], "details": ["x"]})
    info = pd.DataFrame({"mlb_id": [0], "name": ["P0"],
                         "birth_date": [pd.Timestamp("2000-01-01")], "position": ["SS"]})
    return events, info


def test_run_cells_structure_and_registered_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/processed").mkdir(parents=True)
    events, info = _events_info()
    frame = build_class_hold_frame(_synth_panel(), events, info)
    out = run_cells({"hitter": frame}, out_dir="data/processed")
    reg = [c for c in out if c["registered"]]
    assert len(reg) == 1
    cell = reg[0]
    assert cell["grade"] == "ungraded" and cell["horizon"] == 12 and cell["group"] == "hitter"
    gate = cell["gate"]
    assert set(gate) == {"mean_excess", "ci_low", "ci_high", "net_mean", "net_ci_low",
                         "n_years", "verdict"}
    assert gate["verdict"] in {"PASS", "FAIL"}
    assert gate["n_years"] == 3  # 2023-2025 at min_train_years=2
    # planted positive signal -> realized mean should be clearly positive
    assert gate["mean_excess"] > 0
    summary = json.loads((tmp_path / "data/processed/class_gate_summary.json").read_text())
    assert any(c["registered"] for c in summary["cells"])
    picks = pd.read_csv(tmp_path / "data/processed/class_gate_picks.csv")
    assert set(picks["entry_year"]) == {2023, 2024, 2025}
