# scripts/run_class_gate.py
"""Class-panel walk-forward gate: registered cell + descriptive cells.

Registered cell (per the plan's pre-registration): ungraded, 12m, hitters,
top-5/year, min_train_years=2, block bootstrap over years, net of 0.14 log
fees, PASS iff net_ci_low > 0. Everything else is descriptive.
"""

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cardprice.class_hold_frame import (
    CLASS_HOLD_FEATURES,
    CLASS_PITCHER_FEATURES,
    build_class_hold_frame,
)
from cardprice.walkforward_holds import gate_evaluation_years, walk_forward_years

# Quoted verbatim from docs/superpowers/plans/2026-09-19-t2-model.md (pre-registration block)
REGISTERED_GATE_TEXT = (
    '> **Registered 2026-09-19:** 12 m horizon, **ungraded** series, **hitters** '
    '(`position != "P"`), feature list `CLASS_HOLD_FEATURES` (Task 1, verbatim), '
    'top-**5** picks per entry year, `min_train_years=2`, block bootstrap over years '
    '(10,000 resamples, seed 42), net of **0.14** log fees, **PASS iff `net_ci_low > 0`**'
    '. All other grade × horizon × group cells are descriptive only; the descriptive '
    'cells do not bear on the verdict (same anti-horizon-shopping rule as P6c).'
)

CELLS = [
    {"grade": "ungraded", "horizon": 12, "group": "hitter", "registered": True},
    {"grade": "ungraded", "horizon": 6, "group": "hitter", "registered": False},
    {"grade": "ungraded", "horizon": 24, "group": "hitter", "registered": False},
    {"grade": "ungraded", "horizon": 36, "group": "hitter", "registered": False},
    {"grade": "psa_10", "horizon": 12, "group": "hitter", "registered": False},
    {"grade": "ungraded", "horizon": 12, "group": "pitcher", "registered": False},
]
FEATURES = {"hitter": CLASS_HOLD_FEATURES, "pitcher": CLASS_PITCHER_FEATURES}
PICKS_CSV = "class_gate_picks.csv"
SUMMARY_JSON = "class_gate_summary.json"


def run_cells(
    frame_by_group: dict[str, pd.DataFrame], out_dir: str = "data/processed"
) -> list[dict]:
    out_path = Path(out_dir)
    print("Pre-registered gate text (verbatim from the plan's pre-registration block):")
    print(REGISTERED_GATE_TEXT)
    results, picks_all = [], []
    for cell in CELLS:
        frame = frame_by_group.get(cell["group"])
        if frame is None:
            continue
        target = f"ret_{cell['horizon']}m"
        sub = frame[frame["grade"] == cell["grade"]]
        if not len(sub) or target not in sub.columns:
            continue
        # per-cell target filter: the harness does NOT drop NaN targets (the adapter
        # does, per horizon), so slice to this cell's non-null target rows here
        sub = sub[sub[target].notna()]
        if not len(sub):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            picks = walk_forward_years(sub, FEATURES[cell["group"]], cell["horizon"], top_k=5)
        if picks.empty:
            print(f"[skipped] {cell['grade']}/{cell['horizon']}m/{cell['group']}: no picks")
            continue
        gate = gate_evaluation_years(picks)
        yearly = {
            int(y): m
            for y, m in picks.groupby("entry_year")["realized"].mean().round(4).items()
        }
        picks = picks.assign(**cell)
        picks_all.append(picks)
        results.append(
            {**cell, "yearly": yearly, "gate": gate, "picks_csv": str(out_path / PICKS_CSV)}
        )
        label = "PRE-REGISTERED GATE" if cell["registered"] else "descriptive"
        print(f"[{label}] {cell['grade']}/{cell['horizon']}m/{cell['group']}: "
              f"yearly={yearly} gate={gate}")
    if picks_all:
        pd.concat(picks_all).to_csv(out_path / PICKS_CSV, index=False)
    (out_path / SUMMARY_JSON).write_text(json.dumps({"cells": results}, indent=2, default=float))
    return results


def _group_frame(
    panel: pd.DataFrame, events: pd.DataFrame, info: pd.DataFrame, group: str
) -> pd.DataFrame:
    """Stack one adapter frame per horizon.

    The adapter keeps only its own horizon's ret column (and drops that target's
    NaN rows), so stacking all horizons gives every ret column; each cell's
    target filter in run_cells then selects exactly the rows built at its horizon.
    """
    parts = [
        build_class_hold_frame(panel, events, info, horizon=h, group=group)
        for h in (6, 12, 24, 36)
    ]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    panel = pd.read_parquet("data/processed/panel_class.parquet")
    events = pd.read_parquet("data/processed/events_class.parquet")
    info = pd.read_csv("data/reference/player_info_class.csv", parse_dates=["birth_date"])
    frames = {
        "hitter": _group_frame(panel, events, info, "hitter"),
        "pitcher": _group_frame(panel, events, info, "pitcher"),
    }
    run_cells(frames)


if __name__ == "__main__":
    main()
