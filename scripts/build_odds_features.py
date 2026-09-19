# scripts/build_odds_features.py
"""Attach award-odds features to the class hold frame -> class_hold_odds.parquet.

Scope (Task 3 brief): the registered cell's frame — ungraded hitters, all
horizons (6/12/24/36) — plus the ungraded 12m pitcher frame for the pitcher
descriptive cell. The psa_10 descriptive cell is out of scope.

Output shape (chosen, documented): ONE stacked frame at
data/processed/class_hold_odds.parquet (gitignored). One adapter build per
(group, horizon) pair — the adapter keeps only its own target column and drops
that target's NaN rows — with `horizon` and `group` columns added so each row
records which build it came from (same stacking idiom as run_class_gate's
_group_frame; a cell slices on group/horizon and filters its target non-null).
Odds features are attached per build by (mlb_id, entry_month); has_market=0
carries 0.0 odds features ("no listed market ≈ zero priced expectation"),
has_market=1 without a pre-entry snapshot keeps NaN (never filled from later
dates).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cardprice.class_hold_frame import build_class_hold_frame
from cardprice.odds import attach_odds_features

CELLS = [("hitter", h) for h in (6, 12, 24, 36)] + [("pitcher", 12)]
GRADE = "ungraded"
OUT = "data/processed/class_hold_odds.parquet"
ODDS_COLS = ["has_market", "odds_level", "odds_delta_7d", "odds_delta_30d"]


def main() -> None:
    panel = pd.read_parquet("data/processed/panel_class.parquet")
    events = pd.read_parquet("data/processed/events_class.parquet")
    info = pd.read_csv("data/reference/player_info_class.csv", parse_dates=["birth_date"])
    snapshots = pd.read_parquet("data/processed/odds_snapshots.parquet")

    parts = []
    for group, horizon in CELLS:
        frame = build_class_hold_frame(panel, events, info, horizon=horizon, group=group)
        frame = frame[frame["grade"] == GRADE]
        out = attach_odds_features(frame, snapshots)
        out["horizon"] = horizon
        out["group"] = group
        parts.append(out)
        cov = out["has_market"].mean()
        nan = {c: round(float(out[c].isna().mean()), 4) for c in ODDS_COLS}
        print(f"{GRADE} {group} {horizon}m: rows={len(out)} "
              f"has_market={cov:.4f} nan_rates={nan}")

    stacked = pd.concat(parts, ignore_index=True)
    stacked.to_parquet(OUT, index=False)
    dup = stacked.duplicated(["card_slug", "grade", "entry_month", "horizon", "group"]).sum()
    print(f"stacked: {len(stacked)} rows -> {OUT} (dup card/month/horizon rows: {dup})")


if __name__ == "__main__":
    main()
