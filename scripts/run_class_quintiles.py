# scripts/run_class_quintiles.py
"""Quintile-spread evaluation on the ungraded hitter frame (the registered cell).

Three score series: (a) model_score — walk_forward_years(..., all_scores=True)
predictions on CLASS_HOLD_FEATURES, so a test year's scores come only from that
year's walk-forward model; (b) draft_rank; (c) surprise_ops. draft_rank is
signed low = early pick, so a NEGATIVE spread still means early picks win.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cardprice.class_hold_frame import CLASS_HOLD_FEATURES, build_class_hold_frame
from cardprice.quintiles import quintile_spread, spread_summary
from cardprice.walkforward_holds import walk_forward_years

OUT_CSV = "data/processed/class_quintile_spreads.csv"


def model_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """All-rows walk-forward predictions joined back to (entry_month, ret_12m).

    The harness emits exactly [entry_year, card_slug, predicted, realized] in
    (ascending entry_year, frame-order) sequence, so the join is positional;
    it is verified through the identity realized == ret_12m - year-median.
    """
    f = frame.sort_values(["entry_year", "entry_month", "card_slug"], kind="stable")
    f = f.reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        allrows = walk_forward_years(f, CLASS_HOLD_FEATURES, 12, all_scores=True)
    years = sorted(f["entry_year"].unique())
    test = f[f["entry_year"] >= years[2]].reset_index(drop=True)  # min_train_years=2
    assert len(test) == len(allrows), "walk-forward scored a different number of rows"
    scored = test[["entry_month", "ret_12m"]].copy()
    scored["predicted"] = allrows["predicted"].to_numpy()
    bench = scored.groupby(scored["entry_month"].dt.year)["ret_12m"].transform("median")
    assert np.allclose(
        (scored["ret_12m"] - bench).to_numpy(), allrows["realized"].to_numpy()
    ), "positional join to walk-forward scores failed verification"
    return scored


def main() -> None:
    panel = pd.read_parquet("data/processed/panel_class.parquet")
    events = pd.read_parquet("data/processed/events_class.parquet")
    info = pd.read_csv("data/reference/player_info_class.csv", parse_dates=["birth_date"])
    frame = build_class_hold_frame(panel, events, info, horizon=12, group="hitter")
    frame = frame[frame["grade"] == "ungraded"]
    print(f"ungraded hitter 12m frame: {len(frame)} rows")
    series = {"model_score": (model_scores(frame), "predicted")}
    series["draft_rank"] = (frame, "draft_rank")
    series["surprise_ops"] = (frame, "surprise_ops")
    out = []
    for name, (df, score_col) in series.items():
        spreads = quintile_spread(df, score_col)
        spreads = spreads.rename(columns={"entry_month": "month"})
        print(f"{name}: {spread_summary(spreads)}")
        out.append(spreads.assign(series=name))
    pd.concat(out)[["series", "month", "spread", "n_scored"]].to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
