"""Run the full modeling pass on the real panels and print everything the report needs."""

import pandas as pd

from cardprice.features import HITTER_FEATURES, build_modeling_frame, hitter_matrix
from cardprice.model_consistency import (
    player_random_effects,
    position_interaction_test,
    season_consistency,
)
from cardprice.model_gbm import gbm_shap_importance
from cardprice.model_lasso import lasso_path_summary, stability_selection
from cardprice.walkforward import gate_evaluation, walk_forward

if __name__ == "__main__":
    monthly = pd.read_parquet("data/processed/panel_monthly.parquet")
    weekly = pd.read_parquet("data/processed/panel_weekly.parquet")

    for grain, panel in [("monthly", monthly), ("weekly", weekly)]:
        frame = build_modeling_frame(panel, grain)
        hit = frame[frame["position_group"] != "P"]
        X, y, groups = hitter_matrix(frame)
        print(f"\n=== {grain}: {len(frame)} rows after filters ({len(hit)} hitter rows) ===")
        print("hitters by season:", hit.groupby("stats_season").size().to_dict())

        freq = stability_selection(X, y)
        print("\nstability selection:", freq.round(2).to_dict())
        print("\nlasso path:\n", lasso_path_summary(X, y))
        print("\ngbm shap:\n", gbm_shap_importance(X, y))

        for stat in ["ops", "form_ops_delta"]:
            print(f"\nseason consistency ({stat}):\n", season_consistency(hit, stat))
            print(f"position interaction ({stat}):", position_interaction_test(hit, stat))
        print("\nplayer random effects (ops):", player_random_effects(hit, "ops"))

        if grain == "monthly":
            picks = walk_forward(hit, HITTER_FEATURES)
            print("\nwalk-forward gate:", gate_evaluation(picks, hit))
