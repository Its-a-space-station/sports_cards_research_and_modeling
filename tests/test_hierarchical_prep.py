# tests/test_hierarchical_prep.py
import numpy as np
import pandas as pd
import pytest

from cardprice.hierarchical import build_model_frame


def _panel(n_per_class=20, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    stages = ["prospect", "rookie_year", "sophomore", "established"]
    for cy in (2019, 2020, 2021):
        for i in range(n_per_class):
            pos = "P" if i % 5 == 0 else "SS"
            so = rng.normal(0, 1) if pos != "P" else None
            se = rng.normal(0, 1) if pos == "P" else None
            rows.append(
                {"month": pd.Timestamp("2024-06-01"), "grade": "ungraded",
                 "ret_12m": rng.normal(0.1, 0.3), "mlb_id": cy * 100 + i,
                 "player_name": f"P{i}", "rookie_year": cy, "position": pos,
                 "career_stage": stages[i % 4], "surprise_ops": so,
                 "surprise_era": se, "market_ret_3m": rng.normal(0, 0.05),
                 "price_level": rng.normal(2, 0.3), "entry_price": 10.0,
                 "card_slug": f"s/p-{cy}-{i}", "card_type": "bowman_1st_base",
                 "awards_to_date": 0, "age": 22.0}
            )
    return pd.DataFrame(rows)


def test_hitters_frame_schema_scaling_and_sign():
    frame, params = build_model_frame(_panel(), "hitter")
    assert set(frame.columns) == {"y", "surprise", "career_stage", "market_ret_3m",
                                  "price_level", "class_year", "class_position", "mlb_id"}
    assert (frame["position" if "position" in frame else "class_position"].str.contains("SS")).all()
    # z-scored predictors
    assert frame["surprise"].mean() == pytest.approx(0, abs=1e-9)
    assert frame["surprise"].std(ddof=0) == pytest.approx(1, abs=1e-9)
    # y unscaled
    assert abs(frame["y"].mean() - 0.1) < 0.2
    # class keys
    assert set(frame["class_year"]) == {"2019", "2020", "2021"}
    assert frame["class_position"].str.contains("2020_SS").any()
    # scaling params present for back-transformation
    assert "surprise" in params and len(params["surprise"]) == 2


def test_pitchers_sign_flip_and_filter():
    frame, _ = build_model_frame(_panel(), "pitcher")
    raw = _panel()
    raw_p = raw[raw["position"] == "P"].reset_index(drop=True)
    assert len(frame) == len(raw_p)
    assert (frame["class_position"].str.contains("_P")).all()
    # surprise == -surprise_era (z-scored): verify the whole column by hand
    eras = raw_p["surprise_era"].astype(float)
    expected = ((-eras) - (-eras).mean()) / (-eras).std(ddof=0)
    got = frame.sort_values("mlb_id")["surprise"].to_numpy()
    assert got == pytest.approx(expected.to_numpy()[raw_p["mlb_id"].argsort()], abs=1e-9)


def test_target_and_grade_filters():
    panel = _panel()
    # rows 1 and 2 are both hitters (i % 5 != 0); row 0 is a pitcher, so
    # mutating it would leave the grade filter unexercised within the group
    panel.loc[1, "grade"] = "psa_10"
    panel.loc[2, "ret_12m"] = None
    frame, _ = build_model_frame(panel, "hitter")
    assert len(frame) == len(panel[panel["position"] != "P"]) - 2
