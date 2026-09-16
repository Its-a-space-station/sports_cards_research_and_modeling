"""Panel parquets -> standardized modeling frames. Horizon rules per Plan 3 review."""

import pandas as pd

HITTER_FEATURES = [
    "ops",
    "slg",
    "home_runs",
    "strikeouts",
    "games",
    "form_games",
    "form_ops_delta",
    "age",
    "playoff",
    "years_since_rookie",
]

POSITION_MAP = {
    "P": "P",
    "C": "C",
    "1B": "IF",
    "2B": "IF",
    "3B": "IF",
    "SS": "IF",
    "OF": "OF",
    "LF": "OF",
    "CF": "OF",
    "RF": "OF",
    "DH": "OF",
}


def build_modeling_frame(panel: pd.DataFrame, grain: str) -> pd.DataFrame:
    df = panel.copy()
    if grain == "monthly":
        df = df[(df["months_since_prev"].isna()) | (df["months_since_prev"] <= 2)]
    elif grain == "weekly":
        df = df[(df["days_since_prev"].isna()) | (df["days_since_prev"] <= 21)]
    else:
        raise ValueError(f"unknown grain: {grain}")
    df = df[df["excess_ret"].notna()]
    df["years_since_rookie"] = df["stats_season"] - df["rookie_year"]
    df["position_group"] = df["position"].map(POSITION_MAP)
    return df.reset_index(drop=True)


def standardize(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = pd.DataFrame({"mean": X.mean(), "sd": X.std(ddof=0)})
    Z = (X - params["mean"]) / params["sd"].replace(0, 1)
    Z.loc[:, params["sd"] == 0] = 0.0
    return Z, params


def hitter_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    hit = frame[frame["position_group"] != "P"]
    X = hit[HITTER_FEATURES].copy()
    X = X.fillna(X.median())
    Z, _ = standardize(X)
    y = hit["excess_ret"]
    groups = hit[["mlb_id", "position_group", "stats_season"]].reset_index(drop=True)
    return Z.reset_index(drop=True), y.reset_index(drop=True), groups
