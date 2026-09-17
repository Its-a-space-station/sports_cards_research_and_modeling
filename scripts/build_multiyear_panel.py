# scripts/build_multiyear_panel.py
"""Assemble the multi-year panel: card x entry-month with lagged predictors + hold outcomes."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.career import (
    awards_to_date,
    career_stage,
    career_to_date,
    minors_pedigree,
)
from cardprice.multiyear import hold_returns, series_month_ends, trailing_features

HITTER_CAREER = {"games": "career_games", "ops": "career_ops", "home_runs": "career_home_runs"}
PITCHER_CAREER = {
    "games": "career_games",
    "era": "career_era",
    "k_bb_pct": "career_k_bb_pct",
    "innings_pitched": "career_innings_pitched",
}


def build_panel(me, outcomes, trailing, game_logs, events, info) -> pd.DataFrame:
    mlb_logs = game_logs[game_logs["level"] == "mlb"]
    minor_logs = game_logs[game_logs["level"] != "mlb"]
    # empty/object-dtyped event_date breaks .dt in awards_to_date; no-op on real data
    events = events.assign(event_date=pd.to_datetime(events["event_date"]))
    tf = trailing.rename(columns={"month": "entry_month"})
    df = outcomes.merge(
        tf, on=["card_slug", "grade", "entry_month"], how="left", validate="one_to_one"
    )
    meta = me[["card_slug", "grade", "month", "mlb_id", "player_name", "rookie_year", "card_type"]]
    df = df.merge(
        meta,
        left_on=["card_slug", "grade", "entry_month"],
        right_on=["card_slug", "grade", "month"],
        how="left",
        validate="one_to_one",
    ).drop(columns="month")
    market = tf.groupby("entry_month")["ret_3m"].median().rename("market_ret_3m").reset_index()
    df = df.merge(market, on="entry_month", how="left", validate="many_to_one")
    debuts = events[events["event_type"] == "debut"][["mlb_id", "event_date"]].rename(
        columns={"event_date": "debut_date"}
    )

    rows = []
    for r in df.itertuples(index=False):
        lag = (r.entry_month - pd.Timedelta(days=1)).date()
        line = career_to_date(mlb_logs, int(r.mlb_id), lag)
        ped = minors_pedigree(minor_logs, int(r.mlb_id), lag)
        row = r._asdict()
        for src, dst in (HITTER_CAREER | PITCHER_CAREER).items():
            row[dst] = line.get(src) if line else None
        row.update(
            max_level=ped["max_level"],
            max_level_rank=ped["max_level_rank"],
            rate_at_max_level=ped["rate_at_max_level"],
            minor_games=ped["minor_games"],
            career_stage=career_stage(mlb_logs, int(r.mlb_id), r.entry_month),
            awards_to_date=awards_to_date(events, int(r.mlb_id), lag),
        )
        p_info = info[info["mlb_id"] == r.mlb_id]
        if len(p_info):
            birth = p_info.iloc[0]["birth_date"]
            row["age"] = round((r.entry_month - birth).days / 365.25, 2)
            row["position"] = p_info.iloc[0]["position"]
            d = debuts[debuts["mlb_id"] == r.mlb_id]
            if len(d) and d.iloc[0]["debut_date"].date() <= lag:
                row["age_at_debut"] = round((d.iloc[0]["debut_date"] - birth).days / 365.25, 2)
            else:
                row["age_at_debut"] = row["age"]  # pre-debut: age-now semantics (spec §6)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    chart = pd.read_parquet("data/processed/universe_chart_monthly.parquet")
    game_logs = pd.read_parquet("data/processed/game_logs_universe.parquet")
    events = pd.read_parquet("data/processed/events_universe.parquet")
    info = pd.read_csv("data/reference/player_info_universe.csv", parse_dates=["birth_date"])
    cards = pd.read_csv("data/reference/cards_modeling.csv")

    me = series_month_ends(chart)
    # restrict to the liquidity-selected (card, series) universe
    sel = cards[cards["grade"].isin(["ungraded", "psa_10"])][
        ["card_slug", "grade"]
    ].drop_duplicates()
    me = me.merge(sel, on=["card_slug", "grade"], how="inner", validate="many_to_one")

    panel = build_panel(me, hold_returns(me), trailing_features(me), game_logs, events, info)
    panel.to_parquet("data/processed/panel_multiyear.parquet", index=False)
    print(f"panel_multiyear: {len(panel)} rows, {panel['card_slug'].nunique()} cards")
    print("\nrows per horizon (non-NaN):")
    for h in (6, 12, 24, 36):
        print(f"  ret_{h}m: {panel[f'ret_{h}m'].notna().sum()}")
    print("\nrows per career_stage:")
    print(panel["career_stage"].value_counts())
    print("\nrows per grade:")
    print(panel["grade"].value_counts())


if __name__ == "__main__":
    main()
