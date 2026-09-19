# scripts/build_class_panel.py
"""Class panel: card x entry-month with surprise features, expectations, era flag.

No liquidity SELECTION here (thresholds are T2-model's); liquidity stats are
produced for all cards. rookie_year column carries class_year (panel convention).
Join semantics (left joins, validate=, lag = entry - 1 day, market regime from
the trailing frame) mirror scripts/build_multiyear_panel.py exactly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cardprice.career import (
    awards_to_date,
    career_stage,
    career_to_date,
    minors_pedigree,
    season_year,
)
from cardprice.expectations import marcel_projection
from cardprice.multiyear import hold_returns, series_month_ends, trailing_features

EXPECTATIONS_DTYPES = {
    "player_name": "object",
    "mlb_id": "Int64",
    "season": "int64",
    "source": "object",
    "rank": "Int64",
    "fv": "Float64",
    "as_of": "datetime64[ns]",
}


def season_line(logs: pd.DataFrame, group: str) -> tuple[float, float]:
    """(rate, playing_time) from summed counting stats; degenerate -> (NaN, 0)."""
    if group == "hitting":
        h = logs["hits"].sum()
        bb = logs["baseOnBalls"].sum()
        hbp = logs["hitByPitch"].sum()
        ab = logs["atBats"].sum()
        sf = logs["sacFlies"].sum()
        tb = logs["totalBases"].sum()
        denom = ab + bb + hbp + sf
        if ab == 0 or denom == 0:
            return float("nan"), 0.0
        obp = (h + bb + hbp) / denom
        return float(obp + tb / ab), float(logs["plateAppearances"].sum())
    er = logs["earnedRuns"].sum()
    ip = logs["outs"].sum() / 3.0
    if ip == 0:
        return float("nan"), 0.0
    return float(9.0 * er / ip), float(logs["battersFaced"].sum())


def league_means(game_logs: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """Per (season, group): playing-time-weighted mean rate over all players,
    from games strictly before `before` — the Marcel regression target must not
    see games played after the entry month (pace-style truncation league-wide)."""
    rows = []
    for (season, group), sub in game_logs[game_logs["date"] < before].groupby(["season", "group"]):
        rate, pt = 0.0, 0.0
        for _, g in sub.groupby("mlb_id"):
            r, p = season_line(g, group)
            if not np.isnan(r) and p > 0:
                rate += r * p
                pt += p
        rows.append({"season": season, "group": group,
                     "league_rate": rate / pt if pt else np.nan})
    return pd.DataFrame(rows, columns=["season", "group", "league_rate"])


def pace_line(
    logs: pd.DataFrame, group: str, season: int, before: pd.Timestamp
) -> tuple[float, float] | None:
    """season_line over rows of `season` strictly before `before`; None when empty."""
    sub = logs[(logs["season"] == season) & (logs["date"] < before)]
    if not len(sub):
        return None
    return season_line(sub, group)


def build_panel(me, outcomes, trailing, game_logs, expectations, info, events) -> pd.DataFrame:
    """Assemble card x entry-month rows with surprise/expectation/era features."""
    mlb_logs = game_logs[game_logs["level"] == "mlb"]
    minor_logs = game_logs[game_logs["level"] != "mlb"]
    # empty/object-dtyped event_date breaks .dt in awards_to_date; no-op on real data
    events = events.assign(event_date=pd.to_datetime(events["event_date"]))
    exp = expectations.copy()
    exp["as_of"] = pd.to_datetime(exp["as_of"])
    # NA mlb_id (unmapped names) can never match a panel player and would put
    # pd.NA in the boolean masks below
    exp = exp[exp["mlb_id"].notna()]

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

    group_of = game_logs.groupby("mlb_id")["group"].first().to_dict()
    first_pro = game_logs.groupby("mlb_id")["season"].min().to_dict()
    # league tables truncated at each entry month, computed once per month and
    # reused across players (never recomputed per player-row)
    means_by_month = {m: league_means(mlb_logs, m) for m in df["entry_month"].unique()}
    # per-player frames, grouped once: the row loop would otherwise full-frame
    # scan by mlb_id several times per row (~94k rows x ~700k log rows)
    mlb_by_id = {int(k): g for k, g in mlb_logs.groupby("mlb_id")}
    minors_by_id = {int(k): g for k, g in minor_logs.groupby("mlb_id")}
    empty_mlb = mlb_logs.iloc[0:0]
    empty_minors = minor_logs.iloc[0:0]
    info_ix = info.set_index("mlb_id")
    births = info_ix["birth_date"].to_dict()
    positions = info_ix["position"].to_dict()

    rows = []
    for r in df.itertuples(index=False):
        mid = int(r.mlb_id)
        player_mlb = mlb_by_id.get(mid, empty_mlb)
        player_minors = minors_by_id.get(mid, empty_minors)
        group = group_of.get(mid, "hitting")
        season = season_year(r.entry_month)
        lag = (r.entry_month - pd.Timedelta(days=1)).date()
        row = r._asdict()
        row["month"] = r.entry_month
        del row["entry_month"]
        # career helpers re-filter by mlb_id internally — a no-op on these
        # per-player frames, so results are identical to full-frame calls
        row.update(career_to_date(player_mlb, mid, lag))
        row.update(minors_pedigree(player_minors, mid, lag))
        row.update(
            career_stage=career_stage(player_mlb, mid, r.entry_month),
            awards_to_date=awards_to_date(events, mid, lag),
            price_visible_breakout=first_pro.get(mid, 9999) >= 2020,
        )
        birth = births.get(mid)
        row["age"] = (
            (r.entry_month - birth).days / 365.25
            if birth is not None and pd.notna(birth)
            else np.nan
        )
        row["position"] = positions.get(mid)
        # expectations for this season, as_of strictly before entry month
        e = exp[
            (exp["mlb_id"] == mid) & (exp["season"] == season) & (exp["as_of"] < r.entry_month)
        ]
        pipe = e[e["source"] == "pipeline"]
        row["prospect_rank"] = pipe["rank"].min() if len(pipe) else pd.NA
        prev = exp[
            (exp["mlb_id"] == mid)
            & (exp["season"] == season - 1)
            & (exp["source"] == "pipeline")
            & (exp["as_of"] < r.entry_month)
        ]
        prev_min = prev["rank"].min() if len(prev) else pd.NA
        row["rank_change"] = (
            int(row["prospect_rank"]) - int(prev_min)
            if not pd.isna(row["prospect_rank"]) and not pd.isna(prev_min)
            else pd.NA
        )
        fg = e[e["source"] == "fg_draft_board"]
        row["fg_draft_fv"] = fg["fv"].max() if len(fg) else pd.NA
        # draft_rank: the player's own draft-year rank (mlb_draft source), keyed
        # to the player's CLASS year (rookie_year), not the entry's season — a
        # class-2019 draftee keeps the rank on 2023+ entries. Separate feature
        # from the Pipeline pre-debut expectations term (that join is season-correct)
        draft = exp[
            (exp["mlb_id"] == mid)
            & (exp["season"] == r.rookie_year)
            & (exp["source"] == "mlb_draft")
            & (exp["as_of"] < r.entry_month)
        ]
        row["draft_rank"] = draft["rank"].min() if len(draft) else pd.NA
        # Marcel projection for `season` from the 3 prior MLB seasons (<= Y-1)
        prior = []
        for y in range(season - 1, season - 4, -1):
            sub = player_mlb[player_mlb["season"] == y]
            if len(sub):
                rate, pt = season_line(sub, group)
                if not np.isnan(rate) and pt > 0:
                    prior.append((rate, pt))
        lg = means_by_month[r.entry_month]
        lg = lg[(lg["season"] == season) & (lg["group"] == group)]
        league_rate = lg["league_rate"].iloc[0] if len(lg) else np.nan
        marcel = marcel_projection(prior, league_rate) if not np.isnan(league_rate) else None
        pace = pace_line(
            player_mlb[player_mlb["season"] == season],
            group,
            season,
            r.entry_month,
        )
        row["marcel_rate"] = marcel
        row["pace_rate"] = pace[0] if pace else np.nan
        surprise = (pace[0] - marcel) if pace and marcel is not None else np.nan
        row["surprise_ops"] = surprise if group == "hitting" else np.nan
        row["surprise_era"] = surprise if group == "pitching" else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    chart = pd.read_parquet("data/processed/class_chart_monthly.parquet")
    # chart parquets carry `key:*` uncalibrated rows — every consumer re-filters
    chart = chart[~chart["grade"].astype(str).str.startswith("key:")]
    me = series_month_ends(chart)
    sales = pd.read_parquet("data/processed/class_sales.parquet")
    game_logs = pd.read_parquet("data/processed/game_logs_class.parquet")
    events = pd.read_parquet("data/processed/events_class.parquet")
    exp_path = Path("data/processed/expectations.parquet")
    if exp_path.exists():
        expectations = pd.read_parquet(exp_path)
    else:  # collection rate-limited (Task 6): build with NaN expectation columns
        expectations = pd.DataFrame(
            {c: pd.Series(dtype=t) for c, t in EXPECTATIONS_DTYPES.items()}
        )
    info = pd.read_csv("data/reference/player_info_class.csv", parse_dates=["birth_date"])

    panel = build_panel(
        me, hold_returns(me), trailing_features(me), game_logs, expectations, info, events
    )
    panel.to_parquet("data/processed/panel_class.parquet", index=False)

    weeks = (sales["sale_date"].max() - sales["sale_date"].min()).days / 7.0
    liq = (
        sales.groupby(["card_slug", "grade"])
        .agg(n_sales=("price", "size"))
        .reset_index()
        .assign(sales_per_week=lambda d: d["n_sales"] / max(weeks, 1.0))
    )
    pts = me.groupby(["card_slug", "grade"]).size().rename("n_chart_points").reset_index()
    liq = liq.merge(pts, on=["card_slug", "grade"], how="outer")
    liq.to_csv("data/processed/class_liquidity.csv", index=False)
    print("panel rows:", len(panel), "| cards:", panel["card_slug"].nunique(),
          "| players:", panel["mlb_id"].nunique())
    print("surprise coverage:", panel["surprise_ops"].notna().mean(),
          panel["surprise_era"].notna().mean())


if __name__ == "__main__":
    main()
