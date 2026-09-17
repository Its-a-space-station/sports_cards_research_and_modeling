"""Price incorporation lag: sale-level event study at daily grain.

Each sale is normalized by its card's EVENT-ANCHORED baseline (median of
same-grade-class sales strictly before the event) so post-event repricing is
visible instead of absorbed by a moving baseline. Market index + adjustment
live in part 2 (Task 3); pooled curves + cluster bootstrap, the lag/half-life
estimators, and per-event adjustment classes live in part 3 (Task 4).
"""

import numpy as np
import pandas as pd

from cardprice.career import career_stage

GRADE_CLASSES = {"ungraded": None, "psa_10": "psa_10"}

WINDOW_COLUMNS = [
    "event_id", "mlb_id", "card_slug", "event_date", "sale_date", "t_days", "rel_price",
]


def _grade_mask(sales: pd.DataFrame, grade_class: str) -> pd.Series:
    grade = GRADE_CLASSES[grade_class]
    return sales["grade"].isna() if grade is None else (sales["grade"] == grade)


def event_sale_windows(
    sales: pd.DataFrame,
    events: pd.DataFrame,
    grade_class: str = "ungraded",
    baseline_days: int = 56,
    window_days: int = 21,
    min_baseline: int = 3,
    min_window: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Align a card's sales around each event; normalize by pre-event baseline.

    Returns (windows, drop_log); drop_log counts event-card pairs dropped for
    too few baseline sales / too few window sales, and kept pairs.
    """
    s = sales[_grade_mask(sales, grade_class) & sales["price"].notna()]
    rows: list[dict] = []
    drop_log = {"dropped_baseline": 0, "dropped_window": 0, "kept": 0}
    by_card = {c: g.sort_values("sale_date") for c, g in s.groupby("card_slug")}
    cards_of = s.groupby("mlb_id")["card_slug"].unique()
    for event_id, e in enumerate(events.itertuples()):
        for card in cards_of.get(e.mlb_id, []):
            g = by_card.get(card)
            if g is None:
                continue
            d = g["sale_date"]
            pre = g[(d >= e.event_date - pd.Timedelta(days=baseline_days)) & (d < e.event_date)]
            if len(pre) < min_baseline:
                drop_log["dropped_baseline"] += 1
                continue
            baseline = float(pre["price"].median())
            win = g[(d >= e.event_date - pd.Timedelta(days=window_days))
                    & (d <= e.event_date + pd.Timedelta(days=window_days))]
            if len(win) < min_window:
                drop_log["dropped_window"] += 1
                continue
            drop_log["kept"] += 1
            t = (win["sale_date"] - e.event_date).dt.total_seconds() / 86400.0
            for sale_date, t_days, price in zip(win["sale_date"], t, win["price"]):
                rows.append(
                    {
                        "event_id": event_id, "mlb_id": e.mlb_id, "card_slug": card,
                        "event_date": e.event_date, "sale_date": sale_date,
                        "t_days": float(t_days), "rel_price": float(price) / baseline,
                    }
                )
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS), drop_log


def market_relative_index(
    sales: pd.DataFrame,
    grade_class: str = "ungraded",
    baseline_days: int = 28,
    min_cards: int = 3,
    max_gap_days: int = 3,
) -> pd.Series:
    """Daily cross-card median of (card daily median price / card trailing baseline).

    A card contributes to a day only when it both trades that day and has >= 1
    baseline sale in the trailing `baseline_days` (strictly pre-date). Days
    with < `min_cards` contributing cards are NaN; NaN gaps of <= `max_gap_days`
    are linearly interpolated.
    """
    s = sales[_grade_mask(sales, grade_class) & sales["price"].notna()].copy()
    s["d"] = s["sale_date"].dt.normalize()
    per_day: dict[pd.Timestamp, list[float]] = {}
    for _, g in s.groupby("card_slug"):
        g = g.sort_values("d")
        days = g["d"].to_numpy()
        prices = g["price"].to_numpy(dtype=float)
        daily = g.groupby("d")["price"].median()
        for day, med in daily.items():
            lo = np.datetime64(day - pd.Timedelta(days=baseline_days))
            i0 = np.searchsorted(days, lo, side="left")
            i1 = np.searchsorted(days, np.datetime64(day), side="left")  # strictly pre-date
            if i1 - i0 < 1:
                continue
            base = float(np.median(prices[i0:i1]))
            if base > 0:
                per_day.setdefault(day, []).append(float(med) / base)
    idx = pd.date_range(s["d"].min(), s["d"].max(), freq="D")
    mkt = pd.Series(
        [float(np.median(per_day[d])) if len(per_day.get(d, [])) >= min_cards else np.nan
         for d in idx],
        index=idx,
    )
    return mkt.interpolate(limit=max_gap_days, limit_direction="both")


def adjust_for_market(windows: pd.DataFrame, mkt: pd.Series) -> tuple[pd.DataFrame, int]:
    """rel_adj = rel_price / market_index(sale day). NaN index -> NaN, counted."""
    out = windows.copy()
    day = out["sale_date"].dt.normalize()
    m = day.map(mkt)
    out["rel_adj"] = out["rel_price"] / m
    n_unadj = int(out["rel_adj"].isna().sum())
    return out, n_unadj


def pooled_lag_curve(
    windows: pd.DataFrame,
    value_col: str = "rel_adj",
    bin_min: int = -14,
    bin_max: int = 14,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Pooled daily median curve with cluster-bootstrap CIs (resample events)."""
    w = windows.dropna(subset=[value_col]).copy()
    w["t_bin"] = np.floor(w["t_days"] + 0.5).astype(int)
    w = w[(w["t_bin"] >= bin_min) & (w["t_bin"] <= bin_max)]
    rng = np.random.default_rng(seed)
    event_ids = w["event_id"].unique()
    boots = {k: [] for k in range(bin_min, bin_max + 1)}
    grouped = {e: g for e, g in w.groupby("event_id")}
    for _ in range(n_boot):
        sample = rng.choice(event_ids, size=len(event_ids), replace=True)
        frame = pd.concat([grouped[e] for e in sample])
        meds = frame.groupby("t_bin")[value_col].median()
        for k, vals in boots.items():
            if k in meds.index:
                vals.append(meds[k])
    rows = []
    for k in range(bin_min, bin_max + 1):
        obs = w[w["t_bin"] == k][value_col]
        b = np.asarray(boots[k])
        rows.append(
            {
                "t_bin": k,
                "median": float(obs.median()) if len(obs) else np.nan,
                "ci_lo": float(np.percentile(b, 2.5)) if len(b) >= 20 else np.nan,
                "ci_hi": float(np.percentile(b, 97.5)) if len(b) >= 20 else np.nan,
                "n_sales": len(obs),
            }
        )
    return pd.DataFrame(rows)


def estimate_lag(curve: pd.DataFrame, threshold: float = 1.0, hold_bins: int = 3) -> dict:
    """First post-event bin whose CI floor clears `threshold` for `hold_bins`
    consecutive bins. None when adjustment never reaches significance."""
    c = curve.set_index("t_bin")
    pre = c.loc[c.index < 0]
    cover = (pre["ci_lo"] <= threshold) & (pre["ci_hi"] >= threshold)
    share = float(cover.mean()) if len(cover) else 0.0
    pre_ok = bool(share >= 0.8)  # calibrated: joint 95% coverage is miscalibrated
    post_bins = sorted(c.index[c.index >= 0])
    for k in post_bins:
        seq = [k + j for j in range(hold_bins)]
        if all(b in c.index for b in seq) and all(c.loc[b, "ci_lo"] > threshold for b in seq):
            return {"lag_days": int(k), "pre_ok": pre_ok, "pre_cover_share": share}
    return {"lag_days": None, "pre_ok": pre_ok, "pre_cover_share": share}


def fit_half_life(
    curve: pd.DataFrame,
    h_min: float = 0.25,
    h_max: float = 14.0,
    h_step: float = 0.25,
) -> dict:
    """Fit median-1 = A*(1 - 2^(-t/h)) on post bins (t >= 0); grid over h, OLS for A."""
    post = curve[curve["t_bin"] >= 0].dropna(subset=["median"])
    t = post["t_bin"].to_numpy(dtype=float)
    y = (post["median"] - 1.0).to_numpy(dtype=float)
    if y.sum() <= 0 or len(t) < 5:
        return {"half_life_days": np.nan, "amplitude": np.nan}
    best = None
    for h in np.arange(h_min, h_max + 1e-9, h_step):
        g = 1.0 - np.power(2.0, -t / h)
        a = float((y * g).sum() / (g * g).sum())
        sse = float(((y - a * g) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, h, a)
    if best[2] <= 0.01:  # fitted move is noise-level -> report no adjustment
        return {"half_life_days": np.nan, "amplitude": np.nan}
    return {"half_life_days": float(best[1]), "amplitude": float(best[2])}


def classify_adjustment(
    windows: pd.DataFrame,
    early_hi: float = 3.5,
    late_lo: float = 7.0,
    no_move: float = 0.02,
    late_move: float = 0.05,
    fast_ratio: float = 0.8,
) -> pd.DataFrame:
    """Per-event adjustment class from adjusted relative prices (see gate in spec §8)."""
    rows = []
    for event_id, g in windows.dropna(subset=["rel_adj"]).groupby("event_id"):
        pre = g[g["t_days"] < 0]["rel_adj"]
        early = g[(g["t_days"] >= 0) & (g["t_days"] < early_hi)]["rel_adj"]
        late = g[(g["t_days"] >= late_lo)]["rel_adj"]
        if min(len(pre), len(early), len(late)) < 2:
            rows.append({"event_id": event_id, "pre_med": np.nan, "early_med": np.nan,
                         "late_med": np.nan, "cls": "insufficient"})
            continue
        pre_med, early_med, late_med = pre.median(), early.median(), late.median()
        move = late_med - pre_med
        if move <= no_move:
            cls = "no_adjustment"
        elif (early_med - pre_med) / move >= fast_ratio:
            cls = "fast"
        elif abs(early_med - pre_med) <= no_move and move >= late_move:
            cls = "late"
        else:
            cls = "intermediate"
        rows.append({"event_id": event_id, "pre_med": float(pre_med),
                     "early_med": float(early_med), "late_med": float(late_med), "cls": cls})
    return pd.DataFrame(rows)


def assign_strata(events: pd.DataFrame, game_logs_mlb: pd.DataFrame) -> pd.Series:
    """'prospect' when career stage at the event is prospect/rookie_year, else 'established'.

    `game_logs_mlb` must already be restricted to level == "mlb" (the caller's
    job); career_stage assumes MLB-only rows when deriving debut season.
    """
    out = []
    for e in events.itertuples():
        stage = career_stage(game_logs_mlb, e.mlb_id, pd.Timestamp(e.event_date))
        out.append("prospect" if stage in ("prospect", "rookie_year") else "established")
    return pd.Series(out, index=events.index, name="stratum")


def main() -> None:
    """Run the lag study end to end on the universe data."""
    import argparse
    import json

    from cardprice.breakout_events import dedupe_overlaps, detect_breakouts, merge_debuts

    ap = argparse.ArgumentParser()
    ap.add_argument("--sales", default="data/processed/universe_sales.parquet")
    ap.add_argument("--game-logs", default="data/processed/game_logs_universe.parquet")
    ap.add_argument("--events", default="data/processed/events_universe.parquet")
    ap.add_argument("--out-events", default="data/processed/breakout_events.parquet")
    ap.add_argument("--out-curves", default="data/processed/lag_curves.csv")
    ap.add_argument("--out-summary", default="data/processed/lag_summary.json")
    args = ap.parse_args()

    sales = pd.read_parquet(args.sales)
    logs = pd.read_parquet(args.game_logs)
    registry = pd.read_parquet(args.events)

    events = merge_debuts(detect_breakouts(logs), registry)
    events = dedupe_overlaps(events)
    events["stratum"] = assign_strata(events, logs[logs["level"] == "mlb"]).to_numpy()
    events.to_parquet(args.out_events, index=False)
    print(f"events: {len(events)} "
          f"({events['event_type'].value_counts().to_dict()}, "
          f"levels {events['level'].value_counts(dropna=False).to_dict()})")

    summary = {"n_events": len(events), "runs": {}}
    curves_all = []
    for grade_class in ("ungraded", "psa_10"):
        mkt = market_relative_index(sales, grade_class=grade_class)
        for stratum in ("all", "prospect", "established"):
            ev = events if stratum == "all" else events[events["stratum"] == stratum]
            windows, drop_log = event_sale_windows(sales, ev, grade_class=grade_class)
            # empty windows (kept == 0) have object-dtype columns; nothing to adjust
            windows, n_unadj = adjust_for_market(windows, mkt) if len(windows) else (windows, 0)
            key = f"{grade_class}/{stratum}"
            print(f"{key}: events={len(ev)} drop_log={drop_log} unadjusted_sales={n_unadj}")
            if drop_log["kept"] < 5:
                print(f"{key}: too few event-card pairs, skipped")
                summary["runs"][key] = {
                    "skipped": True, "drop_log": drop_log, "n_unadjusted": n_unadj,
                }
                continue
            curve = pooled_lag_curve(windows)
            lag = estimate_lag(curve)
            hl = fit_half_life(curve)
            cls = classify_adjustment(windows)
            shares = cls["cls"].value_counts(normalize=True).round(4).to_dict()
            curve["grade_class"], curve["stratum"] = grade_class, stratum
            curves_all.append(curve)
            summary["runs"][key] = {
                "drop_log": drop_log, "n_unadjusted": n_unadj, "lag": lag,
                "half_life": hl, "adjustment_shares": shares,
                "n_event_card_pairs": int(windows["event_id"].nunique()),
            }
            print(f"{key}: lag={lag} half_life={hl} shares={shares}")

    if curves_all:
        pd.concat(curves_all).to_csv(args.out_curves, index=False)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    primary = summary["runs"].get("ungraded/all", {})
    if primary.get("adjustment_shares"):
        s = primary["adjustment_shares"]
        fast = s.get("fast", 0.0)
        late = s.get("late", 0.0)
        print("\nGATE (spec §8, primary = ungraded/all):")
        print(f"  fast share (>=80% adjusted within 72h): {fast:.3f}")
        print(f"  late share (>=20% adjusting at >=7d):   {late:.3f}")
        if fast >= 0.8:
            print("  VERDICT: reaction-timing DEAD -> reframe T3 to anticipation")
        elif late >= 0.2:
            print("  VERDICT: timing edge PLAUSIBLE -> proceed as designed")
        else:
            print("  VERDICT: intermediate -> report as measured")


if __name__ == "__main__":
    main()
