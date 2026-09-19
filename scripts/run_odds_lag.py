# scripts/run_odds_lag.py
"""Odds-spike incorporation-lag study (descriptive; T1's lag_study reused verbatim).

Events: cardprice.odds.odds_spike_events over the POLYMARKET rows of
data/processed/odds_snapshots.parquet (mapped players only) — the primary,
vig-normalized source covering both collected seasons (2025 closed + 2026
live). Kalshi rows are excluded by design: they are raw single-outcome binary
closes (no vig normalization), exist only for 2026 (settled 2025 markets are
unretrievable unauthenticated), and mixing both sources in one daily-diff
series injects cross-source artifacts (31 of 130 mixed-source spike events
have a source switch vs the previous daily observation; mean same-day
|pm - kalshi| divergence is 4.6pp against the 0.10 spike floor). Kalshi
remains the documented 2026 cross-source check (findings doc, Data section).

Sales: class_sales.parquet U universe_sales.parquet, deduped on (card_slug,
sale_date, title, price) — the frames share the SCP source and overlap on
some bowman_1st cards. Market index per grade class over the UNION frame.
Config mirrors T1's recalibrated defaults (event_sale_windows: 56d baseline,
±21d window, min 3 sales); primary cell = ungraded/all; strata via
assign_strata over game_logs_class (level == "mlb" — players absent from the
class logs fall to career_stage's "prospect" fallback; acceptable, noted).

Everything here is DESCRIPTIVE: T3 has no pre-registered lag gate; the spec
§8 gate language is quoted at the end for comparability with T1 only.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cardprice.lag_study import (
    adjust_for_market,
    assign_strata,
    classify_adjustment,
    estimate_lag,
    event_sale_windows,
    market_relative_index,
    pooled_lag_curve,
)
from cardprice.odds import odds_spike_events

SNAPSHOTS = "data/processed/odds_snapshots.parquet"
CLASS_SALES = "data/processed/class_sales.parquet"
UNIVERSE_SALES = "data/processed/universe_sales.parquet"
GAME_LOGS = "data/processed/game_logs_class.parquet"
OUT_EVENTS = "data/processed/odds_spike_events.parquet"
OUT_JSON = "data/processed/odds_lag_summary.json"
DEDUP_KEY = ["card_slug", "sale_date", "title", "price"]


def build_events(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Spike events from the polymarket rows (mapped players), mlb_id as int."""
    pm = snapshots[(snapshots["source"] == "polymarket") & snapshots["mlb_id"].notna()].copy()
    pm["mlb_id"] = pm["mlb_id"].astype(int)
    return odds_spike_events(pm)


def union_sales() -> tuple[pd.DataFrame, dict]:
    """class_sales U universe_sales deduped on DEDUP_KEY, plus build stats."""
    cs = pd.read_parquet(CLASS_SALES)
    us = pd.read_parquet(UNIVERSE_SALES)
    union = pd.concat([cs, us], ignore_index=True).drop_duplicates(subset=DEDUP_KEY)
    stats = {
        "class_rows": len(cs),
        "universe_rows": len(us),
        "union_rows": len(union),
        "duplicates_removed": len(cs) + len(us) - len(union),
    }
    return union, stats


def main() -> None:
    snapshots = pd.read_parquet(SNAPSHOTS)
    events = build_events(snapshots)
    sales, sales_stats = union_sales()
    logs = pd.read_parquet(GAME_LOGS)
    events["stratum"] = assign_strata(events, logs[logs["level"] == "mlb"]).to_numpy()
    events.to_parquet(OUT_EVENTS, index=False)
    print(
        f"events: {len(events)} over {events['mlb_id'].nunique()} players "
        f"({events['event_type'].value_counts().to_dict()}, "
        f"strata {events['stratum'].value_counts().to_dict()}, "
        f"years {events['event_date'].dt.year.value_counts().to_dict()})"
    )
    print(
        f"union sales: {sales_stats['union_rows']} rows "
        f"(class {sales_stats['class_rows']} + universe {sales_stats['universe_rows']} "
        f"- {sales_stats['duplicates_removed']} dups on {DEDUP_KEY})"
    )

    summary = {
        "label": "descriptive",
        "event_source": "polymarket (vig-normalized; kalshi excluded — see module docstring)",
        "config": {
            "spike": {"min_abs_delta": 0.10, "rel_std_window": 30, "rel_std_mult": 3.0},
            "windows": {"baseline_days": 56, "window_days": 21, "min_baseline": 3, "min_window": 3},
            "curve": {"bin_min": -14, "bin_max": 14, "n_boot": 2000, "seed": 42},
            "dedup_key": DEDUP_KEY,
        },
        "n_events": len(events),
        "n_players": int(events["mlb_id"].nunique()),
        "events_per_year": {int(y): int(n) for y, n in
                            events["event_date"].dt.year.value_counts().items()},
        "event_date_range": [str(events["event_date"].min().date()),
                             str(events["event_date"].max().date())],
        "strata": {k: int(v) for k, v in events["stratum"].value_counts().items()},
        "union_sales": sales_stats,
        "runs": {},
    }
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
                print(f"{key}: too few event-card pairs ({drop_log['kept']}), skipped")
                summary["runs"][key] = {
                    "n_events": len(ev), "skipped": True,
                    "drop_log": drop_log, "n_unadjusted": n_unadj,
                }
                continue
            curve = pooled_lag_curve(windows)
            lag = estimate_lag(curve)
            cls = classify_adjustment(windows)
            shares = cls["cls"].value_counts(normalize=True).round(4).to_dict()
            summary["runs"][key] = {
                "n_events": len(ev),
                "drop_log": drop_log,
                "n_unadjusted": n_unadj,
                "lag": lag,
                "adjustment_shares": shares,
                "adjustment_counts": {k: int(v) for k, v in cls["cls"].value_counts().items()},
                "n_event_card_pairs": int(windows["event_id"].nunique()),
                "n_events_classified": int(cls["event_id"].nunique()),
                "curve": curve.to_dict("records"),
            }
            print(f"{key}: lag={lag} shares={shares}")

    Path(OUT_JSON).write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {OUT_EVENTS} and {OUT_JSON}")

    primary = summary["runs"].get("ungraded/all", {})
    if primary.get("adjustment_shares"):
        s = primary["adjustment_shares"]
        fast = s.get("fast", 0.0)
        late = s.get("late", 0.0)
        print("\nDESCRIPTIVE read in spec §8 language (primary = ungraded/all; "
              "T3 has no registered lag gate):")
        print(f"  fast share (>=80% adjusted within 72h): {fast:.3f}")
        print(f"  late share (>=20% adjusting at >=7d):   {late:.3f}")
        if fast >= 0.8:
            print("  READ: reaction-timing dead — as measured (descriptive)")
        elif late >= 0.2:
            print("  READ: timing edge plausible — as measured (descriptive)")
        else:
            print("  READ: intermediate — reported as measured (descriptive)")


if __name__ == "__main__":
    main()
