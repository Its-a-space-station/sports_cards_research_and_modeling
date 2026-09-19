"""Award-futures odds: parsing, vig normalization, feature math (pure functions).

Sources (spike-governed): Polymarket gamma/CLOB APIs (2025+2026), Kalshi
candlesticks (2026 cross-source). Everything here is offline; network lives in
scripts/collect_odds.py.
"""

import numpy as np
import pandas as pd

SNAP_COLUMNS = [
    "source", "event", "market_id", "player_name", "mlb_id", "season",
    "date", "raw_price", "implied_prob", "volume",
]


def parse_polymarket_event(payload: dict) -> pd.DataFrame:
    """Outcome markets of an award event -> [market_id, player_name, volume]."""
    rows = []
    for m in payload.get("markets", []):
        label = m.get("groupItemTitle") or m.get("question") or ""
        rows.append(
            {
                "market_id": str(m.get("conditionId") or m.get("condition_id") or m.get("id")),
                "player_name": label.strip(),
                "volume": float(m.get("volume") or m.get("volumeNum") or 0.0),
            }
        )
    return pd.DataFrame(rows, columns=["market_id", "player_name", "volume"])


def parse_polymarket_history(payload: dict, market_id: str) -> pd.DataFrame:
    """prices-history chunk -> [market_id, ts, raw_price] (UTC, ascending)."""
    hist = payload.get("history", payload if isinstance(payload, list) else [])
    rows = [
        {"market_id": market_id, "ts": pd.to_datetime(int(p["t"]), unit="s", utc=True),
         "raw_price": float(p["p"])}
        for p in hist
    ]
    out = pd.DataFrame(rows, columns=["market_id", "ts", "raw_price"])
    return out.sort_values("ts").reset_index(drop=True)


def vig_normalize(prices: pd.DataFrame) -> pd.DataFrame:
    """implied_prob = raw_price / event-date total (overround removal).

    Dates whose event total is 0 are dropped (counted via the .attrs tally).
    """
    out = prices.copy()
    totals = out.groupby(["event", "date"])["raw_price"].transform("sum")
    valid = totals > 0
    out = out[valid].copy()
    out["implied_prob"] = out["raw_price"] / totals[valid]
    out.attrs["n_dropped"] = int((~valid).sum())
    return out.reset_index(drop=True)


def daily_prices(history: pd.DataFrame) -> pd.DataFrame:
    """Last price per (market_id, UTC date)."""
    h = history.assign(date=history["ts"].dt.floor("D"))
    return h.groupby(["market_id", "date"], as_index=False).last()[
        ["market_id", "date", "raw_price"]
    ]


def month_grain_features(
    snapshots: pd.DataFrame,
    entry_months: pd.Series,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per (mlb_id, entry_month): odds_level/delta_7d/delta_30d/has_market.

    odds_level = last implied_prob with date < entry_month (strict; a snapshot
    exactly at entry_month is excluded). delta_Nd = odds_level minus the last
    implied_prob strictly before min(entry_month - N days, level_date) — the
    reference is always an earlier snapshot than odds_level's own, never the
    level snapshot itself; NaN when no snapshot precedes that cutoff. Players
    with no market in the season: has_market=0 and all odds features 0.0
    (documented semantics). `players` (mlb_id, entry_month, season) supplies
    the full target grid; when omitted, the grid is every mlb_id in snapshots
    x entry_months with season = entry year.
    """
    cols = [
        "mlb_id", "entry_month", "has_market",
        "odds_level", "odds_delta_7d", "odds_delta_30d",
    ]
    if players is None:
        players = pd.DataFrame(
            [
                {"mlb_id": mlb_id, "entry_month": entry, "season": int(entry.year)}
                for mlb_id in snapshots["mlb_id"].unique()
                for entry in entry_months
            ]
        )
    out_rows = []
    snap = snapshots.sort_values("date")
    for p in players.itertuples():
        g = snap[snap["mlb_id"] == p.mlb_id]
        entry = p.entry_month
        before = g[g["date"] < entry]
        has = int(len(g[g["season"] == p.season]) > 0)
        if not len(before):
            out_rows.append(
                {"mlb_id": p.mlb_id, "entry_month": entry, "has_market": has,
                 "odds_level": 0.0 if not has else np.nan,
                 "odds_delta_7d": 0.0 if not has else np.nan,
                 "odds_delta_30d": 0.0 if not has else np.nan}
            )
            continue
        level = float(before["implied_prob"].iloc[-1])
        level_date = before["date"].iloc[-1]

        def _level_at(cutoff, g=g):
            b = g[g["date"] < cutoff]
            return float(b["implied_prob"].iloc[-1]) if len(b) else np.nan

        l7 = _level_at(min(entry - pd.Timedelta(days=7), level_date))
        l30 = _level_at(min(entry - pd.Timedelta(days=30), level_date))
        out_rows.append(
            {"mlb_id": p.mlb_id, "entry_month": entry, "has_market": has,
             "odds_level": level,
             "odds_delta_7d": (level - l7) if not np.isnan(l7) else np.nan,
             "odds_delta_30d": (level - l30) if not np.isnan(l30) else np.nan}
        )
    return pd.DataFrame(out_rows, columns=cols)


def parse_kalshi_candles(payload: dict, market_ticker: str) -> pd.DataFrame:
    """Kalshi daily candlesticks -> [market_id, ts, raw_price, volume].

    Current payloads carry string fields already in dollars (`price.close_dollars`,
    `volume_fp`); legacy payloads carried integer cents (`price.close`, `volume`).
    Cents are converted to dollars (pinned by the fixture golden: raw_price in
    [0, 1]).
    """
    candles = payload.get("candlesticks", [])
    rows = []
    for c in candles:
        price = c.get("price", {})
        raw = price.get("close_dollars", price.get("close")) if isinstance(price, dict) else None
        close = float(raw)
        if close > 1.0:  # legacy cents -> dollars
            close = close / 100.0
        rows.append(
            {"market_id": market_ticker,
             "ts": pd.to_datetime(int(c.get("end_period_ts", c.get("ts", 0))), unit="s", utc=True),
             "raw_price": close,
             "volume": float(c.get("volume_fp", c.get("volume", 0)) or 0)}
        )
    out = pd.DataFrame(rows, columns=["market_id", "ts", "raw_price", "volume"])
    return out.sort_values("ts").reset_index(drop=True)


def odds_spike_events(
    snapshots: pd.DataFrame,
    min_abs_delta: float = 0.10,
    rel_std_window: int = 30,
    rel_std_mult: float = 3.0,
) -> pd.DataFrame:
    """|1-day delta| >= max(min_abs_delta, rel_std_mult x rolling-30d std of
    daily deltas); per player, the largest |delta| is kept per 7-day cluster
    of same-direction spikes (an up-spike and a down-spike within 7 days are
    distinct events)."""
    cols = ["mlb_id", "event_date", "event_type", "delta", "implied_prob"]
    events = []
    for mlb_id, g in snapshots.sort_values("date").groupby("mlb_id"):
        g = g.drop_duplicates("date").set_index("date")["implied_prob"]
        delta = g.diff()
        std = delta.rolling(rel_std_window, min_periods=5).std()
        thresh = np.maximum(min_abs_delta, rel_std_mult * std)
        spikes = delta[delta.abs() >= thresh]
        taken: list[tuple[pd.Timestamp, float]] = []
        for d, v in spikes.sort_values(key=abs, ascending=False).items():
            if all(abs((d - t).days) > 7 or v * tv < 0 for t, tv in taken):
                taken.append((d, v))
        for d, _v in sorted(taken):
            events.append(
                {"mlb_id": int(mlb_id), "event_date": d, "event_type": "odds_spike",
                 "delta": float(delta[d]), "implied_prob": float(g[d])}
            )
    return pd.DataFrame(events, columns=cols)
