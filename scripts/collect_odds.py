# scripts/collect_odds.py
"""Odds collector: Polymarket chunked prices-history (2025+2026) + Kalshi 2026 daily candles.

Endpoints are the spike-verified unauthenticated ones
(docs/superpowers/spikes/2026-09-19-odds-history.md); no others are used:
- gamma public-search (the spike's exact queries) + /events?slug= for discovery
  (the response is a LIST — the slug-matched event dict is extracted);
- clob /prices-history per outcome YES token, 14-day chunks at fidelity=60 over
  the market's active window (startTs/endTs); daily_prices; vig-normalize per
  (event, date) over ALL outcome markets (mapped or not);
- Kalshi /events (series tickers, `-26` events) + /markets + daily candlesticks
  as a 2026 cross-source: source="kalshi" rows carry raw close as implied_prob
  directly (single-outcome binaries — NO vig normalization), flagged by source.

Snapshot-resume: a (market, chunk) whose snapshot exists is LOADED from
data/raw/odds/ (no fetch, no sleep); only misses hit the API, so the weekly
cron fetches just the moved trailing chunks. --since Nd additionally skips
FETCHING chunks that end before today-N (unsnapshotted old chunks stay missing
— they were recorded as gaps when first attempted). Raw payloads are saved via
save_raw("odds", key, payload) BEFORE parsing. Politeness: --sleep between
calls (0.3-0.5s band); 403/429 -> one retry after 10s backoff, then a recorded
gap (never hammer, never fabricate).

Player mapping: exact norm_name match against player_info_class.csv U
player_info_universe.csv; unmatched outcome labels -> odds_unmapped_audit.csv
(label, event, volume), never fuzzy-matched. Dates are UTC calendar dates
stored tz-naive. Volume semantics: polymarket = gamma market lifetime volume
(repeated per date row); kalshi = daily candle contracts.

Run from the repo root: `.venv/bin/python scripts/collect_odds.py [--since 14d]`
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from resolve_seed_cards import norm_name

from cardprice.odds import (
    SNAP_COLUMNS,
    daily_prices,
    parse_kalshi_candles,
    parse_polymarket_event,
    parse_polymarket_history,
    vig_normalize,
)
from cardprice.storage import load_latest, save_raw, snapshot_exists

ROOT = Path(__file__).resolve().parents[1]
CLASS_INFO = ROOT / "data" / "reference" / "player_info_class.csv"
UNIVERSE_INFO = ROOT / "data" / "reference" / "player_info_universe.csv"
AUDIT_OUT = ROOT / "data" / "reference" / "odds_unmapped_audit.csv"
REGISTRY_OUT = ROOT / "data" / "reference" / "odds_events_registry.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "odds_snapshots.parquet"

UA = {"User-Agent": "cardprice-research/0.1 (odds collector)"}
BACKOFF_S = 10.0

GAMMA_SEARCH = "https://gamma-api.polymarket.com/public-search"
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

SEASONS = (2025, 2026)
LEAGUES = ("AL", "NL")
AWARDS = ("MVP", "CY", "ROY")
CELLS = [(s, lg, aw) for s in SEASONS for lg in LEAGUES for aw in AWARDS]

# Spike's verbatim event inventory (slug per season x league x award cell).
POLYMARKET_SLUGS = {
    (2025, "AL", "MVP"): "mlb-al-mvp",
    (2025, "NL", "MVP"): "mlb-nl-mvp",
    (2025, "AL", "CY"): "mlb-al-cy-young",
    (2025, "NL", "CY"): "mlb-nl-cy-young",
    (2025, "AL", "ROY"): "al-rookie-of-the-year",
    (2025, "NL", "ROY"): "nl-rookie-of-the-year",
    (2026, "NL", "MVP"): "mlb-2026-nl-mvp",
    (2026, "AL", "CY"): "mlb-2026-al-cy-young-winner",
    (2026, "NL", "CY"): "mlb-2026-nl-cy-young-winner",
    (2026, "AL", "ROY"): "mlb-al-rookie-of-the-year",
    (2026, "NL", "ROY"): "mlb-nl-rookie-of-the-year",
}
# The spike's search inventory has no 2026 AL MVP event; the slug surfaced by
# the spike's own search pattern is pro-baseball-2026-al-mvp (verified live
# 2026-09-19) — probed via the same spike-verified /events?slug= endpoint.
SLUG_PROBES = {
    (2026, "AL", "MVP"): ["pro-baseball-2026-al-mvp", "mlb-2026-al-mvp", "mlb-2026-al-mvp-winner"]
}
SEARCH_QUERIES = ["MLB MVP", "2025 MLB MVP", "MLB Cy Young", "MLB Rookie of the Year"]

KALSHI_SERIES = {  # spike-verbatim tickers; 2026 (`-26`) events only
    ("NL", "MVP"): "KXMLBNLMVP",
    ("AL", "MVP"): "KXMLBALMVP",
    ("AL", "CY"): "KXMLBALCY",
    ("NL", "CY"): "KXMLBNLCY",
    ("AL", "ROY"): "KXMLBALROTY",
    ("NL", "ROY"): "KXMLBNLROTY",
}
KALSHI_SEASON = 2026
KALSHI_SEASON_START = pd.Timestamp("2026-01-01", tz="UTC")  # before the earliest market open

CHUNK_DAYS = 14
FIDELITY_MIN = 60

REGISTRY_COLUMNS = [
    "event", "source", "event_id", "season", "league", "award", "n_outcomes", "collected_through",
]


def plan_chunks(start: pd.Timestamp, end: pd.Timestamp, days: int = CHUNK_DAYS):
    """Contiguous [start, end) windows of at most `days` days each."""
    chunks = []
    a = start
    while a < end:
        b = min(a + pd.Timedelta(days=days), end)
        chunks.append((a, b))
        a = b
    return chunks


def map_players(outcomes: pd.DataFrame, info: pd.DataFrame):
    """Exact normalized-name match of outcome labels against the player info.

    Returns (mapped, audit): mapped = outcomes + mlb_id (NA when unmatched);
    audit = the unmatched rows (player_name, event, volume when present) —
    never fuzzy-matched."""
    lookup: dict[str, int] = {}
    for mlb_id, name in zip(info["mlb_id"], info["name"], strict=True):
        lookup.setdefault(norm_name(str(name)), int(mlb_id))
    mapped = outcomes.copy()
    mapped["mlb_id"] = [lookup.get(norm_name(str(n))) for n in outcomes["player_name"]]
    audit_cols = [c for c in ("player_name", "event", "volume", "source") if c in outcomes.columns]
    audit = mapped[mapped["mlb_id"].isna()][audit_cols].reset_index(drop=True)
    return mapped, audit


def _get(session, url, params, sleep_s, gaps, label):
    """GET JSON politely; one 10s-backoff retry on 403/429, then a recorded gap."""
    for attempt in (1, 2):
        time.sleep(sleep_s)
        try:
            resp = session.get(url, params=params, headers=UA, timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except (requests.RequestException, ValueError) as e:
            gaps.append({"event": label, "gap": f"request/JSON error: {e}"})
            return None
        if resp.status_code in (403, 429) and attempt == 1:
            print(f"  HTTP {resp.status_code} on {label} — {BACKOFF_S}s backoff, one retry")
            time.sleep(BACKOFF_S)
            continue
        gaps.append({"event": label, "gap": f"HTTP {resp.status_code}"})
        return None
    return None


def _yes_token(market: dict) -> str | None:
    """CLOB token of the YES outcome (capture-script pattern); None when absent."""
    try:
        tokens = json.loads(market["clobTokenIds"])
        outcomes = json.loads(market["outcomes"])
        return str(tokens[outcomes.index("Yes")])
    except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError):
        try:
            return str(json.loads(market["clobTokenIds"])[0])
        except (KeyError, TypeError, IndexError, json.JSONDecodeError):
            return None


def _market_id(market: dict) -> str:
    return str(market.get("conditionId") or market.get("condition_id") or market.get("id"))


def _window(market: dict, event: dict, now: pd.Timestamp):
    """Active window of an outcome market (event dates as fallback), capped at now."""
    start = pd.to_datetime(market.get("startDate") or event.get("startDate"), utc=True)
    end_raw = market.get("endDate") or event.get("endDate")
    end = min(pd.to_datetime(end_raw, utc=True), now) if end_raw else now
    return start, end


def discover_slugs(get) -> dict:
    """(season, league, award) -> gamma slug; the spike inventory plus probes."""
    slugs = dict(POLYMARKET_SLUGS)
    found = set()
    for q in SEARCH_QUERIES:
        payload = get(GAMMA_SEARCH, {"q": q, "limit_per_type": 10}, label=f"search {q!r}")
        if isinstance(payload, dict):
            found |= {e.get("slug") for e in payload.get("events", []) if e.get("slug")}
    for cell, probes in SLUG_PROBES.items():
        cands = sorted(s for s in found if "2026" in s and "mvp" in s and "-al-" in s)
        for slug in cands + [p for p in probes if p not in cands]:
            payload = get(GAMMA_EVENTS, {"slug": slug}, label=f"probe {slug}")
            events = payload if isinstance(payload, list) else []
            if any(e.get("slug") == slug for e in events):
                slugs[cell] = slug
                break
    return slugs


def collect_polymarket(get, now, since_cutoff, gaps):
    """Per-cell event payloads + chunked histories -> (daily frames, outcomes, registry)."""
    frames, outcomes_all, registry = [], [], []
    slugs = discover_slugs(get)
    for season, league, award in CELLS:
        slug = slugs.get((season, league, award))
        cell = f"{season} {league} {award}"
        base = {"source": "polymarket", "season": season, "league": league, "award": award}
        payload = slug and get(GAMMA_EVENTS, {"slug": slug}, label=f"event {slug}")
        events = payload if isinstance(payload, list) else []
        event = next((e for e in events if e.get("slug") == slug), None)
        if event is None:
            print(f"MISSING CELL: {cell} (slug={slug!r} not found via gamma)")
            registry.append({**base, "event": slug or "", "event_id": "",
                             "n_outcomes": 0, "collected_through": ""})
            continue
        save_raw("odds", f"pm-event/{slug}", event)
        outcomes = parse_polymarket_event(event)
        outcomes = outcomes.assign(event=slug, source="polymarket", season=season)
        outcomes_all.append(outcomes)
        dailies = []
        for m in event.get("markets", []):
            token = _yes_token(m)
            start, end = _window(m, event, now)
            if token is None or pd.isna(start) or start >= end:
                gaps.append({"event": slug, "gap": f"skipped market {_market_id(m)} (no token/window)"})
                continue
            hist = []
            n_fetched = 0
            for a, b in plan_chunks(start, end):
                key = f"pm-history/{token}/{int(a.timestamp())}_{int(b.timestamp())}"
                if snapshot_exists("odds", key):
                    payload = load_latest("odds", key)
                elif since_cutoff is not None and b < since_cutoff:
                    continue  # --since: old unsnapshotted chunks stay for a full run
                else:
                    params = {"market": token, "startTs": int(a.timestamp()),
                              "endTs": int(b.timestamp()), "fidelity": FIDELITY_MIN}
                    payload = get(CLOB_HISTORY, params, label=f"history {slug}/{token[-6:]}")
                    if payload is None:
                        continue
                    save_raw("odds", key, payload)
                    n_fetched += 1
                hist.append(parse_polymarket_history(payload, _market_id(m)))
            hist = [h for h in hist if len(h)]  # empty chunks carry object dtypes
            if hist:
                dailies.append(daily_prices(pd.concat(hist, ignore_index=True)))
        daily = pd.concat(dailies, ignore_index=True) if dailies else pd.DataFrame()
        if len(daily):
            meta = outcomes[["market_id", "player_name", "volume", "event", "source", "season"]]
            daily = daily.merge(meta, on="market_id", how="left")
            frames.append(daily)
        through = str(daily["date"].max().date()) if len(daily) else ""
        registry.append({**base, "event": slug, "event_id": str(event.get("id", "")),
                         "n_outcomes": len(outcomes), "collected_through": through})
        print(f"{cell}: {slug} — {len(outcomes)} outcomes, {len(daily)} market-days, "
              f"through {through}")
    return frames, outcomes_all, registry


def _kalshi_label(market: dict) -> str:
    # yes_sub_title carries the bare player name ("Pete Alonso"); title is a question.
    return str(market.get("yes_sub_title") or market.get("title") or market.get("ticker") or "")


def _kalshi_markets(get, event_ticker, gaps):
    markets, cursor = [], ""
    while True:
        params = {"event_ticker": event_ticker, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        payload = get(f"{KALSHI_BASE}/markets", params, label=f"kalshi markets {event_ticker}")
        if not isinstance(payload, dict):
            break
        markets.extend(payload.get("markets", []))
        cursor = payload.get("cursor") or ""
        if not cursor:
            break
    return markets


def collect_kalshi(get, now, gaps):
    """2026 daily candles per award series -> (frames, outcomes, registry)."""
    frames, outcomes_all, registry = [], [], []
    start_ts = int(KALSHI_SEASON_START.timestamp())
    for (league, award), series in KALSHI_SERIES.items():
        base = {"source": "kalshi", "season": KALSHI_SEASON, "league": league, "award": award}
        payload = get(f"{KALSHI_BASE}/events", {"series_ticker": series, "limit": 20},
                      label=f"kalshi events {series}")
        events = payload.get("events", []) if isinstance(payload, dict) else []
        ticker = f"{series}-26"
        if not any(e.get("event_ticker") == ticker for e in events):
            print(f"MISSING CELL: kalshi {KALSHI_SEASON} {league} {award} (no {ticker} event)")
            registry.append({**base, "event": ticker, "event_id": "",
                             "n_outcomes": 0, "collected_through": ""})
            continue
        markets = _kalshi_markets(get, ticker, gaps)
        outcomes = pd.DataFrame(
            [{"market_id": str(m.get("ticker")), "player_name": _kalshi_label(m).strip(),
              "volume": float(m.get("volume_fp") or m.get("volume") or 0.0)} for m in markets],
            columns=["market_id", "player_name", "volume"],
        ).assign(event=ticker, source="kalshi", season=KALSHI_SEASON)
        outcomes_all.append(outcomes)
        candles = []
        n_null = 0
        for m in markets:
            t = str(m.get("ticker"))
            params = {"start_ts": start_ts, "end_ts": int(now.timestamp()), "period_interval": 1440}
            payload = get(f"{KALSHI_BASE}/series/{series}/markets/{t}/candlesticks",
                          params, label=f"kalshi candles {t}")
            if not isinstance(payload, dict):
                if payload is not None:
                    gaps.append({"event": ticker, "gap": f"non-dict candlesticks payload {t}"})
                continue
            save_raw("odds", f"kalshi/{t}/{start_ts}_{int(now.timestamp())}", payload)
            # No-trade days carry an empty `price` dict (null close) — drop those
            # candles before the parser, which requires a close. The raw snapshot
            # keeps them; price conversion stays inside parse_kalshi_candles.
            all_c = payload.get("candlesticks", [])
            ok_c = [
                c for c in all_c
                if isinstance(c.get("price"), dict)
                and c["price"].get("close_dollars", c["price"].get("close")) is not None
            ]
            n_null += len(all_c) - len(ok_c)
            parsed = parse_kalshi_candles({**payload, "candlesticks": ok_c}, t)
            if len(parsed):  # skip empty payloads (object dtypes contaminate concat)
                candles.append(parsed)
        daily = pd.concat(candles, ignore_index=True) if candles else pd.DataFrame()
        if len(daily):
            daily = daily.assign(date=daily["ts"].dt.floor("D")).drop(columns=["ts"])
            meta = outcomes[["market_id", "player_name", "event", "source", "season"]]
            daily = daily.merge(meta, on="market_id", how="left")
            daily["implied_prob"] = daily["raw_price"]  # flagged by source; no vig
            frames.append(daily)
        through = str(daily["date"].max().date()) if len(daily) else ""
        registry.append({**base, "event": ticker, "event_id": ticker,
                         "n_outcomes": len(markets), "collected_through": through})
        print(f"kalshi {KALSHI_SEASON} {league} {award}: {ticker} — {len(markets)} markets, "
              f"{len(daily)} market-days, through {through} ({n_null} no-trade candles dropped)")
    return frames, outcomes_all, registry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="snapshot parquet path")
    ap.add_argument("--since", default=None, help="e.g. 14d — only fetch recent chunks")
    ap.add_argument("--sleep", type=float, default=0.35, help="seconds between calls (0.3-0.5)")
    args = ap.parse_args()
    if not 0.3 <= args.sleep <= 0.5:
        sys.exit("--sleep must be within the 0.3-0.5s politeness band")

    now = pd.Timestamp.now(tz="UTC")
    since_cutoff = None
    if args.since:
        since_cutoff = now - pd.Timedelta(days=int(args.since.rstrip("d")))
    info = pd.concat(
        [pd.read_csv(CLASS_INFO), pd.read_csv(UNIVERSE_INFO)], ignore_index=True
    ).drop_duplicates("mlb_id")
    session = requests.Session()
    gaps: list[dict] = []

    def get(url, params, label):
        return _get(session, url, params, args.sleep, gaps, label)

    poly_frames, poly_outcomes, poly_reg = collect_polymarket(get, now, since_cutoff, gaps)
    kalshi_frames, kalshi_outcomes, kalshi_reg = collect_kalshi(get, now, gaps)

    outcomes_all = pd.concat(poly_outcomes + kalshi_outcomes, ignore_index=True)
    poly = pd.concat(poly_frames, ignore_index=True) if poly_frames else pd.DataFrame()
    parts = [vig_normalize(poly)] if len(poly) else []
    n_dropped = parts[0].attrs.get("n_dropped", 0) if parts else 0
    parts += [f for f in kalshi_frames if len(f)]
    snaps = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    mapped, audit = map_players(outcomes_all, info)
    if len(snaps):
        ids = mapped[["market_id", "mlb_id"]].drop_duplicates("market_id")
        snaps = snaps.merge(ids, on="market_id", how="left")
        snaps["mlb_id"] = snaps["mlb_id"].astype("Int64")
        snaps["date"] = snaps["date"].dt.tz_localize(None)  # UTC calendar dates, tz-naive
        snaps = snaps[SNAP_COLUMNS].sort_values(
            ["source", "event", "market_id", "date"]).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    snaps.to_parquet(out, index=False)
    gap_rows = pd.DataFrame(gaps, columns=["event", "gap"]) if gaps else pd.DataFrame(
        columns=["event", "gap"])
    audit_out = pd.concat([audit, gap_rows], ignore_index=True).reindex(
        columns=["player_name", "event", "volume", "source", "gap"]).fillna("")
    audit_out.to_csv(AUDIT_OUT, index=False)
    registry = pd.DataFrame(poly_reg + kalshi_reg, columns=REGISTRY_COLUMNS)
    registry.to_csv(REGISTRY_OUT, index=False)

    print("\n=== ACCEPTANCE ===")
    print(registry.to_string(index=False))
    missing = registry[(registry["source"] == "polymarket") & (registry["n_outcomes"] == 0)]
    print(f"\npolymarket cells missing: {len(missing)}"
          + (f" — {missing[['season', 'league', 'award']].values.tolist()}" if len(missing) else ""))
    n_gap = len(gap_rows)
    print(f"audit: {len(audit)} unmapped outcomes, {n_gap} gaps -> {AUDIT_OUT}")
    print(f"vig: {n_dropped} rows dropped (zero event-date totals)")
    print(f"snapshots: {len(snaps)} rows -> {out}")
    ok = True
    if len(snaps):
        bad = snaps[~snaps["implied_prob"].between(0.0, 1.0)]
        nat = int(snaps["date"].isna().sum())
        print(f"implied_prob range: [{snaps['implied_prob'].min():.4f}, "
              f"{snaps['implied_prob'].max():.4f}]; out-of-range rows: {len(bad)}; NaT dates: {nat}")
        ok = len(bad) == 0 and nat == 0
    if len(missing):
        print("STOP: whole award x league x season cells missing from discovery — see report")
    if not ok:
        sys.exit("ACCEPTANCE FAILED: implied_prob out of [0,1] or NaT dates present")


if __name__ == "__main__":
    main()
