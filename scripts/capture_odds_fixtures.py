"""One-off: capture Polymarket + Kalshi odds API payloads as committed test fixtures.

Run from repo root:
    .venv/bin/python scripts/capture_odds_fixtures.py

This is the only odds script allowed on the network for fixtures; all tests are
offline against the committed fixtures. Endpoints are the spike-verified
unauthenticated ones (docs/superpowers/spikes/2026-09-19-odds-history.md):
Polymarket gamma event `mlb-al-mvp` (2025 AL MVP, closed), one CLOB
prices-history chunk for its Aaron Judge outcome token (June 2025, daily), and
Kalshi daily candlesticks for the 2026 market KXMLBALMVP-26-YALV.

Politeness: >= 5 s between calls; on 403/429 retry once after a longer backoff,
then STOP (never fabricate fixture content). Golden checks run before saving;
a failed check aborts without writing.
"""

import json
import sys
import time
from pathlib import Path

import requests

OUT = Path("tests/fixtures/odds")
SLEEP_S = 5.0
BACKOFF_S = 20.0
UA = {"User-Agent": "cardprice-research/0.1 (odds fixture capture)"}

GAMMA_EVENT_URL = "https://gamma-api.polymarket.com/events?slug=mlb-al-mvp"
HISTORY_URL = (
    "https://clob.polymarket.com/prices-history"
    "?market={token}&startTs=1748736000&endTs=1749340800&fidelity=1440"  # 2025-06-01..08 UTC
)
KALSHI_URL = (
    "https://api.elections.kalshi.com/trade-api/v2/series/KXMLBALMVP/markets/"
    "KXMLBALMVP-26-YALV/candlesticks?start_ts=1786000000&end_ts=1789776000&period_interval=1440"
)


def _get(url: str) -> dict | list:
    """GET JSON; one backoff retry on 403/429, then stop."""
    for attempt in (1, 2):
        resp = requests.get(url, headers=UA, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (403, 429) and attempt == 1:
            print(f"  HTTP {resp.status_code} — backing off {BACKOFF_S}s and retrying once")
            time.sleep(BACKOFF_S)
            continue
        raise SystemExit(
            f"CAPTURE BLOCKED: HTTP {resp.status_code} for {url} — stopped, nothing fabricated"
        )
    raise SystemExit("unreachable")


def _label(market: dict) -> str:
    return (market.get("groupItemTitle") or market.get("question") or "").strip()


def _yes_token(market: dict) -> str:
    tokens = json.loads(market["clobTokenIds"])
    try:
        outcomes = json.loads(market["outcomes"])
        return tokens[outcomes.index("Yes")]
    except (ValueError, KeyError, TypeError):
        return tokens[0]


def capture_event() -> dict:
    payload = _get(GAMMA_EVENT_URL)
    events = payload if isinstance(payload, list) else [payload]
    event = next((e for e in events if e.get("slug") == "mlb-al-mvp"), None)
    if event is None:
        raise SystemExit("GOLDEN CHECK FAILED: no mlb-al-mvp event in gamma response")
    markets = event.get("markets", [])
    if len(markets) < 3:
        raise SystemExit(f"GOLDEN CHECK FAILED: only {len(markets)} markets in event")
    labels = [_label(m) for m in markets]
    if "Aaron Judge" not in labels:
        raise SystemExit(f"GOLDEN CHECK FAILED: no 'Aaron Judge' outcome; labels={labels[:8]}")
    volumes = [float(m.get("volume") or m.get("volumeNum") or 0.0) for m in markets]
    if any(v < 0 for v in volumes):
        raise SystemExit("GOLDEN CHECK FAILED: negative volume in event markets")
    print(f"  event: {len(markets)} markets, Aaron Judge outcome present, volumes >= 0")
    return event


def capture_history(event: dict) -> dict:
    judge = next(m for m in event["markets"] if _label(m) == "Aaron Judge")
    token = _yes_token(judge)
    payload = _get(HISTORY_URL.format(token=token))
    hist = payload.get("history", []) if isinstance(payload, dict) else []
    if not hist:
        raise SystemExit("GOLDEN CHECK FAILED: empty prices-history chunk")
    prices = [float(p["p"]) for p in hist]
    if not all(0.0 <= p <= 1.0 for p in prices):
        raise SystemExit(f"GOLDEN CHECK FAILED: prices out of [0,1]: {prices[:5]}")
    print(f"  history: {len(hist)} points for Judge YES token, prices in [0,1]")
    return payload


def _kalshi_close(candle: dict) -> float:
    """Close in dollars: current API gives `close_dollars` strings; legacy gave cents."""
    price = candle.get("price", {})
    raw = price.get("close_dollars", price.get("close")) if isinstance(price, dict) else None
    close = float(raw)
    return close / 100.0 if close > 1.0 else close


def capture_kalshi() -> dict:
    payload = _get(KALSHI_URL)
    candles = payload.get("candlesticks", []) if isinstance(payload, dict) else []
    if not candles:
        raise SystemExit("GOLDEN CHECK FAILED: empty Kalshi candlesticks")
    closes = [_kalshi_close(c) for c in candles]
    if not all(0.0 <= p <= 1.0 for p in closes):
        raise SystemExit(f"GOLDEN CHECK FAILED: Kalshi closes out of [0,1]: {closes[:5]}")
    units = "close_dollars strings" if "close_dollars" in candles[0].get("price", {}) else "legacy"
    print(f"  kalshi: {len(candles)} daily candles, closes in [0,1]; payload units: {units}")
    return payload


def _save(name: str, payload: dict) -> None:
    text = json.dumps(payload, indent=1)
    (OUT / name).write_text(text + "\n")
    print(f"wrote {OUT / name} ({len(text) + 1} bytes)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("capturing polymarket event (gamma, mlb-al-mvp)...")
    event = capture_event()
    time.sleep(SLEEP_S)
    print("capturing polymarket prices-history chunk (Judge YES, June 2025)...")
    history = capture_history(event)
    time.sleep(SLEEP_S)
    print("capturing kalshi candlesticks (KXMLBALMVP-26-YALV, daily)...")
    kalshi = capture_kalshi()
    _save("polymarket_event.json", event)
    _save("polymarket_history_chunk.json", history)
    _save("kalshi_candlesticks.json", kalshi)


if __name__ == "__main__":
    sys.exit(main())
