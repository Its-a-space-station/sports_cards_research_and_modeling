# Spike: Dated MLB award-futures odds history (2026-09-19)

**Verdict: PASS — ≥2 full seasons of dated MVP / Cy Young / ROY odds at weekly-or-better cadence are freely and programmatically obtainable from Polymarket (2025 closed season + 2026 live season, hourly candles, no auth, no key). Kalshi is a strong secondary for the current season (daily candles back to market open, free, no auth) but its settled 2025 award markets are NOT retrievable via the unauthenticated API. Build the collector on Polymarket `prices-history`, with Kalshi `candlesticks` as a cross-source for 2026.**

Consequence for T3: odds-delta feature set is viable. Primary = Polymarket CLOB `prices-history` (chunked ≤14-day windows). Secondary = Kalshi daily candlesticks for the live season going forward. The Odds API free tier and Wayback Machine contribute nothing (see §3, §4).

## 1. Kalshi public API — PARTIAL (current season deep, settled seasons gone)

Base: `https://api.elections.kalshi.com/trade-api/v2` (api.kalshi.com returned empty for the same query; elections host is canonical). All calls below made **without any auth header**, HTTP 200.

**Award series exist** — `GET /series?limit=200&category=Sports` (paginated, ~6 pages) returned, verbatim ticker|title pairs:

```
KXMLBNLMVP | Pro Baseball National League MVP
KXMLBALMVP | Pro Baseball American League MVP
KXMLBALCY  | Pro Baseball American League Cy Young
KXMLBNLCY  | Pro Baseball National League Cy Young
KXMLBALROTY | Pro Baseball American League Rookie of the Year
KXMLBNLROTY | Pro Baseball National League Rookie of the Year
```
(plus Manager/Comeback-Player/Reliever of the Year, WS MVP, Hank Aaron, Silver Slugger, Gold Glove, Triple Crown). Note: `KXALMVP`/`KXNLMVP` legacy tickers exist but have swapped titles and zero events — ignore.

**Two seasons of events exist** — `GET /events?series_ticker=KXMLBALMVP&limit=20`:

```
KXMLBALMVP-26 | AL MVP Winner?
KXMLBALMVP-25 | AL MVP Winner?
```
Identical `-25`/`-26` pairs confirmed for `KXMLBNLROTY`, `KXMLBALCY`, `KXMLBNLCY`, `KXMLBALROTY`.

**Settled 2025 markets not retrievable unauthenticated:**
- `GET /markets?event_ticker=KXMLBALMVP-25&limit=30` → `"n_markets: 0"` (also 0 with `status=settled`, `status=closed`, and `series_ticker=KXMLBALMVP&status=closed|settled`).
- Candlestick guesses `KXMLBALMVP-25-AJUDGE`, `-JUDGE`, `-BWITT` → `{"error":{"code":"not_found","message":"not found"}}`.

**Live 2026 markets: full daily history, free.** 49 active markets under `KXMLBALMVP-26`; top by price (verbatim): `KXMLBALMVP-26-YALV | last: 0.9000 | vol: 460810.75 | Yordan Álvarez`.

`GET /series/KXMLBALMVP/markets/KXMLBALMVP-26-YALV/candlesticks?start_ts=1786000000&end_ts=1789776000&period_interval=1440` (verbatim sample):

```
n_days: 43
08-07 close: 0.7900 vol: 536.86  oi: 225733.55
08-14 close: 0.8700 vol: 2565.37 oi: 225269.38
08-21 close: 0.8900 vol: 2425.32 oi: 225413.83
08-28 close: 0.9100 vol: 203.10  oi: 227599.64
09-04 close: 0.9100 vol: 2531.20 oi: 238017.62
09-11 close: 0.9200 vol: 385.45  oi: 238654.18
09-18 close: 0.9000 vol: 20651.95 oi: 296762.57
```

Candles carry `price` (trade OHLC), `yes_bid`/`yes_ask` OHLC, `volume_fp`, `open_interest_fp`. Depth verified to market open: `KXMLBALMVP-26-PALO` returned **206 daily candles, first 2026-02-03, last 2026-09-18** (market opened 2026-02-02). `period_interval=10080` (weekly) is rejected/empty — valid intervals are minute/hour/day (1/60/1440); daily is better than the required weekly cadence.

Kalshi sports only launched Jan 2025, so 2025 is the earliest possible season regardless; and its settled-season history appears purged/hidden from the unauthenticated API.

## 2. Polymarket — GOOD (2 full seasons, hourly cadence, free)

Gamma search: `https://gamma-api.polymarket.com/public-search?q=...&limit_per_type=10` — no auth. Verbatim event inventory (slug | closed | startDate | volume USD):

```
# q=MLB MVP / 2025 MLB MVP
mlb-2026-nl-mvp            | False | 2026-02-19 | 1372549
mlb-al-mvp                 | True  | 2025-03-25 | 672058
mlb-nl-mvp                 | True  | 2025-03-25 | 557186
# q=MLB Cy Young
mlb-2026-al-cy-young-winner | False | 2026-02-19 | 1285635
mlb-2026-nl-cy-young-winner | False | 2026-02-19 | 1104343
mlb-al-cy-young            | True  | 2025-03-25 | 217532
mlb-nl-cy-young            | True  | 2025-04-01 | 158592
# q=MLB Rookie of the Year
mlb-nl-rookie-of-the-year  | False | 2026-03-26 | 1020429
mlb-al-rookie-of-the-year  | False | 2026-03-26 | 1654358
al-rookie-of-the-year      | True  | 2025-05-06 | 295465
nl-rookie-of-the-year      | True  | 2025-05-06 | 204493
```

No 2024 award markets exist (searches for `2024 MVP` / `2024 Cy Young` / `2024 rookie` return nothing MLB; earliest baseball futures on the platform are one-off 2020/2022 WS markets). **Polymarket coverage ceiling = 2025 season onward — exactly 2 seasons today.**

Each event's nested markets expose `clobTokenIds` (verified via `GET /events?slug=mlb-al-mvp` → 26 markets, e.g. "Will Aaron Judge win the 2025 AL MVP?" → token `39626864953731...`).

History: `https://clob.polymarket.com/prices-history?market=<clobTokenId>&startTs=...&endTs=...&fidelity=<minutes>` — no auth. **Works for closed 2025 markets** (verbatim, Judge 2025 AL MVP, daily fidelity, June 2025):

```json
{"history":[{"t":1748736007,"p":0.845},{"t":1748822406,"p":0.835},{"t":1748908807,"p":0.82},
{"t":1748995206,"p":0.88},{"t":1749081606,"p":0.85},{"t":1749168006,"p":0.885},{"t":1749254407,"p":0.835}]}
```

Window limits (all verbatim HTTP results, Judge token):
- `interval=all` or `interval=max` → 200 but **only ~1 month of points** (live Ohtani market: 4457 pts, 2026-08-19→09-19) — NOT full history.
- 7 days @ fidelity=60 → 200, 168 hourly points. 14 days @ fidelity=60 → 200, 335 points.
- 25 days @ fidelity=60, 20/30 days @ fidelity=1440, 7 months @ any fidelity → 400 `{"error":"invalid filters: 'startTs' and 'endTs' interval is too long"}`.

→ Collector must chunk into ≤14-day windows (~13 requests per market per season at hourly fidelity; ~14 at daily). Trivial.

## 3. The Odds API — FAIL for history (confirming the paid-wall assumption)

Docs (`https://the-odds-api.com/liveapi/guides/v4/`), verbatim:
- `GET /v4/historical/sports/{sport}/odds`: "Historical odds data is available from June 6th 2020... **This endpoint is only available on paid usage plans.**" (same sentence on historical events + historical event odds). Cost 10 credits/region/market even when paid.
- Free tier = current `/v4/sports/{sport}/odds` only; futures supported via `markets=outrights`, but **no dated history at any free level**. Requires an API key even for free tier. Nothing for T3 here.

## 4. Wayback Machine — FAIL this session (rate-limited, as warned)

```
cdx oddschecker.com/baseball/mlb*  → HTTP 429
cdx vegasinsider.com/mlb/odds/futures* → HTTP 429
```
archive.org 429'd both CDX queries immediately. Recorded per instructions; retry is moot given §2 fully satisfies the requirement.

## 5. Do liquid MLB award-futures markets exist at all? — Yes, post-2024

MVP/Cy Young/ROY futures are listed by major sportsbooks (oddschecker/vegasinsider carry the pages; The Odds API supports `outrights`), but **no sportsbook source offers free dated history**. ROY is the thinnest book market. The free, dated, programmatic liquidity is on prediction markets: 2025 Polymarket award volumes were $158k–$672k per event; 2026 is $1.0M–$1.7M (AL ROY $1.65M). Kalshi 2026 AL MVP favorite alone shows 460k contracts traded. Sufficient for a weekly odds-delta signal.

## Coverage table

| Source | Seasons | Cadence | Free & programmatic | Verdict |
|---|---|---|---|---|
| Polymarket `prices-history` | 2025 (closed) + 2026 (live) | hourly (1-min possible in short windows); chunked ≤14d pulls | yes, no auth | **GOOD — primary** |
| Kalshi `candlesticks` | 2026 only (2025 settled markets not retrievable unauthenticated) | daily (min/hr also), back to market open | yes, no auth | PARTIAL — secondary/forward-looking |
| The Odds API | current only (history paid-only) | n/a | free tier: current odds w/ key | FAIL |
| Wayback CDX | unknown | — | 429 rate-limited | FAIL (this session) |
| Sportsbook ROY futures | thin market, no free dated API | — | — | FAIL |

## Recommendation (for T3)

- **Build the odds-delta collector on Polymarket**: enumerate award events via `gamma-api.polymarket.com/public-search` (slugs: `mlb-al-mvp`, `mlb-nl-mvp`, `mlb-al-cy-young`, `mlb-nl-cy-young`, `al-rookie-of-the-year`, `nl-rookie-of-the-year`, + `mlb-2026-*` variants), pull `clob.polymarket.com/prices-history` per candidate token in 14-day windows at fidelity=60, resample to weekly.
- **Add Kalshi daily candlesticks as a second source for the live season** (free, includes volume + open interest; settled-season gap noted). Re-check after the 2026 awards settle whether settled markets remain readable — if yes, Kalshi becomes a full-history source too.
- 2-season ceiling is a platform reality (Polymarket MLB awards began 2025; Kalshi sports began Jan 2025) — no free source yields 2024 or earlier award-futures odds. If T3 needs >2 seasons, fallback stays rankings/prediction-markets-only regardless of collector effort.
