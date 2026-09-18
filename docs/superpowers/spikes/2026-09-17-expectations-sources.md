# Spike: Expectations data sources (2026-09-17)

**Verdict: PARTIAL — Pipeline Top-100 GOOD (free, dated, plain-HTTP); Steamer/ZiPS historical POOR (paywalled + Cloudflare); BA Top-100 POOR (paywall + CF); FG draft boards PARTIAL (dated, draft classes only).**

Consequence (spec amendment 2026-09-17, user-approved): expectations term = Pipeline rank (pre-debut) + in-house Marcel projections (post-debut); BA dropped; FG membership recorded as optional paid upgrade.

## 1. FanGraphs preseason projections (Steamer/ZiPS) — POOR

- Data exists for every season 2015–2026 (FG advertises "Historical Projections — Members Exclusive": ZiPS 2010+, Steamer 2012+), hitters + pitchers, full rate lines.
- **Paywall:** historical seasons are members-only; the free `/api/projections` endpoint is current-season only (verified via `baseballr` source).
- **Cloudflare:** `projections?type=steamer&season=2021` and `/api/projections?...season=2015` both return HTTP 403 `cf-mitigated: challenge` to plain HTTP and to a rendering proxy. Needs authenticated Playwright + membership.
- No server-side CSV; JSON API only.

## 2. MLB Pipeline Top-100, 2015–2025 — GOOD (two free routes verified)

- **Route A (2020–2025):** `https://www.mlb.com/milb/prospects/YYYY/top100/` — server-rendered, fully extractable table. Fetch-verified for 2023 (1. Henderson, 2. Carroll, 3. Alvarez, 4. Walker, 5. Volpe…). mlb.com itself links year lists for 2020–2025. Caveat: Akamai intermittently 403s bot-looking fetches (2020/2025 attempts 403'd in the same session 2023 succeeded) → retry/backoff.
- **Route B (2015–2019):** mlb.com news articles with complete ranked lists. Fetch-verified 2015: `mlb.com/news/2015-top-100-mlb-prospects-list-c301609384` (full 1–100, rank/name/position/team, explicitly preseason). 2016–2019 unveil/full-list articles confirmed to exist (content IDs unpredictable — resolve via search per year). Article display dates unreliable (republish dates) — trust in-body dateline.
- **Route C (Wayback):** UNVERIFIED — archive.org returned HTTP 429 to all endpoints for the entire ~30-minute probe window. Retry templates recorded in the probe; not on the critical path given Routes A/B.

Per-year coverage:

| Year | Source | Status |
|---|---|---|
| 2015 | news article c301609384 | ✅ fetch-verified |
| 2016–2018 | unveil/full-list articles | ✅ existence verified |
| 2019 | article c303077544 | ✅ existence verified |
| 2020–2022, 2024–2025 | `/milb/prospects/YYYY/top100/` | ⚠️ exists, fetch flaky (Akamai) |
| 2023 | `/milb/prospects/2023/top100/` | ✅ fetch-verified, full SSR table |

## 3. Baseball America Top 100 — POOR

- `baseballamerica.com/rankings/YYYY-top-100-prospects/` resolves but is Cloudflare-challenged; the one passing fetch returned stub content — list table is subscriber-gated. No free programmatic route. Third-party reposts exist but are non-canonical. **Dropped from the design.**

## 4. FanGraphs "The Board" — PARTIAL

- `/prospects/the-board` current-state only (no dated archive of the pro board).
- **Dated MLB Draft boards exist**: `/prospects/the-board/YYYY-mlb-draft` — 2023 fetch-verified (181 players, rank/FV/school + cross-referenced Top-100 column). Useful as draft-class expectations, not preseason pro rankings. CF intermittency → retry.

## 5. Fallbacks — not needed

- Wikipedia has no per-year list pages (opensearch verified). Baseball-Reference 403s everything and carries no rankings. Route A/B fully reconstructs preseason Pipeline membership 2015–2025.

## Recommendation (adopted)

- Primary: Pipeline Routes A/B (free, dated, plain-HTTP, retry through Akamai 403s).
- Secondary: FG dated draft boards for draft classes.
- Post-debut expectations: in-house Marcel-style lagged projections from own season stats (5/4/3 weights, mean regression).
- Paid upgrade recorded, not purchased: FG membership for true Steamer/ZiPS historical.
- Open item: re-run Wayback CDX queries when archive.org rate-limiting clears (secondary assurance only).
