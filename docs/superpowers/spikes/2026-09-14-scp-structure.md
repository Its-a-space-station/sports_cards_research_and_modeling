# Spike: SportsCardsPro page structure (2026-09-14)

Evidence source: `tests/fixtures/scp/henderson_2023_topps_chrome.html` (652,338 bytes,
fetched 2026-09-15 via `cardprice.web.fetch_page` with `PLAYWRIGHT_BROWSERS_PATH=.pw-browsers`)
plus live probes of console/category/search/sitemap endpoints the same day.

## (a) Dual-price cell semantics: first span = ACCEPTED, second span = LIST

A sold listing that went through eBay Best Offer renders two prices in the price
cell (`td.numeric`, the 4th `td`). In DOM order:

1. **first** `<span class="js-price" title="best offer accepted price">` — the price
   actually paid (accepted offer). This is the `price` the pipeline must store.
2. **second** `<span class="js-price listed-price-inline" title="best offer list price">`
   — the original list price (rendered struck-through/inline). Store as `list_price`.

A separate 5th cell `<td class="numeric listed-price">` repeats the list price as its
own column. Single-price rows have one bare `<span class="js-price">$7.50</span>`.

Verbatim fixture row (eBay item 157186956574, sale date 2026-03-17, TAG 10 — the
accepted $35.00 / list $40.00 row referenced by the Task 2 test):

```html
<tr id="ebay-157186956574">
    <td class="date">2026-03-17</td>
    <td class="image"> ... </td>
    <td class="title">
        <a target="_blank" class="js-ebay-completed-sale" href="https://www.ebay.it/itm/157186956574?...">
            2023 TOPPS CHROME  #2  GUNNAR HENDERSON  RC  TAG 10 MT BALTIMORE ORIOLES #2</a>
        [eBay]
    </td>
    <td class="numeric">
        <span class="js-price" title="best offer accepted price">$35.00</span>
        <br>
        <span class="js-price listed-price-inline" title="best offer list price">$40.00</span>
    </td>
    <td class="numeric listed-price">
        <span class="js-price" title="best offer list price">$40.00</span>
    </td>
    ...
</tr>
```

All 8 dual-price rows in the fixture follow this exact pattern (accepted $45/list $60,
accepted $30/list $50, accepted $40/list $44.99, accepted $38/list $49.99,
accepted $37.50/list $44.99, accepted $24/list $29.99, accepted $30/list $34.99,
accepted $35/list $40). **Correction to the plan:** Task 2's sketched `_parse_price_cell`
assumes "first is list, second is accepted" — that is inverted. The fixture shows
accepted-first, list-second; parsers should key on the span `title`/`listed-price-inline`
class rather than position. The Task 2 test values (list 40.00, accepted 35.00) remain
valid — only the positional interpretation changes.

## (b) `VGPC.chart_data` keys for the fixture page

Verbatim line start (one long line inside an inline `<script>`):

```
VGPC.chart_data = {"boxonly":[[1690869600000,2876],[1693548000000,2848],[1696140000000,1550],...
```

6 keys, each a monthly `[ms-epoch, price-cents]` series of 38 points spanning
2023-08-01 → 2026-09-01. Last pair per key:

| key         | n  | last [ts, cents]        | date (UTC) | $      |
|-------------|----|-------------------------|------------|--------|
| boxonly     | 38 | [1788242400000, 1400]   | 2026-09-01 | 14.00  |
| cib         | 38 | [1788242400000, 353]    | 2026-09-01 | 3.53   |
| graded      | 38 | [1788242400000, 1238]   | 2026-09-01 | 12.38  |
| manualonly  | 38 | [1788242400000, 3450]   | 2026-09-01 | 34.50  |
| new         | 38 | [1788242400000, 852]    | 2026-09-01 | 8.52   |
| used        | 38 | [1788242400000, 163]    | 2026-09-01 | 1.63   |

Note `used` last = $1.63 and `manualonly` last = $34.50, matching `table#price_data`
"Ungraded" and "PSA 10" respectively — the Task 3 cent-exact calibration is feasible.

**Bonus (contradicts the plan header's "buckets are not grade-labeled"):** the page's
tab bar + `#completed-auctions-condition` select label every `completed-auctions-*`
bucket. Verbatim from the fixture select:

```html
<option value="completed-auctions-used">Ungraded (30)</option>
<option value="completed-auctions-grade-twenty">BGS 10 Black (0)</option>
<option value="completed-auctions-grade-nineteen">CGC 10 Prist. (0)</option>
<option value="completed-auctions-manual-only">PSA 10 (30)</option>
<option value="completed-auctions-loose-and-box">BGS 10 (0)</option>
<option value="completed-auctions-grade-seventeen">CGC 10 (5)</option>
<option value="completed-auctions-grade-eighteen">SGC 10 (30)</option>
<option value="completed-auctions-grade-twenty-one">TAG 10 (2)</option>
<option value="completed-auctions-grade-twenty-two">ACE 10 (0)</option>
<option value="completed-auctions-box-only">Grade 9.5 (30)</option>
<option value="completed-auctions-graded">Grade 9 (30)</option>
<option value="completed-auctions-new">Grade 8 (30)</option>
<option value="completed-auctions-cib">Grade 7 (4)</option>
```

This mapping is per-page (bucket names vary by which grade buckets have sales), so
parsing the select gives a dynamic bucket→grade cross-check against Task 3's
price-match calibration. Grade-per-sale should still come from the listing title
(the buckets mix graders at "Grade 9" granularity for raw-grade buckets).

## (c) Category listing URL scheme

- `https://www.sportscardspro.com/category/<set-slug>` **does not exist**: soft-404
  with empty `<title></title>` and body `<div id="not_found" class="market">\n
  <h1>Page Not Found</h1>` ("We couldn't find the page you were looking for.").
  The plan's category URL is wrong.
- **Working scheme: `https://www.sportscardspro.com/console/<set-slug>`**, confirmed
  by the card page's own breadcrumb (verbatim):

  ```html
  <a href="/brand/baseball-cards/topps" style="text-transform: capitalize;">topps</a> &gt;
  <a href="/console/baseball-cards-2023-topps-chrome">2023 Topps Chrome</a>
  ```

- The default console view is sorted by sales volume and capped (~150 anchors ≈ 75
  unique cards for 2022 Topps Chrome); further pages require a POST form
  (`<form method="POST" action="" class="next_page js-next-page">` with hidden
  `cursor=150`), so plain GET pagination params do not exist.
- **Complete rookie base-card listing in one GET** (no cursor): append
  `?rookies-only=true&exclude-variants=true`. Observed counts: 2022 Topps Chrome → 58
  cards, 2024 → 120, 2025 → 100. This is what `scripts/resolve_seed_cards.py` uses.
- Search: `/search?q=...` hard-404s (170-byte "404 page not found"). The real endpoint
  is `/search-products?q=...` (verbatim form: `<form method="GET" action="/search-products">`
  with `<input id="game_search_box" name="q">`), but its Cloudflare challenge does not
  auto-resolve headless — see (d).
- `/sitemap.xml` fetches unchallenged (~57 MB, 74,280 `<loc>`) but indexes only
  `/console/`, `/user/`, `/brand/` etc. — **no `/game/` card URLs**; useless for
  card-level discovery.

## (d) Challenge wait/retry behavior (confirmed)

- Card pages, console pages, `/robots.txt`, `/sitemap.xml`: Cloudflare's auto-challenge
  resolves headless with **one 8 s wait** — every such fetch in this spike returned the
  real page on the first `wait_for_timeout(8000)` cycle (title becomes the real page
  title, e.g. `Gunnar Henderson #2 Prices [Rookie] | 2023 Topps Chrome | Baseball Cards`).
- `/search-products?q=...`: challenge **never** resolved — three probe configurations
  all raised `ChallengeError`: (8 s × 3 retries), (12 s × 3), (20 s × 2). Treat SCP
  search as unavailable headless (same failure mode the plan recorded for 130point).
- `fetch_page` retry path itself verified by those failures: 3 attempts with 5 s pauses,
  then `ChallengeError(url)`.
- Caveat: SCP soft-404s return an **empty** `<title></title>` — not a challenge title —
  so `is_challenge_title` passes and `fetch_page` returns the 404 page as if it were
  content. Callers must validate content (e.g. expected anchors present), as
  `resolve_seed_cards.py` does by falling through to a warning when no anchor matches.
