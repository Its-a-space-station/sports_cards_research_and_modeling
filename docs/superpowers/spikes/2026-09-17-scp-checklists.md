# Spike: SCP 1st Bowman Chrome Prospect Auto checklists (2026-09-17)

**Verdict: FEASIBLE.** Full Bowman Chrome Prospect Autograph checklists (player
name + card number + card URL) are extractable from sportscardspro.com, in most
years with **one GET per set** (`?exclude-variants=true`), falling back to the
known cursor-POST pagination for sets with >150 base entries. Cross-audit
against cardboardconnection.com confirms completeness for 2015.

Evidence: ~25 live fetches 2026-09-17 via headless Playwright (repo conventions:
`src/cardprice/web.py` UA, 8 s settle wait, 3× challenge retry — never needed,
every page resolved on first wait — ≥6 s politeness gaps). Raw HTML + extracted
JSON checklists under `/tmp/scp_probe_html/` (temp; regenerate on demand).

## 1. URL patterns (verified)

- Set page: `https://www.sportscardspro.com/console/<set-slug>` (the `/game/<slug>`
  form without a card name soft-404s; `/category/...` does not exist — consistent
  with `2026-09-14-scp-structure.md`).
- **Insert sets are their own set pages**, including the Prospect Autographs.
  Slugs are **not consistent year to year** — do not guess, discover via the brand
  page `https://www.sportscardspro.com/brand/baseball-cards/bowman` (single GET,
  lists every Bowman set as `/console/...` anchors). Verified slugs:

  | year | set | slug |
  |------|-----|------|
  | 2015 | BC Prospect Autograph | `baseball-cards-2015-bowman-chrome-prospect-autograph` |
  | 2019 | BC Prospects Autographs | `baseball-cards-2019-bowman-chrome-prospects-autographs` |
  | 2024 | BC Prospects Autograph  | `baseball-cards-2024-bowman-chrome-prospects-autograph` |
  | 2015 | BC Prospects (base)     | `baseball-cards-2015-bowman-chrome-prospects` |
  | 2019 | BC Prospects (base)     | `baseball-cards-2019-bowman-chrome-prospects` |
  | 2019 | Draft Chrome Picks Auto | `baseball-cards-2019-bowman-draft-chrome-picks-autograph` |

- Card page: `/game/<set-slug>/<player-slug>-<number-slug>`, e.g.
  `/game/baseball-cards-2015-bowman-chrome-prospect-autograph/cody-bellinger-bcap-cbe`.

## 2. Extraction recipe (production recommendation)

1. GET `/console/<slug>?exclude-variants=true` with `cardprice.web.fetch_page`.
2. Parse `table#games_table` rows; each card title anchor looks like
   `<a href="/game/...">Cody Bellinger #BCAP-CBE</a>`. Parallel rows carry a
   bracket (`Aaron Brown [Gold Refractor] #BCAP-ABR`) — with
   `exclude-variants=true` they are absent; if parsing a full page, drop any
   title containing `[`.
3. If the page still contains `<form ... class="next_page js-next-page">`
   (sets with >150 base entries), paginate: POST to the **same URL including the
   query string**, form body `sort=&when=none&release-date=<tomorrow>&cursor=N`,
   N += 150 per page, until the form disappears. The POST returns the full HTML
   page for that cursor; works via in-page `fetch()` inside the Playwright
   session (shares cookies/CF clearance). `web.py`'s `fetch_page` is GET-only —
   a small POST-capable helper is needed for this path.

Verified results:

| set | entries | unique #s | requests needed | complete? |
|-----|---------|-----------|-----------------|-----------|
| 2015 BC Prospect Autograph | 81 | 78 | 1 GET | yes (`more=False`), incl. **Cody Bellinger #BCAP-CBE** |
| 2019 BC Prospects Autographs | 69 | 67 | 1 GET | yes |
| 2024 BC Prospects Autograph | 228 | 208 | 2 pages (GET ev + 1 POST) | yes — full cursor walk (39 pages without `ev`) ended `more=False` at cursor 5700 |
| 2019 BC Prospects (base) | 251 | 251 (249 BCP + 2 plain #) | 2 pages | yes |
| 2019 Draft Chrome Picks Auto | page 1 only | CDA prefix confirmed | — | not fully pulled |

Without `exclude-variants=true`, pages mix ~10-40 parallel rows per player
(150 rows/page), so the cursor walk costs e.g. 39 requests for 2024 — always use
`exclude-variants=true`.

## 3. Parsing notes / data caveats

- Row shape (verbatim, 2015 autos page 1):

  ```html
  <tr id="product-1450786" data-product="1450786"> <td class="image"> ... </td>
  <td class="title" title="1450786"> <a href="/game/baseball-cards-2015-bowman-chrome-prospect-autograph/aaron-brown-bcap-abr">Aaron Brown #BCAP-ABR</a> </td>
  <td class="price numeric used_price"> <span class="js-price">$1.05</span> </td> ...
  ```

  `id="product-<n>"` gives SCP's numeric product id; the same page also exposes
  per-parallel print runs as `title="250 copies printed"` etc.
- Numbering scheme by year: 2015 = `BCAP-XX`; 2019 & 2024 = `CPA-XX`; draft-pick
  autos (separate Bowman Draft sets) = `CDA-XX`; base chrome prospects = `BCP-N`.
- **SCP has data quirks — dedupe and cross-audit:**
  - Same number listed for two players (2024: 20 collisions, e.g.
    `Paul Skenes #CPA-PS` *and* `Paulino Santana #CPA-PS`,
    `Agustin Ramirez #CPA-AR` *and* `Adriel Radney #CPA-AR`; 2015:
    `BCAP-DS` Dansby Swanson/Darnell Sweeney, `BCAP-MC` Conforto/Castro).
  - Duplicate same-player rows under spelling variants: `Corelle Prime #BCAP-CP` /
    `Correlle Prime #BCAP-CPR`; `Wilmer Difo #BCAP-WD` / `Wimer Difo #BCAP-WD`;
    `Ozhaino Albies #BCAP-DAL` (2015, likely mis-tag).
  - Consequence: 2015 yields 81 rows / 78 unique numbers; treat SCP as generous
    (superset) and reconcile against Cardboard Connection.
- **"1st Bowman" is NOT flagged per card** on set pages (rows carry only name,
  number, price). Derivation must come from earliest-year appearance across sets
  or another source.

## 4. Cross-audit source: cardboardconnection.com (verified)

- `https://www.cardboardconnection.com/2015-bowman-chrome-baseball-cards` fetched
  clean (no challenge) with the same Playwright setup, 301 KB, full checklist in
  static HTML: lines of the form `BCAP-CBE Cody Bellinger - Los Angeles Dodgers`.
- 79 unique `BCAP-` codes on the page (75 matched by a strict
  `BCAP-XX Name - Team` regex). Set comparison: **CC ⊂ SCP** (`CC-not-SCP = ∅`),
  i.e. SCP's 2015 extraction is complete; SCP's 3-6 extras are the quirks above.
- CC pages are per-set (all inserts on one page), no pagination — good as the
  authoritative numbering cross-check; SCP remains better for price linkage
  (same product ids the price pipeline uses).

## 5. Per-class caveat: Bowman Chrome vs Bowman Draft (Witt correction)

- The probe assumption "Bobby Witt Jr. auto in 2019 Bowman Chrome" is **wrong**:
  the full 69-entry 2019 BC Prospects Autographs checklist has no Witt, and the
  full 251-entry 2019 BC Prospects (base) has no Witt either —
  **`#BCP-25` is Fernando Tatis Jr.**, not Witt. Witt (2019 draft) first appears
  in **2019 Bowman Draft**: autos in
  `baseball-cards-2019-bowman-draft-chrome-picks-autograph` (`CDA-` numbering,
  page 1 shows `Gunnar Henderson #CDA-GH`), base chrome in the
  `...-2019-bowman-draft-chrome*` family.
- General rule for classes 2015-2025: a player's 1st Bowman Chrome (auto or base)
  lives in *Bowman Chrome* (spring release, CPA/BCAP/BCP) if they were already in
  pro ball, else in *Bowman Draft* (fall release, CDA/BDC). Checklist extraction
  must cover both product families per class year.
- Task 5 answer: yes — base non-auto 1st Bowman Chrome Prospects are listed with
  `BCP-N` numbers on the `...-prospects` set pages (e.g. `Julio Rodriguez #BCP-33`,
  `Wander Franco #BCP-100`); existence per player is derivable, with the
  Chrome-vs-Draft caveat above.

## 6. Risks / effort

- Cloudflare: zero challenges observed across ~25 fetches with the repo's
  standard settings; the earlier spike's finding that only `/search-products`
  hard-blocks headless still holds (a `/search?q=` attempt returned a 170-byte
  404 — search is unusable, use brand-page discovery instead).
- Effort per class year: 1 brand-page GET (cached for all years) + 1-2 GETs per
  auto set + 1-2 GETs per base prospects set (+ same for the Draft family) ≈
  **≤6 requests per class year**, well within politeness norms.
- Remaining unknowns: slugs for 2016-2018, 2020-2023, 2025 not individually
  probed (pattern + brand-page discovery makes this mechanical); whether very
  recent years (2025) list cards before release was not checked.
