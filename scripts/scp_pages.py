# scripts/scp_pages.py
"""Thin SCP fetch layer for set pages: GET + cursor-POST pagination (spike §2).
POSTs run as in-page fetch() inside a shared Playwright session so Cloudflare
clearance carries. web.fetch_page is GET-only — this module adds the POST path
for checklist work only.
"""

import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cardprice.checklist import needs_pagination
from cardprice.web import USER_AGENT, ChallengeError, fetch_page, is_challenge_title

CURSOR_STEP = 150

POST_JS = """
async ([url, body]) => {
  const r = await fetch(url, {method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'}, body});
  return await r.text();
}
"""


def _post_body(cursor: int) -> str:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()  # noqa: DTZ011  # SCP form date
    return f"sort=&when=none&release-date={tomorrow}&cursor={cursor}"


def fetch_set_pages(url: str, sleep_s: float = 6.0, max_pages: int = 50) -> list[str]:
    """Return [page1_html, page2_html, ...] following cursor POSTs until the
    next-page form disappears."""
    from playwright.sync_api import sync_playwright

    first = fetch_page(url)
    if not needs_pagination(first):
        return [first]
    pages = [first]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            # same UA/context + challenge retry as web.fetch_page: a bare
            # headless session draws a Cloudflare challenge and the POSTs fail
            ctx = browser.new_context(
                user_agent=USER_AGENT, viewport={"width": 1400, "height": 900}
            )
            page = ctx.new_page()
            for _ in range(3):
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(8000)
                if not is_challenge_title(page.title()):
                    break
                time.sleep(5)
            else:
                raise ChallengeError(url)
            cursor = CURSOR_STEP
            while len(pages) < max_pages:
                time.sleep(sleep_s)
                html = page.evaluate(POST_JS, [url, _post_body(cursor)])
                if not html or "games_table" not in html:
                    break
                pages.append(html)
                if not needs_pagination(html):
                    break
                cursor += CURSOR_STEP
        finally:
            browser.close()
    return pages
