# src/cardprice/web.py
"""Browser-based page fetching via Playwright (tolerates Cloudflare auto-challenges)."""

import time

from playwright.sync_api import sync_playwright

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class ChallengeError(RuntimeError):
    """Bot challenge did not resolve after retries."""


def is_challenge_title(title: str) -> bool:
    t = title.lower()
    return "just a moment" in t or "security verification" in t


def fetch_page(url: str, wait_ms: int = 8000, retries: int = 3) -> str:
    """Return fully-rendered HTML for url. Raises ChallengeError if the
    bot challenge never resolves."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            ctx = browser.new_context(
                user_agent=USER_AGENT, viewport={"width": 1400, "height": 900}
            )
            page = ctx.new_page()
            for _ in range(retries):
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(wait_ms)
                if not is_challenge_title(page.title()):
                    return page.content()
                time.sleep(5)
            raise ChallengeError(url)
        finally:
            browser.close()
