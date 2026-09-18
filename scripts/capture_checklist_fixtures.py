"""One-off: capture SCP/CC checklist pages as committed test fixtures.

Run from repo root:
    PLAYWRIGHT_BROWSERS_PATH=.pw-browsers python scripts/capture_checklist_fixtures.py

This is the only checklist script allowed on the network; all tests are offline
against the committed fixtures. Reuses the spike's verified captures from
/tmp/scp_probe_html when present (content markers checked); otherwise fetches
live via cardprice.web.fetch_page with >= 6 s politeness gaps.
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

OUT = Path("tests/fixtures/scp_checklists")
SPIKE = Path("/tmp/scp_probe_html")
SLEEP_S = 6.0
SCP = "https://www.sportscardspro.com"

# fixture name -> (url, content markers that must be present)
PAGES = {
    "brand_bowman.html": (
        f"{SCP}/brand/baseball-cards/bowman",
        [
            "/console/baseball-cards-2015-bowman-chrome-prospect-autograph",
            "/console/baseball-cards-2019-bowman-draft-chrome-picks-autograph",
        ],
    ),
    "2015_chrome_auto.html": (
        (
            f"{SCP}/console/baseball-cards-2015-bowman-chrome-prospect-autograph"
            "?exclude-variants=true"
        ),
        ["cody-bellinger-bcap-cbe", "games_table"],
    ),
    "2019_chrome_auto.html": (
        (
            f"{SCP}/console/baseball-cards-2019-bowman-chrome-prospects-autographs"
            "?exclude-variants=true"
        ),
        ["/game/baseball-cards-2019-bowman-chrome-prospects-autographs/", "games_table"],
    ),
    "2019_draft_auto_p1.html": (
        f"{SCP}/console/baseball-cards-2019-bowman-draft-chrome-picks-autograph",
        ["gunnar-henderson-cda-gh"],
    ),
    "2015_cc.html": (
        "https://www.cardboardconnection.com/2015-bowman-chrome-baseball-cards",
        ["BCAP-CBE Cody Bellinger"],
    ),
    "2024_chrome_auto_p1.html": (
        (
            f"{SCP}/console/baseball-cards-2024-bowman-chrome-prospects-autograph"
            "?exclude-variants=true"
        ),
        ["js-next-page", "games_table"],
    ),
}


def _spike_name(url: str) -> str:
    """Reproduce the spike's capture filename for a URL (non-alnum -> '_')."""
    slug = re.sub(r"[^a-z0-9]+", "_", url.replace("https://", "").lower()).strip("_")
    return slug + ".html"


def _verify(html: str, markers: list[str], name: str) -> None:
    missing = [m for m in markers if m not in html]
    if missing:
        raise SystemExit(f"MARKER CHECK FAILED for {name}: {missing} — bad fixture, not saved")


def main() -> None:
    from cardprice.web import fetch_page  # deferred: network only when run

    OUT.mkdir(parents=True, exist_ok=True)
    for name, (url, markers) in PAGES.items():
        reuse = SPIKE / _spike_name(url)
        if reuse.exists():
            html = reuse.read_text()
            source = f"reused {reuse}"
        else:
            time.sleep(SLEEP_S)
            html = fetch_page(url)
            source = f"fetched {url}"
        _verify(html, markers, name)
        (OUT / name).write_text(html)
        print(f"wrote {OUT / name} ({len(html)} bytes, {source})")


if __name__ == "__main__":
    main()
