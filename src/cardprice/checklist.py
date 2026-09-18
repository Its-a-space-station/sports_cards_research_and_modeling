# src/cardprice/checklist.py
"""SCP set-page (checklist) parsing: 1st Bowman auto checklists and base-prospect
lists. Pure functions — network lives in scripts/scp_pages.py. All card URLs come
from real anchors (the no-guess rule); slugs are discovered from the brand page,
never constructed.
"""

import re

import pandas as pd
from bs4 import BeautifulSoup

SCP_BASE = "https://www.sportscardspro.com"
COLUMNS = ["product_id", "name", "number", "card_url", "raw_title"]
TITLE_RE = re.compile(r"^(?P<name>.+?)\s+#(?P<number>\S+)$")
CC_LINE_RE = r"{prefix}-(?P<code>[A-Z0-9]+)\s+(?P<name>.+?)\s+-\s+"

# Canonical 1st-Bowman set shapes, verified against the live brand page
# (2026-09-17): exactly one auto/base set slug per family per year 2015-2025.
# The brand page lists dozens of OTHER autograph inserts per year (sterling,
# inception, sapphire, rookie, mega-box, ...); those classify as "other".
CHROME_AUTO_RE = re.compile(r"^baseball-cards-\d{4}-bowman-chrome-prospects?-autographs?$")
DRAFT_AUTO_RE = re.compile(
    r"^baseball-cards-\d{4}-bowman-(chrome-draft-pick-autograph"
    r"|draft-chrome-picks?-autographs?"
    r"|draft-picks?-chrome-autographs"
    r"|draft-chrome-autographs"
    r"|draft-chrome-prospect-autographs?)$"
)
CHROME_BASE_RE = re.compile(r"^baseball-cards-\d{4}-bowman-chrome-prospects$")
DRAFT_BASE_RE = re.compile(r"^baseball-cards-\d{4}-bowman-draft-chrome$")


def parse_set_page(html: str) -> pd.DataFrame:
    """Card rows of `table#games_table`; parallel rows (title contains '[') dropped."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("table#games_table tr[id^='product-']"):
        a = tr.select_one("td.title a[href^='/game/']")
        if a is None:
            continue
        title = a.get_text(strip=True)
        if "[" in title:
            continue
        m = TITLE_RE.match(title)
        if not m:
            continue
        rows.append(
            {
                "product_id": int(tr["id"].split("-")[1]),
                "name": m["name"],
                "number": m["number"],
                "card_url": SCP_BASE + a["href"],
                "raw_title": title,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def discover_set_slugs(brand_html: str) -> pd.DataFrame:
    """Classify Bowman set anchors from the brand page by year/family/kind."""
    soup = BeautifulSoup(brand_html, "html.parser")
    rows = []
    for a in soup.select("a[href^='/console/baseball-cards-']"):
        slug = a["href"].split("/console/")[1].strip("/")
        ym = re.search(r"baseball-cards-(\d{4})", slug)
        if not ym:
            continue
        family = "draft" if "draft" in slug else "chrome"
        if CHROME_AUTO_RE.match(slug) or DRAFT_AUTO_RE.match(slug):
            kind = "auto"
        elif CHROME_BASE_RE.match(slug) or DRAFT_BASE_RE.match(slug):
            kind = "base"
        else:
            kind = "other"
        rows.append(
            {
                "year": int(ym.group(1)), "family": family, "kind": kind,
                "slug": slug, "url": f"{SCP_BASE}/console/{slug}",
            }
        )
    out = pd.DataFrame(rows, columns=["year", "family", "kind", "slug", "url"])
    return out.drop_duplicates(subset=["slug"]).reset_index(drop=True)


def dedupe_checklist(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop exact (name, number) duplicates (keep lowest product_id — SCP lists
    spelling variants); KEEP same-number-different-name collisions, audited."""
    norm = rows.assign(_n=rows["name"].str.lower().str.strip(), _c=rows["number"].str.lower())
    exact_dup = norm.duplicated(subset=["_n", "_c"], keep=False)
    keep = norm[~exact_dup]
    dupes = norm[exact_dup].sort_values("product_id").drop_duplicates(subset=["_n", "_c"])
    deduped = pd.concat([keep, dupes]).sort_values("product_id").drop(columns=["_n", "_c"])
    counts = deduped.assign(_c=deduped["number"].str.lower()).groupby("_c")["name"].nunique()
    colliding = counts[counts > 1].index
    audit = deduped[deduped["number"].str.lower().isin(colliding)].copy()
    return deduped.reset_index(drop=True), audit.reset_index(drop=True)


def cross_audit_cc(cc_html: str, scp_rows: pd.DataFrame, prefix: str) -> dict:
    """Compare an SCP checklist against a Cardboard Connection page (CC is the
    authoritative numbering reference; SCP is the price-linked superset)."""
    soup = BeautifulSoup(cc_html, "html.parser")
    found = re.findall(CC_LINE_RE.format(prefix=re.escape(prefix)), soup.get_text(" "))
    # CC lists BCAP-MC twice (Conforto/Castro collision): total counts lines,
    # set logic below dedupes.
    cc = {f"{prefix}-{code}" for code, _ in found}
    scp = set(scp_rows["number"])
    return {
        "cc_total": len(found),
        "scp_total": len(scp),
        "cc_only": sorted(cc - scp),
        "scp_only": sorted(scp - cc),
        "ok": not (cc - scp),
    }


def needs_pagination(html: str) -> bool:
    return "js-next-page" in html
