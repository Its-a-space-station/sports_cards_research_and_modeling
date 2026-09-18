# scripts/build_class_checklists.py
"""Build 1st Bowman auto checklists (Chrome + Draft, 2015-2025) from SCP set pages.

Slugs discovered from the brand page (no-guess rule); raw snapshots saved under
data/raw/scp_setpages/<slug>/<date>.json; cross-audit vs Cardboard Connection
for 2015 Chrome (spike-verified CC subset of SCP).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scp_pages import fetch_set_pages

from cardprice.checklist import (
    cross_audit_cc,
    dedupe_checklist,
    discover_set_slugs,
    parse_set_page,
)
from cardprice.storage import save_raw
from cardprice.web import fetch_page

YEARS = range(2015, 2026)
BRAND_URL = "https://www.sportscardspro.com/brand/baseball-cards/bowman"
CC_2015_URL = "https://www.cardboardconnection.com/2015-bowman-chrome-baseball-cards"
OUT = Path("data/reference/class_checklists.parquet")
AUDIT = Path("data/reference/class_checklist_audit.csv")


def main() -> None:
    brand = fetch_page(BRAND_URL)
    save_raw("scp_setpages", "brand_bowman", {"url": BRAND_URL, "html": brand})
    sets = discover_set_slugs(brand)
    wanted = sets[(sets["year"].isin(YEARS)) & (sets["kind"] == "auto")]
    missing = sorted(set(YEARS) - set(wanted[wanted["family"] == "chrome"]["year"]))
    missing_d = sorted(set(YEARS) - set(wanted[wanted["family"] == "draft"]["year"]))
    print("auto set pages found:", len(wanted), "| missing chrome years:", missing,
          "| missing draft years:", missing_d)

    frames, audits = [], []
    for s in wanted.itertuples():
        pages = fetch_set_pages(f"{s.url}?exclude-variants=true")
        rows = pd.concat([parse_set_page(h) for h in pages])
        deduped, collisions = dedupe_checklist(rows)
        deduped = deduped.assign(year=s.year, family=s.family)
        frames.append(deduped)
        if len(collisions):
            audits.append(collisions.assign(year=s.year, family=s.family, kind="number_collision"))
        print(f"{s.year} {s.family}: {len(deduped)} autos ({len(pages)} page(s), "
              f"{len(collisions)} collision rows)")
    checklists = pd.concat(frames)[["year", "family", "name", "number", "product_id", "card_url"]]
    checklists.to_parquet(OUT, index=False)

    # cross-audit 2015 chrome vs Cardboard Connection (spike-verified relationship)
    cc = fetch_page(CC_2015_URL)
    save_raw("scp_setpages", "cc_2015_bowman_chrome", {"url": CC_2015_URL, "html": cc})
    scp2015 = checklists[(checklists["year"] == 2015) & (checklists["family"] == "chrome")]
    res = cross_audit_cc(cc, scp2015, prefix="BCAP")
    print("cross-audit 2015 chrome vs CC:", res)
    if not res["ok"]:
        audits.append(pd.DataFrame([{"kind": "cross_audit_fail", **res}]))
    if audits:
        pd.concat(audits).to_csv(AUDIT, index=False)
    print("wrote", OUT, len(checklists), "rows")


if __name__ == "__main__":
    main()
