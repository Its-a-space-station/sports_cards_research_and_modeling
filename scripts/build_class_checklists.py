# scripts/build_class_checklists.py
"""Build 1st Bowman auto checklists (Chrome + Draft, 2015-2025) from SCP set pages.

Slugs discovered from the brand page (no-guess rule); only the brand page and
the CC audit pages are snapshotted under data/raw/scp_setpages/ (the auto set
pages themselves are not snapshotted); cross-audit vs Cardboard Connection
for 2015 Chrome (spike-verified CC subset of SCP) and a completeness audit for
2019 Draft (spec amendment ed7a7db — SCP draft checklists may be incomplete;
STOP below 85% CDA coverage). Task 2 extension: also fetches each year/family's
kind=="base" set pages and snapshots them (snapshot only; the class-universe
runner parses them via resolve_class_universe._load_base_rows).
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
CC_2019_DRAFT_URL = "https://www.cardboardconnection.com/2019-bowman-draft-baseball-cards"
DRAFT_COMPLETENESS_FLOOR = 0.85
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

    # completeness audit 2019 draft vs Cardboard Connection (spec amendment ed7a7db:
    # SCP's draft-auto checklists may be incomplete — Witt Jr. absent from SCP's 2019
    # draft pull). CC CDA entries are the numbering reference; STOP-and-report when
    # SCP covers < 85% of them (fraction of CC lines whose code SCP also lists).
    cc19 = fetch_page(CC_2019_DRAFT_URL)
    save_raw("scp_setpages", "cc_2019_bowman_draft", {"url": CC_2019_DRAFT_URL, "html": cc19})
    scp2019d = checklists[(checklists["year"] == 2019) & (checklists["family"] == "draft")]
    res19 = cross_audit_cc(cc19, scp2019d, prefix="CDA")
    res19["completeness"] = round(
        (res19["cc_total"] - len(res19["cc_only"])) / max(res19["cc_total"], 1), 4
    )
    print("cross-audit 2019 draft vs CC:", res19)
    audits.append(pd.DataFrame([{"kind": "cross_audit_cc_2019_draft", **res19}]))
    if res19["completeness"] < DRAFT_COMPLETENESS_FLOOR:
        pd.concat(audits).to_csv(AUDIT, index=False)
        raise SystemExit(
            f"STOP: SCP 2019 draft completeness {res19['completeness']:.1%} "
            f"< {DRAFT_COMPLETENESS_FLOOR:.0%} — investigate before proceeding"
        )

    # Task 2 extension: fetch + snapshot base set pages (snapshots only, no checklist
    # output — resolve_class_universe._load_base_rows parses them). Years with no
    # base slug on the brand page are audit-recorded gaps, never guessed.
    base = sets[(sets["year"].isin(YEARS)) & (sets["kind"] == "base")]
    for family in ("chrome", "draft"):
        have = set(base[base["family"] == family]["year"])
        for y in sorted(set(YEARS) - have):
            audits.append(pd.DataFrame([{"kind": "missing_base_set", "year": y,
                                         "family": family}]))
    print("base set pages found:", len(base), "| missing base year/family rows audited")
    for s in base.itertuples():
        pages = fetch_set_pages(f"{s.url}?exclude-variants=true")
        save_raw("scp_setpages", s.slug,
                 {"url": s.url, "html": pages[0], "pages": pages, "kind": "base"})
        n_rows = sum(len(parse_set_page(h)) for h in pages)
        print(f"base {s.year} {s.family}: {n_rows} rows ({len(pages)} page(s)) -> {s.slug}")

    if audits:
        pd.concat(audits).to_csv(AUDIT, index=False)
    print("wrote", OUT, len(checklists), "rows")


if __name__ == "__main__":
    main()
