# tests/test_checklist.py
from pathlib import Path

import pandas as pd

from cardprice.checklist import (
    cross_audit_cc,
    dedupe_checklist,
    discover_set_slugs,
    needs_pagination,
    parse_set_page,
)

FIX = Path(__file__).parent / "fixtures" / "scp_checklists"


def test_parse_set_page_2015_autos():
    df = parse_set_page((FIX / "2015_chrome_auto.html").read_text())
    assert len(df) == 81  # spike-verified row count (incl. quirks)
    bell = df[df["name"] == "Cody Bellinger"]
    assert len(bell) == 1
    assert bell.iloc[0]["number"] == "BCAP-CBE"
    assert bell.iloc[0]["card_url"].endswith("/cody-bellinger-bcap-cbe")
    assert "[" not in " ".join(df["raw_title"])  # exclude-variants page: no parallels


def test_parse_set_page_2019_chrome_has_no_witt():
    df = parse_set_page((FIX / "2019_chrome_auto.html").read_text())
    assert len(df) == 69  # spike-verified
    names = " ".join(df["name"])
    assert "Witt" not in names  # Witt is 2019 Bowman DRAFT, not Chrome (spike §5)


def test_parse_set_page_2019_draft_contains_henderson():
    df = parse_set_page((FIX / "2019_draft_auto_p1.html").read_text())
    assert len(df) > 0
    assert (df["number"].str.startswith("CDA-")).any()
    assert (df["name"] == "Gunnar Henderson").any()


def test_parse_set_page_drops_parallel_rows():
    html = """<table id="games_table">
    <tr id="product-1"><td class="title"><a href="/game/s/x-1">Aaron Brown #BCAP-ABR</a></td></tr>
    <tr id="product-2"><td class="title"><a href="/game/s/x-2">Aaron Brown [Gold Refractor]
    #BCAP-ABR</a></td></tr>
    </table>"""
    df = parse_set_page(html)
    assert len(df) == 1
    assert df.iloc[0]["product_id"] == 1


def test_dedupe_checklist_drops_exact_dupes_keeps_number_collisions():
    rows = pd.DataFrame(
        [
            {
                "product_id": 2, "name": "Wilmer Difo", "number": "BCAP-WD",
                "card_url": "u2", "raw_title": "t",
            },
            {
                "product_id": 1, "name": "Wimer Difo", "number": "BCAP-WD",
                "card_url": "u1", "raw_title": "t",
            },
            {
                "product_id": 3, "name": "Dansby Swanson", "number": "BCAP-DS",
                "card_url": "u3", "raw_title": "t",
            },
            {
                "product_id": 4, "name": "Darnell Sweeney", "number": "BCAP-DS",
                "card_url": "u4", "raw_title": "t",
            },
            {
                "product_id": 5, "name": "Wimer Difo", "number": "BCAP-WD",
                "card_url": "u5", "raw_title": "t",
            },
        ]
    )
    deduped, audit = dedupe_checklist(rows)
    # exact dupe (Wimer/Wimer) dropped keeping lowest product_id (1);
    # number collisions across DIFFERENT names (Wimer/Wilmer, Swanson/Sweeney) kept + audited
    assert len(deduped) == 4
    assert 5 not in set(deduped["product_id"])
    assert 1 in set(deduped["product_id"])
    assert set(audit["number"]) == {"BCAP-WD", "BCAP-DS"}


def test_cross_audit_cc_2015():
    df = parse_set_page((FIX / "2015_chrome_auto.html").read_text())
    deduped, _ = dedupe_checklist(df)
    res = cross_audit_cc((FIX / "2015_cc.html").read_text(), deduped, prefix="BCAP")
    assert res["cc_total"] >= 75  # spike: 75-79 regex-matched codes
    assert res["ok"]  # spike-verified: CC subset of SCP


def test_discover_set_slugs_classifies_families():
    df = discover_set_slugs((FIX / "brand_bowman.html").read_text())
    got = {(r.year, r.family, r.kind): r.slug for r in df.itertuples()}
    assert got[(2015, "chrome", "auto")] == "baseball-cards-2015-bowman-chrome-prospect-autograph"
    assert (
        got[(2019, "chrome", "auto")] == "baseball-cards-2019-bowman-chrome-prospects-autographs"
    )
    assert (
        got[(2024, "chrome", "auto")] == "baseball-cards-2024-bowman-chrome-prospects-autograph"
    )
    assert got[(2019, "chrome", "base")] == "baseball-cards-2019-bowman-chrome-prospects"
    assert (
        got[(2019, "draft", "auto")] == "baseball-cards-2019-bowman-draft-chrome-picks-autograph"
    )
    # every classified URL is a real anchor href, never constructed
    assert df["url"].str.startswith("https://www.sportscardspro.com/console/").all()


def test_needs_pagination_flag():
    assert needs_pagination((FIX / "2024_chrome_auto_p1.html").read_text())  # 228 entries
    assert not needs_pagination((FIX / "2015_chrome_auto.html").read_text())  # 81 entries
