# tests/test_collect_class_prices.py
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import collect_class_prices as ccp

# NOTE: fixture models the real SCP row layout (date | image | title | price tds —
# see docs/superpowers/spikes/2026-09-14-scp-structure.md); parse_sales_tables reads
# the title from tds[2] and the price cell from tds[3].
CARD_HTML = """
<html><head><title>ok</title></head><body>
<div class="completed-auctions-used">
<table>
<tr><td class="date">2024-06-01</td><td class="image"></td>
<td class="title">2019 Bowman Draft Gunnar Henderson #BDC-99 [eBay]</td>
<td class="numeric"><span class="js-price">$5.00</span></td></tr>
<tr><td class="date">2024-06-03</td><td class="image"></td>
<td class="title">2019 Bowman Draft Gunnar Henderson #BDC-99 PSA 10 [eBay]</td>
<td class="numeric"><span class="js-price">$25.00</span></td></tr>
</table></div>
<table id="price_data"><tr><th>Ungraded</th><th>PSA 10</th></tr>
<tr><td>$5.00</td><td>$25.00</td></tr></table>
<script>VGPC.chart_data = {"used": [[1717200000000, 500]], "manualonly": [[1717200000000, 2500]]};</script>
</body></html>
"""


def _cards(n=2):
    return pd.DataFrame(
        {
            "player_name": [f"P{i}" for i in range(n)],
            "mlb_id": list(range(1, n + 1)),
            "class_year": [2019] * n,
            "card_type": ["bowman_1st_base"] * n,
            "set_slug": ["baseball-cards-2019-bowman-draft-chrome"] * n,
            "scp_url": [
                f"https://www.sportscardspro.com/game/baseball-cards-2019-bowman-draft-chrome/p{i}-bdc-{i}"
                for i in range(n)
            ],
        }
    )


def test_collect_writes_parts_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(ccp, "fetch_page", lambda url: CARD_HTML)
    monkeypatch.setattr(ccp, "save_raw", lambda *a, **k: None)
    part_root = str(tmp_path / "parts")
    res = ccp.collect_class_prices(_cards(), part_root, sleep_s=0)
    assert res["done"] == 2 and res["failed"] == []
    # parts exist even if a frame is empty (_cards() generates slugs p0-bdc-0, p1-bdc-1)
    for i in range(2):
        safe = f"baseball-cards-2019-bowman-draft-chrome__p{i}-bdc-{i}"
        assert (Path(part_root) / "sales" / f"{safe}.parquet").exists()
        assert (Path(part_root) / "chart" / f"{safe}.parquet").exists()
    # resume: fetch_page must not be called again
    monkeypatch.setattr(
        ccp, "fetch_page", lambda url: (_ for _ in ()).throw(AssertionError("refetch"))
    )
    res2 = ccp.collect_class_prices(_cards(), part_root, sleep_s=0)
    assert res2["done"] == 0 and res2["skipped"] == 2


def test_collect_records_failures_without_parts(tmp_path, monkeypatch):
    from cardprice.web import ChallengeError

    def boom(url):
        raise ChallengeError("nope")

    monkeypatch.setattr(ccp, "fetch_page", boom)
    part_root = str(tmp_path / "parts")
    res = ccp.collect_class_prices(_cards(1), part_root, sleep_s=0)
    assert res["done"] == 0 and len(res["failed"]) == 1
    assert not list((Path(part_root) / "sales").glob("*.parquet"))


def test_merge_parts_concatenates(tmp_path, monkeypatch):
    monkeypatch.setattr(ccp, "fetch_page", lambda url: CARD_HTML)
    monkeypatch.setattr(ccp, "save_raw", lambda *a, **k: None)
    part_root = str(tmp_path / "parts")
    ccp.collect_class_prices(_cards(), part_root, sleep_s=0)
    sales, chart = ccp.merge_parts(part_root)
    assert len(sales) == 4  # 2 sales rows per card x 2 cards
    assert set(sales["player_name"]) == {"P0", "P1"}
    assert set(chart["grade"]) == {"ungraded", "psa_10"}
    assert (sales["price"] > 0).all()


# Passes PAGE_MARKERS (price_data table + completed-auctions- div) but its empty
# sales table yields zero parsed rows -> parse_sales_tables raises KeyError.
POISON_HTML = """
<html><head><title>ok</title></head><body>
<div class="completed-auctions-used">
<table><tr><th>Date</th><th></th><th>Title</th><th>Price</th></tr></table>
</div>
<table id="price_data"><tr><th>Ungraded</th></tr><tr><td>$5.00</td></tr></table>
</body></html>
"""


def test_parse_failure_recorded_and_run_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ccp, "fetch_page", lambda url: POISON_HTML if url.endswith("p0-bdc-0") else CARD_HTML
    )
    monkeypatch.setattr(ccp, "save_raw", lambda *a, **k: None)
    part_root = str(tmp_path / "parts")
    res = ccp.collect_class_prices(_cards(), part_root, sleep_s=0)
    assert res["done"] == 1  # poison card did not block the tail
    assert len(res["failed"]) == 1
    slug, reason = res["failed"][0]
    assert slug == "baseball-cards-2019-bowman-draft-chrome/p0-bdc-0"
    assert reason.startswith("parse:")
    # poison card wrote no parts (retried next run); the healthy card after it did
    assert not list((Path(part_root) / "sales").glob("*p0-bdc-0.parquet"))
    ok_part = Path(part_root) / "sales" / "baseball-cards-2019-bowman-draft-chrome__p1-bdc-1.parquet"
    assert ok_part.exists()
