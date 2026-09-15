# tests/test_collect_prices.py
import pandas as pd
import pytest

from cardprice import collect_prices, storage

MINI_HTML = """
<html><head><title>x</title></head><body>
<table id="attribute"><tr><td>Is Rookie Card:</td><td>Yes</td></tr></table>
<table id="price_data"><tr><th>PSA 10</th></tr><tr><td>$34.50</td></tr></table>
<div class="completed-auctions-graded"><table>
<tr><th>Sale Date</th><th>TW</th><th>Title</th><th>Price</th><th></th></tr>
<tr><td>2026-09-05</td><td></td><td>2023 Topps Chrome Gunnar Henderson RC PSA 10</td><td>$45.99</td><td></td></tr>
</table></div>
<script>VGPC.chart_data = {"graded":[[1756600000000,4599],[1759200000000,3450]]};</script>
</body></html>
"""


@pytest.fixture
def cards():
    return pd.DataFrame(
        [
            {
                "player_name": "Gunnar Henderson",
                "role": "hitter",
                "rookie_year": 2023,
                "set_slug": "baseball-cards-2023-topps-chrome",
                "mlb_id": 683002,
                "scp_url": "https://www.sportscardspro.com/game/baseball-cards-2023-topps-chrome/gunnar-henderson-2",
            },
            {
                "player_name": "Unresolved Player",
                "role": "hitter",
                "rookie_year": 2025,
                "set_slug": "baseball-cards-2025-topps-chrome",
                "mlb_id": 999999,
                "scp_url": "",
            },
        ]
    )


def test_card_slug():
    assert (
        collect_prices.card_slug(
            "https://www.sportscardspro.com/game/baseball-cards-2023-topps-chrome/gunnar-henderson-2"
        )
        == "baseball-cards-2023-topps-chrome/gunnar-henderson-2"
    )


def test_collect_cards_happy_path_and_skip(cards, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(collect_prices, "fetch_page", lambda url: MINI_HTML)
    sales, chart = collect_prices.collect_cards(cards, sleep_s=0)
    assert len(sales) == 1
    assert sales.iloc[0]["grade"] == "psa_10"
    assert sales.iloc[0]["mlb_id"] == 683002
    assert len(chart) == 2
    assert (
        storage.load_latest("prices_scp", "baseball-cards-2023-topps-chrome/gunnar-henderson-2")[
            "url"
        ]
        == cards.iloc[0]["scp_url"]
    )


def test_collect_cards_challenge_continues(cards, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_ROOT", tmp_path)

    def flaky(url):
        raise collect_prices.ChallengeError(url)

    monkeypatch.setattr(collect_prices, "fetch_page", flaky)
    sales, chart = collect_prices.collect_cards(cards, sleep_s=0)
    assert len(sales) == 0 and len(chart) == 0  # no exception propagated
