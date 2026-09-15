# tests/test_web.py
import pytest

from cardprice import web


def test_is_challenge_title_detects_cloudflare():
    assert web.is_challenge_title("Just a moment...")
    assert web.is_challenge_title("Performing security verification")
    assert not web.is_challenge_title(
        "Gunnar Henderson #2 Prices [Rookie] | 2023 Topps Chrome | Baseball Cards"
    )


@pytest.mark.live
def test_fetch_page_real_card():
    html = web.fetch_page(
        "https://www.sportscardspro.com/game/baseball-cards-2023-topps-chrome/gunnar-henderson-2"
    )
    assert "VGPC.chart_data" in html
    assert "completed-auctions-" in html
    assert 'id="price_data"' in html
