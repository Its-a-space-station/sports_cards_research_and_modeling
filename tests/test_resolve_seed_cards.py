# tests/test_resolve_seed_cards.py
"""Offline unit tests for scripts/resolve_seed_cards.py anchor picking."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "resolve_seed_cards",
    Path(__file__).parent.parent / "scripts" / "resolve_seed_cards.py",
)
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)

SLUG = "baseball-cards-2023-topps-chrome"


def _html(*anchors: tuple[str, str]) -> str:
    return "".join(f'<a href="{href}">{text}</a>' for href, text in anchors)


def test_pick_anchor_returns_none_when_only_parallels_match():
    html = _html(
        (f"/game/{SLUG}/gunnar-henderson-refractor-2", "Gunnar Henderson [Refractor] #2"),
        (f"/game/{SLUG}/gunnar-henderson-autograph-2", "Gunnar Henderson [Autograph] #2"),
    )
    assert resolver.pick_anchor(html, "Gunnar Henderson", SLUG) is None


def test_pick_anchor_prefers_base_over_parallel():
    html = _html(
        (f"/game/{SLUG}/gunnar-henderson-refractor-2", "Gunnar Henderson [Refractor] #2"),
        (f"/game/{SLUG}/gunnar-henderson-2", "Gunnar Henderson #2"),
    )
    assert resolver.pick_anchor(html, "Gunnar Henderson", SLUG) == f"/game/{SLUG}/gunnar-henderson-2"


def test_pick_anchor_name_with_red_substring_not_treated_as_parallel():
    html = _html(
        (f"/game/{SLUG}/isaac-paredes-99", "Isaac Paredes #99"),
    )
    assert resolver.pick_anchor(html, "Isaac Paredes", SLUG) == f"/game/{SLUG}/isaac-paredes-99"
