# tests/test_expectations.py
"""Marcel-projection goldens (Task 6 unit tests, committed with Task 7's panel).

The parser goldens (Pipeline Top-100 page/article, FG draft board) depend on
Task 6's live fixture captures, which were blocked by source rate-limiting
(2026-09-18); they land here when the parsers join `cardprice.expectations`.
"""

import pytest

from cardprice.expectations import marcel_projection


def test_marcel_projection_golden():
    # num = 5*.8*500 + 4*.7*300 + 3*.6*100 = 3020; den = 5*500+4*300+3*100 = 4000
    # proj = (3020 + .72*1200) / (4000+1200) = 3884/5200
    got = marcel_projection([(0.8, 500.0), (0.7, 300.0), (0.6, 100.0)], league_mean=0.72)
    assert got == pytest.approx(3884.0 / 5200.0)


def test_marcel_projection_uses_at_most_three_seasons():
    a = marcel_projection([(0.8, 500.0), (0.7, 300.0), (0.6, 100.0), (0.9, 999.0)], 0.72)
    b = marcel_projection([(0.8, 500.0), (0.7, 300.0), (0.6, 100.0)], 0.72)
    assert a == b


def test_marcel_projection_no_seasons_is_none():
    assert marcel_projection([], 0.72) is None


def test_marcel_projection_zero_pt_season_ignored():
    assert marcel_projection([(0.8, 0.0)], 0.72) is None  # den == 0
