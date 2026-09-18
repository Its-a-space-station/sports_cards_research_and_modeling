# src/cardprice/expectations.py
"""Expectations data: Marcel projections (external-source parsers are Task 6's).

Marcel is the in-house post-debut expectations term (5/4/3 rate-weighted,
regressed to the universe-season mean). The Pipeline Top-100 / FG draft-board
parsers were written under Task 6 but never committed: their fixture captures
were blocked by source rate-limiting (2026-09-18) and honest-golden discipline
forbids shipping untested goldens. They join this module when Task 6 lands;
only `marcel_projection` (reviewed, golden-pinned) is committed now because the
class panel builder imports it.
"""


def marcel_projection(
    seasons: list[tuple[float, float]], league_mean: float, regression: float = 1200.0
) -> float | None:
    """5/4/3 rate-weighted projection, regressed toward `league_mean` with
    `regression` units of league-average playing time. None when no usable seasons."""
    weights = (5.0, 4.0, 3.0)
    num = den = 0.0
    for (rate, pt), w in zip(seasons[:3], weights):
        num += w * rate * pt
        den += w * pt
    if den == 0:
        return None
    return (num + league_mean * regression) / (den + regression)
