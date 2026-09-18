# src/cardprice/expectations.py
"""Expectations data: draft-year prospect ranks + Marcel projections.

`parse_draft_prospects` is the v1 draft-year leg of Task 6 (spec amendment 2,
2026-09-18): MLB Stats API `/draft/prospects/{year}` JSON, official statsapi
ids (direct mlb_id join, no name matching). Unranked years (2015-2016 live;
2017+ carry ranks) keep their players with rank=NA. The Pipeline Top-100
parsers remain parked behind mlb.com rate-limiting; they join this module
when their capture clears.

Marcel is the in-house post-debut expectations term (5/4/3 rate-weighted,
regressed to the universe-season mean).
"""

import pandas as pd

DRAFT_PROSPECTS_COLUMNS = ["player_name", "mlb_id", "season", "source", "rank", "fv", "as_of"]


def parse_draft_prospects(payload: dict, year: int) -> pd.DataFrame:
    """Stats API `/draft/prospects/{year}` payload -> expectations rows.

    mlb_id comes from `person.id` (the statsapi id space, verified live
    2026-09-18: 2023 rank-1 Paul Skenes -> /people/694973), never from the
    entry-level `bisPlayerId` (a different id space). Entries without a
    `person` block, or with neither name nor id, are skipped. Entries without
    a `rank` (unranked years such as 2015) are kept with rank=NA. The payload
    carries verbatim duplicated persons (e.g. 2023: 59 ids twice, identical
    fields, never conflicting ranks); the first occurrence is kept. fv is
    always NA for this source; as_of is the conservative post-draft {year}-07-01.
    """
    rows = []
    for entry in payload.get("prospects") or []:
        person = entry.get("person") or {}
        mlb_id, name = person.get("id"), person.get("fullName")
        if mlb_id is None and name is None:
            continue
        rows.append(
            {
                "player_name": name,
                "mlb_id": mlb_id,
                "season": year,
                "source": "mlb_draft",
                "rank": entry.get("rank"),
                "fv": pd.NA,
                "as_of": pd.Timestamp(f"{year}-07-01"),
            }
        )
    df = pd.DataFrame(rows, columns=DRAFT_PROSPECTS_COLUMNS).astype(
        {
            "mlb_id": "Int64",
            "season": "int64",
            "rank": "Int64",
            "fv": "Float64",
            "as_of": "datetime64[ns]",
        }
    )
    # duplicated() treats NA == NA, so mask id-less rows out of the dedupe
    dup = df["mlb_id"].notna() & df.duplicated(subset=["mlb_id"], keep="first")
    return df[~dup].reset_index(drop=True)


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
