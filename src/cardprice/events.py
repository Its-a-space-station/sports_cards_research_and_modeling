"""Event registry: discrete performance events per player."""

import pandas as pd


def events_from_game_logs(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Build the event registry from game logs.

    Returns one row per event with columns
    ``mlb_id, event_date, event_type, details``. Event types:

    - ``debut``: first game in the loaded logs; caller must ensure logs cover
      the player's career start.
    - ``three_hr_game``: hitter game with ``homeRuns >= 3``.
    - ``four_hit_game``: hitter game with ``hits >= 4`` (not already a
      three_hr_game).
    - ``ten_k_game``: pitcher game with ``strikeOuts >= 10``.

    Doubleheaders collapse to one player-day (max stat line governs), and
    milestone rules are group-scoped: hitting rules apply only to hitting
    rows, pitching rules only to pitching rows (pitcher ``hits``/``homeRuns``
    are allowed, not produced).
    """
    df = game_logs.copy()
    # collapse doubleheaders: one row per player-day with max of counting stats
    day = df.groupby(["mlb_id", "date", "group"], as_index=False)[
        ["homeRuns", "hits", "strikeOuts"]
    ].max()
    rows = []
    for mlb_id, grp in day.groupby("mlb_id"):
        debut = grp["date"].min()
        rows.append(
            {
                "mlb_id": mlb_id,
                "event_date": debut,
                "event_type": "debut",
                "details": "first game in dataset",
            }
        )
        hit = grp[grp["group"] == "hitting"]
        pit = grp[grp["group"] == "pitching"]
        for r in hit[hit["homeRuns"] >= 3].itertuples():
            rows.append(
                {
                    "mlb_id": mlb_id,
                    "event_date": r.date,
                    "event_type": "three_hr_game",
                    "details": f"{r.homeRuns} HR",
                }
            )
        four_hit = hit[(hit["hits"] >= 4) & (hit["homeRuns"] < 3)]
        for r in four_hit.itertuples():
            rows.append(
                {
                    "mlb_id": mlb_id,
                    "event_date": r.date,
                    "event_type": "four_hit_game",
                    "details": f"{r.hits} H",
                }
            )
        for r in pit[pit["strikeOuts"] >= 10].itertuples():
            rows.append(
                {
                    "mlb_id": mlb_id,
                    "event_date": r.date,
                    "event_type": "ten_k_game",
                    "details": f"{r.strikeOuts} K",
                }
            )
    return pd.DataFrame(rows).sort_values(["mlb_id", "event_date"]).reset_index(drop=True)
