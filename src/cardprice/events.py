"""Event registry: discrete performance events per player."""

import time

import pandas as pd
import requests

from cardprice.stats_api import BASE

EVENT_COLUMNS = ["mlb_id", "event_date", "event_type", "details"]

_MIN_CALL_GAP_S = 0.3  # seconds between Stats API calls
_last_api_call = 0.0


def _get(url, **kwargs):
    """Throttled GET: at least ``_MIN_CALL_GAP_S`` between Stats API calls."""
    global _last_api_call
    wait = _MIN_CALL_GAP_S - (time.monotonic() - _last_api_call)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, **kwargs)
    _last_api_call = time.monotonic()
    return resp


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


def fetch_playoff_events(
    mlb_ids: list[int], seasons: list[int], players: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Postseason participation events: one row per player-season with playoff games.

    Uses the player-level game log with ``gameType=P``; players without a
    postseason return an empty stat block and are skipped. Columns match
    :func:`events_from_game_logs`. ``players`` is accepted for interface
    compatibility and not used.
    """
    rows = []
    for mlb_id in mlb_ids:
        for season in seasons:
            for group in ("hitting", "pitching"):
                resp = _get(
                    f"{BASE}/people/{mlb_id}/stats",
                    params={
                        "stats": "gameLog",
                        "group": group,
                        "season": season,
                        "gameType": "P",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                stats = resp.json().get("stats", [])
                if not stats:
                    continue
                splits = stats[0].get("splits", [])
                if splits:
                    rows.append(
                        {
                            "mlb_id": mlb_id,
                            "event_date": pd.Timestamp(splits[0]["date"]),
                            "event_type": "playoff_appearance",
                            "details": f"{len(splits)} postseason games ({group})",
                        }
                    )
                    break  # one event per player-season
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return df.sort_values(["mlb_id", "event_date"]).reset_index(drop=True)


def _discover_major_award_ids() -> list[tuple[str, str]]:
    """Find the canonical league award ids (MVP, Cy Young, ROY) from the API.

    Restricts to AL/NL (league ids 103/104) so team-level, minor-league and
    MLB.com-awards entries with similar names are excluded.
    """
    resp = _get(f"{BASE}/awards", timeout=30)
    resp.raise_for_status()
    found = []
    for a in resp.json().get("awards", []):
        if (a.get("league") or {}).get("id") not in (103, 104):
            continue
        name = a.get("name", "")
        low = name.lower()
        if low in ("al mvp", "nl mvp") or "cy young" in low or "rookie of the year" in low:
            found.append((a["id"], name))
    return found


def fetch_award_events(mlb_ids: list[int], seasons: list[int]) -> pd.DataFrame:
    """Award wins (MVP, Cy Young, Rookie of the Year) for the given players.

    Award ids are discovered live from ``GET /awards`` (never hardcoded).
    The recipients payload is a flat list under ``awards`` — one entry per
    winner, carrying the announcement ``date``; when absent, the date falls
    back to Nov 15 of the season and the approximation is noted in
    ``details``. Award-seasons not yet announced return HTTP 404 and are
    skipped. Columns match :func:`events_from_game_logs`.
    """
    rows = []
    for award_id, award_name in _discover_major_award_ids():
        for season in seasons:
            resp = _get(
                f"{BASE}/awards/{award_id}/recipients",
                params={"season": season},
                timeout=30,
            )
            if resp.status_code == 404:  # award not yet announced for this season
                continue
            resp.raise_for_status()
            for rec in resp.json().get("awards", []):
                pid = rec.get("player", {}).get("id")
                if pid not in mlb_ids:
                    continue
                name = rec.get("name", award_name)
                if rec.get("date"):
                    event_date = pd.Timestamp(rec["date"])
                    details = f"{name} {season}"
                else:
                    event_date = pd.Timestamp(f"{season}-11-15")
                    details = f"{name} {season} (announcement date approximated: Nov 15)"
                rows.append(
                    {
                        "mlb_id": pid,
                        "event_date": event_date,
                        "event_type": "award_win",
                        "details": details,
                    }
                )
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return df.sort_values(["mlb_id", "event_date"]).reset_index(drop=True)
