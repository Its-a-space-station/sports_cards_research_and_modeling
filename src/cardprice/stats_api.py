# src/cardprice/stats_api.py
"""Thin client for the official MLB Stats API (no key required)."""

import time

import pandas as pd
import requests

BASE = "https://statsapi.mlb.com/api/v1"

MINOR_LEAGUE_LEVELS = {11: "aaa", 12: "aa", 13: "a_plus", 14: "a"}


def fetch_game_log(mlb_id: int, group: str, season: int, sport_id: int | None = None) -> list[dict]:
    params = {"stats": "gameLog", "group": group, "season": season}
    if sport_id is not None:
        params["sportId"] = sport_id
    resp = requests.get(f"{BASE}/people/{mlb_id}/stats", params=params, timeout=30)
    resp.raise_for_status()
    stats = resp.json().get("stats", [])
    if not stats:
        return []
    return stats[0].get("splits", [])


def fetch_minor_league_logs(mlb_id: int, group: str, season: int) -> dict[str, list[dict]]:
    out = {}
    for sport_id, level in MINOR_LEAGUE_LEVELS.items():
        splits = fetch_game_log(mlb_id, group, season, sport_id=sport_id)
        if splits:
            out[level] = splits
        time.sleep(0.3)
    return out


def game_log_to_frame(splits: list[dict], mlb_id: int, group: str, season: int) -> pd.DataFrame:
    rows = []
    for split in splits:
        row = {"mlb_id": mlb_id, "group": group, "season": season, "date": split["date"]}
        row.update(split["stat"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_player_info(mlb_ids: list[int]) -> pd.DataFrame:
    resp = requests.get(
        f"{BASE}/people",
        params={"personIds": ",".join(str(i) for i in mlb_ids), "hydrate": ""},
        timeout=30,
    )
    resp.raise_for_status()
    rows = [
        {
            "mlb_id": p["id"],
            "name": p["fullName"],
            "birth_date": p.get("birthDate"),
            "position": p.get("primaryPosition", {}).get("abbreviation"),
        }
        for p in resp.json().get("people", [])
    ]
    df = pd.DataFrame(rows)
    df["birth_date"] = pd.to_datetime(df["birth_date"])
    return df
