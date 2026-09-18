"""Immutable dated JSON snapshots. Collectors add files; nothing edits or deletes them."""

import json
from datetime import date
from pathlib import Path

RAW_ROOT = Path("data/raw")


def save_raw(dataset: str, key: str, payload: dict, on: date | None = None) -> Path:
    day = (on or date.today()).isoformat()  # noqa: DTZ011  # local calendar date is intended
    path = RAW_ROOT / dataset / key / f"{day}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path.exists() and path.read_text() == text:
        return path
    path.write_text(text)
    return path


def snapshot_exists(dataset: str, key: str) -> bool:
    d = RAW_ROOT / dataset / key
    return d.is_dir() and any(d.glob("*.json"))


def load_latest(dataset: str, key: str) -> dict:
    files = sorted((RAW_ROOT / dataset / key).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no snapshots for {dataset}/{key}")
    return json.loads(files[-1].read_text())
