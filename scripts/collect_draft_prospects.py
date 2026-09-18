# scripts/collect_draft_prospects.py
"""Collect draft-year expectations from MLB Stats API /draft/prospects/{year}.

V1 draft leg of Task 6 (spec amendment 2): years 2015-2025, official JSON,
direct statsapi ids (no name matching). Unranked years (2015-2017) keep their
players with rank=NA. One retry per year, then the year is recorded as a gap
(never fabricated).

Outputs: data/processed/expectations_draft.parquet (this source only) and
data/processed/expectations.parquet (combined: any existing non-mlb_draft
rows preserved, mlb_draft rows replaced by this run — additive-by-source, so a
later Pipeline collector appends source="pipeline" rows without clobbering).
Full per-year payloads are snapshotted under data/raw/expectations/ via
save_raw.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.expectations import parse_draft_prospects
from cardprice.stats_api import BASE
from cardprice.storage import save_raw

YEARS = list(range(2015, 2026))
OUT_DIR = Path("data/processed")


def fetch_year(year: int, timeout: int = 60) -> dict:
    """One GET of /draft/prospects/{year}; one retry, then raise."""
    url = f"{BASE}/draft/prospects/{year}"
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == 1:
                raise
            time.sleep(2)
    raise AssertionError("unreachable")


def collect(years: list[int], sleep_s: float = 0.4) -> tuple[pd.DataFrame, list[dict]]:
    frames, gaps = [], []
    for year in years:
        try:
            payload = fetch_year(year)
        except requests.RequestException as e:
            gaps.append({"season": year, "source": "mlb_draft", "reason": repr(e)})
            continue
        save_raw("expectations", f"draft_prospects_{year}", payload)
        df = parse_draft_prospects(payload, year)
        frames.append(df)
        mapped = df["mlb_id"].notna().mean() if len(df) else float("nan")
        print(
            f"{year}: {len(df)} prospects ({df['rank'].notna().sum()} ranked), "
            f"mapped-id share {mapped:.1%}"
        )
        time.sleep(sleep_s)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, gaps


def combine(draft: pd.DataFrame) -> pd.DataFrame:
    """Merge draft rows with any existing expectations.parquet, by source."""
    path = OUT_DIR / "expectations.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        other = existing[existing["source"] != "mlb_draft"]
        combined = pd.concat([other, draft], ignore_index=True)
    else:
        combined = draft
    return combined


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draft, gaps = collect(YEARS)
    draft.to_parquet(OUT_DIR / "expectations_draft.parquet", index=False)
    combine(draft).to_parquet(OUT_DIR / "expectations.parquet", index=False)
    print(f"wrote {len(draft)} mlb_draft rows to {OUT_DIR}/expectations_draft.parquet")
    print(f"gaps: {gaps if gaps else 'none'}")


if __name__ == "__main__":
    main()
