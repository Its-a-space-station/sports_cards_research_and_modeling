# tests/test_scp_chart_calibrate.py
from pathlib import Path

import pandas as pd
import pytest

from cardprice.scp_parse import calibrate_chart_grades, parse_price_summary

FIXTURE = (
    Path(__file__).parent / "fixtures" / "scp" / "henderson_2023_topps_chrome.html"
).read_text()


def test_calibrated_grades_cover_psa10():
    df = calibrate_chart_grades(FIXTURE)
    grades = set(df["grade"])
    assert "psa_10" in grades
    psa10 = df[df["grade"] == "psa_10"].sort_values("date")
    assert len(psa10) >= 24  # ~2+ years of monthly points
    # final chart point equals the current price summary value
    assert psa10.iloc[-1]["price"] == pytest.approx(parse_price_summary(FIXTURE)["PSA 10"])


def test_history_reaches_back():
    df = calibrate_chart_grades(FIXTURE)
    psa10 = df[df["grade"] == "psa_10"]
    assert psa10["date"].min() <= pd.Timestamp("2024-06-01")  # card tracked since 2023
