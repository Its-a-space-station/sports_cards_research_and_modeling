from pathlib import Path

import pandas as pd
import pytest

from cardprice import scp_parse

FIXTURE = (
    Path(__file__).parent / "fixtures" / "scp" / "henderson_2023_topps_chrome.html"
).read_text()


def test_parse_grade_from_title():
    assert (
        scp_parse.parse_grade_from_title("2023 Topps Chrome Gunnar Henderson RC PSA 10 GEM MT")
        == "psa_10"
    )
    assert scp_parse.parse_grade_from_title("GUNNAR HENDERSON 2023 TOPPS CHROME PSA 9") == "psa_9"
    assert (
        scp_parse.parse_grade_from_title("2023 Topps Chrome Henderson BGS 9.5 GEM MINT")
        == "bgs_9.5"
    )
    assert scp_parse.parse_grade_from_title("2023 Topps Chrome Gunnar Henderson #2") is None
    assert (
        scp_parse.parse_grade_from_title("Henderson RC GEM MT") is None
    )  # ambiguous without grader


def test_parse_sales_tables_rows():
    df = scp_parse.parse_sales_tables(FIXTURE)
    assert len(df) >= 100  # fixture has ~9 grade buckets x <=30 rows
    assert set(df.columns) == {
        "sale_date",
        "title",
        "price",
        "list_price",
        "best_offer",
        "grade",
        "bucket",
    }
    assert df["sale_date"].notna().all()
    assert (df["price"] > 0).all()
    # "sgc_9.5" added to the brief's allowlist: the fixture's Grade 9.5 (box-only)
    # bucket is 26 sales all titled "SGC 9.5", a legal output of the specified regex.
    grades_seen = {
        "psa_10",
        "psa_9",
        "psa_8.5",
        "psa_8",
        "bgs_10",
        "bgs_9.5",
        "bgs_9",
        "bgs_8.5",
        "sgc_10",
        "sgc_9.5",
        "sgc_9",
        "sgc_8.5",
        "sgc_8",
        "cgc_10",
        "cgc_9",
        "cgc_9.5",
        "cgc_8.5",
        None,
    }
    assert df["grade"].isin(grades_seen).all()
    # every row title mentions Henderson (SCP pre-filters to this card)
    assert df["title"].str.lower().str.contains("henderson").all()


def test_parse_sales_tables_best_offer():
    df = scp_parse.parse_sales_tables(FIXTURE)
    bo = df[df["best_offer"]]
    assert len(bo) >= 1  # fixture contains a dual-price row
    assert (bo["list_price"] > bo["price"]).all()  # accepted < list
    # the spike-documented row: list $40.00, accepted $35.00
    assert ((bo["list_price"] == 40.00) & (bo["price"] == 35.00)).any()


def test_parse_attributes():
    attrs = scp_parse.parse_attributes(FIXTURE)
    assert attrs["is_rookie_card"] is True


def test_parse_price_summary():
    summary = scp_parse.parse_price_summary(FIXTURE)
    assert summary["PSA 10"] == pytest.approx(34.50)
    assert summary["Ungraded"] == pytest.approx(1.63)


def test_parse_chart_data():
    chart = scp_parse.parse_chart_data(FIXTURE)
    assert len(chart) >= 3
    for series in chart.values():
        ts, price = series[-1]
        assert isinstance(ts, pd.Timestamp)
        assert price > 0
        assert series == sorted(series, key=lambda p: p[0])  # chronological
