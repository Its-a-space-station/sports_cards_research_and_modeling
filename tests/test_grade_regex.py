import pytest

from cardprice.scp_parse import parse_grade_from_title


@pytest.mark.parametrize(
    "title,expected",
    [
        # regression: formats that already worked
        ("2023 Topps Chrome Gunnar Henderson RC PSA 10 GEM MT", "psa_10"),
        ("GUNNAR HENDERSON 2023 TOPPS CHROME PSA 9", "psa_9"),
        ("2023 Topps Chrome Henderson BGS 9.5 GEM MINT", "bgs_9.5"),
        ("2023 Topps Chrome Gunnar Henderson #2", None),
        # new: hyphenated
        ("2022 Topps Chrome Bobby Witt Jr RC SGC-10", "sgc_10"),
        ("Henderson PSA-9", "psa_9"),
        # new: word-form between grader and digit
        ("2023 Topps Chrome Henderson PSA GEM MT 10", "psa_10"),
        ("Witt RC PSA MINT 9", "psa_9"),
        ("Henderson SGC PERFECT 10", "sgc_10"),
        # new: reversed order
        ("2023 Topps Chrome Henderson Gem Mint 10 PSA", "psa_10"),
    ],
)
def test_grade_variants(title, expected):
    assert parse_grade_from_title(title) == expected
