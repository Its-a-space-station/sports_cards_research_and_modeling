# tests/test_resolve_universe.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from resolve_universe import _name_matches, norm_name, pick_bowman_1st, pick_flagship


def test_norm_name_handles_accents_and_suffixes():
    assert norm_name("Ronald Acuna Jr.") == norm_name("Ronald Acuña Jr")
    assert norm_name("Michael Harris II") == norm_name("Michael Harris II")


def test_name_matches():
    assert _name_matches("Ronald Acuna Jr.", "Ronald Acuna Jr. #193")
    assert _name_matches("Michael Harris II", "Michael Harris II #251")
    assert not _name_matches("Bobby Witt Jr", "Bobby Witt III #99")
    assert not _name_matches("Nick Kurtz", "Nick Kurtzney #1")


def test_pick_flagship_prefers_base_over_parallel():
    anchors = [
        ("Aaron Judge [Refractor] #99", "/game/set/aaron-judge-refractor-99"),
        ("Aaron Judge #99", "/game/set/aaron-judge-99"),
    ]
    assert pick_flagship("Aaron Judge", anchors) == "/game/set/aaron-judge-99"
    assert pick_flagship("Aaron Judge", [anchors[0]]) is None  # never a parallel


def test_pick_bowman_1st_earliest_year_no_autos():
    candidates = [
        (2019, "Juan Soto #BCP52", "/game/baseball-cards-2019-bowman-chrome/juan-soto-bcp52"),
        (2016, "Juan Soto #BCP52", "/game/baseball-cards-2016-bowman-chrome/juan-soto-bcp52"),
        (2016, "Juan Soto [Autograph] #BCP52", "/game/x"),
    ]
    url, audit = pick_bowman_1st("Juan Soto", candidates)
    assert url == "/game/baseball-cards-2016-bowman-chrome/juan-soto-bcp52"
    assert len(audit) >= 2  # audit trail keeps candidates
