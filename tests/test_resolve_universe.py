# tests/test_resolve_universe.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import resolve_universe
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


def _html(*anchors: tuple[str, str]) -> str:
    return "".join(f'<a href="{href}">{text}</a>' for href, text in anchors)


def test_pick_bowman_1st_excludes_talent_pipeline():
    # "Talent Pipeline" inserts are not a Bowman 1st. Covers the tagged form and
    # the bare "#TP-<team>" card number — the live 2021 Bowman Chrome console
    # anchor for Wander Franco is the bare form (verified 2026-09-16).
    candidates = [
        (2021, "Wander Franco [Talent Pipeline] #TP-TBR", "/game/x/tagged"),
        (2021, "Wander Franco #TP-TBR", "/game/x/tp-tbr"),
    ]
    url, audit = pick_bowman_1st("Wander Franco", candidates)
    assert url is None
    assert all(e["decision"] == "skip:parallel-keyword" for e in audit)


def test_pick_bowman_1st_talent_pipeline_falls_through_to_earlier_year():
    candidates = [
        (2021, "Wander Franco #TP-TBR", "/game/x/tp-tbr"),
        (2019, "Wander Franco #BCP150", "/game/baseball-cards-2019-bowman-chrome/wf-bcp150"),
    ]
    url, _audit = pick_bowman_1st("Wander Franco", candidates)
    assert url == "/game/baseball-cards-2019-bowman-chrome/wf-bcp150"


def test_resolve_flagship_full_listing_fallback_prefers_plain(monkeypatch):
    # Rookies-only listings miss the player on both sets; full listings match on
    # both. The fallback must stay on the same rule-2 set and the plain
    # topps-chrome match must win, recorded as coming from the full listing.
    plain = "baseball-cards-2015-topps-chrome"
    update = "baseball-cards-2015-topps-chrome-update"
    base = resolve_universe.SCP_BASE
    pages = {
        f"{base}/console/{plain}?rookies-only=true&exclude-variants=true": _html(
            (f"/game/{plain}/other-guy-1", "Other Guy #1")
        ),
        f"{base}/console/{plain}?exclude-variants=true": _html(
            (f"/game/{plain}/kris-bryant-112", "Kris Bryant #112")
        ),
        f"{base}/console/{update}?rookies-only=true&exclude-variants=true": _html(
            (f"/game/{update}/other-guy-us1", "Other Guy #US1")
        ),
        f"{base}/console/{update}?exclude-variants=true": _html(
            (f"/game/{update}/kris-bryant-us283", "Kris Bryant #US283")
        ),
    }
    monkeypatch.setattr(resolve_universe, "_console_page", lambda key, url: pages[url])
    url, slug, audit = resolve_universe._resolve_flagship("Kris Bryant", 2015)
    assert url == f"{base}/game/{plain}/kris-bryant-112"
    assert slug == plain
    chosen = [e for e in audit if e.get("result") == "chosen"]
    assert len(chosen) == 1
    assert chosen[0]["listing"] == "full"
    assert chosen[0]["anchor"] == "Kris Bryant #112"


def test_resolve_flagship_rookies_only_match_needs_no_fallback(monkeypatch):
    plain = "baseball-cards-2023-topps-chrome"
    update = "baseball-cards-2023-topps-chrome-update"
    base = resolve_universe.SCP_BASE
    pages = {
        f"{base}/console/{plain}?rookies-only=true&exclude-variants=true": _html(
            (f"/game/{plain}/gunnar-henderson-2", "Gunnar Henderson #2")
        ),
        f"{base}/console/{update}?rookies-only=true&exclude-variants=true": _html(
            (f"/game/{update}/other-guy-usc1", "Other Guy #USC1")
        ),
        f"{base}/console/{update}?exclude-variants=true": _html(
            (f"/game/{update}/other-guy-usc1", "Other Guy #USC1")
        ),
        # plain full listing intentionally absent: a rookies-only match on the
        # plain set must not trigger the fallback fetch.
    }
    monkeypatch.setattr(resolve_universe, "_console_page", lambda key, url: pages[url])
    url, slug, audit = resolve_universe._resolve_flagship("Gunnar Henderson", 2023)
    assert url == f"{base}/game/{plain}/gunnar-henderson-2"
    assert slug == plain
    chosen = [e for e in audit if e.get("result") == "chosen"]
    assert len(chosen) == 1
    assert chosen[0]["listing"] == "rookies-only"
