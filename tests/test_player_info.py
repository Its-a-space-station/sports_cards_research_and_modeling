import pandas as pd

from cardprice import stats_api

PEOPLE_RESPONSE = {
    "people": [
        {
            "id": 683002,
            "fullName": "Gunnar Henderson",
            "birthDate": "2001-06-29",
            "primaryPosition": {"abbreviation": "SS"},
        },
        {
            "id": 592450,
            "fullName": "Aaron Judge",
            "birthDate": "1992-04-26",
            "primaryPosition": {"abbreviation": "RF"},
        },
    ]
}


def test_fetch_player_info_parses(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return PEOPLE_RESPONSE

    calls = {}

    def fake_get(url, params=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        return FakeResp()

    monkeypatch.setattr(stats_api.requests, "get", fake_get)
    df = stats_api.fetch_player_info([683002, 592450])
    assert calls["url"] == "https://statsapi.mlb.com/api/v1/people"
    assert calls["params"]["personIds"] == "683002,592450"
    assert list(df.columns) == ["mlb_id", "name", "birth_date", "position"]
    row = df[df["mlb_id"] == 683002].iloc[0]
    assert row["name"] == "Gunnar Henderson"
    assert row["birth_date"] == pd.Timestamp("2001-06-29")
    assert row["position"] == "SS"
