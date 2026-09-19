# tests/test_walkforward_all_scores.py
import numpy as np
import pandas as pd

from cardprice.walkforward_holds import walk_forward_years


def _frame(n_per_year=30, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    # 4 entry years: with the harness default min_train_years=2 the test years are
    # {2023, 2024} (same shape as tests/test_walkforward_holds.py's pinned semantics)
    for y in (2021, 2022, 2023, 2024):
        for c in range(n_per_year):
            s = rng.normal(0, 1)
            rows.append({"entry_year": y, "card_slug": f"c{c}", "sig": s,
                         "ret_12m": 0.5 * s + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_all_scores_returns_every_test_row_default_unchanged():
    frame = _frame()
    top = walk_forward_years(frame, ["sig"], 12, top_k=5)
    assert len(top) == 10  # 2 test years x top-5
    allrows = walk_forward_years(frame, ["sig"], 12, top_k=5, all_scores=True)
    assert len(allrows) == 60  # 2 test years x 30
    assert set(allrows.columns) == {"entry_year", "card_slug", "predicted", "realized"}
    # top-k picks are the max-predicted rows of the full score set
    for y in (2023, 2024):
        got = top[top.entry_year == y]["card_slug"].tolist()
        want = allrows[allrows.entry_year == y].nlargest(5, "predicted")["card_slug"].tolist()
        assert got == want
