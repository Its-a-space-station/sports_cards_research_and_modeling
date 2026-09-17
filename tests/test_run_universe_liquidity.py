# tests/test_run_universe_liquidity.py
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_universe_liquidity import select_modeling_cards


def test_selection_rule():
    report = pd.DataFrame(
        [
            # deep history, thin recent sales -> kept (chart rule)
            {
                "card_slug": "card/old",
                "grade": "ungraded",
                "sales_per_week": 0.1,
                "n_chart_points": 60,
            },
            # thin history, active recent sales -> kept (sales rule)
            {
                "card_slug": "card/new",
                "grade": "psa_10",
                "sales_per_week": 1.5,
                "n_chart_points": 10,
            },
            # neither -> dropped
            {
                "card_slug": "card/thin",
                "grade": "ungraded",
                "sales_per_week": 0.1,
                "n_chart_points": 10,
            },
        ]
    )
    out = select_modeling_cards(report)
    assert set(out["card_slug"]) == {"card/old", "card/new"}
