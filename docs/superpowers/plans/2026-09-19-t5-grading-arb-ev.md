# T5 — Grading-Arbitrage EV Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate the expected value of the **buy-raw → grade → sell-PSA-10** pipeline per card across the class universe: Monte Carlo EV with grade-outcome posteriors from GemRate pop counts and empirical price distributions, a sensitivity tornado, and a ranked, honestly-flagged output. Explicitly a **desk model** — not realized P&L, not submission advice.

**Architecture:** New pure module `src/cardprice/grading_ev.py` (Dirichlet posteriors, empirical samplers, MC EV, sensitivity — offline-tested); new runners under `scripts/`; the reviewed GemRate collector (`src/cardprice/pop_gemrate.py`) reused verbatim for pop collection.

**Tech Stack:** pandas 2.x, numpy 2.x, requests, beautifulsoup4, playwright (pop collection only), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-t5-grading-arb-ev-design.md`
**Spike (GemRate mechanics):** `docs/superpowers/spikes/2026-09-15-gemrate.md`

## Global Constraints

- Ruff line-length **100**; `.venv/bin/ruff check src tests scripts` before every commit; **never** `ruff format`.
- Honest-golden discipline: goldens verified by hand/recomputation; a mismatch stops the run; never adjust expected values to force a pass.
- **No invented fees:** PSA fees/shipping are captured LIVE from psacard.com at run time into `data/reference/grading_fees.json` (schema below; `captured_date` + `source_url` are part of the schema; unknown tier → validation error). If the live capture is blocked or the structure changed, the implementer **stops and escalates** (to the controller/user for manual entry) — never types fees from memory.
- **Desk-model honesty:** the findings doc and the output CSV label every EV as a *model estimate*; the pop→future-grade-outcome representativeness limitation (past submitters' selection is optimistic-biased) is stated wherever EVs are presented.
- **No fabrication:** cards with < 5 sales in a needed grade's 12-month distribution get `thin_price_evidence` flags (EV still computed), never borrowed prices from other cards; a card with no GemRate match is excluded from the ranked list and counted in findings (never pop-guessed).
- Seed every RNG (`seed=42`); MC deterministic in tests.
- Politeness: GemRate scrape follows the reviewed collector's conventions (shared session, ≥ 5 s gaps, ~30–35 s/card); resumable.
- Findings numbers trace to artifacts (`grading_ev.csv`, `pop_snapshots_class.csv`, `grading_fees.json`, run logs) — no remembered numbers.

## Existing code this plan reuses (do not re-implement)

- `src/cardprice/pop_gemrate.py`: `fetch_pop(query, match={name, year, set_name, card_number})`, `collect_pops(cards_seed)`, `build_query(scp_url)` — the reviewed collector with its no-guess matching rules (digits-only fallback, `BLOCKED_SET_TOKENS`).
- `data/processed/class_sales.parquet` (47,892 per-sale rows with `grade` (`psa_10`/`psa_9`/`psa_8`/…/NaN = ungraded), `sale_date`, `price`, `card_slug`) and `data/processed/class_liquidity.csv`.
- `data/reference/pop_snapshots.csv` (13 seed cards, single snapshot 2026-09-15 — the reference set for pop-growth context; no measured growth exists yet → scenario grid, documented).

## Model definition (binding, per spec §4)

Per card, Monte Carlo (10,000 draws, seed 42):

1. **Grade outcomes:** posterior `Dirichlet(1 + n_10, 1 + n_9, 1 + n_le8)` over (PSA 10, PSA 9, PSA ≤8), from the card's GemRate PSA grade counts (uniform prior Dirichlet(1,1,1)). Limitation stated everywhere: historical pops reflect past submitters' selection (optimistic-biased for a new submission).
2. **Proceeds:** sale price drawn from the card's empirical 12-month sale-price distribution for the drawn grade class (`psa_10` / `psa_9` / `psa_8` as the ≤8 proxy — documented), times `(1 − 0.14)` marketplace fees.
3. **Cost:** raw acquisition price drawn from the card's empirical 12-month ungraded distribution + the fee tier's `fee_per_card` + `shipping_per_card`.
4. **Outputs:** `EV = mean(profit)`, `P(profit > 0)`, 90 % CI (5th/95th percentiles of profit), `thin_price_evidence` flag per distribution with < 5 sales.

**Sensitivity tornado (per card, reported for the top-20 EV cards + median-EV card):**
- **(a) gem-rate estimation:** posterior-spread EV vs point-MLE EV (MLE probabilities, no Dirichlet sampling).
- **(b) pop inflation:** PSA-10 proceeds deflated by monthly rates {0 %, 2 %, 5 %} (scenario grid — no measured per-card growth exists yet; the 13-seed-card reference set is one-snapshot; documented).
- **(c) price slippage:** proceeds × {1.0, 0.9, 0.8}.
- **(d) fee tier:** value ↔ bulk ↔ regular per the captured config.

**Henderson golden (from the spike, binding):** PSA counts (10: 2,070 · 9: 896 · ≤8: 163 of 3,129 total) → posterior mean p10 = 2,071 / 3,132 ≈ 0.66134 (PSA-specific 0.6616 rounded); the Dirichlet unit test pins this.

---

### Task 1: Fee config + liquidity subset + GemRate verification probe

**Files:**
- Create: `scripts/capture_grading_fees.py`
- Create: `scripts/build_grading_subset.py`
- Test: `tests/test_grading_fees.py`, `tests/test_grading_subset.py`

**Interfaces:**
- Consumes: `class_sales.parquet`, `class_liquidity.csv`, psacard.com (live).
- Produces:
  - `data/reference/grading_fees.json` — schema:
    ```json
    {"captured_date": "YYYY-MM-DD", "source_url": "https://www.psacard.com/pricing",
     "tiers": {"bulk": {"fee_per_card": 0.0, "min_cards": 20, "shipping_per_card": 0.0, "notes": ""},
               "value": {"fee_per_card": 0.0, "min_cards": 1, "shipping_per_card": 0.0, "notes": ""},
               "regular": {"fee_per_card": 0.0, "min_cards": 1, "shipping_per_card": 0.0, "notes": ""}},
     "default_tier": "value",
     "shipping_model": "per-card allocation; see notes"}
    ```
    `validate_fees_config(cfg: dict) -> None` (in the runner, tested): raises on unknown tier in `default_tier`, missing keys, non-positive fee, missing `captured_date`/`source_url`.
  - `data/reference/grading_subset.csv` — cards with ≥ 24 ungraded sales in the trailing 365 days (from `class_sales.parquet`: `grade.isna()`, `sale_date >= max(sale_date) - 365 days`), with `player_name, mlb_id, class_year, set_slug, scp_url, card_slug, ungraded_sales_12m`. Sorted by sales desc.
  - **GemRate verification probe:** run `fetch_pop` on 3 subset cards (top-3 by sales) and record match quality (matched description, psa_10_pop, gem_rate). **If 2+ of 3 fail to match (Bowman set coverage or query-construction gap), STOP and report** — do not proceed to the full scrape (escalation options: adjust `build_query` for Bowman set names, or restrict the subset to Chrome-family — controller decision at that point).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grading_fees.py
import pytest

from capture_grading_fees import validate_fees_config


def _cfg(**kw):
    base = {
        "captured_date": "2026-09-19", "source_url": "https://www.psacard.com/pricing",
        "tiers": {
            "bulk": {"fee_per_card": 18.99, "min_cards": 20, "shipping_per_card": 2.0, "notes": ""},
            "value": {"fee_per_card": 24.99, "min_cards": 1, "shipping_per_card": 2.0, "notes": ""},
            "regular": {"fee_per_card": 74.99, "min_cards": 1, "shipping_per_card": 2.0, "notes": ""},
        },
        "default_tier": "value", "shipping_model": "per-card allocation; see notes",
    }
    base.update(kw)
    return base


def test_valid_config_passes():
    validate_fees_config(_cfg())


def test_unknown_default_tier_rejected():
    with pytest.raises(ValueError, match="default_tier"):
        validate_fees_config(_cfg(default_tier="express"))


def test_missing_date_rejected():
    cfg = _cfg()
    del cfg["captured_date"]
    with pytest.raises(ValueError, match="captured_date"):
        validate_fees_config(cfg)


def test_nonpositive_fee_rejected():
    cfg = _cfg()
    cfg["tiers"]["value"]["fee_per_card"] = 0
    with pytest.raises(ValueError, match="fee_per_card"):
        validate_fees_config(cfg)
```

```python
# tests/test_grading_subset.py
import pandas as pd

from build_grading_subset import liquidity_subset


def _sales():
    rows = []
    end = pd.Timestamp("2026-09-19")
    for i in range(30):
        rows.append({"card_slug": "s/a-1", "grade": None, "price": 5.0,
                     "sale_date": end - pd.Timedelta(days=i * 10)})
        rows.append({"card_slug": "s/b-2", "grade": None, "price": 5.0,
                     "sale_date": end - pd.Timedelta(days=i * 12)})
    for i in range(10):
        rows.append({"card_slug": "s/c-3", "grade": None, "price": 5.0,
                     "sale_date": end - pd.Timedelta(days=i * 30)})
    # graded sales never count toward the ungraded subset
    for i in range(40):
        rows.append({"card_slug": "s/d-4", "grade": "psa_10", "price": 50.0,
                     "sale_date": end - pd.Timedelta(days=i * 5)})
    return pd.DataFrame(rows)


def test_subset_counts_trailing_365d_ungraded_only():
    out = liquidity_subset(_sales())
    slugs = set(out["card_slug"])
    assert "s/a-1" in slugs and "s/b-2" in slugs  # 30 / 27 ungraded sales in window
    assert "s/c-3" not in slugs  # 10 < 24
    assert "s/d-4" not in slugs  # graded only
    assert out.iloc[0]["ungraded_sales_12m"] >= out.iloc[-1]["ungraded_sales_12m"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grading_fees.py tests/test_grading_subset.py -v`
Expected: FAIL — module import errors.

- [ ] **Step 3: Implement both scripts; capture fees live**

Implement `validate_fees_config` + `capture_grading_fees.py` (fetch the PSA pricing page with the repo's web conventions; parse the current tier prices; if parsing fails or the page 403s/structure-changed → **STOP and escalate**, no manual typing) and `build_grading_subset.py` (compute `liquidity_subset(sales)` + write the CSV + print the subset size). Run: fee capture (writes grading_fees.json, validated), subset build (writes grading_subset.csv, prints count).

- [ ] **Step 4: GemRate verification probe (3 cards, live)**

Run `fetch_pop` on the top-3 subset cards (playwright per collector conventions, from the repo root: `PLAYWRIGHT_BROWSERS_PATH=.pw-browsers`). Record matched descriptions + pops in the report. Apply the STOP rule above.

- [ ] **Step 5: Commit**

```bash
git add scripts/capture_grading_fees.py scripts/build_grading_subset.py tests/test_grading_fees.py tests/test_grading_subset.py data/reference/grading_fees.json data/reference/grading_subset.csv
git commit -m "feat: grading fee config (live-captured, dated) + liquidity subset + GemRate probe"
```

---

### Task 2: Pop collection for the subset (controller-run)

**Files:**
- Create: `scripts/collect_pops_class.py`

**Interfaces:**
- Consumes: `data/reference/grading_subset.csv`, the reviewed `pop_gemrate` collector.
- Produces: `data/reference/pop_snapshots_class.csv` — columns `date, card_slug, matched_description, psa_10_pop, psa_9_pop, psa_le8_pop, psa_total_pop, gem_rate, gemrate_id` (+ `match_quality` notes for fallback matches per the spike's conventions). Resumable (skip cards with a same-day row; append-mode like the seed collector); per-card failures recorded with empty pop fields and the run continues (reviewed behavior).

- [ ] **Step 1: Implement + smoke-test the runner (3 cards offline-verified)**

`collect_pops_class.py`: iterate the subset; build the GemRate query per `pop_gemrate.build_query` conventions from each card's `scp_url`; `fetch_pop` per card; append rows; failure rows kept with reason. Smoke: run on the same 3 probe cards, verify rows match Task 1's probe output exactly.

- [ ] **Step 2: Controller runs the full collection (background, ~35 s/card)**

Controller command (not the implementer's): `PLAYWRIGHT_BROWSERS_PATH=.pw-browsers .venv/bin/python scripts/collect_pops_class.py` — expected duration = 35 s × subset size. Stop-and-report triggers: match-failure rate > 15 %, or 3 consecutive Cloudflare failures.

- [ ] **Step 3: Acceptance + commit**

Acceptance: every subset card has a row (matched or failed-with-reason); failed rows ≤ 15 %; spot-check 2 known pops against GemRate manually (report). Commit:
```bash
git add scripts/collect_pops_class.py data/reference/pop_snapshots_class.csv
git commit -m "data: GemRate pop snapshots for the grading subset"
```

---

### Task 3: `src/cardprice/grading_ev.py` (Monte Carlo EV + sensitivity)

**Files:**
- Create: `src/cardprice/grading_ev.py`
- Test: `tests/test_grading_ev.py`

**Interfaces:**
- Consumes: pops frame (Task 2), sales frame, fees config.
- Produces (used by Task 4):
  - `dirichlet_posterior(n10: int, n9: int, n_le8: int) -> np.ndarray` — posterior alpha vector `(1+n10, 1+n9, 1+n_le8)`.
  - `grade_price_distributions(sales: pd.DataFrame, card_slug: str, months: int = 12) -> dict[str, np.ndarray]` — `{"psa_10": [...], "psa_9": [...], "psa_8": [...], "raw": [...]}` (trailing months; ≤8 proxy = `psa_8`, documented).
  - `monte_carlo_ev(pops_row: dict, dists: dict[str, np.ndarray], fee_per_card: float, shipping_per_card: float, n_draws: int = 10000, seed: int = 42, marketplace_fee: float = 0.14, mle_only: bool = False, pop_inflation_monthly: float = 0.0, slippage: float = 1.0) -> dict` — returns `{"ev", "p_profit", "ci_5", "ci_95", "ev_mle" (when mle_only), "flags": {"thin_price_evidence": [...]}}`. MC per the model definition; `pop_inflation_monthly` deflates the drawn psa_10 price by `(1 - rate)` per month for 12 months compounded (the grading turnaround horizon — documented as 12-month-forward deflation); `slippage` multiplies proceeds.
  - `sensitivity_tornado(card, pops_row, dists, fees_cfg) -> pd.DataFrame` — the four arms per the model definition: rows `{arm, variant, ev, p_profit}` for (a) posterior vs MLE, (b) inflation {0, 2, 5}% , (c) slippage {1.0, 0.9, 0.8}, (d) each captured fee tier.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grading_ev.py
import numpy as np
import pandas as pd
import pytest

from cardprice.grading_ev import (
    dirichlet_posterior,
    grade_price_distributions,
    monte_carlo_ev,
    sensitivity_tornado,
)


def test_dirichlet_posterior_henderson_golden():
    # spike counts: PSA 10: 2070, 9: 896, <=8: 163 (total 3,129)
    alpha = dirichlet_posterior(2070, 896, 163)
    assert alpha.tolist() == [2071, 897, 164]
    assert (alpha / alpha.sum())[0] == pytest.approx(2071 / 3132, abs=1e-4)  # 0.6613


def test_dirichlet_posterior_uniform_prior_empty_counts():
    alpha = dirichlet_posterior(0, 0, 0)
    assert alpha.tolist() == [1, 1, 1]


def test_grade_price_distributions_trailing_window_and_proxies():
    end = pd.Timestamp("2026-09-19")
    rows = []
    for price, grade in [(100, "psa_10"), (110, "psa_10"), (40, "psa_9"), (15, "psa_8"), (5, None)]:
        rows.append({"card_slug": "s/x", "grade": grade, "price": float(price),
                     "sale_date": end - pd.Timedelta(days=30)})
    rows.append({"card_slug": "s/x", "grade": "psa_10", "price": 999.0,
                 "sale_date": end - pd.Timedelta(days=400)})  # outside window
    dists = grade_price_distributions(pd.DataFrame(rows), "s/x")
    assert sorted(dists["psa_10"]) == [100.0, 110.0]
    assert dists["psa_9"] == [40.0]
    assert dists["psa_8"] == [15.0]  # the <=8 proxy
    assert dists["raw"] == [5.0]


def _dists():
    rng = np.random.default_rng(7)
    return {
        "psa_10": rng.uniform(90, 110, 50), "psa_9": rng.uniform(35, 45, 20),
        "psa_8": rng.uniform(12, 18, 20), "raw": rng.uniform(4, 6, 50),
    }


def test_monte_carlo_ev_deterministic_and_reasonable():
    pops = {"psa_10_pop": 2070, "psa_9_pop": 896, "psa_le8_pop": 163}
    a = monte_carlo_ev(pops, _dists(), fee_per_card=25.0, shipping_per_card=2.0)
    b = monte_carlo_ev(pops, _dists(), fee_per_card=25.0, shipping_per_card=2.0)
    assert a["ev"] == b["ev"]  # seeded determinism
    # hand-computed sign: p10 ~ 0.66 at ~100 * 0.86 = 86; p9 ~ 0.29 at ~40 * 0.86 = 34;
    # p8 ~ 0.05 at ~15 * 0.86 = 13 -> expected proceeds ~ 0.66*86 + 0.29*34 + 0.05*13
    # ~ 56.8 + 9.9 + 0.65 = 67.3; cost ~ 5 + 27 = 32 -> EV ~ +35
    assert 20 < a["ev"] < 50
    assert 0.9 < a["p_profit"] <= 1.0
    assert a["flags"]["thin_price_evidence"] == []


def test_thin_price_evidence_flagged_not_borrowed():
    dists = _dists()
    dists["psa_9"] = np.array([40.0, 41.0])  # < 5 sales
    out = monte_carlo_ev({"psa_10_pop": 10, "psa_9_pop": 5, "psa_le8_pop": 3}, dists,
                         fee_per_card=25.0, shipping_per_card=2.0)
    assert "psa_9" in out["flags"]["thin_price_evidence"]


def test_sensitivity_arms_present_and_directional():
    pops = {"psa_10_pop": 2070, "psa_9_pop": 896, "psa_le8_pop": 163}
    cfg = {"tiers": {"bulk": {"fee_per_card": 19.0}, "value": {"fee_per_card": 25.0},
                     "regular": {"fee_per_card": 75.0}, "shipping_per_card": 2.0}}
    tor = sensitivity_tornado("s/x", pops, _dists(), cfg)
    assert set(tor["arm"]) == {"gem_rate", "pop_inflation", "slippage", "fee_tier"}
    base = tor[(tor["arm"] == "slippage") & (tor["variant"] == "1.0")]["ev"].iloc[0]
    slip80 = tor[(tor["arm"] == "slippage") & (tor["variant"] == "0.8")]["ev"].iloc[0]
    assert slip80 < base  # slippage hurts
    bulk = tor[(tor["arm"] == "fee_tier") & (tor["variant"] == "bulk")]["ev"].iloc[0]
    regular = tor[(tor["arm"] == "fee_tier") & (tor["variant"] == "regular")]["ev"].iloc[0]
    assert bulk > regular
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grading_ev.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/cardprice/grading_ev.py`**

```python
# src/cardprice/grading_ev.py
"""Grading-arbitrage EV model: buy-raw -> grade -> sell-PSA-10, Monte Carlo.

DESK MODEL — every EV is an estimate, not realized P&L. Grade-outcome
probabilities are Dirichlet posteriors over historical pop counts, which
reflect past submitters' selection (optimistic-biased for a new submission);
the limitation is stated wherever EVs are presented. psa_8 prices proxy the
PSA <=8 outcome bucket (most liquid lower grade; documented approximation).
"""

import numpy as np
import pandas as pd

GRADE_BUCKETS = ("psa_10", "psa_9", "psa_8")
MIN_SALES = 5


def dirichlet_posterior(n10: int, n9: int, n_le8: int) -> np.ndarray:
    """Uniform-prior Dirichlet alpha over (PSA 10, PSA 9, PSA <=8)."""
    return np.array([1 + n10, 1 + n9, 1 + n_le8], dtype=float)


def grade_price_distributions(
    sales: pd.DataFrame, card_slug: str, months: int = 12
) -> dict[str, np.ndarray]:
    """Empirical 12-month sale-price arrays per grade class (raw = ungraded)."""
    end = sales["sale_date"].max()
    lo = end - pd.Timedelta(days=months * 30)
    s = sales[(sales["card_slug"] == card_slug) & (sales["sale_date"] >= lo)]
    out = {g: s.loc[s["grade"] == g, "price"].to_numpy(dtype=float) for g in GRADE_BUCKETS}
    out["raw"] = s.loc[s["grade"].isna(), "price"].to_numpy(dtype=float)
    return out


def monte_carlo_ev(
    pops_row: dict,
    dists: dict[str, np.ndarray],
    fee_per_card: float,
    shipping_per_card: float,
    n_draws: int = 10000,
    seed: int = 42,
    marketplace_fee: float = 0.14,
    mle_only: bool = False,
    pop_inflation_monthly: float = 0.0,
    slippage: float = 1.0,
) -> dict:
    """MC EV of the grade pipeline; see module docstring for the model."""
    rng = np.random.default_rng(seed)
    alpha = dirichlet_posterior(
        int(pops_row["psa_10_pop"]), int(pops_row["psa_9_pop"]), int(pops_row["psa_le8_pop"])
    )
    probs = rng.dirichlet(alpha, size=n_draws) if not mle_only else np.tile(
        alpha / alpha.sum(), (n_draws, 1)
    )
    outcomes = rng.choice(3, size=n_draws, p=probs.mean(axis=0)) if mle_only else None
    if mle_only:
        p10, p9, p8 = probs[0]
    else:
        p10, p9, p8 = probs[:, 0], probs[:, 1], probs[:, 2]
    # grade for each draw: sample per-draw outcome from its own prob vector
    u = rng.uniform(size=n_draws)
    if mle_only:
        outcome = np.where(u < p10, 0, np.where(u < p10 + p9, 1, 2))
    else:
        cum = np.cumsum(probs, axis=1)
        outcome = (u[:, None] > cum[:, :2]).sum(axis=1)  # 0,1,2 via thresholds

    def _draw(bucket: str) -> np.ndarray:
        pool = dists.get(bucket, np.array([]))
        if len(pool) == 0:
            return np.full(n_draws, np.nan)
        return pool[rng.integers(0, len(pool), size=n_draws)]

    p10_price = _draw("psa_10") * ((1 - pop_inflation_monthly) ** 12)
    prices = np.select(
        [outcome == 0, outcome == 1], [p10_price, _draw("psa_9")], default=_draw("psa_8")
    )
    proceeds = prices * slippage * (1 - marketplace_fee)
    cost = _draw("raw") + fee_per_card + shipping_per_card
    profit = proceeds - cost
    profit = profit[~np.isnan(profit)]
    flags = {"thin_price_evidence": [b for b in (*GRADE_BUCKETS, "raw")
                                     if len(dists.get(b, [])) < MIN_SALES]}
    return {
        "ev": float(np.mean(profit)),
        "p_profit": float((profit > 0).mean()),
        "ci_5": float(np.percentile(profit, 5)),
        "ci_95": float(np.percentile(profit, 95)),
        "n_draws_used": int(len(profit)),
        "flags": flags,
    }


def sensitivity_tornado(card_slug: str, pops_row: dict, dists: dict, fees_cfg: dict) -> pd.DataFrame:
    """Four-arm sensitivity per the plan's model definition."""
    ship = fees_cfg["tiers"]["shipping_per_card"] if "shipping_per_card" in fees_cfg.get("tiers", {}) else fees_cfg["tiers"]["value"]["shipping_per_card"]
    base_fee = fees_cfg["tiers"]["value"]["fee_per_card"]
    rows = []
    for arm, variant, kw in (
        [("gem_rate", "posterior", {}), ("gem_rate", "mle", {"mle_only": True})]
        + [("pop_inflation", f"{r}%", {"pop_inflation_monthly": r / 100}) for r in (0, 2, 5)]
        + [("slippage", f"{s}", {"slippage": float(s)}) for s in (1.0, 0.9, 0.8)]
        + [("fee_tier", t, {"fee_per_card": fees_cfg["tiers"][t]["fee_per_card"]}) for t in fees_cfg["tiers"]]
    ):
        res = monte_carlo_ev(pops_row, dists, fee_per_card=kw.pop("fee_per_card", base_fee),
                             shipping_per_card=ship, **kw)
        rows.append({"card_slug": card_slug, "arm": arm, "variant": variant,
                     "ev": res["ev"], "p_profit": res["p_profit"]})
    return pd.DataFrame(rows)
```

(Implementer note: the `outcomes`/`p10/p9/p8` lines above contain redundant sketch logic (the `rng.choice` line is dead) — keep ONLY the clean `u`/cumsum implementation; ruff will flag the dead lines otherwise. The shipping lookup in `sensitivity_tornado` should read `fees_cfg["tiers"]["value"]["shipping_per_card"]` (fees live inside tiers) — simplify accordingly and adjust the test's cfg shape to match your implementation.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grading_ev.py -v` → 7 passed; full `pytest` green; `ruff check src tests scripts` clean.

- [ ] **Step 5: Commit**

```bash
git add src/cardprice/grading_ev.py tests/test_grading_ev.py
git commit -m "feat: grading-arb Monte Carlo EV + sensitivity tornado"
```

---

### Task 4: Real run + findings doc

**Files:**
- Create: `scripts/run_grading_ev.py`
- Create: `docs/findings/2026-09-19-t5-grading-arb-ev.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `data/processed/grading_ev.csv` — ranked by EV desc: columns `card_slug, player_name, class_year, ev, p_profit, ci_5, ci_95, psa_10_pop, gem_rate, thin_price_evidence, fee_tier, top_sensitivities` (top-2 EV-delta arms from the tornado for that card); plus `data/processed/grading_ev_tornado.csv` (top-20 + median card).

- [ ] **Step 1: Run the evaluation**

`scripts/run_grading_ev.py`: per subset card with a matched pop row: dists from `class_sales.parquet`, MC EV + tornado for the flagged cards; write both CSVs; print summary (cards scored, mean/median EV, count EV>0, thin flags share). Acceptance: every matched card scored; failed-pop cards counted, not scored; deterministic rerun (two runs produce identical CSVs).

- [ ] **Step 2: Write the findings doc**

Sections: Data (subset size, pop match rate, fee config date/source, sales window) → Headline (ranked top-10 table with EV/p_profit/CI/flags — every number from `grading_ev.csv`) → Distribution (EV stats, share positive, thin-flag share) → Sensitivity (which arms move EV most — tornado summary) → Caveats (**desk model, not realized P&L**; pop representativeness bias; psa_8 proxy; no measured pop growth (scenario grid); fee allocations assumed; slippage/liquidity reality of selling 10-20 graded copies into thin markets; repeat-submission correlation ignored) → Reproduce. Labels: every EV figure carries "model estimate".

- [ ] **Step 3: Final verification + commit**

Run: `pytest && ruff check src tests scripts` → green.
```bash
git add scripts/run_grading_ev.py docs/findings/2026-09-19-t5-grading-arb-ev.md
git commit -m "feat: grading-arb EV real run + findings"
```

---

## Self-Review

**Spec coverage:** §3 data (pop extension with liquidity filter ≥24 sales; paired raw/PSA-10 empirical prices; dated fee config with validation) ✓; §4 model (Dirichlet posterior, MC EV, P(profit>0), 90% CI, 4-arm tornado incl. scenario-grid pop inflation) ✓; §5 outputs (ranked CSV + findings, desk-model labels) ✓; §6 error handling (thin flags, match failures excluded+counted, stale snapshot tolerated — N/A here since pops are fresh) ✓; §7 testing (Henderson golden 0.6613/0.6616, Dirichlet unit tests, fee validation, seeded determinism) ✓.

**Placeholder scan:** two implementer notes flag sketch lines to remove/simplify (dead `outcomes` line; shipping lookup) — the pinned tests define the contract those lines must satisfy; no TBDs. The GemRate Bowman-coverage risk is an explicit STOP-and-escalate checkpoint (Task 1 Step 4), not a hidden assumption.

**Type consistency:** pop row dict keys (`psa_10_pop, psa_9_pop, psa_le8_pop`) flow from Task 2's CSV → `monte_carlo_ev` ✓; fees config `tiers.<name>.fee_per_card/shipping_per_card` flows from Task 1's schema → tornado ✓; flags column `thin_price_evidence` semicolon-joined into the ranked CSV ✓.
