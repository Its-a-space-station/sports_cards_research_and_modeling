"""Is a stat's effect consistent across seasons and positions?"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def season_consistency(frame: pd.DataFrame, stat: str) -> pd.DataFrame:
    rows = []
    for season, grp in frame.groupby("stats_season"):
        grp = grp.dropna(subset=[stat, "games", "age", "excess_ret"])
        if len(grp) < 5:
            rows.append(
                {
                    "season": int(season),
                    "coef": np.nan,
                    "se": np.nan,
                    "p_value": np.nan,
                    "n": len(grp),
                }
            )
            continue
        fit = smf.ols(f"excess_ret ~ {stat} + games + age", data=grp).fit(cov_type="HC1")
        rows.append(
            {
                "season": int(season),
                "coef": fit.params[stat],
                "se": fit.bse[stat],
                "p_value": fit.pvalues[stat],
                "n": len(grp),
            }
        )
    return pd.DataFrame(rows)


def position_interaction_test(frame: pd.DataFrame, stat: str) -> dict:
    df = frame[frame["position_group"].isin(["IF", "OF"])].dropna(
        subset=[stat, "games", "age", "excess_ret"]
    )
    fit = smf.ols(f"excess_ret ~ {stat} * C(position_group) + games + age", data=df).fit(
        cov_type="HC1"
    )
    interaction_terms = [t for t in fit.params.index if ":" in t]
    if not interaction_terms:
        return {"f_stat": np.nan, "p_value": np.nan}
    restriction = " and ".join(f"{t} = 0" for t in interaction_terms)
    ftest = fit.f_test(restriction)  # joint F-test: all interaction terms are zero
    return {"f_stat": float(ftest.fvalue), "p_value": float(ftest.pvalue)}


def player_random_effects(frame: pd.DataFrame, stat: str) -> dict:
    df = frame.dropna(subset=[stat, "games", "age", "excess_ret"])
    # Constant covariates are collinear with the intercept and make the MixedLM
    # fixed-effects design singular; drop them (a no-op when games/age vary).
    covariates = [c for c in [stat, "games", "age"] if df[c].std() > 0]
    fit = smf.mixedlm(f"excess_ret ~ {' + '.join(covariates)}", data=df, groups=df["mlb_id"]).fit(
        reml=True
    )
    ri_var = float(fit.cov_re.iloc[0, 0])
    resid_var = float(fit.scale)
    return {
        "random_intercept_var": ri_var,
        "resid_var": resid_var,
        "icc": ri_var / (ri_var + resid_var) if (ri_var + resid_var) > 0 else 0.0,
    }
