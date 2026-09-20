# scripts/run_hierarchical.py
"""Real hierarchical fits + pooled baselines + LOO comparison (T4 Task 3).

One group per invocation (`--group hitter|pitcher`); the controller runs the
two groups as separate long jobs. Stages (one progress line each):

1. prep      — load panel_class.parquet, drop rows with NaN rookie_year /
               mlb_id (counted; `build_model_frame` fails fast on them), build
               the z-scored model frame via `cardprice.hierarchical`.
2. fit       — `fit_with_ladder` walks SIMPLIFICATION_LADDER (4 chains x
               1000 tune + 1000 draws by default) until the scripted gate
               passes. Priors are never re-specified here: the Task-2 wrapper
               owns them (bambi 0.21 "group" -> "group_specific" translation
               lives inside fit_model). If even the floor rung fails the gate,
               the honest failing convergence JSON is written and the run
               CONTINUES through every artifact (no silent success, no crash).
3. baselines — pooled LASSO and player fixed-effect null, both 5-fold
               out-of-fold (seed 42) on the SAME model frame. LASSO mirrors
               the registered walk-forward harness: per fold, train-median
               imputation + train z-scoring (ddof=0, sd 0 -> 1) +
               LassoCV(cv=5); features are surprise, market_ret_3m,
               price_level, and career_stage dummies (drop_first). The
               player-FE null predicts out-of-fold player means (unseen
               player -> fold-global train mean).
4. loo       — hierarchical ELPD from arviz.loo (PSIS) on the winning rung.
               fit_with_ladder returns only InferenceData, so a
               structure-identical bambi model is rebuilt and
               `compute_log_likelihood` adds the log_likelihood group
               (priors do not enter the log-likelihood). The ELPD is
               normalized to a per-observation mean so it is comparable to
               the baseline LPDs; winner by highest value.
5. artifacts — written to data/processed/ (gitignored):
               hierarchical_convergence_{group}.json, hierarchical_loo_{group}.json,
               hierarchical_variance_components_{group}.csv (param/mean/sd/
               hdi_3%/hdi_97%, residual sigma included, 94% HDI),
               hierarchical_slopes_{group}.csv (per class_position shrunken
               surprise-slope posterior mean + 94% HDI; the TOTAL class slope
               = common surprise effect + group deviation, HDI over the summed
               posterior; header-only when the winning rung has no slope term),
               hierarchical_fit_{group}.nc (full fit archive).

Gaussian-LPD caveat (kept identical to GAUSSIAN_LPD_CAVEAT, which is embedded
in every hierarchical_loo_{group}.json): baseline ELPDs are a Gaussian plug-in
approximation — a constant sigma (OOF residual std, ddof=0) in a Gaussian log
predictive density, not a Bayesian posterior predictive. Baselines score ALL
model-frame rows (median-imputed predictors) while the hierarchical ELPD
scores dropna-complete rows only, so the comparison is indicative, not exact.

Fit-archive format note: the brief asks for az.to_netcdf, but arviz 1.3
InferenceData is an xarray DataTree whose to_netcdf REQUIRES NETCDF4 group
support (netCDF4/h5netcdf), neither installed, and new dependencies are
prohibited in this task. `flatten_fit_to_netcdf` therefore writes ONE netCDF3
file via xarray's scipy engine with groups flattened: posterior and
sample_stats variables stay bare (xr.merge fails loudly on any collision),
observed_data/log_likelihood variables get observed__/loglik__ prefixes.
Attrs are stripped (scipy's netCDF3 writer cannot encode numpy unicode attrs).
"""

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LassoCV

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cardprice.hierarchical import (
    SIMPLIFICATION_LADDER,
    build_model_frame,
    fit_with_ladder,
    variance_components,
)

PANEL_PARQUET = "data/processed/panel_class.parquet"
OUT_DIR = "data/processed"
N_SPLITS = 5
SEED = 42
LASSO_NUMERIC = ["surprise", "market_ret_3m", "price_level"]
SLOPE_TERM = "surprise|class_position"

GAUSSIAN_LPD_CAVEAT = (
    "Baseline ELPDs are a Gaussian plug-in approximation: a constant sigma "
    "(OOF residual std, ddof=0) in a Gaussian log predictive density, not a "
    "Bayesian posterior predictive. Baselines score ALL model-frame rows "
    "(median-imputed predictors); the hierarchical ELPD is PSIS-LOO on "
    "dropna-complete rows only, normalized to a per-observation mean. "
    "Comparison is indicative, not exact."
)

FIT_NETCDF_FORMAT = (
    "netCDF3 single file via xarray scipy engine; arviz groups flattened "
    "(posterior/sample_stats bare, observed__/loglik__ prefixed) because "
    "arviz 1.3 DataTree.to_netcdf requires absent NETCDF4 backends"
)


def pooled_lasso_lpd(
    frame: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series, sigma: float
) -> float:
    """Mean Gaussian log predictive density of the LASSO out-of-fold predictions.

    `-0.5*log(2*pi*sigma^2) - (y-pred)^2/(2*sigma^2)`, averaged over rows,
    with sigma = OOF residual std (ddof=0) supplied by the caller. `frame`
    is accepted for interface symmetry; the metric is a pure function of
    (y_true, y_pred, sigma). Documented as an ELPD approximation for baseline
    comparison — see GAUSSIAN_LPD_CAVEAT.
    """
    del frame  # interface symmetry only
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(-0.5 * np.log(2 * np.pi * sigma**2) - resid**2 / (2 * sigma**2)))


def player_fe_null_lpd(
    frame: pd.DataFrame, group_col: str = "mlb_id", y_col: str = "y", fold_col: str = "fold"
) -> float:
    """Player fixed-effect null: same Gaussian LPD with player-mean predictions.

    Each fold is predicted by the player's mean y in the OTHER folds; a
    player unseen in the training folds gets the fold-global train mean.
    sigma = OOF residual std (ddof=0), identical metric to the LASSO baseline.
    """
    y = frame[y_col].astype(float)
    oof = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in pd.unique(frame[fold_col]):
        val = frame[fold_col] == fold
        train = frame[~val]
        means = train.groupby(group_col)[y_col].mean()
        global_mean = float(train[y_col].mean())
        oof.loc[val] = frame.loc[val, group_col].map(means).fillna(global_mean).astype(float)
    sigma = float((y - oof).std(ddof=0))
    return pooled_lasso_lpd(frame, y, oof, sigma)


def assign_folds(n_rows: int, n_splits: int = N_SPLITS, seed: int = SEED) -> pd.Series:
    """Deterministic fold labels 0..n_splits-1 (seeded shuffle of balanced blocks)."""
    rng = np.random.default_rng(seed)
    fold = np.arange(n_rows) % n_splits
    rng.shuffle(fold)
    return pd.Series(fold, name="fold")


def oof_lasso_predictions(
    frame: pd.DataFrame, fold_col: str = "fold", y_col: str = "y", seed: int = SEED
) -> pd.Series:
    """5-fold out-of-fold pooled-LASSO predictions on the model frame.

    Mirrors the registered walk-forward harness per fold: train-median
    imputation, train mean/sd (ddof=0, sd 0 -> 1) z-scoring, LassoCV(cv=5).
    Features: surprise, market_ret_3m, price_level + career_stage dummies
    (drop_first). Imputation absorbs the NaN predictors that the hierarchical
    fit listwise-deletes, so every frame row receives a prediction.
    """
    dummies = pd.get_dummies(
        frame["career_stage"], prefix="career_stage", drop_first=True, dtype=float
    )
    X = pd.concat([frame[LASSO_NUMERIC].astype(float), dummies], axis=1)
    y = frame[y_col].astype(float)
    oof = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in pd.unique(frame[fold_col]):
        val = (frame[fold_col] == fold).to_numpy()
        X_train, X_val = X[~val], X[val]
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_val = X_val.fillna(medians)
        # standardize with TRAIN mean/sd only (walkforward.py idiom)
        mean, sd = X_train.mean(), X_train.std(ddof=0).replace(0, 1)
        X_train = (X_train - mean) / sd
        X_val = (X_val - mean) / sd
        model = LassoCV(cv=5, random_state=seed, max_iter=10000).fit(X_train, y[~val])
        oof.loc[val] = model.predict(X_val)
    return oof


def _winning_formula(fit: az.InferenceData) -> str:
    """Recover the ladder rung from the posterior's variable set."""
    var_names = set(fit.posterior.data_vars)
    if SLOPE_TERM in var_names:
        return SIMPLIFICATION_LADDER[0]
    if "1|class_position" in var_names:
        return SIMPLIFICATION_LADDER[1]
    return SIMPLIFICATION_LADDER[2]


def hierarchical_mean_elpd(fit: az.InferenceData, frame: pd.DataFrame) -> tuple[float, int]:
    """Per-observation mean PSIS-LOO ELPD for the fitted rung; (elpd, n_points).

    Rebuilds a structure-identical bambi model (default priors — priors do
    not enter the log-likelihood) and calls `compute_log_likelihood`, which
    mutates `fit` in place: the log_likelihood group is added and `mu` is
    appended to the posterior (both kept in the .nc archive).
    """
    model = bmb.Model(_winning_formula(fit), frame, dropna=True)
    model.compute_log_likelihood(fit)
    loo = az.loo(fit)
    n = int(loo["n_data_points"])
    return float(loo["elpd"]) / n, n


def loo_comparison(fit: az.InferenceData, frame: pd.DataFrame) -> dict:
    """{"hierarchical_elpd", "lasso_elpd", "player_fe_elpd", "winner"} by highest ELPD.

    All three values are per-observation mean log predictive densities: the
    hierarchical one is PSIS-LOO on dropna-complete rows; the baselines are
    out-of-fold on all frame rows (folds seeded 42). GAUSSIAN_LPD_CAVEAT
    applies to the baseline side of the comparison. Ties resolve to the
    earlier key (hierarchical first).
    """
    hier_elpd, _ = hierarchical_mean_elpd(fit, frame)
    frame = frame.assign(fold=assign_folds(len(frame)).to_numpy())
    oof = oof_lasso_predictions(frame)
    sigma = float((frame["y"].astype(float) - oof).std(ddof=0))
    elpds = {
        "hierarchical": hier_elpd,
        "lasso": pooled_lasso_lpd(frame, frame["y"], oof, sigma),
        "player_fe": player_fe_null_lpd(frame),
    }
    winner = max(elpds, key=elpds.get)
    return {
        "hierarchical_elpd": elpds["hierarchical"],
        "lasso_elpd": elpds["lasso"],
        "player_fe_elpd": elpds["player_fe"],
        "winner": winner,
    }


def shrunken_slopes(fit: az.InferenceData) -> pd.DataFrame:
    """Per class_position shrunken surprise-slope posterior mean + 94% HDI.

    The slope is the TOTAL class slope: bambi group terms are zero-centered
    deviations, so the common `surprise` effect is added back and the HDI is
    taken over the summed posterior (draw-level correlation preserved).
    Header-only frame when the winning rung dropped the slope term.
    """
    cols = ["class_position", "mean", "sd", "hdi_3%", "hdi_97%"]
    if SLOPE_TERM not in fit.posterior.data_vars:
        return pd.DataFrame(columns=cols)
    total = fit.posterior["surprise"] + fit.posterior[SLOPE_TERM]
    (level_dim,) = [d for d in total.dims if d not in ("chain", "draw")]
    rows = []
    for level in total[level_dim].values:
        draws = total.sel({level_dim: level}).values.ravel()
        hdi = np.asarray(az.hdi(draws, prob=0.94)).ravel()
        rows.append(
            {
                "class_position": str(level),
                "mean": float(draws.mean()),
                "sd": float(draws.std(ddof=1)),
                "hdi_3%": float(hdi[0]),
                "hdi_97%": float(hdi[1]),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def flatten_fit_to_netcdf(fit: az.InferenceData, path: Path) -> None:
    """Write the fit as ONE netCDF3 file (xarray scipy engine).

    Deviation from the brief's az.to_netcdf (see module docstring): arviz 1.3
    InferenceData is a DataTree requiring NETCDF4 group support, and neither
    netCDF4 nor h5netcdf is installed (new dependencies prohibited). Groups
    are flattened into a single Dataset — posterior and sample_stats stay
    bare (xr.merge raises on any name collision), observed_data and
    log_likelihood get observed__/loglik__ prefixes — and all attrs are
    stripped (scipy's netCDF3 writer cannot encode numpy unicode attrs).
    """
    parts = []
    for group, prefix in (
        ("posterior", ""),
        ("sample_stats", ""),
        ("observed_data", "observed__"),
        ("log_likelihood", "loglik__"),
    ):
        if not hasattr(fit, group):
            continue
        ds = getattr(fit, group)
        ds = ds if isinstance(ds, xr.Dataset) else ds.dataset
        parts.append(ds.rename({v: f"{prefix}{v}" for v in ds.data_vars}))
    merged = xr.merge(parts)
    merged.attrs = {}
    for var in merged.variables:
        merged[var].attrs = {}
    merged.to_netcdf(str(path), engine="scipy")


def run_group(
    group: str,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    panel_path: str = PANEL_PARQUET,
    out_dir: str = OUT_DIR,
) -> dict:
    """Full pipeline for one group; returns the LOO dict. Never raises on a
    failed convergence gate — the honest failing JSON is written instead."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{group}] prep: loading {panel_path}", flush=True)
    panel = pd.read_parquet(panel_path)
    key_na = panel["rookie_year"].isna() | panel["mlb_id"].isna()
    panel = panel[~key_na]
    frame, _scaling = build_model_frame(panel, group)
    n_fitted = int(frame.dropna().shape[0])
    print(
        f"[{group}] prep done: panel_rows={len(panel) + int(key_na.sum())} "
        f"dropped_nan_key_rows={int(key_na.sum())} frame_rows={len(frame)} "
        f"fit_rows={n_fitted} (bambi dropna listwise)",
        flush=True,
    )

    print(
        f"[{group}] fit started: ladder over {len(SIMPLIFICATION_LADDER)} rungs, "
        f"{chains} chains x {tune} tune + {draws} draws",
        flush=True,
    )
    fit, formula, rep = fit_with_ladder(frame, draws=draws, tune=tune, chains=chains)
    rung = SIMPLIFICATION_LADDER.index(formula)
    print(
        f"[{group}] fit done: rung={rung} gate_pass={rep['pass']} "
        f"rhat_max={rep['rhat_max']:.4f} ess_min={rep['ess_min']:.0f} "
        f"divergences={rep['divergences']} ({rep['divergence_rate']:.2%})",
        flush=True,
    )
    if not rep["pass"]:
        print(
            f"[{group}] WARNING: convergence gate FAILED on the ladder's floor rung — "
            f"writing the honest failing convergence JSON and continuing",
            flush=True,
        )

    conv = {
        "group": group,
        "rung": rung,
        "formula": formula,
        "pass": rep["pass"],
        "rhat_max": rep["rhat_max"],
        "ess_min": rep["ess_min"],
        "divergences": rep["divergences"],
        "divergence_rate": rep["divergence_rate"],
        "sampler": {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": 0.9,
            "random_seed": SEED,
        },
        "rows": {
            "panel": len(panel) + int(key_na.sum()),
            "dropped_nan_keys": int(key_na.sum()),
            "frame": len(frame),
            "fitted": n_fitted,
        },
        "fit_netcdf_format": FIT_NETCDF_FORMAT,
    }
    conv_path = out / f"hierarchical_convergence_{group}.json"
    conv_path.write_text(json.dumps(conv, indent=2) + "\n")
    vc_path = out / f"hierarchical_variance_components_{group}.csv"
    variance_components(fit).to_csv(vc_path, index=False)
    slopes_path = out / f"hierarchical_slopes_{group}.csv"
    shrunken_slopes(fit).to_csv(slopes_path, index=False)
    print(
        f"[{group}] artifacts: {conv_path.name}, {vc_path.name}, {slopes_path.name} written",
        flush=True,
    )

    print(
        f"[{group}] baselines: 5-fold OOF LASSO + player-FE null on {len(frame)} frame rows",
        flush=True,
    )
    print(
        f"[{group}] loo: rebuilt winning rung + compute_log_likelihood + az.loo (PSIS)",
        flush=True,
    )
    loo = loo_comparison(fit, frame)
    print(
        f"[{group}] loo done: hierarchical={loo['hierarchical_elpd']:.4f} "
        f"lasso={loo['lasso_elpd']:.4f} player_fe={loo['player_fe_elpd']:.4f} "
        f"winner={loo['winner']} (per-observation mean ELPD, highest wins)",
        flush=True,
    )

    loo_json = {
        "group": group,
        **loo,
        "caveat": GAUSSIAN_LPD_CAVEAT,
        "convergence_pass": rep["pass"],
        "rung": rung,
        "n_rows_frame": len(frame),
        "n_obs_fitted": n_fitted,
        "folds": {"n_splits": N_SPLITS, "seed": SEED},
        "normalization": "per-observation mean; hierarchical ELPD sum = value * n_obs_fitted",
    }
    loo_path = out / f"hierarchical_loo_{group}.json"
    loo_path.write_text(json.dumps(loo_json, indent=2) + "\n")
    nc_path = out / f"hierarchical_fit_{group}.nc"
    flatten_fit_to_netcdf(fit, nc_path)
    print(f"[{group}] artifacts: {loo_path.name}, {nc_path.name} written", flush=True)
    return loo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hierarchical real fits + baselines + LOO.")
    parser.add_argument("--group", required=True, choices=("hitter", "pitcher"))
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args(argv)
    run_group(args.group, draws=args.draws, tune=args.tune, chains=args.chains)


if __name__ == "__main__":
    main()
