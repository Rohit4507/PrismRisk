"""M2 — model training & selection.

Champions compared on W70–79 (val), trained on W0–59, isotonically calibrated on
W60–69, and *reported* on the locked W80–91 test weeks (never used for selection):

  1. logreg_baseline      — interpretable linear baseline (median imputer + scaler)
  2. mlp_challenger       — neural net; subsampled, expected to struggle under drift
  3. xgboost_unconstrained — strong tree challenger, no governance constraints
  4. lightgbm_monotone    — LightGBM + Optuna with monotone_constraints (banking governance)

Usage:
  python -m prismrisk.train --features data/processed/features.parquet \
      --out models/champion_bundle.joblib --trials 25
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prismrisk import metrics as M
from prismrisk.config import REPO_ROOT, load_config
from prismrisk.split import make_splits

optuna.logging.set_verbosity(optuna.logging.WARNING)

EXCLUDE_BASE = ("id_col", "week_col", "target", "date_decision")


# ------------------------------------------------------------- table -> X
class TableEncoder:
    """Ordinal-encode string columns; unseen categories -> -1. Fit on TRAIN only."""

    def __init__(self) -> None:
        self.maps: dict[str, dict[str, int]] = {}
        self.feature_names: list[str] = []

    def fit(self, train_df: pl.DataFrame, cfg) -> TableEncoder:
        exclude = {getattr(cfg, k) for k in EXCLUDE_BASE}
        self.feature_names = [c for c in train_df.columns if c not in exclude]
        for c in self.feature_names:
            if train_df[c].dtype == pl.String:
                vals = train_df[c].drop_nulls().cast(pl.String).unique().to_list()
                self.maps[c] = {v: i for i, v in enumerate(sorted(vals))}
        return self

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        cols = []
        for c in self.feature_names:
            col = df[c]
            if c in self.maps:
                mapping = self.maps[c]
                cols.append(np.array([mapping.get(str(v), -1) for v in col.to_list()], dtype=np.float32))
            elif col.dtype == pl.Boolean:
                cols.append(col.cast(pl.Float32).to_numpy())
            elif col.dtype in (pl.Date, pl.Datetime, pl.Duration, pl.Time):
                cols.append(col.cast(pl.Int64).cast(pl.Float32).to_numpy())  # epoch units
            elif col.dtype.is_numeric():
                cols.append(col.cast(pl.Float32).to_numpy())
            else:  # unexpected dtype -> all-missing column rather than crash
                cols.append(np.full(df.height, np.nan, dtype=np.float32))
        return np.column_stack(cols)


def xy(df: pl.DataFrame, cfg, encoder: TableEncoder):
    X = encoder.transform(df)
    y = df[cfg.target].to_numpy().astype(np.int64)
    weeks = df[cfg.week_col].to_numpy()
    if "credamount_770A" in df.columns:
        amounts = df["credamount_770A"].cast(pl.Float32).fill_null(0.0).to_numpy()
    else:
        amounts = np.ones(df.height, dtype=np.float32)
    return X, y, weeks, amounts


def monotone_vector(feature_names: list[str], cfg) -> list[int]:
    plus, minus = set(cfg.monotone_plus), set(cfg.monotone_minus)
    return [1 if n in plus else (-1 if n in minus else 0) for n in feature_names]


# ------------------------------------------------------------ model factory
def make_base_models(seed: int) -> dict[str, object]:
    return {
        "logreg_baseline": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=300, C=1.0, random_state=seed)),
            ]
        ),
        "mlp_challenger": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(128, 64), max_iter=40, early_stopping=True,
                    random_state=seed,
                )),
            ]
        ),
        "xgboost_unconstrained": xgb.XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.8, tree_method="hist",
            eval_metric="aucpr", n_jobs=-1, random_state=seed,
        ),
    }


def fit_lgbm_optuna(X_tr, y_tr, X_val, y_val, mcv, *, seed: int, trials: int):
    def objective(trial):
        params = {
            "n_estimators": 500,
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "monotone_constraints": mcv,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    best = dict(study.best_params)
    best.update({"n_estimators": 500, "bagging_fraction": 0.8, "bagging_freq": 1,
                 "monotone_constraints": mcv, "random_state": seed, "n_jobs": -1, "verbose": -1})
    model = lgb.LGBMClassifier(**best)
    model.fit(X_tr, y_tr)
    return model, best, float(study.best_value)


# ---------------------------------------------------------------- evaluation
def split_metrics(model, X, y, weeks, amounts, *, cfg) -> dict:
    p = model.predict_proba(X)[:, 1]
    gini = M.weekly_gini_stability(weeks, y, p)
    out = {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "gini": 2 * roc_auc_score(y, p) - 1.0,
        "gini_stability_custom": gini["custom_stability"],
        "gini_week_min": gini["min_gini"],
        "brier": brier_score_loss(y, p),
        "ece": M.expected_calibration_error(y, p),
    }
    return {k: float(v) for k, v in out.items()}


def log_run(mlflow_uri, name, *, params, val_metrics, champion, champion_name, bundle_path, model) -> None:
    """Best-effort MLflow logging — never crashes the training run."""
    if not mlflow_uri:
        return
    import mlflow  # noqa: PLC0415  # local import: training must work without mlflow too

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("PrismRisk-Stability")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics(val_metrics)
        if not champion:
            return
        try:
            if champion_name == "lightgbm_monotone":
                mlflow.lightgbm.log_model(model, name="champion_model", registered_model_name="PrismRiskChampion")
            elif champion_name == "xgboost_unconstrained":
                mlflow.xgboost.log_model(model, name="champion_model", registered_model_name="PrismRiskChampion")
            else:  # sklearn pipeline (logreg / mlp)
                mlflow.sklearn.log_model(
                    model,
                    name="champion_model",
                    registered_model_name="PrismRiskChampion",
                    skops_trusted_types=["numpy.dtype", "numpy.ufunc"],
                )
        except Exception as exc:  # registry is nice-to-have; bundle is what the API loads
            print(f"[mlflow] registry step skipped ({exc}); bundle still exported")
            mlflow.log_artifact(str(bundle_path), artifact_path="bundle")



def calibrate(champion, *, Xc, yc, amt_c, X_val, y_val, X_test, y_test, w_test, cfg):
    """Isotonic calibration fit on calib weeks + profit-optimal approval threshold."""
    p_cal_raw = champion.predict_proba(Xc)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal_raw, yc)
    p_val_cal = iso.predict(champion.predict_proba(X_val)[:, 1])
    p_test_cal = iso.predict(champion.predict_proba(X_test)[:, 1])
    p_cal_cal = iso.predict(p_cal_raw)
    tau_star, profit_curve = M.optimal_cutoff(
        yc, p_cal_cal, amt_c, margin=cfg.interest_margin, lgd=cfg.lgd,
        start=cfg.cutoff_start, stop=cfg.cutoff_stop, step=cfg.cutoff_step,
    )

    val_raw = champion.predict_proba(X_val)[:, 1]
    cal_summary = {
        "brier_before": float(brier_score_loss(y_val, val_raw)),
        "brier_after": float(brier_score_loss(y_val, p_val_cal)),
        "ece_before": float(M.expected_calibration_error(y_val, val_raw)),
        "ece_after": float(M.expected_calibration_error(y_val, p_val_cal)),
        "tau_star": float(tau_star),
        "max_profit_calib_usd": float(max(p for _, p in profit_curve)),
    }
    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, p_test_cal)),
        "pr_auc": float(average_precision_score(y_test, p_test_cal)),
        "brier_cal": float(brier_score_loss(y_test, p_test_cal)),
        "ece_cal": float(M.expected_calibration_error(y_test, p_test_cal)),
        **{f"gini_{k}": float(v) for k, v in M.weekly_gini_stability(w_test, y_test, p_test_cal).items()},
    }
    return iso, tau_star, cal_summary, test_metrics


# -------------------------------------------------------------------- driver
def train(
    features_path: Path | str,
    out_path: Path | str,
    *,
    trials: int = 25,
    seed: int | None = None,
    mlflow_uri: str | None = "sqlite:///mlflow.db",
    mlp_subsample: int = 60_000,
) -> dict:
    t0 = time.time()
    cfg = load_config()
    seed = cfg.random_seed if seed is None else seed
    features_path, out_path = Path(features_path), Path(out_path)

    df = pl.read_parquet(features_path)
    splits = make_splits(df, week_col=cfg.week_col)
    encoder = TableEncoder().fit(splits["train"], cfg)
    Xtr, ytr, _, _ = xy(splits["train"], cfg, encoder)
    Xc, yc, _, amt_c = xy(splits["calib"], cfg, encoder)
    Xv, yv, w_val, amt_val = xy(splits["val"], cfg, encoder)
    Xt, yt, w_test, _ = xy(splits["test"], cfg, encoder)
    mcv = monotone_vector(encoder.feature_names, cfg)
    n_plus = mcv.count(1)
    n_minus = mcv.count(-1)
    print(f"features={len(encoder.feature_names)}  train={len(ytr)}  monotone(+1)={n_plus}  monotone(-1)={n_minus}")

    base_models = make_base_models(seed)
    runs: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    for name, model in base_models.items():
        t = time.time()
        if name == "mlp_challenger" and Xtr.shape[0] > mlp_subsample:
            idx = np.random.default_rng(seed).choice(Xtr.shape[0], mlp_subsample, replace=False)
            model.fit(Xtr[idx], ytr[idx])
        else:
            model.fit(Xtr, ytr)
        val = split_metrics(model, Xv, yv, w_val, amt_val, cfg=cfg)
        runs[name] = {"val": val, "fit_seconds": round(time.time() - t, 1)}
        fitted[name] = model
        print(f"{name:24s} val_roc_auc={val['roc_auc']:.4f}  pr_auc={val['pr_auc']:.4f}  ({time.time()-t:.0f}s)")

    t = time.time()
    lgbm, best_params, _ = fit_lgbm_optuna(Xtr, ytr, Xv, yv, mcv, seed=seed, trials=trials)
    runs["lightgbm_monotone"] = {
        "val": split_metrics(lgbm, Xv, yv, w_val, amt_val, cfg=cfg),
        "best_params": best_params,
        "fit_seconds": round(time.time() - t, 1),
    }
    fitted["lightgbm_monotone"] = lgbm
    lgbm_auc = runs["lightgbm_monotone"]["val"]["roc_auc"]
    print(f"{'lightgbm_monotone':24s} val_roc_auc={lgbm_auc:.4f}  ({time.time()-t:.0f}s)")

    champion_name = max(runs, key=lambda n: runs[n]["val"]["roc_auc"])
    champion = fitted[champion_name]
    print(f"\nchampion (by val ROC-AUC): {champion_name}")

    # --- isotonic calibration on W60–69 (never on train!) ---
    iso, tau_star, cal_summary, test_metrics = calibrate(
        champion, Xc=Xc, yc=yc, amt_c=amt_c,
        X_val=Xv, y_val=yv, X_test=Xt, y_test=yt, w_test=w_test, cfg=cfg,
    )
    print("calibration:", {k: round(v, 4) for k, v in cal_summary.items()})
    print("test (locked W80-91):", {k: round(v, 4) for k, v in test_metrics.items()})

    # --- export the production bundle (this is what the API loads) ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": champion,
        "model_name": champion_name,
        "isotonic": iso,
        "feature_names": encoder.feature_names,
        "encode_maps": encoder.maps,
        "tau_star": float(tau_star),
        "trained_weeks": f"{cfg.train_weeks[0]}-{cfg.train_weeks[1]}",
        "metrics": {"val": runs, "calibration": cal_summary, "test_locked": test_metrics},
    }
    joblib.dump(bundle, out_path)
    print(f"bundle written -> {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")

    for name, rec in runs.items():
        log_run(
            mlflow_uri,
            name,
            params=rec.get("best_params", {"model": name}),
            val_metrics=rec["val"],
            champion=name == champion_name,
            champion_name=champion_name,
            bundle_path=out_path,
            model=fitted[name],
        )

    summary = {
        "champion": champion_name,
        "val": {n: rec["val"] for n, rec in runs.items()},
        "calibration": cal_summary,
        "test_locked": test_metrics,
        "bundle": str(out_path),
        "train_seconds": round(time.time() - t0, 1),
    }
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "val"}, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(REPO_ROOT / "data" / "processed" / "features.parquet"))
    parser.add_argument("--out", default=str(REPO_ROOT / "models" / "champion_bundle.joblib"))
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--mlp-subsample", type=int, default=60_000)
    args = parser.parse_args()
    train(
        args.features,
        args.out,
        trials=args.trials,
        seed=args.seed,
        mlflow_uri=None if args.no_mlflow else args.mlflow_uri,
        mlp_subsample=args.mlp_subsample,
    )


if __name__ == "__main__":
    main()
