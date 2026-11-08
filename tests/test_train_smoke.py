"""End-to-end smoke test of the M2 training driver on a tiny synthetic frame.

No Kaggle download needed; CI runs this to prove the pipeline wires up correctly.
"""

from pathlib import Path

import joblib
import numpy as np
import polars as pl

from prismrisk import train as T
from prismrisk.config import load_config


def _features(tmp_path: Path, rows: int = 700, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    week = np.arange(rows) % 92
    dpd = rng.gamma(2.0, 12.0, rows)
    inc = rng.lognormal(10, 1.0, rows)
    amt = rng.lognormal(13, 1.0, rows)
    ann = amt / 12.0
    z = -2.2 + 0.03 * dpd - 0.35 * np.log1p(inc / 1e5) + 0.5 * np.log1p(ann / 1e4)
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(size=rows) < p).astype(np.int64)
    df = pl.DataFrame(
        {
            "case_id": np.arange(1, rows + 1),
            "WEEK_NUM": week.astype(np.int64),
            "target": y,
            "dpd_like_001P": dpd.astype(np.float32),
            "maininc_clean": inc.astype(np.float32),
            "credamount_770A": amt.astype(np.float32),
            "annuity_780A": ann.astype(np.float32),
            "annuity_to_income": (ann / (inc + 1.0)).astype(np.float32),
            "credamount_to_income": (amt / (inc + 1.0)).astype(np.float32),
            "channel": np.where(dpd > 20, "online", "branch"),
        }
    )
    path = tmp_path / "features.parquet"
    df.write_parquet(path)
    return path


def test_train_smoke(tmp_path: Path) -> None:
    feats = _features(tmp_path)
    out = tmp_path / "bundle.joblib"
    summary = T.train(feats, out, trials=2, seed=7, mlflow_uri=None, mlp_subsample=400)
    assert out.exists()
    assert summary["champion"] in summary["val"]
    assert set(summary) >= {"champion", "val", "calibration", "test_locked"}

    bundle = joblib.load(out)
    required = [
        "model", "model_name", "isotonic", "encode_maps",
        "feature_names", "tau_star", "trained_weeks", "metrics",
    ]
    for key in required:
        assert key in bundle
    assert bundle["trained_weeks"] == "0-59"
    assert 0.0 <= float(bundle["isotonic"].predict(np.array([0.2]))[0]) <= 1.0


def test_monotone_vector_maps_config() -> None:
    cfg = load_config()
    vec = T.monotone_vector(["max_dpd_bureau", "maininc_clean", "unrelated_feat"], cfg)
    assert vec == [1, -1, 0]