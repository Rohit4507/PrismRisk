"""M3 tests: decision mapping (deterministic) + tiny end-to-end triage smoke run."""

from pathlib import Path

import numpy as np
import polars as pl

from prismrisk import train as T
from prismrisk import triage as TR


def test_decisions_from_sets_mapping() -> None:
    y_set = np.array(
        [
            [[True], [False]],   # {0}   -> AUTO_APPROVE
            [[False], [True]],   # {1}   -> AUTO_DENY
            [[True], [True]],    # {0,1} -> GREY_ZONE_REVIEW
            [[False], [False]],  # {}    -> GREY_ZONE_REVIEW (conservative)
            [[True], [False]],   # {0}   -> AUTO_APPROVE
        ],
        dtype=bool,
    )
    decisions = TR.decisions_from_sets(y_set)
    assert decisions.tolist() == [
        TR.AUTO_APPROVE,
        TR.AUTO_DENY,
        TR.GREY_ZONE,
        TR.GREY_ZONE,
        TR.AUTO_APPROVE,
    ]


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
        }
    )
    path = tmp_path / "features.parquet"
    df.write_parquet(path)
    return path


def test_triage_smoke(tmp_path: Path) -> None:
    feats = _features(tmp_path)
    bundle_path = tmp_path / "bundle.joblib"
    T.train(feats, bundle_path, trials=1, seed=7, mlflow_uri=None, mlp_subsample=400)

    out = tmp_path / "triage.joblib"
    summary = TR.build_triage(feats, bundle_path, out, confidence=0.90)

    assert out.exists()
    labels = set(summary["decisions"]["decision_counts"])
    assert labels == {TR.AUTO_APPROVE, TR.GREY_ZONE, TR.AUTO_DENY}
    assert 0.0 <= summary["coverage"]["test"] <= 1.0
    assert summary["coverage"]["target"] == 0.90
    assert isinstance(summary["profit"]["conformal_3way_profit_usd"], float)
    assert summary["decisions"]["defaults_total"] >= 0
