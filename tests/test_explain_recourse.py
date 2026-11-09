"""M4 explain_recourse: TreeSHAP adverse-action reasons + DiCE counterfactuals.

CI-safe: trains a tiny LGBM on the tiny synthetic fixture; no Kaggle data, no bundle.
"""

import json
import sys

import numpy as np
import polars as pl
import pytest
from lightgbm import LGBMClassifier

from prismrisk import explain_recourse as ER
from prismrisk import features as F
from prismrisk.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from make_synthetic import generate  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tables = generate(rows=300, seed=7)
    raw = tmp_path_factory.mktemp("raw")
    for name, frame in tables.items():
        frame.write_parquet(raw / name)
    out = raw / "features.parquet"
    F.build_features(raw, out)
    pdf = pl.read_parquet(out).to_pandas()
    num = pdf.select_dtypes(include="number").columns
    X = pdf[num].drop(columns=["target", "case_id", "WEEK_NUM", "MONTH"], errors="ignore")
    X = X.fillna(X.median()).fillna(0.0)
    y = pdf["target"].astype(int)
    model = LGBMClassifier(n_estimators=25, max_depth=3, random_state=42, verbose=-1).fit(X, y)
    return X, y, model


def test_top_risk_reasons_positive_sorted(tiny) -> None:
    X, y, model = tiny
    i = int(model.predict_proba(X)[:, 1].argmax())  # highest-risk applicant
    reasons = ER.top_risk_reasons(model, X.iloc[[i]], k=3)
    assert 1 <= len(reasons) <= 3
    shap_vals = [r["shap"] for r in reasons]
    assert all(v > 0 for v in shap_vals)
    assert shap_vals == sorted(shap_vals, reverse=True)
    assert all(r["label"] for r in reasons)
    json.dumps(reasons)  # must be JSON-serialisable for the API


def test_ratio_wrapper_recomputes_derived_columns(tiny) -> None:
    X, y, model = tiny
    wrapped = ER.RatioAwareClassifier(model)
    row = X.iloc[[0]].copy()
    row["credamount_770A"] = float(row["credamount_770A"].iloc[0]) * 2.0
    fixed = wrapped._consistent(row)
    inc = float(row["maininc_clean"].iloc[0]) + 1.0
    assert np.isclose(fixed["credamount_to_income"].iloc[0], float(row["credamount_770A"].iloc[0]) / inc)
    probs = wrapped.predict_proba(row)
    assert probs.shape == (1, 2) and np.isclose(probs.sum(axis=1), 1.0).all()


def test_recourse_schema_and_immutable_untouched(tiny) -> None:
    X, y, model = tiny
    cfg = load_config()
    i = int(model.predict_proba(X)[:, 1].argmax())
    vary = [c for c in ["credamount_770A", "annuity_780A", "maininc_clean", "static_past_dpd_001P"] if c in X.columns]
    res = ER.recourse(model, X, X.iloc[[i]], features_to_vary=vary, bg_rows=150, total_cfs=2)
    assert set(res) >= {"feasible", "top_reasons", "counterfactuals", "immutable_locked", "features_varied"}
    locked = set(cfg.immutable_features) & set(X.columns)
    assert set(res["immutable_locked"]) == locked
    assert not (set(res["features_varied"]) & locked)
    for cf in res["counterfactuals"]:
        for c in locked:
            assert c not in cf["changes"]  # immutable can never be the proposed action
        json.dumps(cf["changes"])  # serialisable for the API


def test_recourse_pinned_bounds_returns_honest_empty(tiny) -> None:
    X, y, model = tiny
    i = int(model.predict_proba(X)[:, 1].argmax())
    v = float(X["credamount_770A"].iloc[i])
    res = ER.recourse(model, X, X.iloc[[i]],
                      features_to_vary=["credamount_770A"],
                      permitted_range={"credamount_770A": [v, v]},  # zero-width: impossible by construction
                      bg_rows=100, total_cfs=2)
    assert res["feasible"] is False
    assert res["counterfactuals"] == []
    assert res["top_reasons"]  # reasons still provided — recourse honesty includes evidence
