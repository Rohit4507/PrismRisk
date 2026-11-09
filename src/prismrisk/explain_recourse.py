"""M4 — adverse-action reasons (TreeSHAP) + DiCE counterfactual recourse.

Part A (notice evidence, <5 ms): top-k risk-INCREASING SHAP contributions for the
applicant, formatted for the adverse-action letter ("Loan-to-income ratio +0.031").

Part B (action plan, demand-side only): DiCE-ML (genetic) counterfactuals over the
applicant's top *actionable* drivers. Immutable attributes (config:
immutable_features) are never varied, derived ratios are recomputed inside a
ratio-aware wrapper so a change to `credamount_770A` is reflected in
`credamount_to_income`, and permitted ranges come from train quantiles (q05–q95)
unless the caller overrides them. DiCE on 75–300 features is far too slow for the
/predict hot path — wire it behind `/recourse` only.

Honesty rules: no counterfactual, no fabrication — when DiCE finds none inside
the bounds, the result says so plainly (feasible=False).
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import polars as pl
import shap
from sklearn.base import BaseEstimator, ClassifierMixin

from prismrisk.config import load_config

FEATURE_LABELS = {
    "annuity_to_income": "Installment-to-income ratio",
    "credamount_to_income": "Loan-to-income ratio",
    "overdue_to_credamount": "Overdue-to-loan ratio",
    "maininc_clean": "Net income",
    "credamount_770A": "Requested loan amount",
    "annuity_780A": "Monthly installment",
    "max_dpd_bureau": "Bureau max days-past-due",
    "max_dpd_prev": "Previous-loan max days-past-due",
    "max_dpd_total_001P": "Bureau total days-past-due",
    "max_static_past_dpd_001P": "Previous-app max days-past-due",
    "risk_band_pct": "Bureau default-risk band (%)",
    "total_overdue_amt": "Total overdue amount",
    "birth_259_days": "Age (days)",
    "age_years": "Age",
}

NON_ACTIONABLE = {"case_id", "WEEK_NUM", "MONTH", "target", "date_decision",
                  "annuity_to_income", "credamount_to_income", "overdue_to_credamount"}


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    stem = name
    for suffix in ("_days", "_001P", "_002A", "_770A", "_780A"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.replace("_", " ").strip().capitalize()


def _unwrap_tree(model):
    """Return a TreeExplainer for the (possibly pipelined) model's tree learner."""
    target = model
    if hasattr(model, "named_steps"):
        target = list(model.named_steps.values())[-1]
    try:
        return shap.TreeExplainer(target)
    except Exception as exc:  # honest failure, never fake reasons
        raise TypeError(
            f"TreeExplainer supports tree models; got {type(target).__name__}"
        ) from exc


def _shap_contrib(model, X_row: pd.DataFrame) -> np.ndarray:
    """Positive-class SHAP contributions for the single row of X_row."""
    explainer = _unwrap_tree(model)
    raw = explainer.shap_values(X_row)
    if isinstance(raw, list):        # LightGBM >=1.x / sklearn API list: [class0, class1]
        arr = np.asarray(raw[-1])
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3:            # (n, features, classes) -> positive class
            arr = arr[:, :, -1]
    return arr[0]


def top_risk_reasons(model, X_row: pd.DataFrame, k: int = 3) -> list[dict]:
    """Top-k risk-INCREASING contributions for the applicant (adverse-notice evidence)."""
    contrib = _shap_contrib(model, X_row)
    out: list[dict] = []
    for i in np.argsort(-contrib):
        v = float(contrib[i])
        if v <= 0.0:
            break
        name = str(X_row.columns[i])
        out.append({"feature": name, "label": feature_label(name), "shap": v})
        if len(out) >= k:
            break
    return out


class RatioAwareClassifier(BaseEstimator, ClassifierMixin):
    """Recompute derived ratios so DiCE CFs stay self-consistent.

    Mirrors features.add_ratio_features: changing credamount_770A / annuity_780A /
    total_overdue_amt updates credamount_to_income / annuity_to_income /
    overdue_to_credamount automatically. Pre-fitted; fit() is a no-op for sklearn
    compatibility with dice-ml's sklearn backend.
    """

    def __init__(self, base):
        self.base = base
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def _consistent(self, X) -> pd.DataFrame:
        X = X.copy()
        if "maininc_clean" in X.columns:
            inc = pd.to_numeric(X["maininc_clean"], errors="coerce").fillna(0.0) + 1.0
            if "annuity_780A" in X.columns:
                X["annuity_to_income"] = pd.to_numeric(X["annuity_780A"], errors="coerce").fillna(0.0) / inc
            if "credamount_770A" in X.columns:
                X["credamount_to_income"] = pd.to_numeric(X["credamount_770A"], errors="coerce").fillna(0.0) / inc
        if {"total_overdue_amt", "credamount_770A"} <= set(X.columns):
            X["overdue_to_credamount"] = (
                pd.to_numeric(X["total_overdue_amt"], errors="coerce").fillna(0.0)
                / (pd.to_numeric(X["credamount_770A"], errors="coerce").fillna(0.0) + 1.0)
            )
        return X

    def predict_proba(self, X):
        return self.base.predict_proba(self._consistent(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _is_actionable(col: str, series: pd.Series) -> bool:
    if col in NON_ACTIONABLE:
        return False
    return bool(pd.api.types.is_numeric_dtype(series) and series.notna().mean() >= 0.5)


def build_dice(model, X_train: pd.DataFrame, outcome_name: str = "target"):
    """DiCE genetic explainer over a numeric background sample (caller subsamples)."""
    import dice_ml  # noqa: PLC0415 - lazy: heavy import, only when DiCE is actually invoked

    frame = X_train.copy()
    num_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    frame = frame[num_cols].dropna(axis=1, how="all").astype(np.float64)  # DiCE needs int/float dtypes
    frame[outcome_name] = 0  # outcome placeholder required by the data interface
    data = dice_ml.Data(dataframe=frame,
                        continuous_features=[c for c in frame.columns if c != outcome_name],
                        outcome_name=outcome_name)
    dmodel = dice_ml.Model(model=RatioAwareClassifier(model), backend="sklearn")
    return dice_ml.Dice(data, dmodel, method="genetic")


def _empty(reasons, immutable, reason: str) -> dict:
    return {"feasible": False, "reason": reason, "top_reasons": reasons,
            "counterfactuals": [], "immutable_locked": list(immutable)}


def _auto_select(model, X_train, q, imm, max_vary: int) -> list[str]:
    """Applicant's top actionable drivers by |SHAP| (immutable/ratios excluded)."""
    contrib = _shap_contrib(model, q)
    vary: list[str] = []
    for i in np.argsort(-np.abs(contrib)):
        c = str(q.columns[i])
        if c not in imm and _is_actionable(c, X_train[c]) and c not in vary:
            vary.append(c)
        if len(vary) >= max_vary:
            break
    return vary


def _zero_pinned(vary: list[str], pr: dict) -> tuple[list[str], dict]:
    """Drop zero-width permitted bounds (an opposite-class CF is impossible there)."""
    pinned = [c for c, (lo, hi) in pr.items() if hi - lo <= 0]
    if not pinned:
        return vary, pr
    return [c for c in vary if c not in pinned], {c: v for c, v in pr.items() if c not in pinned}


def _build_pr(bg: pd.DataFrame, vary: list[str], permitted_range):
    """(vary, permitted_range). Caller bounds are pruned to zero-width columns;
    the default is train q05-q95, dropping un-finite/empty-span features."""
    if permitted_range is not None:
        pr = {c: [float(v[0]), float(v[1])] for c, v in permitted_range.items() if c in vary}
        return _zero_pinned(vary, pr)
    pr, keep = {}, []
    for c in vary:
        lo, hi = np.nanquantile(bg[c].to_numpy(dtype=np.float64), [0.05, 0.95])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pr[c] = [float(lo), float(hi)]
            keep.append(c)
    return keep, pr


def _first_cf_df(cf_obj):
    for part in getattr(cf_obj, "cf_examples_list", []):
        df_cf = getattr(part, "final_cfs_df", None)
        if df_cf is not None and len(df_cf) > 0:
            return df_cf
    return None


def _extract_cfs(cf_df, vary, scale, q0, imm):
    out = []
    for _, row in cf_df.iterrows():
        # paranoia: DiCE must never touch immutable columns — enforce post-hoc
        if any(not np.isclose(float(row[c]), float(q0[c]), equal_nan=True) for c in imm):
            continue
        changes, rel = {}, 0.0
        for c in vary:
            frm, to = float(q0[c]), float(row[c])
            if not np.isclose(frm, to, rtol=1e-6, atol=1e-9):
                changes[c] = {"from": frm, "to": to}
                rel += abs(to - frm) / (abs(frm) + scale[c])
        if changes:
            out.append({"changes": changes, "relative_change": rel})
    out.sort(key=lambda d: d["relative_change"])
    return out


def recourse(model, X_train: pd.DataFrame, query, *, immutable: list[str] | None = None,
             features_to_vary: list[str] | None = None, permitted_range: dict | None = None,
             total_cfs: int = 4, bg_rows: int = 2000, max_vary: int = 8, seed: int = 42) -> dict:
    """Top actionable counterfactuals that flip the applicant to the opposite class.

    JSON-serialisable dict. Never fabricates: if DiCE finds nothing inside the
    permitted bounds, feasible=False and counterfactuals is empty.
    """
    cfg = load_config()
    imm = [c for c in (immutable if immutable is not None else cfg.immutable_features)
           if c in X_train.columns]
    if isinstance(query, dict):
        query = pd.DataFrame([query])
    q = query.reset_index(drop=True)
    num_q = [c for c in q.columns if pd.api.types.is_numeric_dtype(q[c])]
    q = q.copy()
    q[num_q] = q[num_q].astype(np.float64)  # match DiCE's data interface dtypes

    reasons = top_risk_reasons(model, q, k=3)

    if features_to_vary is None:
        vary = _auto_select(model, X_train, q, imm, max_vary)
        msg = "no actionable feature found"
    else:
        vary = [c for c in features_to_vary if c in X_train.columns and c not in imm]
        msg = "no actionable feature in features_to_vary"
    if not vary:
        return _empty(reasons, imm, msg)

    bg = X_train.iloc[np.random.default_rng(seed).choice(
        X_train.shape[0], size=min(bg_rows, X_train.shape[0]), replace=False)]
    vary, pr = _build_pr(bg, vary, permitted_range)
    if not pr:
        return _empty(reasons, imm, "permitted bounds are pinned to a point; no feasible action exists")

    try:
        np.random.seed(seed)  # dice-ml's genetic search has no explicit seed param
        dice = build_dice(model, bg, outcome_name="target")
        cf_obj = dice.generate_counterfactuals(
            q, total_CFs=total_cfs, desired_class="opposite",
            features_to_vary=vary, permitted_range=pr,
        )
        cf_df = _first_cf_df(cf_obj)
    except Exception as exc:  # DiCE failure -> honest empty result, never a fabricated CF
        return {**_empty(reasons, imm, f"DiCE found no counterfactuals ({exc})"),
                "features_varied": vary}
    if cf_df is None:
        return {**_empty(reasons, imm, "no counterfactual inside permitted bounds"),
                "features_varied": vary}

    scale = {c: float(np.nanmedian(np.abs(bg[c].to_numpy(dtype=np.float64)))) + 1e-6 for c in vary}
    out_cfs = _extract_cfs(cf_df, vary, scale, q.iloc[0], imm)

    return {
        "feasible": bool(out_cfs),
        "top_reasons": reasons,
        "counterfactuals": out_cfs[:2],
        "immutable_locked": list(imm),
        "features_varied": vary,
        "note": None if out_cfs else "counterfactuals violate immutability constraints and were discarded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reasons + recourse for one applicant row (demand-side).")
    parser.add_argument("--features", required=True, help="features parquet from prismrisk.features")
    parser.add_argument("--bundle", required=True, help="champion bundle written by prismrisk.train")
    parser.add_argument("--row", type=int, default=0, help="row index within the features file")
    parser.add_argument("--bg-rows", type=int, default=1000, help="DiCE background subsample size")
    args = parser.parse_args()

    bundle = joblib.load(args.bundle)
    model, iso, feats = bundle["model"], bundle.get("isotonic"), bundle["feature_names"]
    # stride the frame BEFORE widening to pandas (1.5M x 300 is too big otherwise)
    df = pl.read_parquet(args.features)
    df = df[:: max(1, df.height // 20000)]
    pdf = df.to_pandas()
    num_cols = [c for c in pdf.columns if pd.api.types.is_numeric_dtype(pdf[c])]
    pdf[num_cols] = pdf[num_cols].fillna(pdf[num_cols].median()).fillna(0.0)
    pdf = pdf.fillna("")
    X = pdf[feats]
    q = X.iloc[[args.row]].reset_index(drop=True)
    result = recourse(model, X, q, bg_rows=min(args.bg_rows, len(X)))
    if iso is not None:
        raw = float(model.predict_proba(q)[:, 1][0])
        result["pd_raw"] = raw
        result["pd_calibrated"] = float(iso.predict([raw])[0])
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
