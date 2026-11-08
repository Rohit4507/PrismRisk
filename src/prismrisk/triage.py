"""M3 — Conformal "grey-zone" triage (2026 upgrade #1).

Wraps the calibrated champion in a MAPIE SplitConformalClassifier (90% coverage)
and maps prediction sets to three business actions:

  {0}     -> AUTO_APPROVE       (confident non-defaulter)
  {0,1}   -> GREY_ZONE_REVIEW   (model uncertain -> human underwriter, $12 cost)
  {1}     -> AUTO_DENY          (confident defaulter)
  {}      -> GREY_ZONE_REVIEW   (empty set: conservative fallback)

Exchangeability caveat: conformal's marginal guarantee assumes i.i.d. between
calibration and test weeks. Post-COVID weeks are NOT strictly i.i.d. vs March–May
2020, so we *report* measured coverage on W80–91 and never claim the guarantee
survives regime change untouched — we measure it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from mapie.classification import SplitConformalClassifier
from mapie.metrics.classification import classification_coverage_score
from sklearn.base import BaseEstimator, ClassifierMixin

from prismrisk import metrics as M
from prismrisk.config import REPO_ROOT, load_config
from prismrisk.split import make_splits
from prismrisk.train import TableEncoder, xy

AUTO_APPROVE, GREY_ZONE, AUTO_DENY = "AUTO_APPROVE", "GREY_ZONE_REVIEW", "AUTO_DENY"


class BundleClassifier(ClassifierMixin, BaseEstimator):
    """sklearn-compatible view over the calibrated joblib bundle (for MAPIE prefit).

    predict_proba -> isotonic-calibrated [P(0), P(1)]; never calls fit().
    """

    def __init__(self, bundle):
        self.bundle = bundle
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):  # noqa: ARG002 - prefit usage
        return self

    def predict_proba(self, X):
        raw = self.bundle["model"].predict_proba(X)[:, 1]
        cal = np.clip(self.bundle["isotonic"].predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def encoder_from_bundle(bundle) -> TableEncoder:
    """Reconstruct the TRAIN-fitted encoder from the saved bundle (no refit!)."""
    enc = TableEncoder()
    enc.maps = dict(bundle["encode_maps"])
    enc.feature_names = list(bundle["feature_names"])
    return enc


def decisions_from_sets(y_set: np.ndarray) -> np.ndarray:
    """Map MAPIE boolean prediction sets (n, 2, 1) -> decision labels (n,)."""
    s = y_set[:, :, 0]
    has_0, has_1 = s[:, 0], s[:, 1]
    out = np.full(s.shape[0], GREY_ZONE, dtype=object)
    out[has_0 & ~has_1] = AUTO_APPROVE
    out[~has_0 & has_1] = AUTO_DENY
    out[has_0 & has_1] = GREY_ZONE
    # empty set -> conservative grey-zone (never auto-approve on no evidence)
    return out


def _profit(y, *, amounts, p, approve_mask, grey_mask, tau, margin, lgd, review_cost):
    """Net profit for the 3-way policy.

    - AUTO_APPROVE: approve at +margin (repaid) / -lgd (default)
    - GREY_ZONE_REVIEW: $review_cost charged always; reviewer approves iff PD < tau
      (documented proxy for a human applying the same breakeven threshold)
    - AUTO_DENY: 0
    """
    y = np.asarray(y)
    amounts = np.asarray(amounts, dtype=np.float64)
    approved_amt_gain = np.where(y == 0, margin * amounts, -lgd * amounts)

    p_auto = 0.0
    if approve_mask.any():
        p_auto = float(approved_amt_gain[approve_mask].sum())

    reviewer_approve = grey_mask & (np.asarray(p) < tau)
    p_grey = 0.0
    if reviewer_approve.any():
        p_grey = float(approved_amt_gain[reviewer_approve].sum())
    p_grey -= review_cost * float(grey_mask.sum())
    return p_auto + p_grey


def build_triage(
    features_path: Path | str,
    bundle_path: Path | str,
    out_path: Path | str,
    *,
    confidence: float = 0.90,
) -> dict:
    cfg = load_config()
    features_path, bundle_path, out_path = Path(features_path), Path(bundle_path), Path(out_path)

    df = pl.read_parquet(features_path)
    splits = make_splits(df, week_col=cfg.week_col)
    bundle = joblib.load(bundle_path)
    encoder = encoder_from_bundle(bundle)

    Xc, yc, _, _ = xy(splits["calib"], cfg, encoder)
    Xv, yv, _, _ = xy(splits["val"], cfg, encoder)
    Xt, yt, _, amt_t = xy(splits["test"], cfg, encoder)

    base = BundleClassifier(bundle)
    mapie = SplitConformalClassifier(estimator=base, confidence_level=confidence, prefit=True)
    mapie.conformalize(Xc, yc)  # W60–69 ONLY

    p_test = base.predict_proba(Xt)[:, 1]
    _, sets_test = mapie.predict_set(Xt)
    _, sets_val = mapie.predict_set(Xv)
    decisions = decisions_from_sets(sets_test)

    coverage = {
        "val": float(classification_coverage_score(yv, sets_val)[0]),
        "test": float(classification_coverage_score(yt, sets_test)[0]),
        "target": confidence,
    }

    tau = float(bundle["tau_star"])
    binary_profit = M.expected_net_profit(
        yt, p_test, amt_t, tau, margin=cfg.interest_margin, lgd=cfg.lgd
    )
    conformal_profit = _profit(
        yt, amounts=amt_t, p=p_test,
        approve_mask=decisions == AUTO_APPROVE,
        grey_mask=decisions == GREY_ZONE,
        tau=tau,
        margin=cfg.interest_margin,
        lgd=cfg.lgd,
        review_cost=cfg.manual_review_cost,
    )

    counts = {d: int((decisions == d).sum()) for d in (AUTO_APPROVE, GREY_ZONE, AUTO_DENY)}

    def default_rate(mask):
        return float(yt[mask].mean()) if mask.any() else None

    stats = {
        "decision_counts": counts,
        "decision_shares": {d: round(counts[d] / len(decisions), 4) for d in counts},
        "default_rate_auto_approve": default_rate(decisions == AUTO_APPROVE),
        "default_rate_grey_zone": default_rate(decisions == GREY_ZONE),
        "default_rate_auto_deny": default_rate(decisions == AUTO_DENY),
        "defaults_caught_by_auto_deny": int(yt[decisions == AUTO_DENY].sum()),
        "defaults_total": int(yt.sum()),
    }
    profit = {
        "binary_cutoff_profit_usd": float(binary_profit),
        "conformal_3way_profit_usd": float(conformal_profit),
        "uplift_usd": float(conformal_profit - binary_profit),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mapie": mapie,
        "confidence": confidence,
        "tau_star": tau,
        "coverage": coverage,
        "decision_stats": stats,
        "profit": profit,
        "trained_weeks": bundle.get("trained_weeks"),
        "champion": bundle.get("model_name"),
    }
    joblib.dump(payload, out_path)

    summary = {
        "champion": bundle.get("model_name"),
        "coverage": coverage,
        "decisions": stats,
        "profit": profit,
        "artifact": str(out_path),
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(REPO_ROOT / "data" / "processed" / "features.parquet"))
    parser.add_argument("--bundle", default=str(REPO_ROOT / "models" / "champion_bundle.joblib"))
    parser.add_argument("--out", default=str(REPO_ROOT / "models" / "conformal_triage.joblib"))
    parser.add_argument("--confidence", type=float, default=0.90)
    args = parser.parse_args()
    build_triage(args.features, args.bundle, args.out, confidence=args.confidence)


if __name__ == "__main__":
    main()
