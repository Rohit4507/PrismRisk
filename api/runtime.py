"""Model runtime for the PrismRisk API.

Two honest modes, chosen at startup:

1. ``champion_bundle`` — loads ``models/champion_bundle.joblib`` (or
   ``$PRISMRISK_BUNDLE``). PD path is exactly the train-time one: bundle's
   ``TableEncoder`` maps + isotonic calibration. If a MAPIE conformal artifact
   exists (``$PRISMRISK_TRIAGE`` or ``models/conformal_triage.joblib``) the
   3-way decision comes from prediction sets; otherwise we fall back to the
   tau* breakeven rule (no grey zone, never faked) and say so via
   ``decision_source="tau_star_threshold"``.

2. ``synthetic_demo_fixture`` — no bundle found: trains a tiny LightGBM on the
   committed synthetic fixture (``data/processed/sample_500.parquet``) at
   startup, calibrates with isotonic on the calib weeks (60–69), conformalizes
   with the same MAPIE recipe as ``prismrisk.triage``. Data and model are the
   synthetic demo set ONLY — surfaced everywhere as ``model_source``.

DiCE recourse needs a feature-space-compatible background frame: in demo mode
it is the fixture; in bundle mode pass ``$PRISMRISK_BACKGROUND`` (parquet with
all trained feature columns, e.g. data/processed/features.parquet) or /recourse
answers 503 — never a silently degraded result.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import shap
from lightgbm import LGBMClassifier
from mapie.classification import SplitConformalClassifier
from sklearn.isotonic import IsotonicRegression

from prismrisk import explain_recourse as ER
from prismrisk import metrics as M
from prismrisk.config import REPO_ROOT, load_config
from prismrisk.split import make_splits
from prismrisk.train import TableEncoder, xy
from prismrisk.triage import AUTO_APPROVE, AUTO_DENY, BundleClassifier, decisions_from_sets

SAMPLE_FIXTURE = REPO_ROOT / "data" / "processed" / "sample_500.parquet"


def _expected_profit(calibrated_pd: float, amount: float, decision: str, tau: float, cfg) -> float:
    gain = cfg.interest_margin * amount * (1.0 - calibrated_pd) - cfg.lgd * amount * calibrated_pd
    if decision == AUTO_APPROVE:
        return float(gain)
    if decision == AUTO_DENY:
        return 0.0
    approved = calibrated_pd < tau  # grey zone: reviewer applies the breakeven rule
    return float((gain if approved else 0.0) - cfg.manual_review_cost)


class Scorer:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.load_ms: float | None = None
        bundle_path = Path(os.environ.get("PRISMRISK_BUNDLE", REPO_ROOT / "models" / "champion_bundle.joblib"))
        t = time.perf_counter()
        if bundle_path.exists():
            self._init_from_bundle(bundle_path)
        elif SAMPLE_FIXTURE.exists():
            self._init_demo(SAMPLE_FIXTURE)
        else:
            raise RuntimeError(
                f"neither champion bundle ({bundle_path}) nor synthetic fixture ({SAMPLE_FIXTURE}) found"
            )
        self.load_ms = round((time.perf_counter() - t) * 1000, 1)
        # cached explainer — building TreeExplainer per request blows the latency budget
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.shap_explainer = None

    # ------------------------------------------------------------- init paths
    def _init_from_bundle(self, bundle_path: Path) -> None:
        bundle = joblib.load(bundle_path)
        self.source = "champion_bundle"
        self.model_name = str(bundle.get("model_name", "unknown"))
        self.registry_version = bundle.get("registry_version")
        self.trained_weeks = str(bundle.get("trained_weeks", self.cfg.train_weeks))
        self.model = bundle["model"]
        self.isotonic = bundle["isotonic"]
        self.encoder = TableEncoder()
        self.encoder.maps = dict(bundle.get("encode_maps", {}))
        self.encoder.feature_names = list(bundle["feature_names"])
        self.tau_star = float(bundle["tau_star"])
        self.background = self._load_background()
        self.conformal = self._load_conformal()
        self.decision_source = "conformal_triage" if self.conformal is not None else "tau_star_threshold"

    def _load_background(self) -> pd.DataFrame | None:
        path = os.environ.get("PRISMRISK_BACKGROUND")
        if not path:
            return None
        try:
            df = pl.read_parquet(path)
            if all(c in df.columns for c in self.encoder.feature_names):
                X = self.encoder.transform(df)
                return pd.DataFrame(X, columns=self.encoder.feature_names)
        except Exception:
            pass
        return None

    def _load_conformal(self):
        path = Path(os.environ.get("PRISMRISK_TRIAGE", REPO_ROOT / "models" / "conformal_triage.joblib"))
        if not path.exists():
            return None
        try:
            artifact = joblib.load(path)
            return artifact.get("mapie")
        except Exception:
            return None

    def _init_demo(self, fixture: Path) -> None:
        cfg = self.cfg
        df = pl.read_parquet(fixture)
        splits = make_splits(df, week_col=cfg.week_col)
        self.encoder = TableEncoder().fit(splits["train"], cfg)
        Xtr, ytr, _, _ = xy(splits["train"], cfg, self.encoder)
        Xc, yc, _, amt_c = xy(splits["calib"], cfg, self.encoder)
        self.model = LGBMClassifier(n_estimators=60, max_depth=4, random_state=cfg.random_seed,
                                    verbose=-1, deterministic=True)
        self.model.fit(Xtr, ytr)
        raw_c = self.model.predict_proba(Xc)[:, 1]
        self.isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw_c, yc)
        cal_c = self.isotonic.predict(raw_c)
        self.tau_star, _ = M.optimal_cutoff(
            yc, cal_c, amt_c, margin=cfg.interest_margin, lgd=cfg.lgd,
            start=cfg.cutoff_start, stop=cfg.cutoff_stop, step=cfg.cutoff_step)
        p = {"model": self.model, "isotonic": self.isotonic}
        self.conformal = SplitConformalClassifier(
            estimator=BundleClassifier(p), confidence_level=cfg.conformal_confidence, prefit=True)
        self.conformal.conformalize(Xc, yc)
        self.source = "synthetic_demo_fixture"
        self.model_name = "demo-lightgbm-synthetic"
        self.registry_version = None
        self.trained_weeks = f"{cfg.train_weeks[0]}-{cfg.train_weeks[1]}"
        Xtr_df = pd.DataFrame(Xtr, columns=self.encoder.feature_names)
        self.background = Xtr_df.sample(min(200, len(Xtr_df)), random_state=cfg.random_seed).reset_index(drop=True)
        self.decision_source = "conformal_triage"

    # -------------------------------------------------------------- scoring
    def _aligned_frame(self, features: dict) -> pl.DataFrame:
        missing = [c for c in self.encoder.feature_names if c not in features]
        if missing:
            raise KeyError(missing)
        data = {}
        for c in self.encoder.feature_names:
            v = features[c]
            if c in self.encoder.maps:          # known string column -> ordinal map
                data[c] = [None if v is None else str(v)]
            elif isinstance(v, bool):
                data[c] = [v]
            elif v is None:
                data[c] = [None]                # -> Null dtype -> all-NaN downstream (LGBM handles)
            else:
                try:
                    data[c] = [float(v)]        # coerce numpy/int/str-numbers
                except (TypeError, ValueError):
                    raise ValueError(f"feature {c!r}: expected a number, got {v!r}") from None
        return pl.DataFrame(data)

    def _encode_one(self, features: dict):
        row = self._aligned_frame(features)
        X = self.encoder.transform(row)
        return X, pd.DataFrame(X, columns=self.encoder.feature_names)

    def _amount(self, features: dict) -> float:
        raw = features.get("credamount_770A")
        try:
            amt = float(raw)
        except (TypeError, ValueError):
            amt = 0.0
        return amt if amt > 0 else 1.0  # same fallback convention as train.xy

    def _calibrated(self, X: np.ndarray) -> float:
        raw = float(self.model.predict_proba(X)[:, 1][0])
        return float(np.clip(self.isotonic.predict([raw])[0], 0.0, 1.0))

    def _decision(self, X: np.ndarray, pd_cal: float) -> str:
        if self.conformal is not None:
            _, sets = self.conformal.predict_set(X)
            return str(decisions_from_sets(sets)[0])
        return AUTO_DENY if pd_cal >= self.tau_star else AUTO_APPROVE

    def predict(self, *, case_id, features: dict) -> dict:
        t0 = time.perf_counter()
        X, X_row = self._encode_one(features)
        pd_cal = self._calibrated(X)
        decision = self._decision(X, pd_cal)
        reasons = ER.top_risk_reasons(self.model, X_row, k=3, explainer=self.shap_explainer)
        amount = self._amount(features)
        return {
            "case_id": case_id,
            "calibrated_pd": pd_cal,
            "threshold_tau": self.tau_star,
            "amount_usd": amount,
            "decision": decision,
            "decision_source": self.decision_source,
            "model_source": self.source,
            "expected_profit_usd": _expected_profit(pd_cal, amount, decision, self.tau_star, self.cfg),
            "top_3_shap_reasons": reasons,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def recourse(self, *, case_id, features: dict, total_cfs: int, bg_rows: int):
        if self.background is None:
            return None  # caller -> 503, honest: no compatible background available
        X, X_row = self._encode_one(features)
        pd_cal = self._calibrated(X)
        decision = self._decision(X, pd_cal)
        if decision == AUTO_APPROVE:
            # recourse is demand-side help for people about to be/already denied —
            # an approved applicant needs no counterfactual; say so, don't fabricate one
            reasons = ER.top_risk_reasons(self.model, X_row, k=3, explainer=self.shap_explainer)
            imm = [c for c in self.cfg.immutable_features if c in self.encoder.feature_names]
            return {
                "feasible": None,
                "reason": "applicant approved — recourse applies to declined/grey-zone applications",
                "top_reasons": reasons,
                "counterfactuals": [],
                "immutable_locked": list(imm),
                "features_varied": [],
                "imputed_features": [],
            }, pd_cal, decision
        bg = self.background.sample(min(bg_rows, len(self.background)), random_state=self.cfg.random_seed)
        result = ER.recourse(self.model, bg.reset_index(drop=True), X_row, total_cfs=total_cfs,
                             bg_rows=len(bg), seed=self.cfg.random_seed)
        # attach post-action PD (measured, not fabricated)
        for cf in result["counterfactuals"]:
            cf_row = X_row.iloc[0].to_dict()
            for col, delta in cf["changes"].items():
                cf_row[col] = delta["to"]
            X_cf = pd.DataFrame([cf_row], columns=self.encoder.feature_names).to_numpy().astype(np.float32)
            cf["new_pd"] = round(self._calibrated(X_cf), 4)
        return result, pd_cal, decision


_scorer: Scorer | None = None


def get_scorer() -> Scorer:
    global _scorer  # noqa: PLW0603 — deliberate process-wide singleton for the API
    if _scorer is None:
        _scorer = Scorer()
    return _scorer
