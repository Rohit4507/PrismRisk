"""Pydantic v2 schemas for the PrismRisk HTTP API (M6).

Honesty contract:
- `/predict` requires the FULL trained feature vector (`features` must cover
  every name in the bundle's `feature_names`). Missing names are rejected with
  422 — we never impute silently.
- Every numeric in responses is measured, not invented. `decision_source` and
  `model_source` say exactly where each artefact came from.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Decision = Literal["AUTO_APPROVE", "GREY_ZONE_REVIEW", "AUTO_DENY"]


class PredictRequest(BaseModel):
    case_id: str | None = None
    features: dict[str, Any] = Field(..., description="Full feature vector: every trained feature, keyed by name")


class RecourseRequest(BaseModel):
    case_id: str | None = None
    features: dict[str, Any] = Field(..., description="Same full feature vector as /predict")
    total_cfs: int = Field(2, ge=1, le=4)
    bg_rows: int = Field(200, ge=50, le=2000)


class HealthResponse(BaseModel):
    status: str
    model: str
    model_source: str          # "champion_bundle" | "synthetic_demo_fixture"
    registry_version: str | None
    trained_weeks: str
    decision_source: str       # "conformal_triage" | "tau_star_threshold"
    drift_report: str          # "ok" | "missing"


class ShapReason(BaseModel):
    feature: str
    label: str
    shap: float


class PredictResponse(BaseModel):
    case_id: str | None
    calibrated_pd: float
    threshold_tau: float           # profit-maximising breakeven cutoff on calib weeks
    amount_usd: float
    decision: Decision
    decision_source: str
    model_source: str
    expected_profit_usd: float     # p0*margin*amt - p1*lgd*amt (grey zone also subtracts $12 review)
    top_3_shap_reasons: list[ShapReason]
    inference_ms: float


class RecourseResponse(BaseModel):
    case_id: str | None
    feasible: bool | None            # null == applicant approved; recourse not applicable
    decision: Decision
    calibrated_pd: float
    top_reasons: list[ShapReason]
    counterfactuals: list[dict[str, Any]]
    immutable_locked: list[str]
    features_varied: list[str]
    imputed_features: list[str] = []       # columns DiCE searched after median fill (honesty)
    note: str | None = None                # why recourse is empty, when it is
    grounding_passed: bool
    applicant_notice: str
    notice_fallback_triggered: bool
    model_source: str
