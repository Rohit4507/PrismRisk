"""PrismRisk HTTP API (M6) — thin layer over api.runtime.Scorer.

Endpoints:
  GET  /health    readiness + which artefacts are live (never on the hot path: no MLflow)
  POST /predict   calibrated PD, 3-way conformal decision, expected profit, SHAP reasons
  POST /recourse  DiCE counterfactuals + numeric-verified deterministic adverse-action letter
  GET  /drift     latest reports/psi_weekly.csv summary (M5 artefact)

Latency: measured per request (`inference_ms` in /predict response); never asserted in tests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException

from api import schemas
from api.runtime import get_scorer
from prismrisk.config import REPO_ROOT
from prismrisk.notice_guard import EvidenceFacts, deterministic_letter, find_ungrounded_numbers, render_notice


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = get_scorer()  # build/load once at startup
    yield


app = FastAPI(title="PrismRisk API", version="1.0.0", lifespan=lifespan)

DRIFT_CSV = Path(REPO_ROOT / "reports" / "psi_weekly.csv")


@app.get("/health", response_model=schemas.HealthResponse)
def health() -> dict:
    s = get_scorer()
    return {
        "status": "ok",
        "model": "PrismRiskChampion" if s.source == "champion_bundle" else "synthetic-demo-not-a-champion",
        "model_source": s.source,
        "registry_version": s.registry_version,
        "trained_weeks": s.trained_weeks,
        "decision_source": s.decision_source,
        "drift_report": "ok" if DRIFT_CSV.exists() else "missing",
    }


@app.get("/")
def index() -> dict:
    return {"service": "prismrisk", "docs": "/docs", "health": "/health"}


def _encode_or_422(scorer, features: dict):
    try:
        return scorer._encode_one(features)
    except KeyError as exc:
        missing = exc.args[0]
        raise HTTPException(status_code=422, detail={
            "error": "missing required features",
            "missing": list(missing),
            "hint": "features must cover the bundle's trained feature_names; see /health for model_source",
        }) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"features could not be encoded: {exc}") from exc


@app.post("/predict", response_model=schemas.PredictResponse)
def predict(req: schemas.PredictRequest):
    scorer = get_scorer()
    _encode_or_422(scorer, req.features)  # validate before timing
    return scorer.predict(case_id=req.case_id, features=req.features)


@app.post("/recourse", response_model=schemas.RecourseResponse)
def recourse(req: schemas.RecourseRequest):
    scorer = get_scorer()
    _encode_or_422(scorer, req.features)
    out = scorer.recourse(case_id=req.case_id, features=req.features,
                          total_cfs=req.total_cfs, bg_rows=req.bg_rows)
    if out is None:
        raise HTTPException(status_code=503, detail=(
            "recourse unavailable: the champion bundle carries no compatible background frame. "
            "Set PRISMRISK_BACKGROUND to a parquet with all trained feature columns "
            "(e.g. data/processed/features.parquet) and restart."))
    result, pd_cal, decision = out

    # numeric-verified deterministic letter from measured evidence (no LLM call by default)
    amount = scorer._amount(req.features)
    cf0 = result["counterfactuals"][0] if result["counterfactuals"] else None
    target_amt = float(cf0["changes"]["credamount_770A"]["to"]) if (
        cf0 and "credamount_770A" in cf0["changes"]) else amount
    new_pd = float(cf0["new_pd"]) if (cf0 and "new_pd" in cf0) else pd_cal
    reasons = result["top_reasons"] or [{"label": "application profile"}]
    facts = EvidenceFacts(
        pd_pct=f"{pd_cal * 100:.1f}%",
        tau_pct=f"{scorer.tau_star * 100:.1f}%",
        current_loan=f"${amount:,.0f}",
        target_loan=f"${target_amt:,.0f}",
        new_pd_pct=f"{new_pd * 100:.1f}%",
        reason_1=reasons[0]["label"],
        reason_2=reasons[1]["label"] if len(reasons) > 1 else "overall application strength",
        reason_3=reasons[2]["label"] if len(reasons) > 2 else "overall application strength",
        decision=decision,
        applicant_ref=req.case_id or "applicant",
    )
    if decision == "AUTO_APPROVE":
        # adverse-action letter is meaningless post-approval — issue an approval letter instead,
        # numbers built from the same facts and machine-verified for grounding
        candidate = (f"Dear {facts.applicant_ref},\n\nGood news — your application has been approved. "
                     f"The review model estimates your probability of default at {facts.pd_pct}, below the "
                     f"{facts.tau_pct} approval threshold. No adverse action has been taken, and no "
                     f"counterfactual changes are required.\n\n"
                     f"This is a model decision, not a final legal determination.\n\nPrismRisk Credit Review")
        if find_ungrounded_numbers(candidate, facts.allowed_numbers()):
            candidate, fell_back = deterministic_letter(facts), True
        else:
            fell_back = False
        grounding_ok, applicant_notice = True, candidate
    else:
        res = render_notice(facts, llm=None)  # deterministic path: numbers only from facts
        grounding_ok, fell_back, applicant_notice = res.grounding_passed, res.fallback_triggered, res.text

    return {
        "case_id": req.case_id,
        "feasible": result["feasible"],
        "decision": decision,
        "calibrated_pd": pd_cal,
        "top_reasons": result["top_reasons"],
        "counterfactuals": result["counterfactuals"],
        "immutable_locked": result["immutable_locked"],
        "features_varied": result["features_varied"],
        "imputed_features": result.get("imputed_features", []),
        "note": result.get("reason") or result.get("note"),
        "grounding_passed": grounding_ok,
        "applicant_notice": applicant_notice,
        "notice_fallback_triggered": fell_back,
        "model_source": scorer.source,
    }


@app.get("/drift")
def drift() -> dict:
    csv_path = Path(DRIFT_CSV)
    if not csv_path.exists():
        return {"status": "no_report",
                "hint": "run `python -m prismrisk.drift --features data/processed/features.parquet "
                        "--bundle models/champion_bundle.joblib` on the training box"}
    df = pl.read_csv(csv_path)
    counts = df.group_by("band").agg(pl.len().alias("rows")).to_dict(as_series=False)
    retrain = df.filter(pl.col("band") == "retrain").select(
        ["week", "feature", "psi"]).sort("psi", descending=True).head(20).to_dicts()
    return {
        "status": "ok",
        "source": str(csv_path),
        "rows": df.height,
        "weeks": sorted({int(w) for w in df["week"].unique().to_list()}),
        "band_counts": dict(zip(counts["band"], counts["rows"], strict=True)),
        "top_retrain": retrain,
    }
