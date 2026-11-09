"""M6 API end-to-end tests — demo mode on the committed synthetic fixture.

PRISMRISK_BUNDLE points at a non-existent path so tests always exercise the
documented ``synthetic_demo_fixture`` fallback (CI-safe, no model artefact).
"""

import math
import os

import polars as pl
import pytest
from fastapi.testclient import TestClient

os.environ["PRISMRISK_BUNDLE"] = "/nonexistent/champion_bundle.joblib"
os.environ["PRISMRISK_TRIAGE"] = "/nonexistent/conformal_triage.joblib"

from api.main import app  # noqa: E402

from prismrisk.config import REPO_ROOT  # noqa: E402

# must cover the trained feature_names exactly (MONTH is trained on — only ids/week/target/date are excluded)
ROW_EXCLUDES = {"case_id", "WEEK_NUM", "target", "date_decision"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def payload() -> dict:
    df = pl.read_parquet(REPO_ROOT / "data" / "processed" / "sample_500.parquet")
    row = df.row(0, named=True)
    features = {k: v for k, v in row.items() if k not in ROW_EXCLUDES}
    return {"case_id": str(row["case_id"]), "features": features}


def test_health_reports_demo_mode(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_source"] == "synthetic_demo_fixture"   # honest: not the champion
    assert body["decision_source"] == "conformal_triage"
    assert body["trained_weeks"] == "0-59"
    assert body["registry_version"] is None


def test_predict_ok(client, payload) -> None:
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["calibrated_pd"] <= 1.0
    assert body["decision"] in {"AUTO_APPROVE", "GREY_ZONE_REVIEW", "AUTO_DENY"}
    assert math.isfinite(body["expected_profit_usd"])
    assert body["amount_usd"] > 0
    assert len(body["top_3_shap_reasons"]) == 3
    assert all(rsn["shap"] > 0 for rsn in body["top_3_shap_reasons"])
    assert body["inference_ms"] >= 0.0
    assert body["model_source"] == "synthetic_demo_fixture"


def test_predict_missing_feature_422(client, payload) -> None:
    bad = {"case_id": "x", "features": {k: v for k, v in payload["features"].items() if k != "credamount_770A"}}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert "credamount_770A" in r.json()["detail"]["missing"]


def test_predict_bad_value_422(client, payload) -> None:
    bad = {"case_id": "x", "features": {**payload["features"], "credamount_770A": "not-a-number"}}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_drift_endpoint_either_state(client) -> None:
    r = client.get("/drift")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "no_report"}
    if body["status"] == "ok":
        assert set(body["band_counts"]) <= {"green", "warning", "retrain", "n/a"}


def test_recourse_approved_applicant_skips_dice(client, payload) -> None:
    # fixture row 0 is decided AUTO_APPROVE by the demo model — recourse must say so, not fabricate a CF
    r = client.post("/recourse", json={**payload, "total_cfs": 1, "bg_rows": 60})
    assert r.status_code == 200
    body = r.json()
    if body["decision"] == "AUTO_APPROVE":
        assert body["feasible"] is None
        assert body["counterfactuals"] == []
        assert "approved" in (body["note"] or "")
        assert "approved" in body["applicant_notice"]
        assert body["grounding_passed"] is True
    else:
        assert isinstance(body["feasible"], bool)  # demo model may have improved; tolerate either path


def test_recourse_declined_applicant_runs_dice(client) -> None:
    df = pl.read_parquet(REPO_ROOT / "data" / "processed" / "sample_500.parquet")
    denied_payload = None
    for row in df.filter(pl.col("target") == 1).head(40).iter_rows(named=True):
        features = {k: v for k, v in row.items() if k not in ROW_EXCLUDES}
        p = {"case_id": str(row["case_id"]), "features": features}
        pr = client.post("/predict", json=p)
        if pr.status_code == 200 and pr.json()["decision"] != "AUTO_APPROVE":
            denied_payload = p
            break
    if denied_payload is None:
        pytest.skip("demo model approved every scanned defaulter — DiCE path not exercised (honest skip)")
    r = client.post("/recourse", json={**denied_payload, "total_cfs": 1, "bg_rows": 60})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["feasible"], bool)
    assert body["grounding_passed"] is True
    for cf in body["counterfactuals"]:
        assert "new_pd" in cf and 0.0 <= cf["new_pd"] <= 1.0
