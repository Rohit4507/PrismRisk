"""PrismRisk Streamlit dashboard (M6) — try the model on held-out-week borrowers.

Demo mode (no champion bundle in the image): applicants come from the committed
synthetic fixture and the model is a tiny LightGBM trained on it at API startup.
In production (PRISMRISK_BUNDLE + real features) the same UI serves the real
champion — the API tells the browser which one is live.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "data" / "processed" / "sample_500.parquet"
API_URL = os.environ.get("PRISMRISK_API_URL", "http://localhost:8000")

st.set_page_config(page_title="PrismRisk", page_icon="🛡️", layout="centered")


@st.cache_data
def load_applicants() -> pd.DataFrame:
    df = pd.read_parquet(FIXTURE)
    test = df[df["WEEK_NUM"] >= 80].reset_index(drop=True)  # locked test weeks — honest demo
    return test if len(test) else df.reset_index(drop=True)


@st.cache_data
def api_health() -> dict:
    r = httpx.get(f"{API_URL}/health", timeout=5.0)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict) -> tuple[int, dict]:
    r = httpx.post(f"{API_URL}{path}", json=payload, timeout=120.0)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}


def row_to_payload(row: pd.Series) -> dict:
    features = {}
    for k, v in row.items():
        if k in {"case_id", "WEEK_NUM", "target", "date_decision"}:
            continue
        value = v.item() if isinstance(v, (np.floating, np.integer)) else v
        if pd.isna(value):
            value = None
        features[k] = value
    return {"case_id": str(row["case_id"]), "features": features}


BADGE = {"AUTO_APPROVE": ("🟢", "AUTO APPROVE"), "GREY_ZONE_REVIEW": ("🟡", "GREY ZONE — REVIEW"),
         "AUTO_DENY": ("🔴", "AUTO DENY")}

st.title("🛡️ PrismRisk")
st.caption("Temporal credit decision & MLOps — monotone/calibrated PD, conformal grey-zone triage, "
           "counterfactual recourse, weekly PSI drift.")

try:
    health = api_health()
except Exception as exc:
    st.error(f"API unreachable at {API_URL}: {exc}")
    st.stop()

src = health["model_source"]
if src == "synthetic_demo_fixture":
    st.warning("DEMO MODE — model is a tiny LightGBM trained on the committed synthetic fixture "
               "(not the real champion). Real deployment: point PRISMRISK_BUNDLE at champion_bundle.joblib.")
else:
    st.success(f"Champion bundle live · model={health['model']} · trained weeks {health['trained_weeks']} "
               f"· decision_source={health['decision_source']}")

df = load_applicants()
choice = st.selectbox("Applicant (test weeks ≥ 80)", df.index,
                      format_func=lambda i: f"case {df.loc[i, 'case_id']} · week {df.loc[i, 'WEEK_NUM']} · "
                                            f"amount {df.loc[i, 'credamount_770A']:,.0f}")
row = df.loc[choice]

with st.expander("Feature snapshot", expanded=False):
    st.dataframe(row.to_frame("value"), use_container_width=True)

if st.button("Score applicant", type="primary"):
    with st.spinner("Scoring…"):
        status, body = api_post("/predict", row_to_payload(row))
    if status != 200:
        st.error(body)
    else:
        icon, label = BADGE[body["decision"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Calibrated PD", f"{body['calibrated_pd']:.1%}", help="isotonic-calibrated")
        c2.metric("Decision τ*", f"{body['threshold_tau']:.1%}", help="profit-maximising cutoff (calib weeks)")
        c3.metric("Expected profit", f"${body['expected_profit_usd']:,.0f}")
        st.markdown(f"### {icon} {label}")
        st.caption(f"decision_source={body['decision_source']} · model_source={body['model_source']} · "
                   f"inference {body['inference_ms']} ms")
        st.subheader("Top risk-increasing factors (SHAP)")
        for rsn in body["top_3_shap_reasons"]:
            st.markdown(f"- **{rsn['label']}**  ·  SHAP +{rsn['shap']:.3f}")
        st.session_state["payload"] = row_to_payload(row)
        st.session_state["case_id"] = body["case_id"]

if "payload" in st.session_state and st.button("Recourse + adverse-action notice (DiCE, on demand)"):
    with st.spinner("Searching feasible counterfactuals (immutable features locked)…"):
        status, body = api_post("/recourse", {**st.session_state["payload"], "total_cfs": 2, "bg_rows": 200})
    if status != 200:
        st.error(body)
    else:
        if body["feasible"] is None:
            st.info(body["note"] or "Applicant is approved — recourse not applicable.")
        elif body["feasible"]:
            st.success("Feasible counterfactual(s) found — immutable features never moved:")
            for cf in body["counterfactuals"]:
                st.json(cf)
        else:
            st.info("No feasible counterfactual inside the permitted bounds — the honest answer is 'no'.")
        if body["top_reasons"]:
            st.subheader("Top reasons")
            for rsn in body["top_reasons"]:
                st.markdown(f"- **{rsn['label']}**  ·  SHAP +{rsn['shap']:.3f}")
        st.subheader("Adverse-action letter (numeric-verified)")
        st.text(body["applicant_notice"])
        st.caption(f"grounding_passed={body['grounding_passed']} · "
                   f"fallback_triggered={body['notice_fallback_triggered']}")

st.divider()
st.subheader("Weekly PSI drift")
drift_csv = REPO_ROOT / "reports" / "psi_weekly.csv"
if drift_csv.exists():
    drift = pd.read_csv(drift_csv)
    pd_rows = drift[drift["feature"] == "predicted_pd"].sort_values("week")
    if len(pd_rows):
        st.line_chart(pd_rows.set_index("week")["psi"], height=180)
    band = drift["band"].value_counts().to_dict()
    st.caption(f"bands: {band} · source: {drift_csv}")
else:
    st.caption("No drift report yet — run `python -m prismrisk.drift` on the training box "
               "(reports/ is gitignored).")
