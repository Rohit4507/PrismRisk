# PrismRisk

[![CI](https://github.com/Rohit4507/PrismRisk/actions/workflows/ci.yml/badge.svg)](https://github.com/Rohit4507/PrismRisk/actions/workflows/ci.yml)

Temporal credit decision & MLOps system — monotone LightGBM + isotonic calibration,
MAPIE conformal grey-zone triage, DiCE counterfactual recourse, zero-hallucination
adverse-action notices, weekly PSI drift monitoring.

> 📄 Full plan: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)

## Status

- [x] M0 — Packaging (`pyproject.toml`, pinned deps), `.gitattributes`, CI workflow
- [x] M1 — features/split/metrics + **real Kaggle run**: locked W80–91 **ROC-AUC 0.854 / stability 0.684** (12 test weeks), zero val→test drop; see [docs/REAL_RUN.md](docs/REAL_RUN.md) for the verbatim run
- [x] M2 — train.py: LogReg/MLP/XGB/Monotone-LGBM (Optuna) + isotonic calibration (val ECE 0.0132→0.0103 on real data) + MLflow tracking & registry (`PrismRiskChampion` v1)
- [x] M3 — MAPIE SplitConformalClassifier grey-zone triage (AUTO_APPROVE/GREY_ZONE_REVIEW/AUTO_DENY); coverage 0.900/0.901 on the synthetic fixture; real-data profit eval pending
- [x] M4 — SHAP adverse-action reasons + DiCE feasibility-constrained counterfactual recourse (immutable features locked, ratios recomputed inside the model wrapper; demand-side only) + zero-hallucination notice guard (numeric verifier + deterministic fallback)
- [x] M5 — weekly PSI drift (train-locked 10-bin quantile bins frozen on W0–59) with green/warning/retrain bands + self-contained HTML report (evidently dropped — unimportable under pandas 3.x); verified on the synthetic fixture; real-data rerun is user-side (see [docs/REAL_RUN.md](docs/REAL_RUN.md)). Notebook 02 executed 2026-11-09 (synthetic fixture, 0 errors) — see below
- [x] M6 — FastAPI (`/health` `/predict` `/recourse` `/drift`, measured latency in every response) + Streamlit dashboard (test-week borrowers, SHAP reasons, recourse letter, PSI chart) + Dockerfile/compose (api + dashboard + mlflow, non-root, 3.12-slim) + Locust benchmark + CI lint scope extended. Measured (demo model, dev box): 10 u p50 30 ms · p95 56 ms · ~148 RPS; 100 u (single worker, saturated) p50 500 ms · p95 640 ms · ~168 RPS, 0 fails. Docker build not run in this sandbox (no container runtime) — user-side verify: `docker compose up --build`
- [ ] M7 — HF Space live demo · Notebook 01 (EDA) still pending

## Notebook 02 — benchmark · calibration · XAI (executed 2026-11-09, synthetic fixture)

`notebooks/02_benchmark_calibration_xai.ipynb` executed end-to-end (0 errors). All numbers on the **synthetic smoke fixture** — honest process proof, never a public score. Headlines: overall best val ROC-AUC `logreg_baseline` 0.8930; tree reference champion `lightgbm_monotone` 0.8669 (SHAP/DiCE need a tree) — reported side by side, never fudged; locked-week stability 0.7385; conformal 3-way policy **lost** to the binary cutoff by $1.77M on this distribution (reported as-is); DiCE counterfactuals are flip-validated against the served (isotonic-calibrated) decision boundary — a CF that does not change the decision is rejected, never shown as feasible. Figures: [`weekly_gini`](reports/figures/weekly_gini.png) · [`reliability`](reports/figures/reliability.png) · [`conformal_profit`](reports/figures/conformal_profit.png) · [`shap_waterfall`](reports/figures/shap_waterfall.png) · [`psi_weekly`](reports/figures/psi_weekly.png)

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                       # 49 tests, synthetic data — no Kaggle download needed
python scripts/make_synthetic.py --rows 2000   # synthetic smoke stand-in
```

## Run the demo (API + dashboard)

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000   # no champion bundle? falls back to a tiny
                                                     # LightGBM trained on the synthetic fixture at startup
streamlit run dashboard/app.py                        # http://localhost:8501
python -m prismrisk.drift --features data/processed/features.parquet --bundle models/real_champion.joblib  # drift report (gitignored)
locust -f tests/locustfile.py --headless -u 100 -r 20 -t 60s --host http://127.0.0.1:8000
```

API contract: `/predict` requires the full trained feature vector (missing → 422, never silently imputed).
`decision_source` says `conformal_triage` (MAPIE sets) or `tau_star_threshold` (no conformal artefact —
no grey zone, never faked). In demo mode `/health` reports `model_source=synthetic_demo_fixture` —
never presented as the real champion. Production: mount `models/champion_bundle.joblib` (+ optionally
`models/conformal_triage.joblib`) and set `PRISMRISK_BUNDLE`; `docker compose up --build` brings up
api + dashboard + mlflow. See [docs/REAL_RUN.md](docs/REAL_RUN.md) for the real-data caveats.

## Data
Not committed (competition rules — see **data/README.md**).
- **Real-data run**: `make data` (Kaggle login + one-time rules click, ~4 GB), or **(A)** UCI "Default of Credit Card Clients" (30k rows, license-friendly, needs a small column adapter).
- **Dev/CI**: `python scripts/make_synthetic.py --rows 300000` — synthetic ~$33k loans, pre/post Week-60 default shock 12% → 23%; demo-grade numbers, never presented as real.
Tests and CI need no download (synthetic only).
