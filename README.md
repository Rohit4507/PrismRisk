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
- [x] M5 — weekly PSI drift (train-locked 10-bin quantile bins frozen on W0–59) with green/warning/retrain bands + self-contained HTML report (evidently dropped — unimportable under pandas 3.x); verified on the synthetic fixture; real-data rerun is user-side (see [docs/REAL_RUN.md](docs/REAL_RUN.md)). Notebook 02 still pending
- [ ] M6 — FastAPI + Streamlit + Docker + Locust p50/p95/p99 · M7 — HF Space live demo

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                       # 42 tests, synthetic data — no Kaggle download needed
python scripts/make_synthetic.py --rows 2000   # synthetic smoke stand-in
```

## Data
Not committed (competition rules — see **data/README.md**).
- **Real-data run**: `make data` (Kaggle login + one-time rules click, ~4 GB), or **(A)** UCI "Default of Credit Card Clients" (30k rows, license-friendly, needs a small column adapter).
- **Dev/CI**: `python scripts/make_synthetic.py --rows 300000` — synthetic ~$33k loans, pre/post Week-60 default shock 12% → 23%; demo-grade numbers, never presented as real.
Tests and CI need no download (synthetic only).
