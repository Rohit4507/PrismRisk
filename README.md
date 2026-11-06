# PrismRisk

Temporal credit decision & MLOps system — monotone LightGBM + isotonic calibration,
MAPIE conformal grey-zone triage, DiCE counterfactual recourse, zero-hallucination
adverse-action notices, weekly PSI drift monitoring.

> 📄 Full plan: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)

## Status

- [x] M0 — Packaging (`pyproject.toml`, pinned deps), `.gitattributes`, CI workflow
- [x] M1 (core) — `features.py` (DuckDB/Polars depth-1 aggregation, relative dates, ratios), `split.py` (strict W0–59/60–69/70–79/80–91), `metrics.py` (weekly Gini stability, Brier/ECE, net profit sweep) — **18 tests green**
- [x] M4 (partial) — `notice_guard.py` zero-hallucination adverse-action notice (numeric verifier + deterministic fallback)
- [ ] M1 — run on real Kaggle CSVs (needs dataset download) + Notebook 01
- [x] M2 — train.py: LogReg/MLP/XGB/Monotone-LGBM (Optuna) + isotonic calibration + MLflow tracking & registry (verified end-to-end on synthetic data; real-data run pending Kaggle download)
- [ ] M3 — MAPIE conformal triage · M5 — drift/Evidently + Notebook 02
- [ ] M6 — FastAPI + Streamlit + Docker + Locust p50/p95/p99 · M7 — HF Space live demo

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                       # 20 tests, synthetic data — no Kaggle download needed
python scripts/make_synthetic.py --rows 2000   # synthetic smoke stand-in
```

## Data
Not committed (Kaggle competition rules). Download `home-credit-credit-risk-model-stability`
CSVs into `data/raw/`, convert to Parquet, then `python -m prismrisk.features --sample`.
