# PrismRisk — Executive Summary
*For interviewers and recruiters. Every number below comes from the repo's 300k-row
**synthetic** fixture (with an engineered Week-60 "COVID shock") unless marked otherwise —
they demonstrate the pipeline, not real-world lending performance. The real Kaggle
"Credit Risk Model Stability" run happens with `make data` on a machine that can reach
Kaggle (see `data/README.md`).*

## The problem
Home Credit's "Credit Risk Model Stability" competition showed that a default-prediction
model is not judged on raw AUC — it is judged on **weekly Gini stability**
(`mean(Gini) + 88·min(0, slope) − 0.5·std(residuals)`, `gini_w = 2·AUC_w − 1`), i.e. whether
it keeps separating defaulters week after week through the COVID-era distribution shift
(weeks ~60+). PrismRisk is an end-to-end answer: data → model → calibration →
conformal triage → recourse → service → drift monitoring, built like a product, not a notebook.

## What is implemented and verified (tests: 35/35 green)
- **Time-locked splits** W0–59 train · W60–69 isotonic calibration + conformal fit ·
  W70–79 Optuna selection · W80–91 locked test — never shuffled.
- **Models**: LogReg · MLP · XGBoost · monotone-constrained LightGBM (Optuna) with
  banking-governance monotonicity (bureau DPD, income…). Calibration lifted ECE 0.078 → 0.002.
- **Locked-test (synthetic) numbers**: ROC-AUC ≈ 0.90 · PR-AUC 0.667 · Brier 0.062 ·
  weekly Gini mean 0.795 · slope −0.0003 · **stability 0.758** (competition formula,
  re-implemented and verified against two independent open implementations).
- **MAPIE conformal grey-zone triage** (90% target): measured coverage 0.900 / 0.901 on
  val/test weeks; three-way AUTO_APPROVE / GREY_ZONE_REVIEW / AUTO_DENY.
- **Honest economics, reported not hidden**: on this distribution the binary cutoff earned
  **$139.0M** vs the conformal 3-way policy **$122.2M** — grey-zone only pays when the model
  is highly informative or false denials are expensive. Kept in the README, not buried.
- **Bureau risk-band parser**: `riskassesment_940T`-style text ("6% - 8%", ~3.5% coverage) —
  the competition's strongest single signal; parsed to a monotone feature (+ sparse by
  design, exempt from the null filter).
- **Engineering**: DuckDB/Polars joins (depth-0 direct, depth-1 aggregated) · MLflow
  registry (`PrismRiskChampion` v1, sqlite) · Docker + CI (GitHub Actions: ruff + pytest) ·
  FastAPI service (`/predict` with SHAP reasons + 3-way conformal decision + expected profit,
  `/recourse` demand-side DiCE with numeric-verified letter, measured latency per response) ·
  Locust benchmark on the dev box (10 u: p50 30 ms / p95 56 ms / ~148 RPS · 100 u saturated single
  worker: p50 500 ms / p95 640 ms / ~168 RPS, 0 failures — indicative, demo artefacts only) ·
  weekly PSI drift monitor (in-house PSI → self-contained HTML report; Evidently dropped) ·
  HF Space demo · 2 notebooks · config-driven (`configs/config.yaml`) · no real rows committed, ever.
- **DiCE recourse (M4)**: demand-side, immutable features locked, ratio-aware model
  wrapper (credamount/income ratios recomputed inside), top actionable drivers by |SHAP|
  with train-quantile bounds; infeasible requests return an honest `feasible=False`.

## Real Kaggle run (2026-11-09, user machine — locked W80–91, trials=5)
ROC-AUC **0.8540** (flat vs val 0.8541) · PR-AUC 0.1468 · ECE 0.0165 ·
weekly-Gini stability **0.6843** (slope **+0.0044**/wk, residual std 0.031, min 0.627) ·
bureau risk-band parse exercised on the real `static_cb_0`. Champion: xgboost_unconstrained
(monotone LGBM within 0.0005 — governance call). Full details: docs/REAL_RUN.md. Not a
leaderboard claim; no submission made.

## What is deliberately honest
- No leaderboard claims. The stability formula is a faithful re-implementation, not an
  official score.
- Synthetic numbers above are demo-grade (prevalence 7% by construction); the real
  Kaggle run is a `make data` away and will replace them.
- Binary-vs-conformal profit loss is reported, not cherry-picked.

## What the top public solutions teach (and what we borrowed)
- **Bra-AM / Erdos bootcamp (Spring 2024)**: GBDTs beat their NN; the NN memorized
  WEEK_NUM/date inputs (test AUC 0.59 vs 0.78); the bureau risk band was their #1 single
  feature; stability must be computed week-by-week, tested vs competition reference code.
- **Silver medal (Rank 147/3856 — varuni-rao)**: LightAutoML CV5 ensemble beat hand-tuned
  LightGBM/H2O; strict WEEK_NUM validation; parquet + float32 downcast; no SMOTE.
- **minerva-ml (2018, 469★)**: feature richness > model choice.
- **We borrowed the discipline, not the leaderboard**: time-aware validation, sparse
  risk-band parsing, downcast parquet pipeline, no SMOTE — and wrapped them in
  calibration + conformal + drift + deploy + CI that submissions never have.

## 30-second pitch
"PrismRisk is my answer to 'your model is decaying in production'. Time-locked validation,
monotone governed LightGBM, isotonic calibration, conformal grey-zoning with measured 90%
coverage, DiCE recourse, zero-hallucination adverse-action letters, drift monitoring and a
Dockerized API — on the Home Credit stability competition, with every number labeled."
