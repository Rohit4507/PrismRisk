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

## What is implemented and verified (tests: 31/31 green)
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
  registry (`PrismRiskChampion`, sqlite) · Docker + CI (GitHub Actions: ruff + pytest) ·
  weekly PSI drift (Evidently) · HF Space demo · 2 notebooks · config-driven
  (`configs/config.yaml`) · no real rows committed, ever.

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
