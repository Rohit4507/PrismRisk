# Resume bullets — PrismRisk

Every number below is **real** (locked W80–91 Kaggle run, 2026-11-09) unless explicitly
labeled otherwise. Sources: [REAL_RUN.md](REAL_RUN.md) · [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md).
Nothing here is a leaderboard claim: no submission was made, and the stability formula is a
faithful re-implementation, not an official score. Synthetic-fixture numbers are labeled
`[synthetic]` and are demo-grade only.

## Short form (one bullet block)

- Engineered **PrismRisk**, an end-to-end temporal credit-risk decision + MLOps system on
  Home Credit *Credit Risk Model Stability* data (1.22M train rows, 92 weeks): time-locked
  splits (W0–59 train · W60–69 isotonic calibration + MAPIE conformal fit · W70–79 Optuna
  selection · W80–91 locked test), four model families with isotonic calibration, DiCE
  counterfactual recourse, numeric-verified adverse-action notices, a Dockerized FastAPI +
  Streamlit service, and weekly PSI drift monitoring.

## Key numbers (all real, labeled)

- **Champion**: `xgboost_unconstrained` (5 Optuna trials) — val ROC-AUC 0.8541; monotone-constrained
  LightGBM within 0.0005 (0.8536), kept as the governance-compliant pick at a statistical tie.
- **Locked W80–91 test**: ROC-AUC **0.8540** (flat vs val — no COVID-window collapse) · PR-AUC **0.1468**
  (~5× over ~3% prevalence) · Brier (isotonic) 0.0198 · ECE 0.0165.
- **Stability (competition formula, re-implemented & unit-tested)**: **0.6843** — mean weekly Gini
  0.7000, slope +0.0044/wk, residual std 0.031, min 0.627 — under the W60+ COVID-era distribution shift.
- **Calibration**: val ECE 0.0132 → **0.0103** (isotonic); profit-optimal cut **τ\* = 0.16**;
  max modeled profit **$471,078,178** (val).

## Systems & engineering (real, qualitative)

- **MAPIE conformal grey-zone triage** (AUTO_APPROVE / GREY_ZONE_REVIEW / AUTO_DENY), 90% coverage
  target — measured coverage 0.900 / 0.901 `[synthetic fixture]`; **real-data conformal profit
  evaluation is still pending** — say so in interviews.
- **DiCE feasibility-constrained recourse** (demand-side only): immutable features locked, derived
  ratios recomputed inside a **calibrated** ratio-aware wrapper, train-quantile permitted bounds,
  and a flip-validity gate — counterfactuals that don't change the served decision are rejected
  (`feasible=False`), never presented as actionable. NaN rows are median-imputed for the search and
  reported via `imputed_features`.
- **Zero-hallucination adverse-action notices**: numeric-token verifier (every number in the letter
  must appear in the facts) + deterministic template fallback; template never promises approval.
- **Service**: FastAPI with measured latency per response (`/predict` incl. SHAP reasons, `/recourse`,
  `/drift`, `/health` with `model_source` honesty) · Streamlit dashboard · Docker + Compose
  (non-root, 3.12-slim) · GitHub Actions CI (ruff + pytest) · **49/49 tests passing**.
- **Perf (dev box, indicative, demo model)**: 10 concurrent users — p50 30 ms / p95 56 ms / ~148 RPS;
  100 users (single worker, saturated) — p50 500 ms / p95 640 ms / ~168 RPS, 0 failures (Locust).
- **MLflow** tracking + registry (`PrismRiskChampion` v1) · in-house weekly **PSI drift monitor**
  (Evidently dropped — unimportable under pandas 3.x) with train-locked bins.

## Numbers to avoid / honest framing

- Do **not** cite a leaderboard rank — no Kaggle submission was made.
- Stability 0.6843 is "competition-formula re-implemented and verified against open reference
  implementations", never "official score".
- `$471M` is **modeled expected profit at val τ\***, not realized savings.
- "Monotone LightGBM" shipped as champion? **No** — XGB won by 0.0005; keep the honest sentence or
  report the statistical tie.
- Docker build and the HF Space demo are not verified yet (user-side) — don't claim "deployed".
- Conformal "profit lift" has **not** been measured on real data; on the synthetic fixture the 3-way
  policy *lost* to the binary cutoff (−$1.77M, Notebook 02). Report the loss, not a lift.
