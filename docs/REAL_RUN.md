# Real Kaggle run — locked W80–91 results (2026-11-09, user's machine)

First **real** run of PrismRisk on the Kaggle competition data
("Home Credit — Credit Risk Model Stability", training file only; the Kaggle
`test/` folder is for submission and was not used). Command:

```powershell
python -m prismrisk.train --features data/processed/features.parquet --out models/real_champion.joblib --trials 5
```

`features=313 train=1,219,149 monotone(+1)=3 monotone(-1)=1` — 5 Optuna trials was all the
time budget allowed; **more trials will only help**.

## Model selection (val, weeks 70–79)

| Model | val ROC-AUC | val PR-AUC |
|---|---|---|
| logreg_baseline | 0.7911 | 0.1144 |
| mlp_challenger | 0.7717 | 0.0949 |
| xgboost_unconstrained | **0.8541** | 0.1705 |
| lightgbm_monotone | 0.8536 | — |

**Champion: `xgboost_unconstrained`** — the monotone LightGBM was within 0.0005 AUC
(0.8536). Honest read: measured champion wins; the monotone variant would be the
production-governance pick at a statistical tie. Overnight `--trials 25` may flip this.

## Calibration (val)

brier 0.0215 → 0.0211 · ECE 0.0132 → **0.0103** · tau_star (profit-max cut) 0.16 ·
max_profit_calib **$471,078,178**.

## Locked test (weeks 80–91) — the number that matters

| Metric | Value |
|---|---|
| ROC-AUC | **0.8540** (val 0.8541 → flat: no COVID-window collapse) |
| PR-AUC | 0.1468 (≈5× lift over ~3% prevalence) |
| brier_cal | 0.0198 |
| ECE_cal | 0.0165 |
| mean weekly Gini | 0.7000 · std 0.0348 · min 0.6269 |
| slope/week | **+0.0044** (upward — no stability penalty) |
| residual std | 0.0313 |
| **gini_stability** (competition formula) | **0.6843** |
| n test weeks | 12 |

`train_seconds` 1044 · bundle `models/real_champion.joblib` (12.84 MB, local only) ·
MLflow: experiment `PrismRisk-Stability`, registered **`PrismRiskChampion` v1**.

## Honesty notes

- Machine: user's Windows laptop (venv), ~17 min end-to-end including calibration
  and MLflow registration. Raw console log preserved by the user.
- Real numbers are **not yet** woven into CI (no data in CI — by design).
- No leaderboard submission was made and none should be claimed.
- `real_champion.joblib`, `mlflow.db`, `data/processed/features.parquet` stay
  **uncommitted** (gitignored). Only this metrics file and README labels carry the
  measured numbers — never real rows.
- Conformal triage (M3) has **not** been re-run on real data yet — the profit-vs-
  cutoff result is synthetic-only for now.
- DiCE recourse (M4) is verified on the tiny synthetic fixture and on synthetic
  300k-row features; it has **not** been timed on the real 313-feature frame.

## To reproduce / extend

1. `make data` (or the kagglehub route) → `python scripts/prepare_raw.py --src <extracted>/train`
2. `python -m prismrisk.features`
3. Overnight: `python -m prismrisk.train --features data/processed/features.parquet --out models/real_champion.joblib --trials 25`
4. M3 triage eval on real data → then M5 (PSI) → M6 (Docker/API) → M7 (HF Space).
