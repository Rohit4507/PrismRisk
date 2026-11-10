# PrismRisk — Final 2026 Implementation Plan (Step-by-Step)

**Scope:** 4th-year undergrad portfolio project. ~21 days, ~25–30 focused hours, ~24 files.
**Goal:** End-to-end temporal credit decision + MLOps system on the Home Credit stability dataset with 5 "2026 upgrades": conformal grey-zone triage, DiCE counterfactual recourse, monotone LightGBM, zero-hallucination AI adverse-action notice, measured p50/p95/p99 API latency.

This document **supersedes** the chat plan where they disagree. Every API here was verified on 2026-11-06 against PyPI and the official library repos (see §9).

---

## 0. Reality check — what the chat plan got wrong (read first)

| # | Chat plan said | Reality (verified 2026-11-06) | What we do instead |
|---|---|---|---|
| 1 | Dataset is "Parquet tables" | The Kaggle competition ships **CSVs** (`train_base.csv`, `train_static_0_*.csv`, …). Parquet is an optimization you apply locally. | Download CSVs manually → convert to Parquet once with Polars → DuckDB reads Parquet. Never commit data (`data/raw/` in `.gitignore`). |
| 2 | `MAPIE (MapieClassifier, prefit=True)` | Deprecated. Current API (mapie 1.6.0): `from mapie.classification import SplitConformalClassifier` → `SplitConformalClassifier(estimator=…, confidence_level=0.9, prefit=True)`, then `.conformalize(X_cal, y_cal)` and `.predict_set(X)` → `y_pred_set` shape `(n, 2, 1)` booleans. Verify with `classification_coverage_score`. | Code the current API (§6.3). The old call is the #1 breakage risk in the chat plan. |
| 3 | Competition stability formula: `mean(weekly Gini) + 88·min(0, slope) − 0.5·std(residuals)` | ~~invented~~ **WRONG** — this IS the competition stability formula, verified in two independent open implementations matched against the competition's reference code (`Bra-AM/erdos-homecreditmodelstability`, fork of `kwongszehongumdmath/erdos-homecreditmodelstability`, 0★; `varuni-rao/home-credit-risk-stability-silver`, MIT, private LB 147/3856 Silver). Residuals are measured **around the fitted week-line**, not raw weekly std. | Implement the competition formula in `metrics.py::weekly_gini_stability` (returns `stability`, per-week Ginis, fitted `slope`, `residual_std_gini`, keeping `custom_stability` as deprecated alias). Report it as a faithful re-implementation — still no official leaderboard claim without a real submission. |
| 4 | "Bra-AM/erdos-homecreditmodelstability" as a reference | It's a **0-star fork** of `kwongszehongumdmath/erdos-homecreditmodelstability` (0★). Layout is ccds-like, code not production-grade — **but its `src/creditrisk/metrics.py` is our only verified mirror of the competition's stability implementation**. | Use as *metric-formula reference* and rough layout guide; borrow the formula, NOT its modeling approach (its MLP memorized WEEK_NUM/date columns → ~0.59 test AUC — we never feed week/date as model features). |
| 5 | "reference repos to keep open while coding" | All three chat-cited repos are **0-star student projects**. | Keep MAPIE/DiCE official docs + `zulqarnainalipk/Home-Credit---Credit-Risk-Model-Stability` (⭐7) open instead. |
| 6 | `/predict` returns in <15 ms via MLflow | Loading from MLflow at request time is slow and fragile. | Load a **joblib bundle** at FastAPI startup; MLflow registry is for lineage, not hot-path inference. |
| 7 | "Weekly Gini ≥ 0.53" as a guarantee | That's a goal, not a guarantee. | Treat as a target; report actual numbers in README. |
| 8 | `uv init prismrisk` | Repo **already exists** as `Rohit4507/PrismRisk` (this checkout). | Work inside it. No re-init. |
| 9 | Calibration via `CalibratedClassifierCV(FrozenEstimator(...))` only | Valid in sklearn ≥1.6, but the simplest robust route is `IsotonicRegression(out_of_bounds="clip")` fit on weeks 60–69 PDs. | Use direct IsotonicRegression as primary; FrozenEstimator as alternative (§5.3). |
| 10 | `.gitignore` ignores `models/*.joblib` but HF Space needs the champion | Contradiction: Space needs the model file. | Commit `models/champion_bundle.joblib` (LightGBM, typically 1–10 MB) for the Space; ignore other `*.joblib` artifacts. Use git-lfs if >25 MB. |

**Kaggle data access note:** the dataset requires a Kaggle login + competition rules acceptance. It cannot be downloaded in CI or in this sandbox. Download locally, work locally, commit only `data/processed/sample_500.parquet` (<2 MB) for tests/Spaces.

**Rules/licensing:** check the competition rules before publishing a model trained on it. Pin to the competition citation in the README; non-commercial portfolio use is standard for this comp, but verify.

---

## 1. Final repo tree (24 files, unchanged in spirit)

```
PrismRisk/                      # already exists — work inside, don't re-init
├── .github/workflows/ci.yml             # Ruff + Pytest on push/PR
├── .gitattributes             # notebooks/*.ipynb linguist-documentation
├── .gitignore
├── Dockerfile                  # multi-stage FastAPI image
├── docker-compose.yml          # api:8000 + mlflow:5000 + dashboard:8501
├── Makefile                    # data | features | test | up | loadtest
├── data/README.md              # Kaggle download steps; competition data never committed
├── README.md                   # leads with Live Demo → architecture → tables
├── pyproject.toml + uv.lock
├── configs/config.yaml         # single source of truth (weeks, profit, monotone, DiCE bounds)
├── notebooks/
│   ├── 01_eda_and_feature_rationale.ipynb
│   └── 02_benchmark_calibration_xai.ipynb
├── src/prismrisk/
│   ├── __init__.py
│   ├── config.py               # typed Pydantic settings from config.yaml
│   ├── features.py             # DuckDB + Polars: 6 tables → 1 row/case_id
│   ├── split.py                # W0–59 train | W60–69 calib | W70–79 val | W80–91 test
│   ├── metrics.py              # ROC/PR-AUC, weekly Gini stability, ECE, net profit $
│   ├── train.py                # LogReg / MLP / XGB / Monotone LGBM + Isotonic + MLflow
│   ├── triage.py               # MAPIE SplitConformalClassifier → 3-way triage
│   ├── explain_recourse.py     # TreeSHAP top-3 + DiCE constrained counterfactuals
│   ├── notice_guard.py         # grounded applicant letter + numeric verifier
│   └── drift.py                # weekly PSI (train bins) + in-house HTML report
├── api/
│   ├── schemas.py              # Pydantic v2 contracts
│   └── main.py                 # /health /predict /recourse /drift
├── dashboard/app.py            # 3-tab Streamlit (Underwriter Desk | Profit Sim | Drift)
└── tests/
    ├── test_pipeline.py        # leakage, join uniqueness, monotonicity, guard, API 200
    └── locustfile.py           # p50/p95/p99 load benchmark
```

## 2. Toolchain (verified 2026-11-06 on PyPI)

```
polars 2.1.0 · duckdb 2.0.0 · pyarrow 26.0.1 · lightgbm 4.7.0 · xgboost 3.2.0
scikit-learn 1.9.1 · optuna 5.0.0 · mlflow 3.18.0 · mapie 1.6.0 · dice-ml 0.12
shap 0.51.0 · fastapi 0.162.0 · uvicorn 0.54.0 · streamlit 1.67.0  (evidently 0.7.23 dropped — see §8 reality-check)
pydantic 2.14.3 · pyyaml 6.0.3 · joblib 1.6.0 · matplotlib 3.11.2 · seaborn 0.13.2
pytest 9.2.0 · ruff 0.17.4 · locust 2.47.0 · httpx 0.28.1 · ipykernel 7.4.0
```

Pin exactly these in `pyproject.toml` (uv or pip — both fine; `uv.lock` committed if using uv; pip users commit `requirements.txt` exported via `uv pip compile` or `pip-compile`).

**Python version pin (important, verified 2026-11-06):** xgboost ≥3.3 and shap ≥0.52 require Python ≥3.12. We pin `xgboost==3.2.0` and `shap==0.51.0` for Python 3.11. On Python ≥3.12 you may unpin freely.

**Sandbox fallback:** if `uv` isn't installed, use `python3 -m venv .venv && pip install -e ".[dev]"`. The plan works either way. In this sandbox uv was absent; a `.venv` was used. Ruff `line-length = 120`.

---

## 3. Milestone 0 — Repo bootstrap + data (Day 1, ~1.5 h)

```bash
# already in /home/user/PrismRisk on branch arena/e097bd73-prismrisk
# 1) package + tooling
uv init --package prismrisk        # or: pip install -e . after writing pyproject.toml
uv add polars duckdb pyarrow lightgbm xgboost scikit-learn optuna mlflow \
       mapie dice-ml shap fastapi "uvicorn[standard]" streamlit \
       pydantic pyyaml joblib matplotlib seaborn
uv add --dev pytest ruff locust httpx ipykernel
# 2) hygiene files
printf 'notebooks/*.ipynb linguist-documentation\n' > .gitattributes
# .gitignore: data/raw/ data/processed/mlruns/ mlflow.db mlruns/ __pycache__/ .venv/
#              models/*.joblib  ← BUT force-add champion_bundle.joblib later: git add -f
# 3) directories
mkdir -p configs data/raw/parquet data/processed notebooks reports/figures \
         src/prismrisk api dashboard tests .github/workflows
# 4) download data manually from Kaggle (Kaggle account + rules acceptance required)
#    competition: "Home Credit - Credit Risk Model Stability"
#    files used (~6 core tables):
#      train_base.csv
#      train_static_0_1..9.csv   → credamount_770A, annuity_780A, *P dpd cols
#      train_static_cb_0.csv     → external bureau static, *D dates
#      train_person_1.csv        → mainoccupationinc_384A, filter num_group1==0
#      train_applprev_1_1..*.csv → previous application depth-1
#      train_credit_bureau_a_1_1..*.csv → bureau depth-1 (DPD, overdue)
# 5) convert CSVs → Parquet once:
uv run python -c "import polars as pl, glob; [pl.scan_csv(f).sink_parquet(f.replace('.csv','.parquet')) for f in glob.glob('data/raw/*.csv')]"
```

**Done gate:** `uv sync --frozen` OK; `ls data/raw/*.parquet` shows the six tables; `python -c "import prismrisk"` OK; nothing >2 MB staged by git.

## 4. Milestone 1 — Features + chronological split (Days 2–4, ~5 h)

### 4.1 `src/prismrisk/features.py`
- Read with `pl.scan_parquet` (lazy) or `duckdb.sql("...")`.
- **Aggregate depth-1 → 1 row/case_id first**, then 1:1 left join onto `train_base`:
  - `static_0`: `max_dpd_prev = MAX(*P cols)`, `credamount_770A`, `annuity_780A`, …
  - `person_1`: filter `num_group1 == 0` → `maininc_clean = MEDIAN(mainoccupationinc_384A)`, `birth_259D`, `empl_from_D`
  - `applprev_1_*`: `COUNT(*) AS prev_app_count`, `SUM(refused) AS prev_refused_count`, `MEAN(annuity) AS mean_prev_annuity`
  - `credit_bureau_a_1_*`: `COUNT(*) AS bureau_loan_count`, `MAX(dpd) AS max_dpd_bureau`, `SUM(overdue) AS total_overdue_amt`
- **Relative dates:** every `*D` column → `(date_decision - col_D).dt.total_days()`. Drop raw date cols. `age_years = (date_decision - birth_259D) / 365.25`.
- **Inflation-proof ratios:** `annuity_to_income = annuity_780A / (maininc_clean + 1)`; `credamount_to_income = credamount_770A / (maininc_clean + 1)`; `overdue_to_credamount = total_overdue_amt / (credamount_770A + 1)`.
- Drop cols with >95% nulls **measured on weeks 0–59 only** (fit on train fold, apply everywhere → no leakage).
- Cast to Float32; write `data/processed/features.parquet` **and** `data/processed/sample_500.parquet` (stratified 500 rows incl. both classes, <2 MB, committed).
- Runtime reality check: <60 s / <4 GB is the target; DuckDB temp dir on disk if needed (`PRAGMA temp_directory`).

### 4.2 `src/prismrisk/split.py`
```python
TRAIN, CALIB, VAL, TEST = (0,59), (60,69), (70,79), (80,91)   # from config.yaml
assert df.weeks.is_between(...)  # build 4 disjoint frames; export w/ case_id, target, weeks
```
Never shuffle. Assert `max(train.week) < min(calib.week) < min(val.week) < min(test.week)`.

### 4.3 Notebook `01_eda_and_feature_rationale.ipynb`
4 figures → `reports/figures/`: (1) target imbalance (~96.9% / ~3.1% — verify on your data), (2) missingness by table, (3) raw-date drift vs relative-day stability across weeks, (4) weekly volume + default rate showing the Week-60 (March 2020 COVID) shift.

**Done gate:** uniqueness test passes (`n_unique(case_id) == len(df)`); zero week overlap across splits; sample_500 committed.

## 5. Milestone 2 — Metrics + monotone LightGBM + MLflow (Days 5–8, ~6 h)

### 5.1 `src/prismrisk/metrics.py`
- `weekly_gini_stability(df)`: per `WEEK_NUM`, `gini_w = 2·roc_auc_w − 1`; fit `gini_w ~ a·week + b`; return dict(`stability`, `mean_gini`, `std_gini`, `min_gini`, `slope`, `residual_std_gini`, deprecated alias `custom_stability`); single-class weeks skipped. Stability = competition formula — report it as a faithful re-implementation, never as an official leaderboard score.
- `calibration_report(y, p)`: `brier_score_loss` + 10-bin ECE + reliability arrays.
- `expected_net_profit(y, p, credamount, cutoff, margin=0.14, lgd=0.65)`: approve `p < cutoff` → `+margin·amount` if `y==0` else `−lgd·amount`. Sweep cutoff 0.01–0.25 → profit curve + `tau_star` on **calib weeks**.
- `roc_auc`, `pr_auc` wrappers.

### 5.2 `src/prismrisk/train.py`
MLflow: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`, experiment `PrismRisk-Stability`. Train on W0–59, select on **W70–79**:
1. `logreg_baseline` — Imputer + Scaler + LogisticRegression (whiteboard-friendly baseline)
2. `mlp_challenger` — MLPClassifier((128,64)) — expected to degrade under W60+ shift (documents temporal drift)
3. `xgboost_unconstrained` — challenger
4. `lightgbm_monotone` — Optuna, 25 trials. `monotone_constraints` list aligned to `feature_names` from `config.yaml` (`+1` on `max_dpd_bureau`, `max_dpd_prev`, `annuity_to_income`, `credamount_to_income`; `−1` on `maininc_clean`).

Log params/metrics (`val_roc_auc`, `val_pr_auc`, `mean_gini`, `brier_before/after`, `optimal_profit_usd`, `tau_star`) + artifacts; register **two** artifacts: the raw LGBM and the isotonic-calibrated bundle; name champion `PrismRiskChampion` in the registry.

### 5.3 Calibration (weeks 60–69)
```python
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(lgbm_raw_pd[calib], y_calib)
cal_pd = iso.predict(raw_pd)          # cal_pd ∈ [0,1], matches empirical default rate
```
Alternative: `CalibratedClassifierCV(FrozenEstimator(lgbm), method="isotonic")` (sklearn ≥1.6: `from sklearn.frozen import FrozenEstimator`).
Export `models/champion_bundle.joblib` = `{model, isotonic, feature_names, tau_star, config_version, trained_weeks:"0-59"}`.

**Done gate:** `mlflow ui` shows 4 runs side by side; test asserts `pd(max_dpd_bureau=high) ≥ pd(max_dpd_bureau=low)` on synthetic rows (monotone proof, CI-safe).

## 6. Milestone 3 — Conformal grey-zone triage (Days 9–10, ~3 h) · Upgrade #1

### 6.1 `src/prismrisk/triage.py` — current MAPIE 1.6.0 API (verified)
```python
from mapie.classification import SplitConformalClassifier
from mapie.metrics.classification import classification_coverage_score

conf = SplitConformalClassifier(estimator=calibrated_lgbm,   # prefit isotonic-calibrated model
                                confidence_level=0.90, prefit=True)
conf.conformalize(X_cal, y_cal)          # weeks 60–69 only
y_pred, y_set = conf.predict_set(X_test) # y_set.shape == (n, 2, 1), bool
coverage = classification_coverage_score(y_test, y_set)[0]
```
### 6.2 Decision mapping (binary 0/1 problem)
| prediction set | decision | action |
|---|---|---|
| {0} | `AUTO_APPROVE` | approve, no human |
| {0,1} | `GREY_ZONE_REVIEW` | escalate; charge `$12` review cost in profit sim |
| {1} | `AUTO_DENY` | decline |
Report: % auto-decided, default rate inside GREY_ZONE vs AUTO_APPROVE, net profit vs hard cutoff at `tau_star`. Save `models/conformal_triage.joblib`.

**Done gate:** decisions ∈ {APPROVE, GREY_ZONE, DENY}; empirical coverage on W80–91 within ~90% ± 2–3 pts (report whatever you get, don't fake it).

## 7. Milestone 4 — SHAP + DiCE + verified notice (Days 11–14, ~7 h) · Upgrades #2 & #4

### 7.1 `src/prismrisk/explain_recourse.py`
**Part A — adverse reasons (TreeSHAP, <5 ms):** `shap.TreeExplainer(champion)` → top-3 **positive** SHAP contributions with human labels (`"Installment-to-income 46.2% (+0.031 risk)"`).

**Part B — DiCE recourse** (`dice-ml` 0.12):
```python
dice_data = Dice(data_interface=dice_ml.Data(...), model=dice_ml.Model(model=clf, backend="sklearn"),
                 method="genetic")  # constructor form: Dice(data, model, method="genetic")
cf = dice_data.generate_counterfactuals(q, total_CFs=4, desired_class="opposite",
        features_to_vary=["credamount_770A", "annuity_780A", "pmtnum_254L"],
        permitted_range={"credamount_770A": [0.5*q["credamount_770A"], q["credamount_770A"]],
                         "annuity_780A":    [0.5*q["annuity_780A"],    q["annuity_780A"]]})
```
- **Immutable:** `age_years`, `max_dpd_bureau`, `max_dpd_prev`, `bureau_loan_count`, `prev_app_count` (from `config.yaml:governance.immutable_features`) — never in `features_to_vary`.
- Recompute derived ratios **inside** the DiCE model wrapper so changing `credamount_770A` updates `credamount_to_income` automatically.
- Rank CFs by %-change needed; return top-2 that flip DENY/GREY_ZONE → APPROVE.
- DiCE runtime on 75–90 features can be slow — run on demand (API `/recourse`), never in `/predict` hot path.

### 7.2 `src/prismrisk/notice_guard.py` · Upgrade #4
- `EvidenceFacts` dataclass: `pd_pct`, `tau_pct`, `current_loan`, `target_loan`, `new_pd_pct`, `reason_1..3` — **formatted strings**.
- Optional LLM (Groq/Gemini free tier or local Qwen GGUF); deterministic template fallback when offline / no key.
- `verify_numbers(text, allowed) -> bool`: regex `\d+(?:,\d{3})*(?:\.\d+)?` → any token not in `allowed` ⇒ raise `HallucinationBlocked` ⇒ emit deterministic template, log event. **0% numeric hallucination by construction.**

**Done gate (the money test):** unit test feeds a mock LLM letter containing fake `$9,999` → guard rejects → fallback template returned. Also test that immutable features never appear in DiCE output CFs.

## 8. Milestone 5 — PSI drift + Notebook 02 (Days 15–16, ~4 h)

> **Reality check (2026-11-09):** `evidently==0.7.23` is unimportable under pandas 3.x (its legacy layer imports `pandas.core.dtypes.common.infer_dtype_from_object`, removed in pandas 3); no newer evidently release fixes this and downgrading pandas was rejected. **Evidently is dropped from the dependency set.** `src/prismrisk/drift.py` now implements the whole milestone in-house with zero new deps: train-locked 10-quantile bin edges frozen on train weeks (W0–59), vectorized per-week PSI, green/warning/retrain bands, `reports/psi_weekly.csv` + `reports/drift.html` (inline SVG line chart + banded table, inline CSS, no assets). PD is computed exactly as at train time when `--bundle models/real_champion.joblib` is passed (bundle's `TableEncoder` maps + isotonic calibration); otherwise a pre-existing `--pd-col` is used. Covers all post-train weeks W60–91 (CLI/tests), not just W80–91. Verified on the synthetic fixture (`tests/test_drift.py`, 7 tests — identical ⇒ PSI≈0; N(0,1)→N(2,1) ⇒ retrain band); on the real data it must be re-run on the user machine (champion bundle + full `features.parquet` are not in the repo). **Notebook 02 (§8.2) executed 2026-11-09 on the synthetic fixture (0 errors).**

### 8.1 `src/prismrisk/drift.py`
- Freeze 10 quantile bin edges on **train (W0–59)** per top-10 SHAP features + predicted PD.
- For each test week W80–91 (implemented for W60+): `PSI = Σ (p_b − q_b)·ln(p_b/q_b)`; bands `<0.10` green, `0.10–0.25` warning, `>0.25` retrain. Emit `reports/psi_weekly.csv` + `reports/drift.html` (in-house renderer, replaces `evidently_drift.html`).

> **Reality check (2026-11-09):** Notebook 02 (§8.2) is **executed** (synthetic fixture, 17 cells, "Restart & Run All" clean). Deviations vs the plan, in the open: overall best val ROC-AUC on the fixture is `logreg_baseline` (0.8930), but SHAP/DiCE need a tree so all XAI/recourse sections run on the tree reference champion `lightgbm_monotone` (0.8669) — both are printed, never fudged. Shaky-distribution honesty: the conformal 3-way (AUTO_APPROVE / review / AUTO_DENY) policy **loses** $1.77M to the naive binary cutoff on this synthetic distribution ($25.29M vs $27.06M) — kept in the notebook as-is. DiCE got two correctness fixes en route: (1) DiCE is now flipped against the *isotonic-calibrated* ratio-aware wrapper (decision boundary 0.5 calibrated PD — exactly what the served conformal layer compares against), so "opposite class" means "approved", not "raw score below 0.5"; (2) `recourse()` now rejects any returned counterfactual that does not actually flip the wrapper-predicted class (`feasible=False` + honest note otherwise). `tests/test_explain_recourse.py` = 49/49 green. `reports/figures/*` PNGs are committed; `reports/` HTMLs remain gitignored. Evidently stays dropped (unimportable under pandas 3.x).

### 8.2 Notebook `02_benchmark_calibration_xai.ipynb`
One clean results notebook (imports from `src/prismrisk/`), exporting PNGs to `reports/figures/`:
1. 4-model table (ROC-AUC, PR-AUC, mean Gini, latency)
2. Weekly Gini across W70–91
3. Isotonic reliability curve before/after
4. Conformal triage: profit lift vs binary cutoff
5. SHAP waterfall (one applicant) + DiCE recourse table + verified notice sample
6. Weekly PSI chart

**Done gate:** both notebooks "Restart & Run All" clean; figures referenced by README. Notebook 01 (EDA) still pending.

## 9. Milestone 6 — FastAPI + Streamlit + Docker + Locust + CI (Days 17–20, ~7 h) · Upgrade #5

> **Reality check (2026-11-09):** implemented; deviations vs this section:
> - No champion bundle / triage artifact exists in the dev sandbox, so the API boots in an explicit
>   `synthetic_demo_fixture` mode (tiny LightGBM trained at startup on the committed fixture, isotonic
>   on calib weeks, MAPIE conformalize) — surfaced as `model_source` everywhere; never presented as the champion.
> - `/predict` returns full SHAP reasons as designed; measured p50/p95 at 10 concurrent users (dev box):
>   **30 ms / 56 ms** (≈148 RPS); at 100 users on one worker the worker saturates (p50 500 ms / p95 640 ms /
>   ~168 RPS) — plan's "<15 ms" target was not held once SHAP-on-the-hot-path was added; single inference
>   ≈15 ms. Numbers are indicative only; HF Space / real bundle will differ.
> - `/predict` requires the **full** trained feature vector; missing features → 422 (never silently imputed).
>   Strings only allowed for trained string columns; everything else coerced via float().
>   Nulls become NaN (LightGBM-native), reported as such.
> - DiCE cannot ingest missing values — rows with NaN (e.g. `risk_band_pct`, ~3.5% coverage) made recourse
>   fail silently; fixed in `explain_recourse.recourse()`: query+background are median-imputed **for the DiCE
>   search only** and the filled columns are returned as `imputed_features` (never hidden).
> - Approved applicants get HTTP 200 from `/recourse` with `feasible: null` and an approval letter
>   (adverse-action letters for approvals would be nonsense) — counterfactual search is skipped.
> - Adverse-action notice for approvals replaced by a machine-verified approval letter;
>   `applicant_ref` digits are now grounded via `allowed_numbers()` (case_id letters no longer false-fallback).
> - Docker: Dockerfile (multi-stage, python:**3.12-slim** per plan, non-root, uvicorn 1 worker) + compose
>   (api + dashboard + mlflow) written; **no container runtime in this sandbox — `docker compose up --build`
>   not executed here** (user-side verify). Locust file shipped and measured (above).
> - CI: lint scope widened to `api dashboard conftest.py`; CI still ruff + pytest only (docker build not added).

### 9.1 `api/main.py` (+ `schemas.py`, Pydantic v2)
- Startup: `joblib.load("models/champion_bundle.joblib")` once. **No MLflow on the hot path.**
- `GET /health` → `{status, model:"PrismRiskChampion", registry_version, trained_weeks:"0-59"}`
- `POST /predict` → `{calibrated_pd, decision: AUTO_APPROVE|GREY_ZONE_REVIEW|AUTO_DENY, expected_profit_usd, top_3_shap_reasons[]}` target **<15 ms** (measure, don't assert)
- `POST /recourse` → DiCE + `notice_guard` on demand → `{recourse_actions[], applicant_notice, grounding_passed:true}`
- `GET /drift` → latest `reports/psi_weekly.csv` summary

### 9.2 `tests/test_pipeline.py` (CI-safe, uses committed `sample_500.parquet`)
1. `test_join_preserves_unique_case_id`
2. `test_chronological_split_no_overlap`
3. `test_monotone_constraint_respected` — trains a tiny constrained LGBM on synthetic data (no artifact needed)
4. `test_conformal_outputs_three_classes` — on 200-row fixture with the fitted triage artifact **or** skip-if-missing
5. `test_notice_guard_blocks_hallucinated_number`
6. `test_api_predict_and_recourse_200` — `fastapi.testclient.TestClient`

### 9.3 `tests/locustfile.py` + Makefile
```
features / train / triage / drift / test / up / loadtest
loadtest: uv run locust -f tests/locustfile.py --headless -u 100 -r 20 -t 60s --host http://localhost:8000
```
Record **real** p50/p95/p99 + RPS on your machine into README (indicative only; HF Space numbers will differ).

### 9.4 Docker + compose
Multi-stage `python:3.12-slim`, non-root, `uvicorn` 1 worker; compose = api + mlflow + dashboard. `.github/workflows/ci.yml`: ruff then pytest on the 500-row sample.

**Done gate:** `docker compose up --build` serves all three; CI green on GitHub; Locust table pasted in README.

## 10. Milestone 7 — Live demo + launch (Day 21, ~2 h)

1. HF Space (Docker SDK, free): `dashboard/app.py` + `src/prismrisk/` + `models/champion_bundle.joblib` + `data/processed/sample_500.parquet`. Recruiter clicks → try real W80–91 borrowers, SHAP reasons, grey-zone badge, recourse letter, PSI chart. (Needs your HF token + account — your action.)
2. README order: **Live Demo link → architecture diagram → 4-model stability table → profit-lift (grey-zone vs cutoff) → Locust p50/p95/p99 → how to run locally (3 commands)**.
3. Archive the old `Vehicle-Insurance-Claim-Prediction`; pin PrismRisk next to VehicleGPT / ScholarGraph / AI-collusion-sprint-Apart.
4. Resume bullets — use honest phrasing:
   > Engineered a DuckDB/Polars relational pipeline across 1.5M+ loan applications (92 weeks), achieving out-of-time weekly Gini stability of **<report actual>** under the COVID-era W60+ distribution shift; monotone-constrained, isotonic-calibrated LightGBM wrapped in MAPIE conformal triage (AUTO_APPROVE / GREY_ZONE / AUTO_DENY, ~90% coverage, measured <actual>%); DiCE feasibility-constrained recourse with immutable features + zero-hallucination numeric-verified adverse-action notices; served via Dockerized FastAPI (measured p95 <actual> ms, Locust, 100 users) with weekly PSI drift alerts.

---

## 11. 21-day schedule (~25–30 h)

| Days | Milestone | Hours |
|---|---|---|
| 1 | M0 bootstrap + data convert | 1.5 |
| 2–4 | M1 features + split + Notebook 01 | 5 |
| 5–8 | M2 metrics + 4 models + calibration + MLflow | 6 |
| 9–10 | M3 conformal triage | 3 |
| 11–14 | M4 SHAP + DiCE + notice guard | 7 |
| 15–16 | M5 drift + Notebook 02 | 4 |
| 17–20 | M6 API + Streamlit + Docker + Locust + CI | 7 |
| 21 | M7 HF Space + README + pins | 2 |

## 12. Known breakage points (pre-mortem)

| Risk | Fix |
|---|---|
| MAPIE 1.x API differs from old tutorials (`MapieClassifier`) | Use `SplitConformalClassifier` exactly as in §6.1 — verified against mapie 1.6.0 |
| sklearn `CalibratedClassifierCV(cv="prefit")` removed | Use `IsotonicRegression` directly (§5.3) or `FrozenEstimator` |
| LightGBM `monotone_constraints` length mismatch | Build vector from `feature_names` order in config; unit-test it |
| DiCE genetic CF too slow / OOM | Cap `total_CFs=4`, subsample train sample to 20k for DiCE's data interface, demand-side only |
| Kaggle CSVs too big for RAM | `scan_csv(...).sink_parquet(...)` streaming; DuckDB joins on disk; sample dev iterates on `sample_500` |
| Column names differ from plan (`*P`, `*D`, suffixes `_1.._9`) | Inspect schema after download; rename in `config.yaml`, not in code |
| HF Space >1 GB / slow cold start | Ship `sample_500.parquet` + `champion_bundle.joblib` only; Streamlit loads lazily |
| GitHub 100 MB file limit | LightGBM bundle is small; else git-lfs or commit via Space repo only |
| CI leaks absolute paths | `pytest` runs from repo root; paths via `importlib.resources`/relative-to-`__file__` |
| Coverage off-target ≠ failure | Report measured coverage; tune alpha or conformalize set; never hide the number |

## 13. Scope cuts if time runs out (in order)
Drop MLP (keep LogReg/XGB/LGBM) → drop Evidently (keep PSI CSV) → single-model README (LGBM only) → Streamlit before Locust. **Never cut:** monotone constraint, calibration, conformal triage, DiCE recourse, notice guard, leakage tests — those are the 2026 differentiators.

## 14. Verified reading list (all confirmed live 2026-11-06)
- Data/stability reference: `zulqarnainalipk/Home-Credit---Credit-Risk-Model-Stability` ⭐7 · `varuni-rao/home-credit-risk-stability-silver` (top-4% silver, LightGBM+H2O) · `a3nath/Credit-Risk-Model-Stability` ⭐3
- Feature engineering at scale: `minerva-ml/open-solution-home-credit` ⭐469 · `mratsim/home-credit-default-risk` ⭐81 (SQL/DuckDB-style) · `KazukiOnodera/Home-Credit-Default-Risk` ⭐229
- Stability-formula + architecture reference: `Bra-AM/erdos-homecreditmodelstability` (fork; metrics.py formula mirror) · `AbhayJuloori/credit-recourse-engine` (FastAPI+MAPIE+DiCE wiring, 0★) · `AmarjeetJha17/mlops-credit-risk-system` (0★)
- Official docs/examples: `scikit-learn-contrib/MAPIE` ⭐1,670 (`examples/classification/1-quickstart/plot_quickstart_classification.py` — this exact API is what §6 uses) · `interpretml/DiCE` ⭐1,533 · `evidentlyai/evidently` ⭐8,047 · `drivendataorg/cookiecutter-data-science` ⭐10,132 · `scikit-learn-contrib/MAPIE` user guide
- MLOps serving: `JakobLS/mlops-credit-risk` ⭐39 · `vaibhav07772/credit-risk-mlops` ⭐3

## 15. First 5 actions (do these today)
1. `git status` clean; create `.gitattributes` + expand `.gitignore`
2. Install uv (or venv) and scaffold §2 dependencies; commit stub `pyproject.toml`
3. Download Kaggle competition CSVs → convert to Parquet (§3)
4. `configs/config.yaml` with weeks/profit/monotone/immutable (single source of truth)
5. Start `features.py`; run the EDA notebook; commit `sample_500.parquet`
