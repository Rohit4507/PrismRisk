# PrismRisk — 2026 Interview Playbook
*Ranked, honest, rehearsed. BTech 4th-year positioning: no medal, no fake numbers —
engineer who builds the system medal repos never ship.*

## Q: Did repos with this dataset download it? Which is best for me in 2026 interviews?
**Do they download?** Yes — everyone downloads, **nobody commits**. Kaggle license forbids
redistribution; even the silver-medal solution only ships code + instructions.
Standard flow: Kaggle login → accept rules (one-time) → API token →
`kaggle competitions download -c home-credit-credit-risk-model-stability` →
~4 GB CSV chunks (`data/csv_files/{train,test}/`) → aggregate depth-1/2 tables to one
row per `case_id` → parquet + float32 downcast → **time-based `WEEK_NUM` validation** →
GBDT (+ class weights, no SMOTE) → submission.

**Best for me in 2026 interviews — ranked:**

| # | Repo | Why it matters to you | Learning load |
|---|---|---|---|
| 1 | **your PrismRisk** | The flagship "production system" story; 70% of your interview | rehearsed |
| 2 | **Bra-AM/erdos-homecreditmodelstability** | Proof you can verify an undocumented metric against reference code; honest team science | skim metrics.py |
| 3 | **varuni-rao/home-credit-risk-stability-silver** | Silver 147/3856 — what winning actually required (LightAutoML CV5, time-based CV, downcast, no SMOTE) | skim README |
| 4 | minerva-ml/open-solution-home-credit (469★) | 2018 incumbent: feature richness > model choice; leakage/artifacts discipline | skim |
| ✗ | zulqarnainalipk (⭐7), a3nath (⭐3) | Student-clone / empty scaffold — category claim you don't want | skip |

**Do NOT claim** medals, leaderboard positions, or real AUC without a real Kaggle run.
Synthetic numbers stay labeled synthetic — 2026 interviewers punish polish-without-measurement.

## Q: How do I get the data? (answer script)
"The dataset is Kaggle-licensed — accepting the rules is part of the download, so I don't
commit a single row. Repo has `scripts/download_data.sh` (Kaggle CLI + token) and
`scripts/prepare_raw.py` (chunks → parquet, float32 downcast, shard concat → exactly the
filenames my feature builder globs). CI and tests use a 500-row synthetic fixture, not real
rows. One `make data` on my machine and the pipeline runs end-to-end."
(If asked why not UCI instead: "UCI card-default has no WEEK_NUM → no stability story; it's
the fallback for real-number demos, not the competition task.")

## Q: How did you handle the dataset part? (table to recite)

| Concern | What everyone does | What PrismRisk does |
|---|---|---|
| Storage | gitignore + script | ✅ `make data` + `data/README.md`; `.gitignore` guards |
| Format | CSV chunks ~4 GB | ✅ → parquet + **float64→float32** (silver-medal practice) |
| Merge | aggregate depth-1/2 → `case_id` | ✅ depth-0 direct joins, depth-1 aggregated (fixed a real bug here) |
| Validation | time-aware by `WEEK_NUM` | ✅ W0–59 / 60–69 / 70–79 / 80–91 locked |
| Imbalance ~3% | class weights, **no SMOTE** | ✅ class-weighted LGBM |
| Metric | weekly-Gini stability | ✅ competition formula, verified vs 2 implementations |
| Tests | synthetic/small fixtures | ✅ 31 tests + committed 500-row synthetic sample |
| Key feature | bureau risk band (text, 3.5% coverage) | ✅ parsed → `risk_band_pct`, monotone, kept from null-drop |

## Q: Which model won the competition? What did you learn?
"Silver was LightAutoML CV5 ensemble — hand-tuned LightGBM and H2O lost. Erdos bootcamp's
NN memorized WEEK_NUM/date inputs (test 0.59 vs GBDT 0.78). Lessons I adopted: GBDT +
time-based CV + rich features; never feed week/date as features; class weights over SMOTE;
and audit your metric against reference code before trusting it. I rejected AutoML itself
deliberately — banking governance needs monotone-constrained, explainable models."

## Q: What's your best result — real?
"Real run on the Kaggle competition data (locked W80–91 test, 5 Optuna trials,
2026-11-09): **ROC-AUC 0.8540** (val 0.8541 → no generalization drop through the
COVID window), PR-AUC 0.147, ECE 0.017, weekly-Gini stability 0.684 on 12 test
weeks with a **positive** slope (+0.004/week — no decay). Champion was
xgboost_unconstrained; monotone LightGBM was within 0.0005 (governance tie — I report
it as a measured tie, not as a win for governance). Synthetic-fixture numbers
(conformal coverage 0.90) and the real-data run are labeled separately in the
README; no leaderboard claim, no submission."

## Trap questions + honest answers
- **"Why is your binary cutoff beating conformal?"** — "It did, $139.0M vs $122.2M on this
  distribution; grey-zone pays when the model is informative or false denial is costly.
  Reported, not hidden."
- **"Why no SMOTE?"** — "It fabricates minority geometry; distorts prevalence; nobody in the
  competition used it. Class weights + isotonic calibration instead."
- **"Is your sample fixture real?"** — "No — 500 synthetic rows so CI/demo never touch real
  data. Real evaluation needs the Kaggle download."
- **"What would you fix first at scale?"** — "Conformal triage (M3) on real data, then HF Space cold-start tuning and worker scaling. Measured so far (demo artefacts, dev box): 10 concurrent users p50 30 ms / p95 56 ms (~148 RPS); at 100 users a single uvicorn worker saturates (p50 500 ms / p95 640 ms / ~168 RPS, 0 errors) — so the next step is more workers + SHAP kept off the critical path if p95 must hold under load, not a claim that it is fine as-is."

## 60-second pitch (memorize)
"PrismRisk answers 'your model is decaying in production'. Home Credit's stability
competition measured that directly — weekly Gini with a slope penalty. I re-implemented that
metric verified against two independent sources, put it behind time-locked splits, a
monotone-constrained LightGBM with isotonic calibration, MAPIE conformal triage with measured
90% coverage, DiCE recourse, zero-hallucination adverse-action letters, PSI drift monitoring
and a Dockerized API — all on the real competition data path, with every number labeled.
The medal solutions ship a score; I ship the system."

## Your to-do before the interview loop
1. Pick the data path: (A) UCI card-default for real-but-different-task numbers ·
   (B) Kaggle `make data` for the true competition run · (C) stay synthetic, labeled.
2. M7: HF Space live demo — deploy, then put the link at the top of the README.
3. Rewrite resume bullets with real numbers **after** the real run — never before.
4. Attribution: if UCI — credit the dataset; if Kaggle — cite competition + rules.
5. Rehearse §"60-second pitch" until it's under 60 seconds.
