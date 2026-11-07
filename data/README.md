# Data — never committed (competition rules)

The data belongs to Kaggle's
[Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability).
Downloading it requires accepting the competition rules, and the rules do **not**
allow redistribution — so no rows (not even samples) may be committed to git.
`data/csv_files/`, `data/raw/` and `data/raw_synth/` are gitignored. Tests and CI
run on the synthetic fixture (`scripts/make_synthetic.py`) and need no download.

Every public repo for this competition follows the same pattern — nobody commits
the data; they gitignore it and ship a download script + these instructions
(e.g. the Erdos bootcamp team's `scripts/download_data.sh` pattern).

## Layout (what the zip unpacks to)

`data/csv_files/train/` (~1.5M applications decided Jan 2019 – Oct 2020, ~3% default):

| Table | Depth | Content |
|---|---|---|
| `train_base` | 0 | `case_id`, decision date, `WEEK_NUM`, `target` (1 = default) |
| `train_static_0_<n>` | 0 | Internal application data: amounts, instalments, DPD |
| `train_static_cb_0` | 0 | External credit-bureau summary (incl. risk band as text, e.g. `"6% - 8%"`) |
| `train_person_1` | 1 | People attached to the application (`num_group1 == 0` → applicant) |
| `train_applprev_1_<n>` | 1 | This client's previous applications |
| `train_credit_bureau_a_1_<n>` | 1–2 | Credit-bureau contracts & payment history |

Depth-1/2 tables ship as chunks (`..._0`, `..._1`, …) and have many rows per
application; `prismrisk.features` aggregates them to one row per `case_id`
(max / count / "any late payment" flags) and joins depth-0 tables directly.
`data/csv_files/test/` exists only for Kaggle submission (hidden targets) —
PrismRisk evaluates out-of-time on weeks 80–91 of the **train** split, so the
test folder is unused.

## Download (on your machine — ~4 GB)

1. Kaggle account → open the competition → **Join competition / Accept rules** (one-time, unavoidable).
2. Kaggle → Settings → API → **Create New Token** → save `kaggle.json` to `~/.kaggle/`
   (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
3. From the repo root:

   ```bash
   pip install kaggle            # one-off; deliberately NOT a project dependency
   bash scripts/download_data.sh # rules-click must already be done, else this 403s
   python scripts/prepare_raw.py # CSV chunks -> data/raw/*.parquet (float32 downcast)
   python -m prismrisk.features  # -> data/processed/features.parquet
   ```

   `make data` does steps 3+4 equivalents (download + prepare) in one go on machines with `make`.

### Windows without bash / make (PowerShell)

```powershell
pip install kaggle
kaggle competitions download -c home-credit-credit-risk-model-stability -p data
python -m zipfile -e data\home-credit-credit-risk-model-stability.zip data\
Remove-Item data\home-credit-credit-risk-model-stability.zip
python scripts/prepare_raw.py
python -m prismrisk.features
```

### Manual browser download (slow internet / CLI trouble)

Competition page → **Data** tab → **Download All** (needs the same accepted rules) →
move the zip into `data/` → extract with any archiver → you should get `data/csv_files/train/`
and `data/csv_files/test/` → `python scripts/prepare_raw.py`.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `403` / `401` on download | Rules not accepted / bad token | Accept rules on the competition page; re-create `kaggle.json`; on Linux/macOS `chmod 600 ~/.kaggle/kaggle.json` |
| `404` | Wrong competition slug | Use exactly `home-credit-credit-risk-model-stability` |
| Asked for phone verification | Kaggle anti-abuse on some comps | Verify once in Kaggle settings — unavoidable |
| `make: command not found` | Windows default | Use the PowerShell block above |
| Kaggle CLI "not found" after pip install | Scripts dir not on PATH | `python -m kaggle competitions download ...` or re-open terminal |
| Out of memory in features/train | 1.5M rows × wide tables | Close other apps; 16 GB RAM recommended; use `python scripts/make_synthetic.py --rows 50000 --out data/raw_synth` to warm up first |
| Slow download (India) | ~4 GB over residential bandwidth | Use the browser download overnight; resume with the Kaggle CLI if interrupted |

## If you don't want to touch Kaggle

- **A. UCI "Default of Credit Card Clients"** (30k rows, license reuse with
  attribution; mirrors on GitHub) — real but different schema: needs a small
  column adapter (~1 h), and it has no `WEEK_NUM`, so the stability metric and
  time-locked splits don't apply (binary default task only). Expected
  ballpark from public results: ROC-AUC ≈ 0.77–0.78.
- **B. Synthetic fixture** (`python scripts/make_synthetic.py --rows 300000 --out data/raw_synth`)
  — full pipeline, tests, CI and the HF Space demo run offline. Numbers are
  demo-grade, never presented as real.

## What not to do

- Do not `git add data/` — the Kaggle license forbids redistribution.
- Do not commit `features.parquet`; `sample_500.parquet` is the only committed
  exception — 500 **synthetic** rows used by tests/HF Space demos.
- Do not claim leaderboard or "real" metrics from synthetic runs.
