# Deploy to Hugging Face Space (M7) — user-side

Everything in this repo is ready; the deploy itself needs your HF account and stays **your action**.
Time: ~10 minutes on the free CPU tier. Nothing here is committed for you — read, then do.

## 0. Read this first — what the Space may and may not serve

- **Public Space: synthetic demo only.** Ship `data/processed/sample_500.parquet` + a tiny demo
  bundle trained at startup (exactly what `docker-compose` does) and let
  `/health` report `model_source=synthetic_demo_fixture`. The demo proves the product
  (predict → calibrate → triage → SHAP → recourse → letter), not the champion score.
- **Never put the competition data or `models/real_champion.joblib` in a public Space.**
  The Kaggle data license forbids redistribution, and the trained artifact is a derived work of
  that data. If you want a full-power recruiter Space, make it **private** and mount the real
  bundle as a Space-side file (Space settings → Files, or git-lfs) — still keep test/raw data out.
- The **real-champion public demo in the README links here but does not exist until you deploy it.**
  Until then the README says "coming with M7" — leave it that way.

## 1. What goes in the Space

HF Spaces run a single container and expose **port 7860**. Two workable shapes:

**A. Single-process Space (simplest)** — one Dockerfile that starts both uvicorn (internal
`127.0.0.1:8000`) and Streamlit (`0.0.0.0:7860`), with a tiny `start.sh`:

```bash
#!/usr/bin/env bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
streamlit run dashboard/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
```

The dashboard already reads `PRISMRISK_API_URL` (default `http://localhost:8000`) — perfect for
same-container API. `.dockerignore` must exclude `data/raw/`, `mlruns/`, `reports/` (check yours).

**B. Dashboard-only Space** — Streamlit alone, calling a separately hosted API (HF Inference
Endpoint / a small VM). More moving parts; pick A unless you need it.

Space repo skeleton (SDK = **Docker**, not Streamlit-SDK — we need two processes):

```
space-repo/
├── Dockerfile            # adapt the repo root Dockerfile: non-root, expose 7860, start.sh
├── start.sh              # as above
├── pyproject.toml        # slimmed dependency set — see §3
├── src/prismrisk/  api/  dashboard/
├── data/processed/sample_500.parquet   (256 KB fixture — safe)
└── README.md             # HF Space card (front-matter: title, sdk: docker, app_port: 7860)
```

## 2. Deploy

```bash
# user-side, with ~/.huggingface/token or HF_TOKEN env var
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli repo create prismrisk-demo --type space --space_sdk docker
git clone https://huggingface.co/spaces/<you>/prismrisk-demo
# copy the slimmed files in (see §1), commit, push — the Space builds automatically
git add . && git commit -m "prismrisk demo space" && git push
```

Then open `https://huggingface.co/spaces/<you>/prismrisk-demo` and smoke-test:

- [ ] `/` loads the dashboard on port 7860 (or `GET <space-url>/health` if using api-only)
- [ ] `/health` → 200 with `model_source=synthetic_demo_fixture` (not presented as champion)
- [ ] `/predict` on a fixture row → calibrated PD + triage badge + latency field present
- [ ] `/recourse` → either flip-validated CF + grounded letter, or honest `feasible=False`
- [ ] no raw data, no `real_champion.joblib` in the Space repo (`git ls-files` check)

## 3. Image size — the `nvidia_nccl_cu12` problem (measured)

On this machine, `pip install -e ".[dev]"` produced a **2.3 GB** `.venv`. Of that:

- `nvidia/nccl` — **469 MB**, pulled in **unconditionally by `xgboost==3.2.0`** (`Requires: numpy, nvidia-nccl-cu12, scipy`). It exists only for multi-GPU collectives; irrelevant for CPU inference.
- `xgboost` itself — 228 MB.
- `[dev]` extras (locust, nbconvert, ruff…) — not needed in the image at all.

For a Space image, install only runtime deps and strip nccl:

```bash
pip install --no-deps xgboost==3.2.0   # skips nvidia-nccl-cu12 (nothing imports it on CPU)
pip install numpy scipy lightgbm==4.7.0 scikit-learn isotonic... # your runtime pins
```

or build via a `space` extra / `requirements-space.txt` with `xgboost --no-deps`. Verify after:
`python -c "import xgboost; print(xgboost.__version__)"` must import fine — it does (nccl is
loaded lazily only for GPU distributed features). Expect the image to drop well under 1 GB.

If anything in the Space build breaks on missing nccl, the fallback is simple: **serve the
LightGBM monotone model as the Space champion instead of XGB** (it was within 0.0005 AUC);
no xgboost import, no nccl at all.

## 4. HF Spaces limits to respect

- Free CPU tier is small (2 vCPU-class) — the synthetic demo's SHAP-on-`/predict` is fine
  (measured p50 ≈ 30 ms locally) but **cold start + concurrent users will be much slower**;
  show the latency-in-response feature, don't quote dev-box numbers as production SLAs.
- `.dockerignore` and Space repo hygiene matter more than raw size: exclude `data/raw/`,
  `mlruns/`, `models/real_champion.joblib`, `.venv/`, `reports/`.
- Check the Space's Build Logs the first time; SDK must be **docker**, `app_port: 7860` in the
  README front-matter.
- Rules: keep every `model_source` / `synthetic_demo_fixture` label exactly as in the app —
  the point of the demo is that it's honest by construction.

## 5. After deploy

1. Paste the Space URL into README ("Live Demo" section) and into `docs/EXECUTIVE_SUMMARY.md`.
2. Flip `- [ ] M7` → `- [x]` **only** when the Space is public and the checklist above passes.
3. Pin the Space next to your other public work (per the plan, archive the old vehicle-insurance repo).
4. If you later want the real champion on the Space: private Space + consented data policy reviewed
   against the Kaggle competition rules — do not skip that review.
