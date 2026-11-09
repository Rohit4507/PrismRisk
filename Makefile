.PHONY: install data features split test lint up loadtest smoke
PYTHONPATH := src

install:
	pip install -e ".[dev]"

data:                ## Download Kaggle comp data + build data/raw/*.parquet (see data/README.md)
	bash scripts/download_data.sh && PYTHONPATH=$(PYTHONPATH) python scripts/prepare_raw.py

features:            ## Build features.parquet from data/raw/*.parquet (needs data; see data/README.md)
	PYTHONPATH=$(PYTHONPATH) python -m prismrisk.features

split:               ## Sanity-check week splits
	PYTHONPATH=$(PYTHONPATH) python -m prismrisk.split

test:
	pytest -q

lint:
	ruff check src tests scripts

smoke:               ## Synthetic end-to-end smoke run (no Kaggle data needed)
	PYTHONPATH=$(PYTHONPATH) python scripts/make_synthetic.py --rows 2000

up:                  ## docker compose up --build (M6)
	docker compose up --build

api:                 ## API on :8000 (demo fixture fallback if no champion bundle)
	uvicorn api.main:app --host 127.0.0.1 --port 8000

dashboard:           ## Streamlit on :8501
	streamlit run dashboard/app.py

loadtest:            ## Locust headless benchmark (needs API running on :8000)
	locust -f tests/locustfile.py --headless -u 100 -r 20 -t 60s --host http://localhost:8000
