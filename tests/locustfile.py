"""Locust benchmark for the PrismRisk API — real p50/p95/p99, never estimated.

Run (API must be up on :8000):
    locust -f tests/locustfile.py --headless -u 100 -r 20 -t 60s --host http://localhost:8000
"""

import random

import polars as pl
from locust import HttpUser, between, task

_ROW_EXCLUDES = {"case_id", "WEEK_NUM", "target", "date_decision"}


def _load_rows() -> list[dict]:
    df = pl.read_parquet("data/processed/sample_500.parquet")
    return [
        {"case_id": str(r["case_id"]),
         "features": {k: v for k, v in r.items() if k not in _ROW_EXCLUDES}}
        for r in df.head(200).to_dicts()
    ]


try:
    _ROWS = _load_rows()
except Exception:  # locust may import from outside repo root — degrade to health-only load
    _ROWS = []


class PrismRiskUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task(10)
    def predict(self) -> None:
        if _ROWS:
            self.client.post("/predict", json=random.choice(_ROWS), name="/predict")

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")
