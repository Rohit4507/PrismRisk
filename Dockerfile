# PrismRisk — multi-stage, non-root, single uvicorn worker.
# Demo (no champion bundle) works out of the box via the committed synthetic fixture.
# Production: mount models/champion_bundle.joblib (+ models/conformal_triage.joblib)
# and set PRISMRISK_BUNDLE=/app/models/champion_bundle.joblib.

FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir=/wheels .

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd -r prismrisk && useradd -r -g prismrisk -d /app prismrisk
WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY api ./api
COPY dashboard ./dashboard
COPY data/processed/sample_500.parquet ./data/processed/sample_500.parquet
COPY configs ./configs
RUN mkdir -p /app/reports && chown -R prismrisk:prismrisk /app

USER prismrisk
EXPOSE 8000 8501
# default: API. Compose overrides command for dashboard/mlflow.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
