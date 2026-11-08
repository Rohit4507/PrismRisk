"""PrismRisk — temporal credit decision & MLOps system.

Pipeline: raw Kaggle CSVs -> Parquet -> features.py -> split.py -> train.py
(monotone LightGBM + isotonic calibration) -> triage.py (conformal) ->
explain_recourse.py (SHAP + DiCE) -> notice_guard.py -> drift.py -> FastAPI.
"""

__version__ = "0.1.0"
