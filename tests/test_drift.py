"""M5 drift: PSI engine (train-locked bins) + artifacts — CI-safe, tiny fixture."""

import sys

import numpy as np
import polars as pl
import pytest
from lightgbm import LGBMClassifier

from prismrisk import drift as D
from prismrisk import features as F
from prismrisk.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from make_synthetic import generate  # noqa: E402


def test_psi_identical_near_zero() -> None:
    a = np.random.default_rng(0).normal(0, 1, 3000)
    assert D.psi(a, a) == pytest.approx(0.0, abs=1e-9)


def test_psi_shifted_distribution_high_and_band() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 2000), rng.normal(2.0, 1, 2000)
    value = D.psi(a, b)
    assert value > D.PSI_WARN
    assert D.band(value) == "retrain"


def test_psi_requires_minimum_train_values() -> None:
    with pytest.raises(ValueError, match="need >=50"):
        D.psi(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tables = generate(rows=250, seed=3)
    raw = tmp_path_factory.mktemp("raw")
    for name, frame in tables.items():
        frame.write_parquet(raw / name)
    out = raw / "features.parquet"
    F.build_features(raw, out)
    df = pl.read_parquet(out).with_columns(
        (1.0 / (1.0 + np.exp(-2.0 * pl.col("credamount_to_income").cast(pl.Float64)))).alias("predicted_pd")
    )
    return df


def test_weekly_psi_schema_bands_and_weeks(tiny) -> None:
    cols = ["credamount_770A", "annuity_to_income", "maininc_clean"]
    report = D.weekly_psi(tiny, feature_cols=cols, pd_col="predicted_pd", train_hi=59)
    assert set(report.columns) == {"week", "feature", "n_rows", "psi", "band"}
    assert set(report["band"].unique().to_list()) <= {"green", "warning", "retrain", "n/a"}
    assert report["week"].min() >= 60
    # one row per (week, feature) pair
    assert report.height == report.select(pl.struct(["week", "feature"]).n_unique().alias("u")).item()


def test_constant_column_reports_na_not_crash(tiny) -> None:
    df = tiny.with_columns(pl.lit(7.0).cast(pl.Float32).alias("const_feat"))
    report = D.weekly_psi(df, feature_cols=["const_feat", "credamount_770A"], train_hi=59)
    const_rows = report.filter(pl.col("feature") == "const_feat")
    assert const_rows.height > 0
    assert (const_rows["band"] == "n/a").all()
    ok = report.filter(pl.col("feature") == "credamount_770A")
    assert (ok["psi"].is_nan() | (ok["psi"] >= 0)).all()


def test_run_drift_writes_csv_and_html(tiny, tmp_path) -> None:
    pq = tmp_path / "f.parquet"
    tiny.write_parquet(pq)
    csv_path, html_path = tmp_path / "psi.csv", tmp_path / "drift.html"
    report = D.run_drift(pq, pd_col="predicted_pd", top_k=4, csv_path=csv_path, html_path=html_path)
    assert csv_path.exists() and html_path.exists()
    written = pl.read_csv(csv_path)
    assert written.height == report.height and "psi" in written.columns
    html = html_path.read_text()
    assert "PSI" in html and "<table>" in html


def test_select_top_features_returns_unique_names(tiny) -> None:
    num = [c for c, dt in tiny.schema.items()
           if dt.is_numeric() and c not in {"target", "case_id", "WEEK_NUM", "MONTH", "predicted_pd"}]
    pdf = tiny.select(num).to_pandas().fillna(0.0)
    model = LGBMClassifier(n_estimators=10, max_depth=2, random_state=1, verbose=-1).fit(pdf, tiny["target"])
    names = D.select_top_features(model, pdf.to_numpy(), list(pdf.columns), k=3)
    assert len(names) == 3 and len(set(names)) == 3
    assert all(n in set(pdf.columns) for n in names)
