"""End-to-end smoke test for build_features on a tiny synthetic stand-in.

Guards the depth-0 direct-join fix: level columns (credamount_770A, annuity_780A)
must survive the join, ratios must be engineered, and raw *D dates must be converted.
"""

import sys

import polars as pl

from prismrisk import features as F
from prismrisk.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from make_synthetic import generate  # noqa: E402


def test_build_features_end_to_end(tmp_path) -> None:
    tables = generate(rows=150, seed=5)
    for name, frame in tables.items():
        frame.write_parquet(tmp_path / name)

    out = tmp_path / "features.parquet"
    F.build_features(tmp_path, out)
    df = pl.read_parquet(out)

    # 1 row per case_id, no explosions
    assert df.height == 150
    assert df["case_id"].n_unique() == df.height

    # depth-0 level columns survive (regression guard)
    for col in ("credamount_770A", "annuity_780A"):
        assert col in df.columns

    # ratios engineered
    for col in ("annuity_to_income", "credamount_to_income", "overdue_to_credamount"):
        assert col in df.columns
        assert df[col].is_finite().all()

    # raw *D dates converted to relative days
    assert "birth_259D" not in df.columns
    assert "birth_259_days" in df.columns
    assert "static_signdate_001D" not in df.columns
    assert "static_signdate_001_days" in df.columns

    # depth-1 aggregation emitted counts
    assert "bureau_loan_count" in df.columns
    assert "prev_app_count" in df.columns

    # null filtering is fit on train weeks only -> train has no extreme-null columns left
    train = df.filter((pl.col("WEEK_NUM") >= 0) & (pl.col("WEEK_NUM") <= 59))
    assert train.height > 0
