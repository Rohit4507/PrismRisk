"""riskassesment_940T bureau risk-band parser (the competition's strongest
single signal when present, ~3.5% coverage; Bra-AM / Erdos Institute bootcamp).
"""

import sys

import polars as pl

from prismrisk import features as F
from prismrisk.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from make_synthetic import generate  # noqa: E402


def _frame(values):
    return pl.DataFrame({"case_id": list(range(1, len(values) + 1)), "riskassesment_940T": values})


def test_parse_ranges_and_boundaries() -> None:
    df = F.engineer_risk_band(
        _frame(["6% - 8%", "0%-3%", ">95%", "<3%", "20% - 34%", "", None, "not specified"])
    )
    got = df["risk_band_pct"].to_list()
    assert got[0] == 7.0          # midpoint of range
    assert got[1] == 1.5
    assert got[2] == 95.0         # '>' tail -> stated boundary
    assert got[3] == 3.0          # '<' tail -> stated boundary
    assert got[4] == 27.0
    assert got[5] is None and got[6] is None and got[7] is None


def test_raw_text_column_dropped() -> None:
    df = F.engineer_risk_band(_frame(["6% - 8%"]))
    assert "riskassesment_940T" not in df.columns
    assert "risk_band_pct" in df.columns


def test_absent_column_is_noop() -> None:
    df = pl.DataFrame({"case_id": [1], "x": [1.0]})
    out = F.engineer_risk_band(df)
    assert "risk_band_pct" not in out.columns


def test_sparse_coverage_survives_null_filter(tmp_path) -> None:
    """risk_band_pct (~96% null) must NOT be dropped by the >95%-null filter."""
    tables = generate(rows=400, seed=5)
    band = tables["train_static_cb_0.parquet"]["riskassesment_940T"]
    assert band.dtype == pl.String  # emitted as text like the competition

    for name, frame in tables.items():
        frame.write_parquet(tmp_path / name)
    out = tmp_path / "features.parquet"
    F.build_features(tmp_path, out)
    df = pl.read_parquet(out)
    assert "risk_band_pct" in df.columns  # survived the high-null drop
    non_null = df["risk_band_pct"].drop_nulls().len()
    assert non_null < 0.15 * df.height    # genuinely sparse (competition-like)
