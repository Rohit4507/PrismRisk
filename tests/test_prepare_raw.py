"""prepare_raw.py: Kaggle-download -> Parquet adapter, tested end-to-end with
tiny synthetic CSVs (no real data, no download needed).
"""

import sys

import polars as pl

from prismrisk import features as F
from prismrisk.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import prepare_raw  # noqa: E402
from make_synthetic import generate  # noqa: E402


def _write_csv_tree(src, tables):
    src.mkdir(parents=True)
    for name, frame in tables.items():
        frame.write_csv(src / name.replace(".parquet", ".csv"))


def test_prepare_raw_roundtrip_into_features(tmp_path) -> None:
    tables = generate(rows=120, seed=11)
    total_rows = tables["train_base.parquet"].height

    # mimic Kaggle chunking: split train_static_0_1 into two shards
    static0 = tables.pop("train_static_0_1.parquet")
    half = static0.height // 2
    tables["train_static_0_1.parquet"] = static0.head(half)
    tables["train_static_0_2.parquet"] = static0.tail(static0.height - half)

    src = tmp_path / "csv_files" / "train"
    _write_csv_tree(src, tables)
    # must be ignored, not crash, not be converted
    pl.DataFrame({"case_id": [1], "prediction": [0.1]}).write_csv(src / "sample_submission.csv")

    out = tmp_path / "raw"
    assert prepare_raw.main(["--src", str(src), "--out", str(out)]) == 0

    # canonical filenames features.py expects
    assert (out / "train_base.parquet").exists()
    assert (out / "train_static_cb_0.parquet").exists()
    assert (out / "train_person_1.parquet").exists()
    shards = sorted(p.name for p in out.glob("train_static_0_*.parquet"))
    assert shards == ["train_static_0_1.parquet", "train_static_0_2.parquet"]
    assert sum(pl.read_parquet(p).height for p in out.glob("train_static_0_*.parquet")) == static0.height

    # CSV parsing infers Float64; prepare_raw must downcast every float to Float32
    assert not any(
        dt == pl.Float64
        for f in out.glob("*.parquet")
        for dt in pl.read_parquet(f).schema.values()
    )

    # full end-to-end: prepared parquet feeds features.build_features unchanged
    features_path = tmp_path / "features.parquet"
    F.build_features(out, features_path)
    fdf = pl.read_parquet(features_path)
    assert fdf.height == total_rows


def test_downcast_floats_unit() -> None:
    df = pl.DataFrame({"a": [1.5, 2.5], "b": [1, 2], "s": ["x", "y"]})
    out = prepare_raw._downcast_floats(df)
    assert out.schema["a"] == pl.Float32
    assert out.schema["b"] == pl.Int64
    assert out.schema["s"] == pl.String


def test_prepare_raw_missing_required_table(tmp_path, capsys) -> None:
    src = tmp_path / "only_base"
    src.mkdir()
    (src / "train_base.csv").write_text("case_id,WEEK_NUM,target\n1,0,0\n")
    rc = prepare_raw.main(["--src", str(src), "--out", str(tmp_path / "raw")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "train_static_0" in err and "train_person_1" in err


def test_prepare_raw_src_not_found(tmp_path, capsys) -> None:
    rc = prepare_raw.main(["--src", str(tmp_path / "nope"), "--out", str(tmp_path / "raw")])
    assert rc == 1
    assert "download_data.sh" in capsys.readouterr().err
