#!/usr/bin/env python3
"""Convert the Kaggle competition download into Parquet for `prismrisk.features`.

The competition ships the tables inside ``data/csv_files/{train,test}/`` as
plain files (CSVs in the original release; the same layout as
``train_<table>[_<shard>].<ext>``). Large tables are chunked
(``train_static_0_0.csv``, ``train_static_0_1.csv``, ...). This script

1. discovers ``.csv``/``.parquet`` files under ``--src`` (default
   ``data/csv_files/train`` — ``test/`` is for Kaggle submission and is not
   used by this repo),
2. concatenates the chunks per logical table,
3. downcasts ``Float64`` columns to ``Float32`` (~50% memory cut — the
   silver-medal approach) unless ``--no-downcast`` is given,
4. writes ``data/raw/train_<table>[_<shard>].parquet`` in exactly the
   filenames ``features.py`` expects, and ignores everything it does not
   recognise (e.g. ``train_credit_bureau_b_*``, ``sample_submission``) with an
   explicit note.

Usage:
    python scripts/prepare_raw.py                 # data/csv_files/train -> data/raw
    python scripts/prepare_raw.py --src X --out Y

Exits 1 and lists what is missing if a required table has no files.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

REQUIRED = [
    "train_base",
    "train_static_0",
    "train_static_cb_0",
    "train_person_1",
    "train_applprev_1",
    "train_credit_bureau_a_1",
]

# canonical stem -> regex matched on the file stem, lowercased
PATTERNS: dict[str, re.Pattern[str]] = {
    "train_base": re.compile(r"train_base(_\d+)?$"),
    "train_static_cb_0": re.compile(r"train_static_cb_0(_\d+)?$"),
    "train_static_0": re.compile(r"train_static_0_\d+$"),
    "train_person_1": re.compile(r"train_person_1(_\d+)?$"),
    "train_applprev_1": re.compile(r"train_applprev_1_\d+$"),
    "train_credit_bureau_a_1": re.compile(r"train_credit_bureau_a_1_\d+$"),
}
# sharded tables are renumbered 1..k on output (features.py globs `train_<t>_*.parquet`)
SHARDED = {"train_static_0", "train_applprev_1", "train_credit_bureau_a_1"}


def _read_any(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, infer_schema_length=10_000, try_parse_dates=False)


def _downcast_floats(df: pl.DataFrame) -> pl.DataFrame:
    f64 = [c for c, dt in df.schema.items() if dt == pl.Float64]
    return df.with_columns(pl.col(f64).cast(pl.Float32)) if f64 else df


def _natural_key(path: Path):
    m = re.search(r"_(\d+)$", path.stem.lower())
    return (int(m.group(1)) if m else -1, path.name)


def _write_table(table: str, files: list[Path], out_dir: Path, downcast: bool) -> tuple[int, int]:
    frames = [_read_any(f) for f in sorted(files, key=_natural_key)]
    if downcast:
        frames = [_downcast_floats(f) for f in frames]
    if table in SHARDED:  # keep shard boundaries: one output file per input file
        rows = 0
        for pos, frame in enumerate(frames, start=1):
            rows += frame.height
            frame.write_parquet(out_dir / f"{table}_{pos}.parquet", compression="zstd")
        return len(frames), rows
    df = pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]
    df.write_parquet(out_dir / f"{table}.parquet", compression="zstd")
    return 1, df.height


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default="data/csv_files/train", help="directory with the unpacked Kaggle download")
    parser.add_argument("--out", default="data/raw", help="output directory for Parquet files")
    parser.add_argument("--no-downcast", action="store_true", help="keep Float64 instead of downcasting to Float32")
    args = parser.parse_args(argv)

    src, out_dir = Path(args.src), Path(args.out)
    if not src.is_dir():
        print(f"src not found: {src} — run scripts/download_data.sh first (see data/README.md)", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[Path]] = {t: [] for t in PATTERNS}
    ignored: list[str] = []
    for path in src.rglob("*"):
        if path.suffix.lower() not in (".csv", ".parquet") or not path.is_file():
            continue
        stem = path.stem.lower()
        for table, pat in PATTERNS.items():
            if pat.fullmatch(stem):
                buckets[table].append(path)
                break
        else:
            ignored.append(path.name)

    missing = [t for t in REQUIRED if not buckets[t]]
    if missing:
        print("missing required tables under", src, file=sys.stderr)
        for t in missing:
            print(f"  - {t}  (expected e.g. {t}.csv or {t}_<shard>.csv)", file=sys.stderr)
        return 1

    print(f"reading {src} -> {out_dir}")
    for table, files in buckets.items():
        if not files:
            continue
        n_out, rows = _write_table(table, files, out_dir, not args.no_downcast)
        print(f"  {table:28s} {len(files):3d} input file(s) -> {n_out} parquet, {rows:,} rows")
    if ignored:
        print(f"ignored (not used by the pipeline): {', '.join(sorted(set(ignored)))}")
    print("Done. Next: python -m prismrisk.features  ->  data/processed/features.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
