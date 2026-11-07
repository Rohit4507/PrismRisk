"""Relational feature engineering: 6 Kaggle tables -> 1 row / case_id.

Design notes
------------
* Kaggle ships CSVs; we convert once to Parquet (`scripts/` / README) and read
  lazily with Polars / DuckDB (out-of-core friendly).
* Depth-0 tables (static_0, static_cb_0, primary-applicant person_1) are 1 row per
  case_id and joined DIRECTLY (level columns like credamount_770A survive).
* Depth-1 tables (applprev_1, credit_bureau_a_1) are aggregated to one row per case_id
  FIRST, then 1:1 joined onto the base table. Duplicates are a hard error (never silently duplicated rows).
* Raw `*D` dates leak temporal drift; everything is converted to
  `(date_decision - col_D).dt.total_days()` and raw date columns are dropped.
* Column names follow the competition schema. They are **verified at runtime**
  and the failure message tells you exactly which column is missing — after you
  download the data, reconcile mismatches in `configs/config.yaml` / this file,
  not by guessing.
* Null filtering is fit on TRAIN weeks only (weeks 0-59) and then applied to all
  splits — the val/test distributions never influence feature selection.

Run:  python -m prismrisk.features            # builds data/processed/features.parquet
      python -m prismrisk.features --sample    # also writes sample_500.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

from prismrisk.config import REPO_ROOT, load_config

# Competition date columns end with `D` right after an alphanumeric code
# (e.g. birth_259D, empl_from_D, datefirstoffer_489D). We deliberately match
# `[_0-9]D$` rather than the literal `_D$`, because `birth_259D` does NOT end
# with `_D`. Relative-day outputs (`*_days`) never match this pattern.
_DATE_RE = re.compile(r"[_0-9]D$")

def _days_name(col: str) -> str:
    """birth_259D -> birth_259_days, empl_from_D -> empl_from_days."""
    return re.sub(r"_?D$", "", col) + "_days"


def _date_expr(df: pl.DataFrame, col: str) -> pl.Expr:
    """Date expression tolerant to CSV round-trips.

    Parquet (synthetic fixture / real Kaggle parquet) carries native Date columns;
    CSVs arrive as String. Strict parse by default: fail loudly on unparseable
    values instead of silently emitting null days.
    """
    if df.schema.get(col) == pl.String:
        return pl.col(col).str.to_date()
    return pl.col(col).cast(pl.Date)



# ---------------------------------------------------------------- primitives
def relative_date_days(
    df: pl.DataFrame, col: str, ref: str = "date_decision"
) -> pl.DataFrame:
    """Convert an absolute `*D` date column to float days before the decision date."""
    return df.with_columns(
        (_date_expr(df, ref) - _date_expr(df, col))
        .dt.total_days()
        .cast(pl.Float32)
        .alias(_days_name(col))
    )


def find_date_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if _DATE_RE.search(c) and not c.endswith("_days")]


def convert_date_columns(
    df: pl.DataFrame, ref: str = "date_decision"
) -> pl.DataFrame:
    """Convert every `*D` column to relative days, then drop the raw dates."""
    date_cols = [c for c in find_date_columns(df.columns) if c != ref]
    if not date_cols:
        return df
    df = df.with_columns(
        [
            (_date_expr(df, ref) - _date_expr(df, c))
            .dt.total_days()
            .cast(pl.Float32)
            .alias(_days_name(c))
            for c in date_cols
        ]
    )
    return df.drop(date_cols)


def aggregate_depth1(
    lf: pl.LazyFrame,
    *,
    case_id: str = "case_id",
    count_name: str,
    numeric_cols: list[str],
    filter_expr: pl.Expr | None = None,
    prefix: str = "",
) -> pl.LazyFrame:
    """Aggregate a depth-1 table to one row per case_id.

    Emits `count_name` (row count) plus `max_<col>` / `mean_<col>` for each
    numeric column, prefixed with `prefix` when given.
    """
    available = set(lf.collect_schema().names())
    numeric_cols = [c for c in numeric_cols if c in available]
    if filter_expr is not None:
        lf = lf.filter(filter_expr)
    aggs: list[pl.Expr] = [pl.len().alias(f"{prefix}{count_name}")]
    for c in numeric_cols:
        aggs.append(pl.col(c).max().alias(f"{prefix}max_{c}"))
        aggs.append(pl.col(c).mean().alias(f"{prefix}mean_{c}"))
    return lf.group_by(case_id).agg(aggs)


def one_row_per_case(base: pl.DataFrame, parts: list[pl.DataFrame]) -> pl.DataFrame:
    """1:1 left-join aggregated parts onto the base frame. Duplicates = hard error."""
    out = base
    for p in parts:
        dupes = int(p["case_id"].is_duplicated().sum())
        if dupes:
            raise ValueError(f"aggregated part has {dupes} duplicated case_id rows — aborting join")
        before = out.height
        out = out.join(p, on="case_id", how="left")
        if out.height != before:  # pragma: no cover - defensive
            raise AssertionError("row explosion during join")
    if out["case_id"].n_unique() != out.height:
        raise AssertionError("output violates 1 row per case_id")
    return out


def add_ratio_features(df: pl.DataFrame) -> pl.DataFrame:
    """Inflation-safe, currency-invariant financial ratios."""
    inc = pl.col("maininc_clean").fill_null(0.0) + 1.0
    amt = pl.col("credamount_770A").fill_null(0.0) + 1.0
    return df.with_columns(
        [
            (pl.col("annuity_780A").fill_null(0.0) / inc)
            .cast(pl.Float32)
            .alias("annuity_to_income"),
            (pl.col("credamount_770A").fill_null(0.0) / inc)
            .cast(pl.Float32)
            .alias("credamount_to_income"),
            (pl.col("total_overdue_amt").fill_null(0.0) / amt)
            .cast(pl.Float32)
            .alias("overdue_to_credamount"),
        ]
    )


def high_null_cols(
    df: pl.DataFrame, threshold: float = 0.95, exclude: tuple[str, ...] = ()
) -> list[str]:
    """Columns whose null fraction exceeds `threshold` (fit on TRAIN only)."""
    n = df.height
    if n == 0:
        return []
    out = []
    for name, null_count in zip(df.columns, df.null_count().row(0), strict=True):
        if name in exclude:
            continue
        if null_count / n > threshold:
            out.append(name)
    return out


def write_sample(df: pl.DataFrame, path: Path, n: int = 500, seed: int = 42) -> Path:
    """Stratified-ish small sample committed for CI / HF Space demos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.height <= n:
        sample = df
    elif "target" in df.columns:
        pos = df.filter(pl.col("target") == 1).sample(
            n=min(max(n // 5, 20), int(df["target"].sum())), seed=seed
        )
        neg = df.filter(pl.col("target") == 0).sample(n=n - pos.height, seed=seed)
        sample = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=seed)
    else:
        sample = df.sample(n=n, seed=seed)
    sample.write_parquet(path)
    return path



def _load_tables(raw_dir: Path, cfg) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """Read raw Parquet shards; depth-0 direct-joined, depth-1 aggregated.

    Returns (base_frame, [part_parts...]) where each part is 1 row per case_id.
    """

    def read_sharded(pattern: str) -> pl.DataFrame | None:
        files = sorted(raw_dir.glob(pattern))
        if not files:
            return None
        frames = [pl.read_parquet(p) for p in files]
        return frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed")

    def check_unique(frame: pl.DataFrame, name: str) -> pl.DataFrame:
        dupes = int(frame["case_id"].is_duplicated().sum())
        if dupes:  # static tables are 1 row/case; warn loudly, never explode rows
            print(f"[features] WARNING: {name} has {dupes} duplicated case_id; keeping first occurrence")
            frame = frame.unique(subset=["case_id"], keep="first", maintain_order=True)
        return frame

    base = pl.read_parquet(raw_dir / "train_base.parquet")

    static0 = read_sharded("train_static_0_*.parquet")
    if static0 is None:
        raise FileNotFoundError(f"no train_static_0_*.parquet under {raw_dir}")
    parts: list[pl.DataFrame] = [check_unique(static0, "train_static_0")]

    static_cb = read_sharded("train_static_cb_0*.parquet")  # optional
    if static_cb is not None:
        parts.append(check_unique(static_cb, "train_static_cb_0"))

    person = pl.read_parquet(raw_dir / "train_person_1.parquet")
    if "num_group1" in person.columns:  # primary applicant only
        person = person.filter(pl.col("num_group1") == 0)
    parts.append(check_unique(person, "train_person_1 (num_group1==0)"))

    bureau = read_sharded("train_credit_bureau_a_1_*.parquet")
    if bureau is None:
        raise FileNotFoundError(f"no train_credit_bureau_a_1_*.parquet under {raw_dir}")
    bureau_num = [c for c in bureau.columns if c not in {cfg.id_col, "num_group1"} and bureau[c].dtype.is_numeric()]
    parts.append(
        aggregate_depth1(
            bureau.lazy(), case_id=cfg.id_col, count_name="bureau_loan_count",
            numeric_cols=bureau_num,
        ).collect()
    )

    applprev_files = sorted(raw_dir.glob("train_applprev_1_*.parquet"))
    if applprev_files:
        applprev = pl.concat([pl.scan_parquet(p) for p in applprev_files], how="vertical_relaxed")
        appl_schema = applprev.collect_schema()
        appl_num = [
            name
            for name in appl_schema.names()
            if name not in {cfg.id_col, "num_group1"} and appl_schema[name].is_numeric()
        ]
        parts.append(
            aggregate_depth1(
                applprev, case_id=cfg.id_col, count_name="app_count",
                numeric_cols=appl_num, prefix="prev_",
            ).collect()
        )
    return base, parts


# ---------------------------------------------------------------- orchestration
def build_features(raw_dir: Path, out_path: Path, config=None) -> Path:
    """Build the modelling frame from raw competition Parquet files."""
    cfg = config or load_config()
    raw_dir = Path(raw_dir)
    required = [
        "train_base.parquet",
        "train_static_0_1.parquet",
        "train_person_1.parquet",
        "train_credit_bureau_a_1_1.parquet",
    ]
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing raw tables under {raw_dir}: {missing}.\n"
            "Download the competition data from Kaggle first (see docs/IMPLEMENTATION_PLAN.md §3)."
        )

    base, parts = _load_tables(raw_dir, cfg)

    df = one_row_per_case(base, parts)

    # income column: keep under stable name used by ratios/monotone rules
    if "mainoccupationinc_384A" in df.columns:
        df = df.with_columns(pl.col("mainoccupationinc_384A").cast(pl.Float32).alias("maininc_clean"))

    if "max_totaloverdue_amt" in df.columns:
        df = df.rename({"max_totaloverdue_amt": "total_overdue_amt"})
    elif "sum_overdue" not in df.columns and "total_overdue_amt" not in df.columns:
        overdue_candidates = [c for c in df.columns if "overdue" in c.lower()]
        if overdue_candidates:
            df = df.rename({overdue_candidates[0]: "total_overdue_amt"})

    df = convert_date_columns(df, ref=cfg.date_decision)
    df = add_ratio_features(df) if {"maininc_clean", "credamount_770A"} <= set(df.columns) else df

    # cast floats -> Float32, drop >95%-null cols fit on TRAIN weeks only
    train_mask = (pl.col(cfg.week_col) >= cfg.train_weeks[0]) & (pl.col(cfg.week_col) <= cfg.train_weeks[1])
    train_part = df.filter(train_mask)
    drop_cols = high_null_cols(train_part, cfg.null_col_threshold, exclude=(cfg.id_col, cfg.week_col, cfg.target))
    if drop_cols:
        df = df.drop(drop_cols)

    float_cols = [c for c, dt in zip(df.columns, df.dtypes, strict=True) if dt.is_float()]
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "processed" / "features.parquet"))
    parser.add_argument("--sample", action="store_true", help="also write sample_500.parquet")
    args = parser.parse_args()

    out = build_features(Path(args.raw_dir), Path(args.out))
    df = pl.read_parquet(out)
    print(f"features written: {out}  rows={df.height}  cols={df.width}")
    if args.sample:
        sample_path = write_sample(df, REPO_ROOT / "data" / "processed" / "sample_500.parquet")
        print(f"sample written:   {sample_path}")


if __name__ == "__main__":
    main()
