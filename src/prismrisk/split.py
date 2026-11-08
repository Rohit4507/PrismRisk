"""Strict chronological week splitter. NEVER shuffles.

Weeks follow the competition's WEEK_NUM (0..91):
  train 0-59 | calib 60-69 | val 70-79 | test 80-91 (locked, touched last)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from prismrisk.config import REPO_ROOT, load_config

SPLITS = ("train", "calib", "val", "test")


def load_features(path: str | Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    cfg = load_config()
    for col in (cfg.id_col, cfg.week_col, cfg.target):
        if col not in df.columns:
            raise KeyError(f"features frame is missing required column {col!r}")
    return df


def split_by_week(
    df: pl.DataFrame, start: int, end: int, week_col: str = "WEEK_NUM"
) -> pl.DataFrame:
    return df.filter((pl.col(week_col) >= start) & (pl.col(week_col) <= end))


def make_splits(
    df: pl.DataFrame,
    bounds: dict[str, tuple[int, int]] | None = None,
    week_col: str = "WEEK_NUM",
) -> dict[str, pl.DataFrame]:
    bounds = bounds or load_config().week_bounds
    splits = {name: split_by_week(df, a, b, week_col) for name, (a, b) in bounds.items()}
    assert_no_overlap(splits, week_col)
    return splits


def assert_no_overlap(splits: dict[str, pl.DataFrame], week_col: str = "WEEK_NUM") -> None:
    """Fail loudly if any week leaks across the chronological boundary."""
    ranges = []
    for name, frame in splits.items():
        if frame.is_empty():
            raise AssertionError(f"split {name!r} is empty — check WEEK_NUM values vs config")
        lo, hi = frame[week_col].min(), frame[week_col].max()
        assert lo is not None and hi is not None
        ranges.append((name, int(lo), int(hi)))
    order = [name for name, _, _ in ranges]
    assert order == list(SPLITS) or set(order) == set(SPLITS), f"unexpected splits: {order}"
    ordered = [r for name in SPLITS for r in ranges if r[0] == name]
    for prev, nxt in zip(ordered, ordered[1:], strict=False):
        assert prev[2] < nxt[1], (
            f"split overlap: {prev[0]} (weeks ..{prev[2]}) must end before "
            f"{nxt[0]} starts (weeks {nxt[1]}..)"
        )


def split_summary(splits: dict[str, pl.DataFrame], target: str = "target") -> list[dict]:
    out = []
    for name, frame in splits.items():
        rate = frame[target].mean() if target in frame.columns else float("nan")
        out.append(
            {
                "split": name,
                "weeks": f"{frame['WEEK_NUM'].min()}-{frame['WEEK_NUM'].max()}",
                "rows": frame.height,
                "default_rate": round(float(rate), 4),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", default=str(REPO_ROOT / "data" / "processed" / "features.parquet")
    )
    args = parser.parse_args()
    df = load_features(args.features)
    splits = make_splits(df)
    for row in split_summary(splits):
        print(row)


if __name__ == "__main__":
    main()
