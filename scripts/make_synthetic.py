"""Generate a tiny SYNTHETIC stand-in for the Kaggle tables.

Used only for local smoke runs and tests — never commit its output as if it
were real competition data. Column names mimic the real schema where known;
after you download the actual dataset, reconcile in `prismrisk/features.py`.

Usage:  python scripts/make_synthetic.py --rows 2000 --out data/raw_synth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def generate(rows: int = 2000, seed: int = 42) -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    case_id = np.arange(1, rows + 1, dtype=np.int64)
    week = np.arange(rows) % 92  # covers WEEK_NUM 0..91 cyclically
    base_date = np.datetime64("2019-01-01") + (week * 7).astype("timedelta64[D]")

    income = rng.lognormal(mean=11.2, sigma=0.6, size=rows)  # ~ KZT-ish scale
    loan = rng.lognormal(mean=12.5, sigma=0.7, size=rows)
    annuity = loan / rng.uniform(6, 24, size=rows)
    dpd_bureau = rng.gamma(shape=0.6, scale=15.0, size=rows)  # past delinquencies

    # target correlated with delinquency (not for ML quality — just for smoke runs)
    p = np.clip(0.04 + 0.009 * dpd_bureau, 0.02, 0.85)
    target = (rng.uniform(size=rows) < p).astype(np.int64)

    base = pl.DataFrame(
        {
            "case_id": case_id,
            "date_decision": base_date,
            "WEEK_NUM": week.astype(np.int64),
            "MONTH": (base_date.astype("datetime64[M]").astype(np.int64) % 12) + 1,
            "target": target,
        }
    )
    static0 = pl.DataFrame(
        {
            "case_id": case_id,
            "credamount_770A": loan.astype(np.float32),
            "annuity_780A": annuity.astype(np.float32),
            "static_past_dpd_001P": dpd_bureau.astype(np.float32),
            "static_limit_002A": (loan * rng.uniform(0.3, 1.5, size=rows)).astype(np.float32),
            "static_signdate_001D": base_date - rng.integers(30, 400, size=rows).astype("timedelta64[D]"),
        }
    )
    person = pl.DataFrame(
        {
            "case_id": np.repeat(case_id, 2),
            "num_group1": np.tile([0, 1], rows),  # 0 = primary applicant
            "mainoccupationinc_384A": np.concatenate([income, income * 0.7]).astype(np.float32),
            "birth_259D": np.concatenate(
                [base_date - rng.integers(20 * 365, 65 * 365, size=rows).astype("timedelta64[D]")] * 2
            ),
        }
    )
    applprev = pl.DataFrame(
        {
            "case_id": np.repeat(case_id, 2),
            "num_group1": np.tile([0, 1], rows),
            "appl_annuity_001A": np.concatenate([annuity, annuity * 0.8]).astype(np.float32),
            "appl_credamount_002A": np.concatenate([loan, loan * 0.9]).astype(np.float32),
            "appl_decisiondate_001D": np.concatenate(
                [base_date - rng.integers(90, 900, size=rows).astype("timedelta64[D]")] * 2
            ),
        }
    )
    bureau = pl.DataFrame(
        {
            "case_id": np.repeat(case_id, 2),
            "num_group1": np.tile([0, 1], rows),
            "dpd_total_001P": np.concatenate([dpd_bureau, dpd_bureau * 0.5]).astype(np.float32),
            "overdue_total_002A": (np.concatenate([dpd_bureau, dpd_bureau * 0.5]) * 37.0).astype(np.float32),
            "bureau_origamount_003A": np.concatenate([loan * 0.6, loan * 0.4]).astype(np.float32),
        }
    )
    static_cb = pl.DataFrame(
        {
            "case_id": case_id,
            "cb_score_001": rng.normal(650, 90, size=rows).astype(np.float32),
            "birth_cb_002D": base_date - rng.integers(20 * 365, 65 * 365, size=rows).astype("timedelta64[D]"),
        }
    )
    return {
        "train_base.parquet": base,
        "train_static_0_1.parquet": static0,
        "train_static_cb_0.parquet": static_cb,
        "train_person_1.parquet": person,
        "train_applprev_1_1.parquet": applprev,
        "train_credit_bureau_a_1_1.parquet": bureau,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/raw_synth")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in generate(args.rows, args.seed).items():
        frame.write_parquet(out_dir / name)
        print(f"wrote {out_dir / name}  rows={frame.height}")


if __name__ == "__main__":
    main()
