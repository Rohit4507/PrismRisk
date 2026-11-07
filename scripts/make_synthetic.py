"""Generate a SYNTHETIC stand-in for the Home Credit stability dataset.

Two regimes are baked in so the pipeline's drift story is visible:
  * WEEK_NUM 0-59  : steady-state baseline
  * WEEK_NUM 60-91 : "COVID shock" — default odds shift, channel mix shifts online

Also emits `riskassesment_940T` (~3.5% coverage, text bands like "6% - 8%"), the
competition's sparsest-but-strongest bureau signal, so the parser in
features.engineer_risk_band is exercised end-to-end on the synthetic fixture.

Used for smoke runs, CI, and the HF Space demo fixture. Never committed as if it
were real competition data (config-driven: real Kaggle parquet plugs in unchanged).

Usage:  python scripts/make_synthetic.py --rows 300000 --out data/raw_synth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def generate(rows: int = 2000, seed: int = 42) -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    case_id = np.arange(1, rows + 1, dtype=np.int64)
    week = np.arange(rows) % 92  # 0..91 cyclically — covers full comp span
    covid = (week >= 60).astype(np.float64)
    base_date = np.datetime64("2019-01-01") + (week * 7).astype("timedelta64[D]")

    # --- correlated applicant economics -----------------------------------
    income = rng.lognormal(mean=11.2, sigma=0.6, size=rows)
    loan = rng.lognormal(mean=10.4, sigma=0.7, size=rows)  # ~ $33k mean retail loan
    annuity = loan / rng.uniform(6.0, 24.0, size=rows)
    # mixture: 55% clean payers (small gamma), 45% stressed (heavy tail)
    stressed = rng.uniform(size=rows) >= 0.55
    dpd = np.where(stressed, rng.gamma(2.0, 16.0, size=rows), rng.gamma(1.0, 2.0, size=rows))
    # channel mix shifts online after the shock
    p_online = np.where(covid > 0, 0.45, 0.30)
    channel = np.where(rng.uniform(size=rows) < p_online, "online", "branch")
    emp_band = rng.choice(["junior", "mid", "senior"], size=rows, p=[0.45, 0.4, 0.15])

    ratio = annuity / (income + 1.0)
    z = (
        -4.80
        + 0.055 * dpd
        + 1.15 * np.log1p(ratio * 10.0)
        - 0.90 * np.log1p(income / 1e5)
        + 0.55 * covid            # regime shift: odds level
        + 0.030 * dpd * covid     # interaction: delinquency hurts MORE post-shock
        + rng.normal(0.0, 0.35, size=rows)
    )
    p = 1.0 / (1.0 + np.exp(-z))
    target = (rng.uniform(size=rows) < p).astype(np.int64)

    base = pl.DataFrame(
        {
            "case_id": case_id,
            "date_decision": base_date,
            "WEEK_NUM": week.astype(np.int64),
            "MONTH": (base_date.astype("datetime64[M]").astype(np.int64) % 12) + 1,
            "target": target,
            "channel": channel,
            "emp_band": emp_band,
        }
    )
    static0 = pl.DataFrame(
        {
            "case_id": case_id,
            "credamount_770A": loan.astype(np.float32),
            "annuity_780A": annuity.astype(np.float32),
            "static_past_dpd_001P": dpd.astype(np.float32),
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
            "dpd_total_001P": np.concatenate([dpd, dpd * 0.5]).astype(np.float32),
            "overdue_total_002A": (np.concatenate([dpd, dpd * 0.5]) * 37.0).astype(np.float32),
            "bureau_origamount_003A": np.concatenate([loan * 0.6, loan * 0.4]).astype(np.float32),
        }
    )
    # Bureau risk band as TEXT (mirrors the competition's riskassesment_940T):
    # only ~3.5% of applicants have one — the parse is the point (top single
    # feature when present). Defaulters land on higher bands on average.
    bands = ["0% - 3%", "4% - 6%", "7% - 11%", "12% - 19%", "20% - 34%", "35% - 59%", "60% - 95%", ">95%"]
    covered = rng.uniform(size=rows) < 0.035
    band_u = rng.uniform(size=rows)
    exponent = np.clip(1.0 - 1.6 * p, 0.15, 1.0)
    band_idx = np.clip((band_u**exponent * len(bands)).astype(int), 0, len(bands) - 1)
    risk_band = np.where(covered, np.array(bands)[band_idx], "")

    static_cb = pl.DataFrame(
        {
            "case_id": case_id,
            "cb_score_001": rng.normal(650, 90, size=rows).astype(np.float32),
            "birth_cb_002D": base_date - rng.integers(20 * 365, 65 * 365, size=rows).astype("timedelta64[D]"),
            "riskassesment_940T": risk_band,  # ~3.5% non-empty — mirrors competition coverage
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
