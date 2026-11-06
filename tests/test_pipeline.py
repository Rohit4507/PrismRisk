"""CI-safe test suite. Uses synthetic data only — no Kaggle downloads needed."""

from __future__ import annotations

from datetime import date

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from prismrisk import metrics, notice_guard
from prismrisk.config import load_config
from prismrisk.features import (
    add_ratio_features,
    aggregate_depth1,
    convert_date_columns,
    high_null_cols,
    one_row_per_case,
)
from prismrisk.split import make_splits, split_by_week

CFG = load_config()


def make_weeks_frame(rows: int = 920) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": np.arange(1, rows + 1),
            "WEEK_NUM": np.arange(rows) % 92,
            "target": (np.arange(rows) * 2654435761 % 100 < 3).astype(np.int64),
        }
    )


# ------------------------------------------------------------------ splits
def test_split_bounds() -> None:
    df = make_weeks_frame()
    splits = make_splits(df)
    assert {k: (s["WEEK_NUM"].min(), s["WEEK_NUM"].max()) for k, s in splits.items()} == {
        "train": (0, 59),
        "calib": (60, 69),
        "val": (70, 79),
        "test": (80, 91),
    }


def test_split_overlap_detected() -> None:
    df = make_weeks_frame()
    bad = {"train": (0, 70), "calib": (60, 69), "val": (70, 79), "test": (80, 91)}
    with pytest.raises(AssertionError, match="overlap"):
        make_splits(df, bounds=bad)


def test_split_by_week_subset() -> None:
    df = split_by_week(make_weeks_frame(), 60, 69)
    assert set(df["WEEK_NUM"].unique().to_list()) == set(range(60, 70))


# ------------------------------------------------------------------ features
def test_aggregate_depth1_counts_and_stats() -> None:
    lf = pl.LazyFrame(
        {
            "case_id": [1, 1, 2, 2, 2],
            "num_group1": [0, 0, 0, 0, 0],
            "x": [1.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    out = aggregate_depth1(
        lf, case_id="case_id", count_name="n", numeric_cols=["x"], filter_expr=pl.col("num_group1") == 0
    ).collect()
    out = out.sort("case_id")
    assert out["n"].to_list() == [2, 3]
    assert out["max_x"].to_list() == [3.0, 30.0]
    assert out["mean_x"].to_list() == [2.0, 20.0]


def test_join_rejects_duplicates() -> None:
    base = pl.DataFrame({"case_id": [1, 2]})
    dup_part = pl.DataFrame({"case_id": [1, 1], "x": [10, 20]})
    with pytest.raises(ValueError, match="duplicated"):
        one_row_per_case(base, [dup_part])


def test_join_preserves_row_count() -> None:
    base = pl.DataFrame({"case_id": [1, 2, 3]})
    part = pl.DataFrame({"case_id": [1, 2], "x": [10, 20]})
    out = one_row_per_case(base, [part])
    assert out.height == 3
    assert out.sort("case_id")["x"].to_list() == [10, 20, None]


def test_ratio_features_non_negative() -> None:
    df = pl.DataFrame(
        {
            "maininc_clean": [1000.0, None, 50.0],
            "credamount_770A": [5000.0, 2000.0, None],
            "annuity_780A": [100.0, 200.0, None],
            "total_overdue_amt": [0.0, None, 30.0],
        }
    )
    out = add_ratio_features(df)
    for col in ("annuity_to_income", "credamount_to_income", "overdue_to_credamount"):
        assert out[col].is_finite().all()
        assert (out[col] >= 0).all()


def test_convert_date_columns() -> None:
    df = pl.DataFrame(
        {
            "case_id": [1],
            "date_decision": pl.Series([date(2020, 6, 15)], dtype=pl.Date),
            "signdate_001D": pl.Series([date(2020, 1, 15)], dtype=pl.Date),
        }
    )
    out = convert_date_columns(df, ref="date_decision")
    assert "signdate_001D" not in out.columns
    assert out["signdate_001_days"][0] == 152.0


def test_high_null_cols_fit_on_train_only() -> None:
    df = pl.DataFrame(
        {
            "keep": [1, None, 3, 4, 5, 6, 7, 8, 9, 10],  # 10% null
            "drop": [None] * 9 + [1],                    # 90% null
        }
    )
    assert high_null_cols(df, threshold=0.5) == ["drop"]
    assert high_null_cols(df, threshold=0.95) == []


# ------------------------------------------------------------------ metrics
def test_ece_constant_predictor_zero() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    assert metrics.expected_calibration_error(y, np.full(y.shape, 0.5)) == pytest.approx(0.0)


def test_expected_net_profit_toy() -> None:
    y = np.array([0, 1])
    p = np.array([0.4, 0.6])
    amounts = np.array([1000.0, 1000.0])
    # only the first (PD 0.4 < cutoff 0.5) is approved -> +14% margin
    assert metrics.expected_net_profit(y, p, amounts, cutoff=0.5, margin=0.14, lgd=0.65) == pytest.approx(140.0)
    # both approved -> 140 - 650
    assert metrics.expected_net_profit(y, p, amounts, cutoff=0.7, margin=0.14, lgd=0.65) == pytest.approx(-510.0)


def test_weekly_gini_stability_perfect_separation() -> None:
    weeks = np.repeat(np.arange(6), 20)
    y = np.tile([0] * 10 + [1] * 10, 6)
    p = y * 0.99 + (1 - y) * 0.01
    out = metrics.weekly_gini_stability(weeks, y, p)
    assert out["mean_gini"] == pytest.approx(1.0)
    assert out["min_gini"] == pytest.approx(1.0)
    assert out["n_weeks"] == 6.0


def test_optimal_cutoff_returns_grid_max() -> None:
    rng = np.random.default_rng(7)
    y = (rng.uniform(size=400) < 0.2).astype(int)
    p = np.clip(np.where(y == 1, 0.7, 0.1) + rng.normal(0, 0.05, size=400), 0, 1)
    amounts = np.full(400, 1000.0)
    tau_star, curve = metrics.optimal_cutoff(y, p, amounts, start=0.05, stop=0.3, step=0.05)
    assert tau_star == max(curve, key=lambda c: c[1])[0]
    assert len(curve) == 6


# ------------------------------------------------------------- notice guard
FACTS = notice_guard.EvidenceFacts(
    pd_pct="8.4%",
    tau_pct="5.0%",
    current_loan="$4,500",
    target_loan="$3,200",
    new_pd_pct="3.6%",
    reason_1="high installment-to-income ratio",
    reason_2="short employment history",
    reason_3="limited credit bureau coverage",
)


def test_notice_template_is_grounded() -> None:
    result = notice_guard.render_notice(FACTS)
    assert result.grounding_passed and not result.fallback_triggered
    for token in notice_guard.extract_numbers(result.text):
        assert token.replace(",", "") in FACTS.allowed_numbers()


def test_notice_guard_blocks_hallucinated_dollar_amount() -> None:
    fake_llm = lambda _prompt: "Dear applicant, your $9,999 application was declined."  # noqa: E731
    result = notice_guard.render_notice(FACTS, llm=fake_llm)
    assert result.fallback_triggered
    assert not result.grounding_passed
    assert "9,999" in result.ungrounded_numbers[0]


def test_notice_guard_llm_with_only_allowed_numbers_passes() -> None:
    good_llm = lambda _prompt: (  # noqa: E731
        "Your estimated default probability is 8.4% against our 5.0% threshold. "
        "Reducing the request from $4,500 to $3,200 gives 3.6%."
    )
    result = notice_guard.render_notice(FACTS, llm=good_llm)
    assert result.grounding_passed and not result.fallback_triggered


# ----------------------------------------------------- monotone governance
def test_monotone_constraint_never_lowers_risk_with_dpd() -> None:
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 100, size=300)      # dpd-like
    x2 = rng.normal(0, 1, size=300)
    y = (0.05 * x1 + rng.normal(0, 1, size=300) > 1.0).astype(int)
    X = np.column_stack([x1, x2])
    model = lgb.LGBMClassifier(
        n_estimators=40,
        num_leaves=8,
        monotone_constraints=[1, 0],
        random_state=0,
        verbose=-1,
    )
    model.fit(X, y)
    grid = np.column_stack([np.linspace(0, 100, 60), np.zeros(60)])
    probs = model.predict_proba(grid)[:, 1]
    assert (np.diff(probs) >= -1e-9).all()


# ------------------------------------------------------------------- config
def test_config_defaults() -> None:
    assert CFG.train_weeks == (0, 59)
    assert CFG.test_weeks == (80, 91)
    assert CFG.conformal_confidence == pytest.approx(0.90)
    assert "age_years" in CFG.immutable_features
