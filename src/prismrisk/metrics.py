"""Evaluation metrics: weekly Gini stability, calibration (Brier/ECE), net profit $.

NOTE on the competition metric: Kaggle's official "stability" metric is defined per
evaluation sample (weekly). We expose a documented *proxy* —
mean/std/min of weekly Gini plus an optional slope-penalty "custom_stability" —
and never claim it reproduces the official leaderboard number.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def roc_auc(y_true, y_prob) -> float:
    return float(roc_auc_score(y_true, y_prob))


def pr_auc(y_true, y_prob) -> float:
    return float(average_precision_score(y_true, y_prob))


def weekly_gini_stability(weeks, y_true, y_prob) -> dict[str, float]:
    """Per-week Gini with dispersion stats. Weeks with a single class are skipped."""
    weeks_arr = np.asarray(weeks)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    gini_per_week: dict[int, float] = {}
    for w in np.unique(weeks_arr):
        mask = weeks_arr == w
        y_w = y_true[mask]
        if y_w.size == 0 or np.unique(y_w).size < 2:
            continue
        gini_per_week[int(w)] = 2.0 * roc_auc_score(y_w, y_prob[mask]) - 1.0

    if not gini_per_week:
        raise ValueError("no weekly slice contains both classes; cannot compute Gini")

    xs = np.array(sorted(gini_per_week), dtype=np.float64)
    gs = np.array([gini_per_week[int(x)] for x in xs])
    slope = float(np.polyfit(xs, gs, 1)[0]) if xs.size >= 3 else 0.0
    mean_g = float(gs.mean())
    std_g = float(gs.std(ddof=0))
    return {
        "mean_gini": mean_g,
        "std_gini": std_g,
        "min_gini": float(gs.min()),
        "slope_per_week": slope,
        # custom proxy, NOT the official Kaggle metric:
        "custom_stability": mean_g + 88.0 * min(0.0, slope) - 0.5 * std_g,
        "n_weeks": float(len(gini_per_week)),
    }


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Uniform-bin ECE (Naeini et al., 2015)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = y_true.size
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (y_prob >= lo) & ((y_prob < hi) if hi < 1.0 else (y_prob <= hi))
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(y_true, y_prob))


def expected_net_profit(
    y_true,
    y_prob,
    amounts,
    cutoff: float,
    *,
    margin: float = 0.14,
    lgd: float = 0.65,
) -> float:
    """Approve applicants with predicted PD < cutoff.

    Repaid loan  -> +margin * amount
    Default      -> -lgd * amount
    """
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts, dtype=np.float64)
    approved = np.asarray(y_prob, dtype=np.float64) < cutoff
    if not approved.any():
        return 0.0
    profit = np.where(
        y_true[approved] == 0,
        margin * amounts[approved],
        -lgd * amounts[approved],
    )
    return float(profit.sum())


def optimal_cutoff(
    y_true,
    y_prob,
    amounts,
    *,
    margin: float = 0.14,
    lgd: float = 0.65,
    start: float = 0.01,
    stop: float = 0.25,
    step: float = 0.01,
) -> tuple[float, list[tuple[float, float]]]:
    """Sweep the approval threshold; return (tau_star, [(cutoff, profit), ...])."""
    grid = np.arange(start, stop + 1e-9, step)
    curve = [(float(c), expected_net_profit(y_true, y_prob, amounts, c, margin=margin, lgd=lgd)) for c in grid]
    tau_star = max(curve, key=lambda item: item[1])[0]
    return tau_star, curve
