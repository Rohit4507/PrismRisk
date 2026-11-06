"""Typed access to configs/config.yaml (single source of truth)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    train_weeks: tuple[int, int] = (0, 59)
    calib_weeks: tuple[int, int] = (60, 69)
    val_weeks: tuple[int, int] = (70, 79)
    test_weeks: tuple[int, int] = (80, 91)
    interest_margin: float = 0.14
    lgd: float = 0.65
    manual_review_cost: float = 12.0
    conformal_confidence: float = 0.90
    alpha: float = 0.10
    monotone_plus: tuple[str, ...] = (
        "max_dpd_bureau",
        "max_dpd_prev",
        "annuity_to_income",
        "credamount_to_income",
    )
    monotone_minus: tuple[str, ...] = ("maininc_clean",)
    immutable_features: tuple[str, ...] = (
        "age_years",
        "max_dpd_bureau",
        "max_dpd_prev",
        "bureau_loan_count",
        "prev_app_count",
    )
    id_col: str = "case_id"
    week_col: str = "WEEK_NUM"
    target: str = "target"
    date_decision: str = "date_decision"
    null_col_threshold: float = 0.95
    cutoff_start: float = 0.01
    cutoff_stop: float = 0.25
    cutoff_step: float = 0.01
    random_seed: int = 42
    raw_dir: Path = field(default=REPO_ROOT / "data" / "raw")
    processed_dir: Path = field(default=REPO_ROOT / "data" / "processed")

    @property
    def week_bounds(self) -> dict[str, tuple[int, int]]:
        return {
            "train": self.train_weeks,
            "calib": self.calib_weeks,
            "val": self.val_weeks,
            "test": self.test_weeks,
        }


def _pair(node: list[int]) -> tuple[int, int]:
    if len(node) != 2 or node[0] > node[1]:
        raise ValueError(f"invalid week range: {node!r} (expected [start, end], start <= end)")
    return (int(node[0]), int(node[1]))


@lru_cache(maxsize=4)
def load_config(path: str | Path = DEFAULT_CONFIG) -> ProjectConfig:
    raw = yaml.safe_load(Path(path).read_text())
    s, e, c, g, d, p, r = (
        raw["splits"],
        raw["economics"],
        raw["conformal"],
        raw["governance"],
        raw["data"],
        raw["profit"],
        raw["runtime"],
    )
    return ProjectConfig(
        train_weeks=_pair(s["train_weeks"]),
        calib_weeks=_pair(s["calib_weeks"]),
        val_weeks=_pair(s["val_weeks"]),
        test_weeks=_pair(s["test_weeks"]),
        interest_margin=float(e["interest_margin"]),
        lgd=float(e["lgd"]),
        manual_review_cost=float(e["manual_review_cost"]),
        conformal_confidence=float(c["confidence_level"]),
        alpha=float(c["alpha"]),
        monotone_plus=tuple(g["monotone_plus"]),
        monotone_minus=tuple(g["monotone_minus"]),
        immutable_features=tuple(g["immutable_features"]),
        id_col=d["id_col"],
        week_col=d["week_col"],
        target=d["target"],
        date_decision=d["date_decision"],
        null_col_threshold=float(d["null_col_threshold"]),
        cutoff_start=float(p["cutoff_start"]),
        cutoff_stop=float(p["cutoff_stop"]),
        cutoff_step=float(p["cutoff_step"]),
        random_seed=int(r["random_seed"]),
        raw_dir=REPO_ROOT / d["raw_dir"],
        processed_dir=REPO_ROOT / d["processed_dir"],
    )
