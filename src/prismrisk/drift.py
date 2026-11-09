"""M5 — weekly PSI drift report with train-locked binning.

For top-k features (by mean |SHAP|, or a caller-supplied list) and predicted PD:
freeze 10 quantile bin edges on the **train weeks (W0–59)**, then for each later
week compute Population Stability Index::

    PSI = sum_b (p_train[b] - p_week[b]) * ln(p_train[b] / p_week[b])

Bands (industry convention): <0.10 green · 0.10–0.25 warning · >0.25 retrain.

Artifacts (all under ``reports/`` by default, gitignored like mlruns/):
``reports/psi_weekly.csv`` (machine-readable) and ``reports/drift.html``
(self-contained, inline SVG — no external assets, no evidently:
evidently 0.7.23 is unimportable under pandas 3.x, so the report is rendered
in-house with zero new dependencies).

PD for a champion bundle is computed exactly as at train time
(``TableEncoder`` with the bundle's encode_maps + optional isotonic).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import shap

from prismrisk.config import load_config

PSI_GREEN = 0.10
PSI_WARN = 0.25
_EPS = 1e-4


def band(psi: float) -> str:
    if psi < PSI_GREEN:
        return "green"
    if psi < PSI_WARN:
        return "warning"
    return "retrain"


def _finite(vals) -> np.ndarray:
    v = np.asarray(vals, dtype=np.float64).ravel()
    return v[np.isfinite(v)]


def freeze_edges(train_vals, n_bins: int = 10) -> np.ndarray:
    v = _finite(train_vals)
    if v.size < 50:
        raise ValueError(f"need >=50 train values to freeze {n_bins} bins, got {v.size}")
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        raise ValueError("train distribution is (near-)constant; PSI undefined — drop this feature")
    return edges


def _bin_fracs(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, np.clip(vals, edges[0], edges[-1]), side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1).astype(np.float64)
    return np.clip(counts / max(counts.sum(), 1.0), _EPS, None)


def psi(train_vals, test_vals, n_bins: int = 10) -> float:
    """Population Stability Index between two samples; bins frozen on train."""
    tr = _finite(train_vals)
    te = _finite(test_vals)
    if te.size == 0:
        raise ValueError("test week has no finite values for this feature")
    edges = freeze_edges(tr, n_bins)
    p = _bin_fracs(tr, edges)
    q = _bin_fracs(te, edges)
    return float(max(np.sum((p - q) * np.log(p / q)), 0.0))


def select_top_features(model, X: np.ndarray, feature_names: list[str], k: int = 10) -> list[str]:
    """Top-k features by mean |SHAP| on a sample (tree models)."""
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    arr = np.asarray(sv)
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    mean_abs = np.abs(arr).mean(axis=0)
    order = [int(i) for i in np.argsort(-mean_abs)]
    return [feature_names[i] for i in order[:k]]


def weekly_psi(df: pl.DataFrame, *, feature_cols: list[str], pd_col: str | None = None,
               week_col: str = "WEEK_NUM", train_hi: int = 59, n_bins: int = 10) -> pl.DataFrame:
    """Long-format report: one row per (week, feature) with psi + band."""
    train_df = df.filter(pl.col(week_col) <= train_hi)
    test_df = df.filter(pl.col(week_col) > train_hi)
    if train_df.height == 0 or test_df.height == 0:
        raise ValueError("need both train (<=train_hi) and post-train weeks in the frame")
    cols = [c for c in feature_cols if c in df.columns]
    if pd_col and pd_col in df.columns:
        cols = [*cols, pd_col]
    train_np = {c: train_df[c].drop_nulls().to_numpy() for c in cols}
    out: list[dict] = []
    for week in sorted(test_df[week_col].unique().to_list()):
        wdf = test_df.filter(pl.col(week_col) == week)
        for c in cols:
            try:
                value = psi(train_np[c], wdf[c].to_numpy(), n_bins)
            except ValueError:  # e.g. all-null week/feature — report, don't crash
                value = float("nan")
            out.append({"week": int(week), "feature": c, "n_rows": wdf.height, "psi": value})
    report = pl.DataFrame(out)
    return report.with_columns(
        pl.when(pl.col("psi") < PSI_GREEN).then(pl.lit("green"))
        .when(pl.col("psi") < PSI_WARN).then(pl.lit("warning"))
        .when(pl.col("psi").is_nan()).then(pl.lit("n/a"))
        .otherwise(pl.lit("retrain"))
        .alias("band")
    )


_HTML_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<title>PrismRisk — weekly PSI drift</title>
<style>
 body{font-family:system-ui,-apple-system,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#111}
 h1{font-size:1.35rem} h2{font-size:1.05rem;margin-top:1.5rem}
 table{border-collapse:collapse;width:100%;font-size:.85rem;margin:.5rem 0}
 th,td{border:1px solid #ddd;padding:.35rem .5rem;text-align:right} th{background:#f3f4f6}
 td:first-child,th:first-child{text-align:left}
 .green{background:#dcfce7}.warning{background:#fef9c3}.retrain{background:#fee2e2}.na{background:#eee}
 .note{color:#555;font-size:.8rem;margin-top:1rem}
 .pill{display:inline-block;border-radius:999px;padding:.1rem .6rem;font-size:.75rem}
</style></head><body>
<h1>PrismRisk — weekly PSI drift report</h1>
<p>Bins frozen on train weeks ≤ __TRAIN_HI__ (10 quantiles). Flag rule of thumb:
<b>psi &gt; 0.25 on any top feature or PD for 2 consecutive weeks → retrain.</b></p>
"""
_HTML_TAIL = """<p class="note">Generated by src/prismrisk/drift.py (train-locked bins).
Synthetic-fixture numbers are demo-grade — do not read production meaning into them.
CI/tests exercise the same code path on the tiny synthetic fixture.</p></body></html>"""


def _svg_line(weeks: list[int], values: list[float], width: int = 920, height: int = 220) -> str:
    if not weeks:
        return "<p>No PD drift series.</p>"
    pad_l, pad_b, pad_t = 44, 28, 14
    vmax = max(*values, PSI_WARN) * 1.15 or 1.0
    xs = [pad_l + i * (width - pad_l - 10) / max(len(weeks) - 1, 1) for i in range(len(weeks))]
    ys = [height - pad_b - (v / vmax) * (height - pad_b - pad_t) for v in values]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    warn_y = height - pad_b - (PSI_WARN / vmax) * (height - pad_b - pad_t)
    return (
        f'<svg width="{width}" height="{height}" role="img" aria-label="weekly PD PSI">'
        f'<line x1="{pad_l}" y1="{warn_y:.1f}" x2="{width - 10}" y2="{warn_y:.1f}" stroke="#ef4444" '
        f'stroke-dasharray="6 4" stroke-width="1.2"/>'
        f'<text x="{width - 12}" y="{warn_y - 4:.1f}" text-anchor="end" font-size="10" fill="#ef4444">0.25</text>'
        f'<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="2"/>'
        f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3.5" fill="#2563eb"/>'
        f'<text x="{pad_l}" y="12" font-size="11" fill="#333">weekly PSI (predicted PD)</text>'
        f'<text x="{pad_l}" y="{height - 8}" font-size="10" fill="#555">W{weeks[0]}</text>'
        f'<text x="{width - 14}" y="{height - 8}" text-anchor="end" font-size="10" fill="#555">W{weeks[-1]}</text>'
        "</svg>"
    )


def render_html(report: pl.DataFrame, *, train_hi: int, pd_col: str | None, out: Path) -> Path:
    pdf = report.to_pandas()
    bands = {"green": "green", "warning": "warning", "retrain": "retrain", "n/a": "na"}
    parts = [_HTML_HEAD.replace("__TRAIN_HI__", str(train_hi))]
    if pd_col and pd_col in set(pdf["feature"]):
        sub = pdf[(pdf["feature"] == pd_col) & np.isfinite(pdf["psi"])].sort_values("week")
        parts.append(f"<h2>PD drift</h2>{_svg_line(sub['week'].tolist(), [float(v) for v in sub['psi']])}")
    parts.append("<h2>Per-feature weekly PSI</h2><table><tr><th>feature</th><th>week</th>"
                 "<th>n</th><th>PSI</th><th>band</th></tr>")
    for _, r in pdf.sort_values(["feature", "week"]).iterrows():
        cls = bands.get(r["band"], "na")
        psi_txt = "n/a" if not np.isfinite(r["psi"]) else f"{r['psi']:.3f}"
        parts.append(f"<tr><td>{r['feature']}</td><td>{int(r['week'])}</td><td>{int(r['n_rows'])}</td>"
                     f"<td>{psi_txt}</td><td class=\"{cls}\">{r['band']}</td></tr>")
    parts.append("</table>" + _HTML_TAIL)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts))
    return out


def run_drift(features_path, *, bundle_path=None, pd_col: str = "predicted_pd", top_k: int = 10,
              n_bins: int = 10, csv_path: Path = Path("reports/psi_weekly.csv"),
              html_path: Path = Path("reports/drift.html")) -> pl.DataFrame:
    """End-to-end: load features (+ optional champion bundle for PD), write CSV + HTML."""
    cfg = load_config()
    df = pl.read_parquet(features_path)
    week_col, target = cfg.week_col, cfg.target

    if bundle_path:
        from prismrisk.train import TableEncoder  # noqa: PLC0415 — lazy: pulls optuna/xgboost

        bundle = joblib.load(bundle_path)
        enc = TableEncoder()
        enc.maps, enc.feature_names = bundle.get("encode_maps", {}), bundle["feature_names"]
        model = bundle["model"]
        X = enc.transform(df)  # encoder.feature_names already excludes ids/week/target
        pd_raw = model.predict_proba(X)[:, 1]
        iso = bundle.get("isotonic")
        df = df.with_columns(pl.Series(pd_col, iso.predict(pd_raw).astype(np.float64) if iso is not None else pd_raw))
    elif pd_col not in df.columns:
        raise ValueError(f"no --bundle given and no '{pd_col}' column in features; pass --bundle or --pd-col")

    exclude = {week_col, target, "case_id", "MONTH", pd_col}
    num_cols = [c for c, dt in zip(df.columns, df.dtypes, strict=True)
                if dt.is_numeric() and c not in exclude]
    if not bundle_path and len(num_cols) > top_k:
        # no model available: rank by train-week variance (deterministic, no SHAP)
        train_var = df.filter(pl.col(week_col) <= cfg.train_weeks[1]).select(
            [pl.col(c).var().alias(c) for c in num_cols]).row(0)
        order = np.argsort(-np.nan_to_num(np.asarray(train_var, dtype=np.float64)))
        num_cols = [num_cols[int(i)] for i in order[:top_k]]
    if bundle_path and len(num_cols) > top_k:
        # rank by SHAP on a strided sample, as at train time
        sample = df[:: max(1, df.height // 5000)]
        enc_x = enc.transform(sample)
        num_cols = select_top_features(model, enc_x, enc.feature_names, k=top_k)
        num_cols = [c for c in num_cols if c in df.columns and df[c].dtype.is_numeric()]

    report = weekly_psi(df, feature_cols=num_cols, pd_col=pd_col, week_col=week_col,
                        train_hi=cfg.train_weeks[1], n_bins=n_bins)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.write_csv(csv_path)
    render_html(report, train_hi=cfg.train_weeks[1], pd_col=pd_col, out=Path(html_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly PSI drift on post-train weeks.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--bundle", help="champion bundle; PD computed with encode_maps + isotonic")
    parser.add_argument("--pd-col", default="predicted_pd", help="existing PD column if --bundle is absent")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--csv", default="reports/psi_weekly.csv")
    parser.add_argument("--html", default="reports/drift.html")
    args = parser.parse_args()
    report = run_drift(args.features, bundle_path=args.bundle, pd_col=args.pd_col,
                       top_k=args.top_k, n_bins=args.n_bins,
                       csv_path=Path(args.csv), html_path=Path(args.html))
    n_retrain = int((report["band"] == "retrain").sum())
    n_warn = int((report["band"] == "warning").sum())
    print(report.head(20))
    print(f"\nrows={report.height}  warning={n_warn}  retrain={n_retrain}  -> {args.csv}, {args.html}")


if __name__ == "__main__":
    main()
