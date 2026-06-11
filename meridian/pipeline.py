"""
pipeline.py — reproducible end-to-end research run.

    python -m meridian.pipeline            # full run, writes artifacts
    python -m meridian.pipeline --no-lstm  # skip the slow LSTM forecast eval

Loads config + data, runs the ablation (baseline + ML overlays + cost sweep),
and writes auditable artifacts to research/results/:
    variants.csv        per-variant OOS performance
    cost_sweep.csv      Sharpe/return vs cost multiplier
    summary.json        machine-readable headline numbers
    equity_curves.png   strategy variants vs benchmarks
    cost_sweep.png      robustness-to-costs curve

Deterministic: one seed, one config. Re-running reproduces the numbers in
research/report.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config, PROJECT_ROOT
from .seeds import set_global_seed
from .data.loader import load_history
from .ablation import run_ablation

RESULTS_DIR = PROJECT_ROOT / "research" / "results"


def _save_figures(out: dict) -> None:
    """Best-effort PNG figures via Plotly+kaleido; skipped if unavailable."""
    try:
        import plotly.graph_objects as go
    except Exception:
        print("[pipeline] plotly unavailable — skipping figures")
        return

    # Equity curves (log scale so benchmarks don't dwarf the strategy).
    fig = go.Figure()
    for name, eq in out["equity"].items():
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name))
    fig.update_layout(
        title="Equity curves — strategy variants vs benchmarks (net of costs)",
        yaxis_type="log", template="plotly_white", height=500,
    )
    try:
        fig.write_image(str(RESULTS_DIR / "equity_curves.png"), scale=2)
    except Exception as exc:
        print(f"[pipeline] could not write equity_curves.png ({exc}); writing HTML")
        fig.write_html(str(RESULTS_DIR / "equity_curves.html"))

    # Cost sweep.
    cs = out["cost_sweep"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=cs.index, y=cs["baseline_sharpe"], mode="lines+markers", name="baseline"))
    fig2.add_trace(go.Scatter(x=cs.index, y=cs["overlaid_sharpe"], mode="lines+markers", name="+regime+meta"))
    fig2.update_layout(
        title="Annualized Sharpe vs cost multiplier",
        xaxis_title="cost multiplier (x base)", yaxis_title="annualized Sharpe",
        template="plotly_white", height=420,
    )
    try:
        fig2.write_image(str(RESULTS_DIR / "cost_sweep.png"), scale=2)
    except Exception:
        fig2.write_html(str(RESULTS_DIR / "cost_sweep.html"))


def run_pipeline(no_lstm: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    set_global_seed(cfg.seed)

    print("[pipeline] loading data…")
    panel = load_history(cfg)
    print(f"[pipeline] {len(panel.assets)} assets, "
          f"{panel.index.min().date()} -> {panel.index.max().date()}")

    print("[pipeline] running ablation (this trains the LSTM unless --no-lstm)…")
    out = run_ablation(panel, cfg, skip_lstm=no_lstm)

    variants = out["variants"]
    variants.to_csv(RESULTS_DIR / "variants.csv")
    out["cost_sweep"].to_csv(RESULTS_DIR / "cost_sweep.csv")

    lstm = out.get("lstm_result")
    meta = out.get("meta_result")
    summary = {
        "n_trials": out["n_trials"],
        "best_variant_by_sharpe": variants["sharpe"].idxmax(),
        "variants_sharpe": variants["sharpe"].round(4).to_dict(),
        "variants_ann_return": variants["ann_return"].round(4).to_dict(),
        "variants_deflated_sharpe": variants["deflated_sharpe"].round(4).to_dict(),
        "lstm": None if lstm is None else {
            "target": lstm.target, "oos_rmse": round(lstm.oos_rmse, 6),
            "baseline_rmse": round(lstm.baseline_rmse, 6),
            "oos_dir_acc": round(lstm.oos_dir_acc, 4),
            "baseline_dir_acc": round(lstm.baseline_dir_acc, 4),
            "beats_baseline": bool(lstm.beats_baseline),
        },
        "meta": None if meta is None else {
            "n_events": int(meta.n_events), "oos_auc": round(meta.oos_auc, 4),
            "base_rate": round(meta.base_rate, 4),
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    _save_figures(out)

    print("\n=== VARIANTS (net of 1x costs, OOS metrics) ===")
    cols = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "deflated_sharpe"]
    print(variants[cols].round(3).to_string())
    print("\n=== COST SWEEP ===")
    print(out["cost_sweep"].round(3).to_string())
    print(f"\n[pipeline] artifacts written to {RESULTS_DIR}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Meridian research pipeline")
    ap.add_argument("--no-lstm", action="store_true",
                    help="skip the slow LSTM forecast evaluation")
    args = ap.parse_args()
    run_pipeline(no_lstm=args.no_lstm)


if __name__ == "__main__":
    main()
