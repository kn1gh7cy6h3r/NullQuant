"""
pipeline.py — reproducible end-to-end research run.

    python -m nullquant.pipeline

Loads config + data, runs the ablation (baseline + four ML signal sources, each
also conformal-gated, + cost sweep), and writes auditable artifacts to
research/results/:
    variants.csv        per-variant OOS performance
    cost_sweep.csv      Sharpe/return vs cost multiplier
    summary.json        machine-readable headline numbers + ML diagnostics
    equity_curves.png   strategy variants vs benchmarks
    cost_sweep.png      robustness-to-costs curve

Deterministic: one seed, one config. Re-running reproduces research/report.md.
"""

from __future__ import annotations

import argparse
import json

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

    fig = go.Figure()
    for name, eq in out["equity"].items():
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name))
    fig.update_layout(
        title="Equity curves — ML signal variants vs benchmarks (net of costs)",
        yaxis_type="log", template="plotly_white", height=520,
    )
    try:
        fig.write_image(str(RESULTS_DIR / "equity_curves.png"), scale=2)
    except Exception as exc:
        print(f"[pipeline] could not write equity_curves.png ({exc}); writing HTML")
        fig.write_html(str(RESULTS_DIR / "equity_curves.html"))

    cs = out["cost_sweep"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=cs.index, y=cs["baseline_sharpe"],
                              mode="lines+markers", name="baseline"))
    fig2.add_trace(go.Scatter(x=cs.index, y=cs["gated_sharpe"],
                              mode="lines+markers", name="baseline +conformal"))
    fig2.update_layout(
        title="Annualized Sharpe vs cost multiplier",
        xaxis_title="cost multiplier (× base)", yaxis_title="annualized Sharpe",
        template="plotly_white", height=420,
    )
    try:
        fig2.write_image(str(RESULTS_DIR / "cost_sweep.png"), scale=2)
    except Exception:
        fig2.write_html(str(RESULTS_DIR / "cost_sweep.html"))


def run_pipeline(refresh_signals: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    set_global_seed(cfg.seed)

    print("[pipeline] loading data…")
    panel = load_history(cfg)
    print(f"[pipeline] {len(panel.assets)} assets, "
          f"{panel.index.min().date()} -> {panel.index.max().date()}")

    print("[pipeline] running ablation (fits all four ML models walk-forward)…")
    out = run_ablation(panel, cfg, refresh_signals=refresh_signals)

    variants = out["variants"]
    variants.to_csv(RESULTS_DIR / "variants.csv")
    out["cost_sweep"].to_csv(RESULTS_DIR / "cost_sweep.csv")

    summary = {
        "n_trials": out["n_trials"],
        "best_variant_by_sharpe": variants["sharpe"].idxmax(),
        "variants_sharpe": variants["sharpe"].round(4).to_dict(),
        "variants_ann_return": variants["ann_return"].round(4).to_dict(),
        "variants_deflated_sharpe": variants["deflated_sharpe"].round(4).to_dict(),
        "ml_diagnostics": out["diagnostics"],
    }
    with open(RESULTS_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    _save_figures(out)

    print("\n=== VARIANTS (net of 1x costs, OOS metrics) ===")
    cols = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "deflated_sharpe"]
    print(variants[cols].round(3).to_string())
    print("\n=== COST SWEEP ===")
    print(out["cost_sweep"].round(3).to_string())
    print("\n=== ML DIAGNOSTICS (honest truth-tellers) ===")
    for name, d in out["diagnostics"].items():
        print(f"  {name}: {d}")
    print(f"\n[pipeline] artifacts written to {RESULTS_DIR}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="NullQuant research pipeline")
    ap.add_argument("--refresh-signals", action="store_true",
                    help="force a full ML refit, ignoring the cached signals")
    args = ap.parse_args()
    run_pipeline(refresh_signals=args.refresh_signals)


if __name__ == "__main__":
    main()
