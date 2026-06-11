"""
ablation.py — the headline evidence: does each ML overlay actually help?

A model that isn't shown to improve out-of-sample risk-adjusted P&L is
decoration. This module runs the strategy in several configurations and compares
them on the SAME data, the SAME costs, and honest OOS metrics:

    baseline                 cross-sectional L/S crossover, vol-targeted
    + regime filter          IsoForest (walk-forward) scales exposure down in
                             abnormal regimes
    + meta-label gate        RF (purged-CV OOS probabilities) scales exposure by
                             how "takeable" the active crossover signals are
    + both                   regime x meta

Each overlay enters ONLY through a causal/OOS exposure multiplier in [0,1], so
turning it on can never use future information. We also sweep the cost model
(0x .. 4x) to show how much of any edge survives friction, and we report the
LSTM's forecast skill against a random-walk baseline separately (it is a
forecast-quality question, not an exposure overlay).

Every reported Sharpe is deflated by the number of configurations tried, so the
selection of the "best" variant is penalised — the antidote to backtest
overfitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data.loader import Panel
from .signals.base import target_directions
from .portfolio.costs import CostModel
from .portfolio.backtest import run_backtest
from .metrics import performance as perf
from .validation.walk_forward import walk_forward_report


def _exposure_product(*series: pd.Series, index: pd.Index) -> pd.Series:
    out = pd.Series(1.0, index=index)
    for s in series:
        if s is not None:
            out = out * s.reindex(index).fillna(1.0).clip(0.0, 1.0)
    return out


def build_overlays(panel: Panel, cfg: Config) -> dict[str, pd.Series]:
    """
    Compute each overlay's causal exposure multiplier (Series in [0,1] over the
    panel calendar). Imports the ML modules defensively: a model that fails to
    import/run simply contributes a neutral (all-ones) overlay.
    """
    idx = panel.close.index
    overlays: dict[str, pd.Series] = {}

    try:
        from .ml.regime_iforest import compute_regime_filter
        overlays["regime"] = compute_regime_filter(panel, cfg).exposure_scale.reindex(idx).fillna(1.0)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ablation] regime overlay unavailable: {exc}")
        overlays["regime"] = pd.Series(1.0, index=idx)

    try:
        from .ml.rf_meta import fit_eval_meta, exposure_gate
        meta_res = fit_eval_meta(panel, cfg)
        overlays["meta"] = exposure_gate(panel, cfg, meta_res).reindex(idx).fillna(1.0)
        overlays["_meta_result"] = meta_res  # carried for the report
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ablation] meta overlay unavailable: {exc}")
        overlays["meta"] = pd.Series(1.0, index=idx)

    return overlays


def run_ablation(panel: Panel, cfg: Config, skip_lstm: bool = False) -> dict:
    """
    Run baseline + each overlay combination and return a structured result:
        variants : DataFrame of OOS performance metrics, one row per variant
        equity   : dict variant -> equity Series (for plotting)
        walk_forward : dict variant -> walk_forward_report
        meta_result, lstm_result : the ML diagnostics
        cost_sweep : DataFrame of Sharpe/ann_return across cost multipliers
    """
    direction = target_directions(panel, cfg)
    cost = CostModel.from_config(cfg, multiplier=1.0)
    idx = panel.close.index

    overlays = build_overlays(panel, cfg)
    regime = overlays.get("regime")
    meta = overlays.get("meta")

    configs = {
        "baseline": None,
        "+regime": regime,
        "+meta": meta,
        "+regime+meta": _exposure_product(regime, meta, index=idx),
    }
    n_trials = len(configs)

    bench = None  # filled from first backtest's equal_weight benchmark
    rows, equity, wf = [], {}, {}
    for name, exposure in configs.items():
        res = run_backtest(panel, direction, cfg, cost, exposure_scale=exposure)
        if bench is None:
            bench = res.benchmarks["equal_weight"]
        s = perf.summary(res.net_returns, benchmark=bench, n_trials=n_trials)
        s["variant"] = name
        s["avg_leverage"] = float(res.leverage.mean())
        s["ann_turnover"] = float(res.turnover.mean() * perf.TRADING_DAYS)
        rows.append(s)
        equity[name] = res.equity
        wf[name] = walk_forward_report(res.net_returns, cfg)

    # Benchmarks as their own rows for an apples-to-apples comparison.
    for bname, bret in run_backtest(panel, direction, cfg, cost).benchmarks.items():
        s = perf.summary(bret, n_trials=1)
        s["variant"] = f"[bench] {bname}"
        s["avg_leverage"] = 1.0
        s["ann_turnover"] = np.nan
        rows.append(s)
        equity[f"[bench] {bname}"] = (1.0 + bret.fillna(0.0)).cumprod()

    variants = pd.DataFrame(rows).set_index("variant")

    # ── LSTM forecast skill (separate question) ───────────────────────────────
    lstm_result = None
    if not skip_lstm:
        try:
            from .ml.lstm_forecast import train_eval_lstm
            lstm_result = train_eval_lstm(panel, cfg, asset="BTC-USD")
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[ablation] LSTM eval unavailable: {exc}")

    cost_sweep = run_cost_sweep(panel, cfg, direction,
                                best_exposure=configs["+regime+meta"])

    return {
        "variants": variants,
        "equity": equity,
        "walk_forward": wf,
        "meta_result": overlays.get("_meta_result"),
        "lstm_result": lstm_result,
        "cost_sweep": cost_sweep,
        "n_trials": n_trials,
    }


def run_cost_sweep(panel: Panel, cfg: Config, direction: pd.DataFrame,
                   best_exposure: pd.Series | None) -> pd.DataFrame:
    """
    Re-run the baseline and the fully-overlaid strategy across the configured
    cost multipliers, recording annualized Sharpe and return. This is the
    robustness curve: how much edge survives 0x .. 4x friction.
    """
    rows = []
    for mult in cfg.costs.sweep_multipliers:
        cm = CostModel.from_config(cfg, multiplier=float(mult))
        base = run_backtest(panel, direction, cfg, cm)
        full = run_backtest(panel, direction, cfg, cm, exposure_scale=best_exposure)
        rows.append({
            "cost_multiplier": float(mult),
            "per_side_bps": cm.per_side_bps,
            "baseline_sharpe": perf.sharpe_ratio(base.net_returns),
            "baseline_ann_return": perf.annualized_return(base.net_returns),
            "overlaid_sharpe": perf.sharpe_ratio(full.net_returns),
            "overlaid_ann_return": perf.annualized_return(full.net_returns),
        })
    return pd.DataFrame(rows).set_index("cost_multiplier")
