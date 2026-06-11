"""
ablation.py — the headline evidence: does any creative ML model actually help?

We pit four independently-built signal sources against each other and against
buy-and-hold, on the same data, same costs, and honest out-of-sample metrics:

    baseline   hand-coded cross-sectional momentum long/short
    LTR        learning-to-rank cross-sectional selector
    regime     HMM regime-switching meta-controller
    lead-lag   lead-lag contagion network

Each is also run with the **conformal** confidence gate layered on top (an
exposure multiplier in [0,1] that only deploys risk when a calibrated model is
confident). Every Sharpe is deflated by the number of configurations tried, so
selecting the "best" is penalised — the antidote to backtest overfitting.

Each model carries its own truth-teller diagnostic (rank IC, regime mapping,
lead-lag hit-rate, conformal coverage); a model that doesn't beat its honest
baseline is reported as such, not hidden.
"""

from __future__ import annotations

import pandas as pd

from .config import Config
from .data.loader import Panel
from .signals.base import target_directions
from .portfolio.costs import CostModel
from .portfolio.backtest import run_backtest
from .metrics import performance as perf
from .validation.walk_forward import walk_forward_report


def compute_signals(panel: Panel, cfg: Config) -> tuple[dict[str, pd.DataFrame], pd.Series, dict]:
    """
    Build every direction source + the conformal exposure overlay, and collect
    each model's honest diagnostic. Imports the ML modules defensively so one
    failure degrades to a neutral signal instead of taking down the ablation.

    Returns (directions, conformal_exposure, diagnostics).
    """
    idx = panel.close.index
    directions: dict[str, pd.DataFrame] = {"baseline": target_directions(panel, cfg)}
    diagnostics: dict = {}

    try:
        from .ml.rank_model import ltr_signal
        r = ltr_signal(panel, cfg)
        directions["LTR"] = r.direction
        diagnostics["ltr"] = dict(rank_ic=r.rank_ic, ic_hit=r.ic_hit,
                                  n_refits=r.n_refits, status=r.status)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ablation] LTR unavailable: {exc}")

    try:
        from .ml.regime_switch import regime_switch_signal
        g = regime_switch_signal(panel, cfg)
        directions["regime"] = g.direction
        diagnostics["regime"] = dict(n_states=g.n_states, mapping=g.mapping,
                                     occupancy=g.occupancy, status=g.status)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ablation] regime unavailable: {exc}")

    try:
        from .ml.lead_lag import leadlag_signal
        l = leadlag_signal(panel, cfg)
        directions["lead-lag"] = l.direction
        diagnostics["leadlag"] = dict(oos_hit_rate=l.oos_hit_rate,
                                      top_edges=l.top_edges[:6], status=l.status)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ablation] lead-lag unavailable: {exc}")

    conf_exp = pd.Series(1.0, index=idx)
    try:
        from .ml.conformal import conformal_exposure
        c = conformal_exposure(panel, cfg)
        conf_exp = c.exposure_scale.reindex(idx).fillna(1.0).clip(0.0, 1.0)
        diagnostics["conformal"] = dict(empirical_coverage=c.empirical_coverage,
                                        mean_exposure=c.mean_exposure,
                                        n_refits=c.n_refits, status=c.status)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ablation] conformal unavailable: {exc}")

    return directions, conf_exp, diagnostics


def run_ablation(panel: Panel, cfg: Config) -> dict:
    """
    Run every (direction × {plain, +conformal}) configuration plus benchmarks,
    out-of-sample, and bundle metrics, equity curves, walk-forward stability,
    diagnostics and a cost-sensitivity sweep.
    """
    directions, conf_exp, diagnostics = compute_signals(panel, cfg)
    cost = CostModel.from_config(cfg, multiplier=1.0)

    configs: dict[str, tuple[pd.DataFrame, pd.Series | None]] = {}
    for name, d in directions.items():
        configs[name] = (d, None)
        configs[f"{name} +conformal"] = (d, conf_exp)
    n_trials = len(configs)

    bench = None
    rows, equity, wf = [], {}, {}
    for name, (d, exp) in configs.items():
        res = run_backtest(panel, d, cfg, cost, exposure_scale=exp)
        if bench is None:
            bench = res.benchmarks
        s = perf.summary(res.net_returns, benchmark=bench["equal_weight"], n_trials=n_trials)
        s["variant"] = name
        s["avg_leverage"] = float(res.leverage.mean())
        s["ann_turnover"] = float(res.turnover.mean() * perf.TRADING_DAYS)
        rows.append(s)
        equity[name] = res.equity
        wf[name] = walk_forward_report(res.net_returns, cfg)

    for bname, bret in bench.items():
        s = perf.summary(bret, n_trials=1)
        s["variant"] = f"[bench] {bname}"
        s["avg_leverage"] = 1.0
        s["ann_turnover"] = float("nan")
        rows.append(s)
        equity[f"[bench] {bname}"] = (1.0 + bret.fillna(0.0)).cumprod()

    variants = pd.DataFrame(rows).set_index("variant")
    cost_sweep = run_cost_sweep(panel, cfg, directions["baseline"], conf_exp)

    return {
        "variants": variants,
        "equity": equity,
        "walk_forward": wf,
        "diagnostics": diagnostics,
        "cost_sweep": cost_sweep,
        "n_trials": n_trials,
    }


def run_cost_sweep(panel: Panel, cfg: Config, direction: pd.DataFrame,
                   exposure: pd.Series | None) -> pd.DataFrame:
    """
    Re-run the baseline plain and conformal-gated across the configured cost
    multipliers, recording annualized Sharpe and return — the robustness curve.
    """
    rows = []
    for mult in cfg.costs.sweep_multipliers:
        cm = CostModel.from_config(cfg, multiplier=float(mult))
        base = run_backtest(panel, direction, cfg, cm)
        gated = run_backtest(panel, direction, cfg, cm, exposure_scale=exposure)
        rows.append({
            "cost_multiplier": float(mult),
            "per_side_bps": cm.per_side_bps,
            "baseline_sharpe": perf.sharpe_ratio(base.net_returns),
            "baseline_ann_return": perf.annualized_return(base.net_returns),
            "gated_sharpe": perf.sharpe_ratio(gated.net_returns),
            "gated_ann_return": perf.annualized_return(gated.net_returns),
        })
    return pd.DataFrame(rows).set_index("cost_multiplier")
