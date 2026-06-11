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

import hashlib
import json
import pickle

import pandas as pd

from .config import Config, PROJECT_ROOT
from .data.loader import Panel
from .signals.base import target_directions
from .portfolio.costs import CostModel
from .portfolio.backtest import run_backtest
from .metrics import performance as perf
from .validation.walk_forward import walk_forward_report

# The four ML models are the only expensive part of a run (walk-forward refits).
# We cache their output keyed on a fingerprint of (config + price data) so a
# second process — e.g. the dashboard launched after the pipeline — reuses the
# fit instead of retraining. The fingerprint changes (and the cache is rebuilt)
# whenever the config or the underlying prices change.
_SIGNALS_CACHE_DIR = PROJECT_ROOT / "research" / "results" / "cache"


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


def _signals_fingerprint(panel: Panel, cfg: Config) -> str:
    """Stable short hash of everything that affects the ML signals: the full
    config, the universe, the date span, and the close prices themselves (so a
    data revision or a fresh day invalidates the cache)."""
    h = hashlib.sha256()
    h.update(json.dumps(cfg.raw, sort_keys=True, default=str).encode())
    h.update(",".join(panel.assets).encode())
    idx = panel.close.index
    h.update(f"{idx[0]}|{idx[-1]}|{len(idx)}".encode())
    h.update(pd.util.hash_pandas_object(panel.close.fillna(0.0), index=True)
             .values.tobytes())
    return h.hexdigest()[:16]


def compute_signals_cached(panel: Panel, cfg: Config, *,
                           use_cache: bool = True, refresh: bool = False
                           ) -> tuple[dict[str, pd.DataFrame], pd.Series, dict]:
    """`compute_signals` with an on-disk cache keyed on (config + data).

    Reuses a previous fit when the fingerprint matches, so the dashboard (or a
    re-run of the pipeline) skips the walk-forward refits entirely. Pass
    ``refresh=True`` to force a recompute, or ``use_cache=False`` to bypass.
    """
    if not use_cache:
        return compute_signals(panel, cfg)

    fp = _signals_fingerprint(panel, cfg)
    path = _SIGNALS_CACHE_DIR / f"signals_{fp}.pkl"
    if path.exists() and not refresh:
        try:
            with open(path, "rb") as fh:
                directions, conf_exp, diagnostics = pickle.load(fh)
            print(f"[ablation] loaded cached ML signals ({fp}) — skipping refits")
            return directions, conf_exp, diagnostics
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[ablation] signal cache unreadable ({exc}); recomputing")

    directions, conf_exp, diagnostics = compute_signals(panel, cfg)
    try:
        _SIGNALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump((directions, conf_exp, diagnostics), fh)
        print(f"[ablation] cached ML signals -> {path.name}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ablation] could not cache signals ({exc})")
    return directions, conf_exp, diagnostics


def run_ablation(panel: Panel, cfg: Config, *, refresh_signals: bool = False) -> dict:
    """
    Run every (direction × {plain, +conformal}) configuration plus benchmarks,
    out-of-sample, and bundle metrics, equity curves, walk-forward stability,
    diagnostics and a cost-sensitivity sweep.
    """
    directions, conf_exp, diagnostics = compute_signals_cached(
        panel, cfg, refresh=refresh_signals)
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
