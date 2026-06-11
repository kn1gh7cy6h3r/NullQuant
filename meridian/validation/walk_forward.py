"""
walk_forward.py — out-of-sample evaluation of the *strategy* over rolling windows.

The strategy's parameters are fixed by config, so walk-forward here is not about
re-fitting parameters; it answers the question a reviewer actually cares about:
"is the performance stable through time, or is the full-sample Sharpe carried by
one lucky regime?" We slice the realized net-return stream into the walk-forward
test windows and report per-window stats plus the stitched out-of-sample record.

This consumes the splitters in splitters.py and the metrics in
metrics.performance, and is deliberately independent of any ML model so it can
be run on the baseline strategy on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..metrics import performance as perf
from .splitters import walk_forward_splits


def walk_forward_report(net_returns: pd.Series, cfg: Config) -> dict:
    """
    Slice `net_returns` into the configured walk-forward test windows and
    summarise each, plus the concatenated out-of-sample series.

    Returns a dict with:
      windows   : DataFrame (one row per test window) of start, end, n, ann_return,
                  sharpe, max_drawdown
      oos        : performance.summary() over all test windows stitched together,
                  with n_trials = number of windows (a mild deflation acknowledging
                  that each window is effectively a separate look).
      stability  : fraction of windows with a positive Sharpe, and the mean/std of
                  window Sharpes (the dispersion is what feeds Deflated Sharpe).
    """
    wf = cfg.validation.walk_forward
    splits = walk_forward_splits(
        net_returns.index,
        train_years=int(wf.train_years),
        test_months=int(wf.test_months),
        anchored=bool(wf.anchored),
    )

    rows = []
    oos_pieces = []
    for sp in splits:
        seg = net_returns.reindex(sp.test).dropna()
        if seg.empty:
            continue
        oos_pieces.append(seg)
        rows.append({
            "start": sp.test.min(),
            "end": sp.test.max(),
            "n": int(seg.size),
            "ann_return": perf.annualized_return(seg),
            "sharpe": perf.sharpe_ratio(seg),
            "max_drawdown": perf.max_drawdown(seg),
        })

    windows = pd.DataFrame(rows)
    if windows.empty:
        return {"windows": windows, "oos": {}, "stability": {}}

    oos_returns = pd.concat(oos_pieces).sort_index()
    window_sharpes = windows["sharpe"].to_numpy()
    # Per-period (daily) std of window Sharpes, de-annualized, for Deflated Sharpe.
    trial_std = float(np.std(window_sharpes / np.sqrt(perf.TRADING_DAYS), ddof=1)) \
        if len(window_sharpes) > 1 else None

    oos_summary = perf.summary(
        oos_returns, n_trials=len(windows), trial_sharpe_std=trial_std)

    stability = {
        "n_windows": int(len(windows)),
        "pct_positive_sharpe": float((windows["sharpe"] > 0).mean()),
        "mean_window_sharpe": float(windows["sharpe"].mean()),
        "std_window_sharpe": float(windows["sharpe"].std(ddof=1)) if len(windows) > 1 else 0.0,
    }
    return {"windows": windows, "oos": oos_summary, "stability": stability}
