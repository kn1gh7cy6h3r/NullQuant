"""
performance.py — performance & risk statistics for a return stream.

Beyond the usual annualized return / vol / Sharpe / drawdown, this module
implements the two statistics that separate a credible result from a lucky one:

  • Probabilistic Sharpe Ratio (PSR) — the probability that the true Sharpe
    exceeds a benchmark, correcting for track-record length, skew and kurtosis
    (Bailey & Lopez de Prado, 2012).

  • Deflated Sharpe Ratio (DSR) — PSR against a benchmark that accounts for the
    number of strategy configurations tried. Selecting the best of N backtests
    inflates Sharpe; DSR deflates it back. This is the antidote to the
    backtest-overfitting that every serious reviewer is on guard for.

All ratios are computed on per-period (daily) returns and annualized only for
display, via sqrt(365) since crypto trades every day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

TRADING_DAYS = 365
EULER_GAMMA = 0.5772156649015329


def _clean(returns: pd.Series) -> np.ndarray:
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    return r.to_numpy(dtype=float)


def annualized_return(returns: pd.Series) -> float:
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    growth = np.prod(1.0 + r)
    years = r.size / TRADING_DAYS
    if years <= 0 or growth <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def annualized_vol(returns: pd.Series) -> float:
    r = _clean(returns)
    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)) if r.size > 1 else 0.0


def sharpe_ratio(returns: pd.Series, annualize: bool = True) -> float:
    r = _clean(returns)
    if r.size < 2 or np.std(r, ddof=1) == 0:
        return 0.0
    sr = np.mean(r) / np.std(r, ddof=1)
    return float(sr * np.sqrt(TRADING_DAYS)) if annualize else float(sr)


def sortino_ratio(returns: pd.Series) -> float:
    r = _clean(returns)
    downside = r[r < 0]
    if downside.size < 1 or np.std(downside, ddof=1) == 0:
        return 0.0
    return float(np.mean(r) / np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    return float(annualized_return(returns) / mdd) if mdd > 0 else 0.0


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """
    P(true non-annualized Sharpe > sr_benchmark), correcting for length, skew,
    kurtosis. sr_benchmark is a *per-period* Sharpe (use 0 for "better than
    nothing", or a deflated threshold for DSR).
    """
    r = _clean(returns)
    if r.size < 3:
        return 0.0
    sr = sharpe_ratio(r, annualize=False)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))  # normal => 3
    denom = np.sqrt(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2)
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(r.size - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          trial_sharpe_std: float | None = None) -> float:
    """
    Deflated Sharpe Ratio. Builds the expected-maximum-Sharpe benchmark from the
    number of trials and the dispersion of trial Sharpes, then returns PSR
    against it. trial_sharpe_std is the std (per-period) of the Sharpes across
    the N configurations tried; if unknown we fall back to the estimator's own
    standard error, which is conservative.
    """
    r = _clean(returns)
    if r.size < 3 or n_trials < 1:
        return 0.0
    sr = sharpe_ratio(r, annualize=False)
    if trial_sharpe_std is None or trial_sharpe_std <= 0:
        g3 = float(skew(r))
        g4 = float(kurtosis(r, fisher=False))
        var_sr = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2) / (r.size - 1)
        trial_sharpe_std = float(np.sqrt(max(var_sr, 1e-12)))
    n = max(int(n_trials), 1)
    if n == 1:
        sr0 = 0.0
    else:
        z1 = norm.ppf(1.0 - 1.0 / n)
        z2 = norm.ppf(1.0 - 1.0 / n * np.e ** -1)
        sr0 = trial_sharpe_std * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    return probabilistic_sharpe_ratio(r, sr_benchmark=sr0)


def bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 1000,
                        alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for the annualized Sharpe ratio."""
    r = _clean(returns)
    if r.size < 10:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(r, size=r.size, replace=True)
        sd = np.std(sample, ddof=1)
        stats[i] = (np.mean(sample) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def summary(returns: pd.Series, benchmark: pd.Series | None = None,
            n_trials: int = 1, trial_sharpe_std: float | None = None) -> dict:
    """One-stop performance summary for a return stream."""
    r = pd.Series(returns)
    out = {
        "ann_return": annualized_return(r),
        "ann_vol": annualized_vol(r),
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_ratio(r),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r),
        "psr_vs_zero": probabilistic_sharpe_ratio(r, 0.0),
        "deflated_sharpe": deflated_sharpe_ratio(r, n_trials, trial_sharpe_std),
        "n_obs": int(_clean(r).size),
    }
    lo, hi = bootstrap_sharpe_ci(r)
    out["sharpe_ci_low"], out["sharpe_ci_high"] = lo, hi

    if benchmark is not None:
        b = pd.Series(benchmark).reindex(r.index)
        excess = (r - b).dropna()
        out["excess_ann_return"] = annualized_return(excess)
        out["information_ratio"] = sharpe_ratio(excess)
    return out
