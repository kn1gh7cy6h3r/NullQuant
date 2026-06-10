"""
indicators.py — strictly causal, vectorized indicators across the panel.

Every function returns a DataFrame aligned to the input (index=dates,
columns=assets) and uses ONLY past/current information at each timestamp:
rolling windows look backward, and nothing here calls .shift(-k) or otherwise
peeks forward. The one rule for using these in a backtest is enforced upstream:
a value computed at date t may only drive a position held from t+1 onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average; NaN until `window` observations exist."""
    return prices.rolling(window, min_periods=window).mean()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1))


def realized_vol(prices: pd.DataFrame, window: int, annualize: bool = True) -> pd.DataFrame:
    """Rolling realized volatility of daily log returns (annualized by default)."""
    r = log_returns(prices)
    vol = r.rolling(window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(365.0)   # crypto trades 365 days/yr
    return vol


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
        period: int = 14) -> pd.DataFrame:
    """
    Wilder's Average True Range per asset. True Range uses the previous close,
    so it captures overnight gaps. Computed column-by-column to keep the
    per-asset previous-close alignment exact.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=0,
    )
    # The concat above stacks; instead compute elementwise max across the three.
    tr = pd.DataFrame(
        np.maximum.reduce([
            (high - low).abs().values,
            (high - prev_close).abs().values,
            (low - prev_close).abs().values,
        ]),
        index=close.index, columns=close.columns,
    )
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder's RSI on closes, per asset, on a 0–100 scale."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.where(avg_loss != 0, 100.0)


def momentum(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing total return over `lookback` days (the raw momentum signal)."""
    return prices / prices.shift(lookback) - 1.0


def risk_adjusted_momentum(prices: pd.DataFrame, lookback: int,
                           vol_window: int) -> pd.DataFrame:
    """
    Momentum scaled by realized volatility — the cross-sectional ranking signal.
    Dividing by vol puts assets of different volatility on a comparable footing,
    which is what makes a cross-sectional rank meaningful.
    """
    mom = momentum(prices, lookback)
    vol = realized_vol(prices, vol_window, annualize=False)
    return mom / vol.replace(0.0, np.nan)


def cross_sectional_rank(score: pd.DataFrame) -> pd.DataFrame:
    """
    Per-date rank of each asset's score in [0, 1] (1 = strongest), ignoring NaN.
    Used to pick the long (top) and short (bottom) legs each rebalance.
    """
    return score.rank(axis=1, pct=True)
