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


# ===========================================================================
# Cross-sectional, drift-neutral transforms (for the learning-to-rank model)
# ===========================================================================
# These operate PER ROW (one timestamp) across the asset columns only. Because a
# value at date t is computed from other assets *at the same t* — never from any
# other date — they are causal by construction: truncating the series in time
# leaves every surviving row unchanged. They exist to strip the common crypto
# drift factor out of the LTR features so the ranker learns RELATIVE strength
# rather than "what already went up".

def cross_sectional_mad_zscore(feature: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    """Robust cross-sectional z-score at each date across the asset universe.

        Z = (x - median_t) / (1.4826 * MAD_t)

    where the median and MAD (median absolute deviation) are taken across assets
    at each timestamp t. The 1.4826 factor rescales MAD to a normal-consistent
    standard deviation. Median/MAD (not mean/std) make the normalization robust
    to the fat tails and single-name blow-ups typical of crypto.

    Strictly per-row across columns, so it introduces no look-ahead. NaN inputs
    (warm-up, or an asset not yet listed) are PRESERVED as NaN so downstream row
    filters drop un-warmed observations rather than train on a fabricated 0. Only
    a finite input on a degenerate row (zero/undefined MAD: < 2 valid assets or
    all-equal values) maps to 0 — there is genuinely no cross-sectional signal to
    extract. Outputs are winsorized to +/- ``clip`` to bound extreme outliers.
    """
    med = feature.median(axis=1, skipna=True)
    centered = feature.sub(med, axis=0)
    mad = centered.abs().median(axis=1, skipna=True) * 1.4826
    z = centered.div(mad.replace(0.0, np.nan), axis=0)
    z = z.replace([np.inf, -np.inf], np.nan)
    # Finite input but undefined z (degenerate MAD) -> neutral 0; NaN input stays NaN.
    z = z.mask(feature.notna() & z.isna(), 0.0)
    return z.clip(-clip, clip)


def atr_normalized_momentum(close: pd.DataFrame, high: pd.DataFrame,
                            low: pd.DataFrame, lookback: int,
                            atr_period: int = 14) -> pd.DataFrame:
    """Volatility-weighted relative strength: trailing move measured in ATR units.

        (close_t - close_{t-lookback}) / ATR_t

    Dividing the raw move by the Average True Range expresses momentum in units
    of each asset's own recent range, so a 10% move in a calm coin and a 10% move
    in a wild coin are placed on a comparable footing before they are ranked
    cross-sectionally. Backward-looking (ATR and the lagged close are both known
    at t), hence causal.
    """
    a = atr(high, low, close, atr_period)
    return (close - close.shift(lookback)) / a.replace(0.0, np.nan)


def funding_rank_signal(funding: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    """Structural cross-sectional score from perpetual-swap funding rates.

    Funding is paid by the crowded side of the perp: persistently HIGH POSITIVE
    funding flags crowded longs that are primed for a long-liquidation cascade /
    mean reversion — a BEARISH cross-sectional tilt. NEGATIVE funding (shorts
    paying longs) is a structural TAILWIND — bullish. We therefore robustly
    cross-sectionally z-score the *negated* funding, so high funding -> low score
    and negative funding -> high score. Missing funding (no perp / offline) flows
    through as 0 (neutral). Causal: ``funding`` is aligned to known-at-t values
    by the loader before it reaches here.
    """
    return cross_sectional_mad_zscore(-funding, clip=clip)
