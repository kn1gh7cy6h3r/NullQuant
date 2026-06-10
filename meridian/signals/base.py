"""
base.py — trading signals: SMA crossover trigger + cross-sectional overlay.

Two layers, combined:

  1. Trend filter (the kept legacy idea). Per asset, the SMA50/SMA200 relation
     defines a trend regime: +1 when the fast SMA is above the slow (uptrend),
     -1 below (downtrend). Golden/death crosses are the *events* where this
     flips — these are the entries the RF meta-labeler scores.

  2. Cross-sectional overlay (the upgrade). On each rebalance date we rank the
     universe by risk-adjusted momentum and go LONG the strongest names that are
     also in an uptrend, SHORT the weakest names that are in a downtrend. This
     turns a single-asset trend toy into a market-neutral-ish basket strategy.

All quantities are causal: the direction targeted for the holding period
starting at date t is computed from data available at t. The backtest then
applies that direction from t+1, so there is no look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import Panel
from ..features import indicators as ind


def trend_state(close: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """+1 where SMA(short) > SMA(long), -1 where below, NaN during warm-up."""
    s = ind.sma(close, short)
    l = ind.sma(close, long)
    state = np.sign(s - l)
    return state.where(s.notna() & l.notna())


def crossover_events(close: pd.DataFrame, short: int, long: int
                     ) -> dict[str, dict[str, pd.DatetimeIndex]]:
    """
    Detect golden (long) and death (short) crosses per asset.

    Returns {asset: {"long": DatetimeIndex, "short": DatetimeIndex}} — the entry
    dates fed to triple-barrier meta-labeling.
    """
    state = trend_state(close, short, long)
    prev = state.shift(1)
    golden = (prev < 0) & (state > 0)
    death = (prev > 0) & (state < 0)
    out: dict[str, dict[str, pd.DatetimeIndex]] = {}
    for asset in close.columns:
        out[asset] = {
            "long": golden.index[golden[asset].fillna(False)],
            "short": death.index[death[asset].fillna(False)],
        }
    return out


def target_directions(panel: Panel, cfg: Config) -> pd.DataFrame:
    """
    Build the daily target-direction matrix in {-1, 0, +1} per asset.

    On each rebalance date:
      • rank assets by risk-adjusted momentum,
      • LONG the top_k that are in an uptrend (trend_state > 0),
      • SHORT the bottom_k that are in a downtrend (trend_state < 0),
      • flat otherwise.
    Directions are held (forward-filled) until the next rebalance.
    """
    close = panel.close
    s = cfg.strategy

    state = trend_state(close, s.sma_short, s.sma_long)
    ras = ind.risk_adjusted_momentum(close, s.momentum_lookback, s.vol_lookback)

    # Rebalance dates (e.g. weekly) — only those with enough history.
    rebal_dates = close.resample(s.rebalance).last().index
    rebal_dates = [d for d in rebal_dates if d in close.index]

    direction = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    for d in rebal_dates:
        row_score = ras.loc[d]
        row_state = state.loc[d]
        valid = row_score.notna() & row_state.notna()
        if valid.sum() == 0:
            continue
        ranked = row_score[valid].sort_values(ascending=False)

        longs, shorts = [], []
        for asset in ranked.index:
            if row_state[asset] > 0 and len(longs) < s.top_k:
                longs.append(asset)
        for asset in ranked.index[::-1]:
            if row_state[asset] < 0 and len(shorts) < s.bottom_k:
                shorts.append(asset)

        direction.loc[d, :] = 0.0
        if longs:
            direction.loc[d, longs] = 1.0
        if shorts:
            direction.loc[d, shorts] = -1.0

    # Hold each rebalance's directions until the next rebalance. Rows that are
    # not rebalance dates are blanked then forward-filled from the last decision.
    rebal_set = set(rebal_dates)
    direction.loc[[d not in rebal_set for d in direction.index], :] = np.nan
    return direction.ffill().fillna(0.0)
