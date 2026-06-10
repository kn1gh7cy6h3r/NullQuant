"""
backtest.py — long/short, vol-targeted, multi-asset backtest.

Pipeline (all causal — a weight applied to the return over [t-1, t] is decided
using information available no later than t-1):

  1. direction[t] in {-1,0,+1} per asset (from signals), held between rebalances.
  2. vol-parity raw weights: w_raw = direction / asset_vol, normalized to gross 1.
     (inverse-vol so a calm coin and a wild coin contribute comparable risk.)
  3. portfolio vol targeting: scale the whole book by target_vol / trailing
     realized vol of the gross-1 strategy, capped at max_gross_leverage.
  4. optional exposure_scale[t] in [0,1] — the regime filter / meta-label gate
     can de-risk the book on a given day.
  5. per-asset weight cap and gross-leverage cap.
  6. net return = sum_i w[t-1]*r_i[t]  -  turnover_cost[t].

Returns a BacktestResult bundling the net/gross series, equity curve, the
actually-held weights, turnover, and costs, plus equal-weight and BTC
buy-and-hold benchmarks for honest comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import Panel
from ..features import indicators as ind
from .costs import CostModel

TRADING_DAYS = 365  # crypto trades every day


@dataclass
class BacktestResult:
    net_returns: pd.Series      # daily net simple returns of the strategy
    gross_returns: pd.Series    # before costs
    equity: pd.Series           # cumulative net equity (starts at 1.0)
    weights: pd.DataFrame       # actually-held weights per asset per day
    turnover: pd.Series         # sum |Δw| per day
    costs: pd.Series            # daily return drag from costs
    asset_returns: pd.DataFrame
    benchmarks: dict[str, pd.Series]  # name -> daily returns
    leverage: pd.Series         # gross leverage per day (sum |w|)


def _normalize_gross(w: pd.DataFrame) -> pd.DataFrame:
    """Scale each row so sum|w| == 1 (gross 1); all-zero rows stay zero."""
    gross = w.abs().sum(axis=1)
    gross = gross.replace(0.0, np.nan)
    return w.div(gross, axis=0).fillna(0.0)


def _apply_caps(w: pd.DataFrame, max_asset: float, max_gross: float) -> pd.DataFrame:
    """Cap per-asset |weight|, then cap total gross leverage."""
    w = w.clip(lower=-max_asset, upper=max_asset)
    gross = w.abs().sum(axis=1)
    scale = (max_gross / gross).clip(upper=1.0).replace([np.inf, np.nan], 1.0)
    return w.mul(scale, axis=0)


def run_backtest(
    panel: Panel,
    direction: pd.DataFrame,
    cfg: Config,
    cost_model: CostModel,
    exposure_scale: pd.Series | None = None,
) -> BacktestResult:
    close = panel.close
    s, risk = cfg.strategy, cfg.risk

    asset_ret = close.pct_change()
    asset_vol = ind.realized_vol(close, s.vol_lookback, annualize=True)

    # 1) Vol-parity raw weights from direction, rebalanced on the configured
    #    cadence and held in between.
    raw = direction / asset_vol.replace(0.0, np.nan).clip(lower=risk.vol_floor)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw = _normalize_gross(raw)

    rebal_dates = set(close.resample(s.rebalance).last().index)
    raw.loc[[d not in rebal_dates for d in raw.index], :] = np.nan
    w_base = raw.ffill().fillna(0.0)

    # 2) Portfolio vol targeting from the gross-1 strategy's trailing vol.
    r_base = (w_base.shift(1) * asset_ret).sum(axis=1)
    trail_vol = r_base.rolling(s.vol_lookback, min_periods=s.vol_lookback).std() * np.sqrt(TRADING_DAYS)
    scale = (risk.target_portfolio_vol / trail_vol.replace(0.0, np.nan))
    scale = scale.shift(1).clip(upper=risk.max_gross_leverage).fillna(0.0)

    w = w_base.mul(scale, axis=0)

    # 3) Optional daily exposure gate (regime filter / meta-label), in [0, 1].
    if exposure_scale is not None:
        w = w.mul(exposure_scale.reindex(w.index).fillna(1.0).clip(0.0, 1.0), axis=0)

    # 4) Caps.
    w = _apply_caps(w, risk.max_asset_weight, risk.max_gross_leverage)

    # 5) Returns and costs. Weight decided at t-1 earns the return over [t-1, t].
    gross_returns = (w.shift(1) * asset_ret).sum(axis=1)
    costs = cost_model.turnover_cost(w)
    net_returns = (gross_returns - costs).fillna(0.0)

    equity = (1.0 + net_returns).cumprod()
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    leverage = w.abs().sum(axis=1)

    benchmarks = _benchmarks(panel)

    return BacktestResult(
        net_returns=net_returns,
        gross_returns=gross_returns.fillna(0.0),
        equity=equity,
        weights=w,
        turnover=turnover,
        costs=costs,
        asset_returns=asset_ret,
        benchmarks=benchmarks,
        leverage=leverage,
    )


def _benchmarks(panel: Panel) -> dict[str, pd.Series]:
    """Equal-weight (rebalanced daily) long-only basket, and BTC buy-and-hold."""
    asset_ret = panel.close.pct_change()
    avail = panel.close.notna()
    # Equal-weight across currently-listed assets each day.
    ew_weights = avail.div(avail.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    ew = (ew_weights.shift(1) * asset_ret).sum(axis=1).fillna(0.0)
    out = {"equal_weight": ew}
    if "BTC-USD" in panel.close.columns:
        out["btc_hold"] = asset_ret["BTC-USD"].fillna(0.0)
    return out
