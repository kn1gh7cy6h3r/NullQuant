"""
costs.py — transaction-cost model.

A backtest without costs is not a backtest. Every change in target weight
implies trading notional, and that notional pays:

    • exchange taker fee       (a real, quoted number)
    • half the bid/ask spread  (crossing the book)
    • slippage                 (market impact / walking the book)

We charge the full stack on the *turnover* |Δw| at each rebalance, expressed in
basis points of the traded notional and converted to a portfolio return drag.
The whole stack is scaled by one multiplier so the sensitivity sweep can ask the
key question: how much edge survives 0×, 1×, 2×, 4× costs?
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import Config


@dataclass
class CostModel:
    taker_fee_bps: float
    slippage_bps: float
    spread_bps: float
    multiplier: float = 1.0

    @classmethod
    def from_config(cls, cfg: Config, multiplier: float = 1.0) -> "CostModel":
        c = cfg.costs
        return cls(
            taker_fee_bps=float(c.taker_fee_bps),
            slippage_bps=float(c.slippage_bps),
            spread_bps=float(c.spread_bps),
            multiplier=float(multiplier),
        )

    @property
    def per_side_bps(self) -> float:
        """Total cost charged per unit of traded notional, in basis points."""
        return (self.taker_fee_bps + self.slippage_bps + self.spread_bps) * self.multiplier

    def turnover_cost(self, weights: pd.DataFrame) -> pd.Series:
        """
        Per-date portfolio return drag from rebalancing to `weights`.

        weights: DataFrame (index=dates, columns=assets) of *target* weights
        actually held each day. Turnover at t is sum_i |w[t] - w[t-1]|; the cost
        is turnover * per_side_bps / 1e4, returned as a positive daily drag.
        """
        prev = weights.shift(1).fillna(0.0)
        turnover = (weights - prev).abs().sum(axis=1)
        return turnover * (self.per_side_bps / 1e4)
