"""
test_backtest.py — accounting identity tests for nullquant.portfolio.backtest.run_backtest.

Tests:
  1. equity == (1 + net_returns).cumprod() (within float tolerance).
  2. On days with turnover > 0, net_returns <= gross_returns (costs are positive).
  3. Gross leverage never exceeds cfg.risk.max_gross_leverage + epsilon.
  4. Per-asset |weight| never exceeds cfg.risk.max_asset_weight + epsilon.

We use a hand-built direction matrix so we don't depend on the SMA200 warm-up
from target_directions (the synthetic panel is 420 rows; SMA200 takes 200 rows
to warm up, so there would be very few or zero signals on a 3-asset universe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nullquant.portfolio.backtest import run_backtest
from nullquant.portfolio.costs import CostModel

EPSILON = 1e-8


def _manual_direction(panel, seed: int = 99) -> pd.DataFrame:
    """
    Build a simple {-1, 0, 1} direction DataFrame of the same shape as
    panel.close, with a sparse but non-trivial pattern — enough turnover to
    exercise cost deduction.
    """
    rng = np.random.default_rng(seed)
    close = panel.close
    n, m = close.shape
    # Every 7 days, randomly assign {-1, 0, 1} to each asset.
    values = np.zeros((n, m), dtype=float)
    choices = [-1.0, 0.0, 1.0]
    for i in range(0, n, 7):
        row = rng.choice(choices, size=m)
        end = min(i + 7, n)
        values[i:end, :] = row
    return pd.DataFrame(values, index=close.index, columns=close.columns)


# ---------------------------------------------------------------------------
# Fixtures local to this module (supplement the session fixtures in conftest.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backtest_result(panel, cfg):
    """Run a single backtest and cache the result for the module."""
    cost_model = CostModel.from_config(cfg)
    direction = _manual_direction(panel)
    return run_backtest(panel, direction, cfg, cost_model)


# ---------------------------------------------------------------------------
# Test 1: equity identity
# ---------------------------------------------------------------------------

class TestEquityIdentity:
    def test_equity_equals_cumprod(self, backtest_result):
        """equity == (1 + net_returns).cumprod() element-wise."""
        result = backtest_result
        expected = (1.0 + result.net_returns).cumprod()
        np.testing.assert_allclose(
            result.equity.values,
            expected.values,
            rtol=1e-9,
            atol=1e-12,
            err_msg="equity curve must equal (1+net_returns).cumprod()",
        )

    def test_equity_starts_near_one(self, backtest_result):
        """The first valid equity value should be close to 1."""
        eq = backtest_result.equity.dropna()
        assert abs(eq.iloc[0] - 1.0) < 0.1, (
            f"Equity curve should start near 1.0, got {eq.iloc[0]}"
        )


# ---------------------------------------------------------------------------
# Test 2: positive costs reduce net vs gross
# ---------------------------------------------------------------------------

class TestCostDeduction:
    def test_net_le_gross_on_turnover_days(self, backtest_result):
        """On days with non-zero turnover, net_returns must be <= gross_returns."""
        result = backtest_result
        turnover_days = result.turnover > 0.0
        if not turnover_days.any():
            pytest.skip("No turnover days found — cannot test cost deduction")

        net = result.net_returns[turnover_days]
        gross = result.gross_returns[turnover_days]
        # Allow a tiny float tolerance in case of numerical rounding
        assert (net.values <= gross.values + EPSILON).all(), (
            "net_returns must be <= gross_returns on turnover days (costs are positive)"
        )

    def test_zero_cost_model_equals_gross(self, panel, cfg):
        """With a zero-cost model, net_returns should equal gross_returns."""
        cost_model = CostModel(
            taker_fee_bps=0.0, slippage_bps=0.0, spread_bps=0.0, multiplier=0.0
        )
        direction = _manual_direction(panel)
        result = run_backtest(panel, direction, cfg, cost_model)
        np.testing.assert_allclose(
            result.net_returns.values,
            result.gross_returns.values,
            atol=1e-12,
            err_msg="Zero-cost model must give net == gross returns",
        )


# ---------------------------------------------------------------------------
# Test 3: leverage cap respected
# ---------------------------------------------------------------------------

class TestLeverageCap:
    def test_leverage_within_cap(self, backtest_result, cfg):
        """Gross leverage must never exceed max_gross_leverage + epsilon."""
        max_lev = float(cfg.risk.max_gross_leverage)
        lev = backtest_result.leverage
        exceeded = lev[lev > max_lev + EPSILON]
        assert exceeded.empty, (
            f"Leverage exceeded cap on {len(exceeded)} days; "
            f"max cap={max_lev}, worst={lev.max():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 4: per-asset weight cap respected
# ---------------------------------------------------------------------------

class TestAssetWeightCap:
    def test_per_asset_weight_within_cap(self, backtest_result, cfg):
        """Per-asset |weight| must not exceed max_asset_weight + epsilon."""
        max_w = float(cfg.risk.max_asset_weight)
        abs_w = backtest_result.weights.abs()
        violating = abs_w[abs_w > max_w + EPSILON].stack().dropna()
        assert violating.empty, (
            f"|weight| exceeded cap for {len(violating)} (date, asset) pairs; "
            f"cap={max_w}, worst={abs_w.max().max():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 5: BacktestResult has all expected fields
# ---------------------------------------------------------------------------

class TestBacktestResultShape:
    def test_result_fields_present(self, backtest_result, panel):
        """BacktestResult must expose the documented attributes."""
        r = backtest_result
        n = len(panel.close)
        assert len(r.net_returns) == n
        assert len(r.gross_returns) == n
        assert len(r.equity) == n
        assert r.weights.shape == panel.close.shape
        assert len(r.turnover) == n
        assert len(r.costs) == n
        assert isinstance(r.benchmarks, dict)
        assert "equal_weight" in r.benchmarks

    def test_weights_sum_consistent_with_leverage(self, backtest_result):
        """leverage == weights.abs().sum(axis=1) by definition."""
        r = backtest_result
        computed_lev = r.weights.abs().sum(axis=1)
        np.testing.assert_allclose(
            r.leverage.values,
            computed_lev.values,
            rtol=1e-9,
            err_msg="leverage must equal weights.abs().sum(axis=1)",
        )
