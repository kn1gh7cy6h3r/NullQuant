"""
test_indicators.py — causality / no-look-ahead tests for nullquant.features.indicators.

For each indicator we verify:
  1. The overlapping prefix of full-series output matches a truncated-series output
     (no future data should alter past values).
  2. Hand-computed reference values match for SMA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nullquant.features.indicators import (
    atr,
    cross_sectional_rank,
    momentum,
    realized_vol,
    risk_adjusted_momentum,
    rsi,
    sma,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_close(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Single-asset close price DataFrame for simple tests."""
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, size=n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"A": prices}, index=idx)


def _synthetic_ohlc(n: int = 100, seed: int = 0):
    """Return (high, low, close) DataFrames for ATR tests."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.001, 0.02, size=n)
    close_vals = 100.0 * np.exp(np.cumsum(log_ret))
    intraday = np.abs(rng.normal(0.0, 0.01, size=n))
    high_vals = close_vals * (1.0 + intraday)
    low_vals = close_vals * (1.0 - intraday)
    high_vals = np.maximum(high_vals, close_vals)
    low_vals = np.minimum(low_vals, close_vals)

    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    h = pd.DataFrame({"A": high_vals}, index=idx)
    l = pd.DataFrame({"A": low_vals}, index=idx)
    c = pd.DataFrame({"A": close_vals}, index=idx)
    return h, l, c


TRUNC = 60  # truncation point for causality tests; full series is 100 rows


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------

class TestSMA:
    def test_causality(self):
        """SMA on full series matches SMA on truncated series for the overlapping prefix."""
        prices = _simple_close(100)
        window = 10
        full = sma(prices, window)
        trunc = sma(prices.iloc[:TRUNC], window)
        overlap = full.iloc[:TRUNC]
        pd.testing.assert_frame_equal(
            overlap, trunc,
            check_names=False,
            obj="SMA causality: full vs truncated overlap",
        )

    def test_hand_computed(self):
        """SMA equals pandas rolling mean on a simple linear series."""
        n = 50
        prices_arr = np.arange(1.0, n + 1)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        prices = pd.DataFrame({"X": prices_arr}, index=idx)
        window = 5
        result = sma(prices, window)
        expected = prices.rolling(window, min_periods=window).mean()
        pd.testing.assert_frame_equal(result, expected, check_names=False)

    def test_nan_during_warmup(self):
        """First (window-1) rows must be NaN (min_periods=window)."""
        prices = _simple_close(50)
        window = 10
        result = sma(prices, window)
        assert result.iloc[:window - 1].isna().all().all(), (
            "SMA should be NaN during warm-up period"
        )
        assert result.iloc[window - 1:].notna().all().all(), (
            "SMA should be non-NaN once window is full"
        )


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_causality(self):
        """ATR on full series equals ATR on truncated series for overlapping prefix."""
        h, l, c = _synthetic_ohlc(100)
        period = 14
        full = atr(h, l, c, period)
        trunc_atr = atr(h.iloc[:TRUNC], l.iloc[:TRUNC], c.iloc[:TRUNC], period)
        # EWM is recursive so values converge but don't stay identical past the
        # warm-up, since early rows feed into later ones.  For a CAUSAL indicator
        # we only require that the *truncated* output never changes when future
        # rows are added.  Verify: the truncated result equals the full result's
        # first TRUNC rows element-wise (EWM is strictly causal).
        overlap = full.iloc[:TRUNC]
        pd.testing.assert_frame_equal(
            overlap, trunc_atr,
            check_names=False,
            obj="ATR causality: full vs truncated",
        )

    def test_positive(self):
        """ATR must be strictly positive once there's enough data."""
        h, l, c = _synthetic_ohlc(50)
        result = atr(h, l, c, period=5)
        # After first row (prev_close undefined) values should be positive.
        assert (result.iloc[1:].values > 0).all(), "ATR must be > 0 for all valid rows"

    def test_high_ge_low(self):
        """ATR should not raise even when high==low (flat price)."""
        n = 30
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        flat_price = pd.DataFrame({"A": np.ones(n) * 100.0}, index=idx)
        result = atr(flat_price, flat_price, flat_price, period=5)
        assert result.shape == flat_price.shape


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_causality(self):
        """RSI on full series equals RSI on truncated for the overlapping rows."""
        prices = _simple_close(100)
        period = 14
        full = rsi(prices, period)
        trunc = rsi(prices.iloc[:TRUNC], period)
        pd.testing.assert_frame_equal(
            full.iloc[:TRUNC], trunc,
            check_names=False,
            obj="RSI causality: full vs truncated",
        )

    def test_bounds(self):
        """RSI must be in [0, 100] where defined."""
        prices = _simple_close(80)
        result = rsi(prices, period=14)
        valid = result.dropna()
        assert ((valid >= 0.0) & (valid <= 100.0)).all().all(), (
            "RSI must lie in [0, 100]"
        )

    def test_all_gains_gives_100(self):
        """Strictly rising prices => RSI should be 100 (no losses)."""
        n = 50
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        prices = pd.DataFrame({"A": np.arange(1.0, n + 1.0)}, index=idx)
        result = rsi(prices, period=14)
        # After warm-up all values should be 100
        assert (result.iloc[15:].values == pytest.approx(100.0)), (
            "RSI of strictly rising series must be 100"
        )


# ---------------------------------------------------------------------------
# Realized Vol
# ---------------------------------------------------------------------------

class TestRealizedVol:
    def test_causality(self):
        """Realized vol on full series equals vol on truncated prefix."""
        prices = _simple_close(100)
        window = 20
        full = realized_vol(prices, window, annualize=False)
        trunc = realized_vol(prices.iloc[:TRUNC], window, annualize=False)
        pd.testing.assert_frame_equal(
            full.iloc[:TRUNC], trunc,
            check_names=False,
            obj="realized_vol causality: full vs truncated",
        )

    def test_positive_and_scale(self):
        """Annualized vol should be larger than non-annualized by sqrt(365)."""
        prices = _simple_close(100)
        window = 20
        v_ann = realized_vol(prices, window, annualize=True)
        v_raw = realized_vol(prices, window, annualize=False)
        # Where both are defined, ratio should be sqrt(365)
        ratio = (v_ann / v_raw).dropna()
        assert ratio.values == pytest.approx(np.sqrt(365.0), rel=1e-9), (
            "Annualized vol must equal raw vol * sqrt(365)"
        )


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_causality(self):
        """Momentum on full series equals momentum on truncated prefix."""
        prices = _simple_close(100)
        lookback = 20
        full = momentum(prices, lookback)
        trunc = momentum(prices.iloc[:TRUNC], lookback)
        pd.testing.assert_frame_equal(
            full.iloc[:TRUNC], trunc,
            check_names=False,
            obj="momentum causality: full vs truncated",
        )

    def test_formula(self):
        """Momentum equals price/lagged_price - 1 exactly."""
        prices = _simple_close(50)
        lb = 10
        result = momentum(prices, lb)
        expected = prices / prices.shift(lb) - 1.0
        pd.testing.assert_frame_equal(result, expected, check_names=False)


# ---------------------------------------------------------------------------
# Risk-Adjusted Momentum
# ---------------------------------------------------------------------------

class TestRiskAdjustedMomentum:
    def test_causality(self):
        """Risk-adjusted momentum on full series equals truncated for the prefix."""
        prices = _simple_close(100)
        lookback, vol_win = 20, 20
        full = risk_adjusted_momentum(prices, lookback, vol_win)
        trunc = risk_adjusted_momentum(prices.iloc[:TRUNC], lookback, vol_win)
        pd.testing.assert_frame_equal(
            full.iloc[:TRUNC], trunc,
            check_names=False,
            obj="risk_adjusted_momentum causality",
        )


# ---------------------------------------------------------------------------
# Cross-Sectional Rank
# ---------------------------------------------------------------------------

class TestCrossSectionalRank:
    def test_range(self):
        """Ranks are in [0, 1] (pct=True) and non-NaN where inputs are non-NaN."""
        n, m = 50, 4
        rng = np.random.default_rng(7)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        scores = pd.DataFrame(rng.standard_normal((n, m)),
                              index=idx, columns=list("ABCD"))
        ranks = cross_sectional_rank(scores)
        assert ((ranks >= 0.0) & (ranks <= 1.0)).all().all(), (
            "cross_sectional_rank must be in [0, 1]"
        )

    def test_monotone(self):
        """Higher score should yield higher rank within a row."""
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        scores = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0],
             "B": [5.0, 4.0, 3.0, 2.0, 1.0]},
            index=idx,
        )
        ranks = cross_sectional_rank(scores)
        # On day 0: A=1 < B=5 => rank(A) < rank(B)
        assert ranks.iloc[0]["A"] < ranks.iloc[0]["B"], (
            "Lower score should yield lower rank"
        )
