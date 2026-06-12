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
    atr_normalized_momentum,
    cross_sectional_mad_zscore,
    cross_sectional_rank,
    funding_rank_signal,
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


# ---------------------------------------------------------------------------
# Cross-Sectional MAD Z-Score (LTR drift-neutral feature transform)
# ---------------------------------------------------------------------------

def _panel(n: int = 80, m: int = 8, seed: int = 3) -> pd.DataFrame:
    """A multi-asset numeric panel for cross-sectional transform tests."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    cols = [f"A{i}" for i in range(m)]
    return pd.DataFrame(rng.standard_normal((n, m)) * 0.05, index=idx, columns=cols)


class TestCrossSectionalMadZscore:
    def test_causality(self):
        """Per-row across assets => truncating time leaves the prefix identical."""
        feat = _panel(100, 8)
        full = cross_sectional_mad_zscore(feat)
        trunc = cross_sectional_mad_zscore(feat.iloc[:TRUNC])
        pd.testing.assert_frame_equal(
            full.iloc[:TRUNC], trunc, check_names=False,
            obj="cross_sectional_mad_zscore causality",
        )

    def test_row_centering(self):
        """The cross-sectional median maps to ~0 each row; spread is symmetric-ish."""
        feat = _panel(30, 9)
        z = cross_sectional_mad_zscore(feat)
        # The per-row median asset should sit very close to 0 after centering.
        row_med = z.median(axis=1).abs()
        assert (row_med < 1e-9).all(), "row median of z-scores must be ~0"

    def test_winsorized_bounds(self):
        """Outputs are bounded to +/- clip even with an extreme outlier."""
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        feat = pd.DataFrame(
            {"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0],
             "C": [0.01, 0.01, 0.01], "D": [1e6, 1e6, 1e6]},  # D is a blow-up
            index=idx,
        )
        z = cross_sectional_mad_zscore(feat, clip=5.0)
        assert (z.abs() <= 5.0 + 1e-12).all().all(), "z must be winsorized to +/-clip"

    def test_nan_input_preserved(self):
        """NaN inputs (warm-up / unlisted) stay NaN so downstream rows are dropped."""
        feat = _panel(20, 6)
        feat.iloc[0, :] = np.nan          # whole warm-up row missing
        feat.iloc[5, 2] = np.nan          # a single unlisted asset
        z = cross_sectional_mad_zscore(feat)
        assert z.iloc[0].isna().all(), "all-NaN row must remain NaN"
        assert np.isnan(z.iloc[5, 2]), "NaN input cell must remain NaN"

    def test_no_cross_sectional_drift(self):
        """Adding a common per-date level to every asset leaves the z-score
        unchanged — exactly the market-drift neutralization we rely on."""
        feat = _panel(40, 8)
        drift = pd.Series(np.linspace(0.0, 1.0, len(feat)), index=feat.index)
        shifted = feat.add(drift, axis=0)
        pd.testing.assert_frame_equal(
            cross_sectional_mad_zscore(feat),
            cross_sectional_mad_zscore(shifted),
            check_names=False, obj="CS-MAD-Z must be invariant to common drift",
        )


# ---------------------------------------------------------------------------
# ATR-Normalized Momentum (volatility-weighted relative strength)
# ---------------------------------------------------------------------------

class TestAtrNormalizedMomentum:
    def test_causality(self):
        """Full series equals truncated series on the overlapping prefix."""
        h, l, c = _synthetic_ohlc(100)
        full = atr_normalized_momentum(c, h, l, lookback=10, atr_period=14)
        trunc = atr_normalized_momentum(
            c.iloc[:TRUNC], h.iloc[:TRUNC], l.iloc[:TRUNC], lookback=10, atr_period=14)
        pd.testing.assert_frame_equal(
            full.iloc[:TRUNC], trunc, check_names=False,
            obj="atr_normalized_momentum causality",
        )

    def test_sign_tracks_move(self):
        """A strictly rising series gives positive ATR-normalized momentum."""
        n = 60
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        c = pd.DataFrame({"A": np.arange(100.0, 100.0 + n)}, index=idx)
        h = c * 1.001
        l = c * 0.999
        out = atr_normalized_momentum(c, h, l, lookback=10).dropna()
        assert (out["A"] > 0).all(), "rising price must give positive momentum"


# ---------------------------------------------------------------------------
# Funding Rank Signal (structural tilt: high funding => bearish)
# ---------------------------------------------------------------------------

class TestFundingRankSignal:
    def test_high_funding_scores_low(self):
        """High positive funding -> low (negative) score; negative funding -> high."""
        idx = pd.date_range("2020-01-01", periods=4, freq="D")
        funding = pd.DataFrame(
            {"HOT": [0.0010, 0.0012, 0.0011, 0.0009],   # crowded longs
             "MID": [0.0001, 0.0001, 0.0000, 0.0001],
             "COLD": [-0.0008, -0.0009, -0.0007, -0.0008]},  # shorts pay
            index=idx,
        )
        score = funding_rank_signal(funding)
        assert (score["HOT"] < score["MID"]).all(), "high funding must score below mid"
        assert (score["MID"] < score["COLD"]).all(), "negative funding must score above mid"

    def test_missing_funding_is_nan(self):
        """A missing per-asset funding value flows through as NaN (neutralized
        to 0 by the consumer, never fabricated here)."""
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        funding = pd.DataFrame(
            {"A": [0.001, np.nan, 0.001], "B": [-0.001, -0.001, -0.001],
             "C": [0.0, 0.0, 0.0]},
            index=idx,
        )
        score = funding_rank_signal(funding)
        assert np.isnan(score.iloc[1]["A"]), "missing funding must stay NaN"
