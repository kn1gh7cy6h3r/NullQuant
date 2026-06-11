"""
test_labeling.py — correctness tests for nullquant.features.labeling.triple_barrier_labels_one.

Key guarantees checked:
  1. Strictly rising close + long event => label=1, ret>0.
  2. Strictly falling close + long event => label=0, ret<0.
  3. t1 >= event_date and within max_hold_days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nullquant.features.labeling import triple_barrier_labels_one


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_close(prices: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(prices), freq="D")
    return pd.Series(prices, index=idx, name="TEST")


def _make_atr(close: pd.Series, value: float = 2.0) -> pd.Series:
    """Constant ATR series aligned to close."""
    return pd.Series(value, index=close.index, name="TEST")


# ---------------------------------------------------------------------------
# Rising series: long event must hit profit-take
# ---------------------------------------------------------------------------

class TestRisingSeries:
    """Strictly monotonically rising prices with a long entry."""

    def setup_method(self):
        # 50-element series rising 1 point per day from 100 to 149
        n = 50
        prices = list(np.arange(100.0, 100.0 + n))
        self.close = _make_close(prices)
        self.atr = _make_atr(self.close, value=2.0)
        # Enter on day 5 (price=105)
        self.event_date = self.close.index[5]
        self.events = pd.DatetimeIndex([self.event_date])
        self.pt_mult = 2.0   # profit-take at entry + 2*ATR = 105 + 4 = 109
        self.sl_mult = 2.0   # stop-loss at entry - 2*ATR = 105 - 4 = 101
        self.max_hold = 30

    def test_label_is_1(self):
        result = triple_barrier_labels_one(
            self.close, self.atr, self.events,
            side=1, pt_mult=self.pt_mult, sl_mult=self.sl_mult,
            max_hold_days=self.max_hold,
        )
        assert not result.empty, "Expected at least one labeled event"
        label = result.loc[self.event_date, "label"]
        assert label == 1, (
            f"Rising series + long should give label=1, got {label}"
        )

    def test_ret_positive(self):
        result = triple_barrier_labels_one(
            self.close, self.atr, self.events,
            side=1, pt_mult=self.pt_mult, sl_mult=self.sl_mult,
            max_hold_days=self.max_hold,
        )
        ret = result.loc[self.event_date, "ret"]
        assert ret > 0.0, (
            f"Rising series + long should give ret > 0, got {ret}"
        )

    def test_t1_timing(self):
        result = triple_barrier_labels_one(
            self.close, self.atr, self.events,
            side=1, pt_mult=self.pt_mult, sl_mult=self.sl_mult,
            max_hold_days=self.max_hold,
        )
        t1 = result.loc[self.event_date, "t1"]
        assert t1 >= self.event_date, "t1 must be >= event_date"
        assert t1 <= self.event_date + pd.Timedelta(days=self.max_hold), (
            "t1 must be within max_hold_days of event_date"
        )


# ---------------------------------------------------------------------------
# Falling series: long event must hit stop-loss
# ---------------------------------------------------------------------------

class TestFallingSeries:
    """Strictly monotonically falling prices with a long entry."""

    def setup_method(self):
        n = 50
        prices = list(np.arange(149.0, 149.0 - n, -1.0))  # 149 down to 100
        self.close = _make_close(prices)
        self.atr = _make_atr(self.close, value=2.0)
        # Enter on day 5 (price=144)
        self.event_date = self.close.index[5]
        self.events = pd.DatetimeIndex([self.event_date])
        self.pt_mult = 2.0   # profit-take at 144 + 4 = 148 (never reached)
        self.sl_mult = 2.0   # stop-loss  at 144 - 4 = 140 (reached on day 5+4=9)
        self.max_hold = 30

    def test_label_is_0(self):
        result = triple_barrier_labels_one(
            self.close, self.atr, self.events,
            side=1, pt_mult=self.pt_mult, sl_mult=self.sl_mult,
            max_hold_days=self.max_hold,
        )
        assert not result.empty, "Expected at least one labeled event"
        label = result.loc[self.event_date, "label"]
        assert label == 0, (
            f"Falling series + long should give label=0, got {label}"
        )

    def test_ret_negative(self):
        result = triple_barrier_labels_one(
            self.close, self.atr, self.events,
            side=1, pt_mult=self.pt_mult, sl_mult=self.sl_mult,
            max_hold_days=self.max_hold,
        )
        ret = result.loc[self.event_date, "ret"]
        assert ret < 0.0, (
            f"Falling series + long should give ret < 0, got {ret}"
        )

    def test_t1_timing(self):
        result = triple_barrier_labels_one(
            self.close, self.atr, self.events,
            side=1, pt_mult=self.pt_mult, sl_mult=self.sl_mult,
            max_hold_days=self.max_hold,
        )
        t1 = result.loc[self.event_date, "t1"]
        assert t1 >= self.event_date, "t1 must be >= event_date"
        assert t1 <= self.event_date + pd.Timedelta(days=self.max_hold), (
            "t1 must be within max_hold_days"
        )


# ---------------------------------------------------------------------------
# Vertical barrier: flat price => hit time stop
# ---------------------------------------------------------------------------

class TestVerticalBarrier:
    """Flat price never touches a barrier — must hit vertical (time) stop."""

    def setup_method(self):
        n = 60
        prices = [100.0] * n
        self.close = _make_close(prices)
        # Very wide barriers so they can't be hit
        self.atr = _make_atr(self.close, value=1.0)
        self.event_date = self.close.index[5]
        self.events = pd.DatetimeIndex([self.event_date])
        self.max_hold = 10

    def test_t1_at_vertical(self):
        result = triple_barrier_labels_one(
            self.close, self.atr, self.events,
            side=1, pt_mult=100.0, sl_mult=100.0,  # unreachably wide barriers
            max_hold_days=self.max_hold,
        )
        assert not result.empty
        t1 = result.loc[self.event_date, "t1"]
        # Should hit the vertical barrier exactly at event_date + max_hold
        expected_t1 = self.event_date + pd.Timedelta(days=self.max_hold)
        assert t1 == expected_t1, (
            f"Flat price should hit vertical barrier at {expected_t1}, got {t1}"
        )


# ---------------------------------------------------------------------------
# Empty result for missing event dates
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_event_not_in_index(self):
        """An event_date not present in the close index should be silently skipped."""
        prices = list(np.arange(100.0, 120.0))
        close = _make_close(prices)
        atr_s = _make_atr(close, value=1.0)
        # Use a date far outside the close index
        bad_event = pd.DatetimeIndex(["2099-01-01"])
        result = triple_barrier_labels_one(
            close, atr_s, bad_event,
            side=1, pt_mult=1.0, sl_mult=1.0, max_hold_days=5,
        )
        assert result.empty, "Event not in index should produce empty result"

    def test_side_column_preserved(self):
        """Result must include the 'side' column with the input side value."""
        n = 30
        prices = list(np.arange(100.0, 100.0 + n))
        close = _make_close(prices)
        atr_s = _make_atr(close, value=2.0)
        event_date = close.index[5]
        events = pd.DatetimeIndex([event_date])
        result = triple_barrier_labels_one(
            close, atr_s, events,
            side=1, pt_mult=2.0, sl_mult=2.0, max_hold_days=20,
        )
        assert "side" in result.columns
        assert result.loc[event_date, "side"] == 1
