"""
test_conformal.py — split-conformal confidence sizer (nullquant.ml.conformal).

Guarantees checked:
  1. Empirical OOS coverage lands near the 1 - alpha target. This is the
     no-leakage canary: a leaking calibration would not produce honest ~90%
     coverage on genuinely out-of-sample days.
  2. Continuous mode is a real sizer: exposure is bounded to [floor, 1], varies
     across dates, and never fully gates to cash (>= floor).
  3. Binary (legacy) mode reproduces the on/off gate in [0, 1].
  4. Graceful degradation: too little history / bad config -> exposure == 1.0,
     never raises.

A local multi-asset panel with enough history (and heterogeneous volatility so
the forest's local scale varies) is built here; the session fixtures are too
short to leave the conformal warm-up.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from nullquant.config import load_config
from nullquant.data.loader import Panel
from nullquant.ml.conformal import conformal_exposure


ASSETS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD",
          "ADA-USD", "SOL-USD", "DOGE-USD", "LTC-USD"]


def _panel(n: int = 1500, seed: int = 11) -> Panel:
    """8-asset OHLCV panel with per-asset, time-varying volatility regimes so the
    RandomForest's local disagreement (and thus the conformal interval width)
    genuinely varies across assets and dates."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", periods=n, freq="D")
    fields = {k: {} for k in ["Open", "High", "Low", "Close", "Volume"]}
    for i, a in enumerate(ASSETS):
        # Volatility cycles at an asset-specific period -> heterogeneous texture.
        period = 90 + 25 * i
        vol = 0.012 + 0.025 * (np.sin(np.arange(n) / period) > 0)
        lr = rng.normal(0.0003, 1.0, n) * vol
        c = 100.0 * np.exp(np.cumsum(lr))
        intra = np.abs(rng.normal(0.0, 0.01, n))
        hi = np.maximum(c * (1.0 + intra), c)
        lo = np.minimum(c * (1.0 - intra), c)
        op = lo + rng.random(n) * (hi - lo)
        fields["Open"][a] = op
        fields["High"][a] = hi
        fields["Low"][a] = lo
        fields["Close"][a] = c
        fields["Volume"][a] = np.abs(rng.normal(1e6, 2e5, n))
    frames = {k: pd.DataFrame(v, index=idx)[ASSETS] for k, v in fields.items()}
    return Panel(fields=frames, assets=ASSETS, funding=None)


@pytest.fixture(scope="module")
def big_panel() -> Panel:
    return _panel()


@pytest.fixture(scope="module")
def continuous_result(big_panel):
    return conformal_exposure(big_panel, load_config())


# ---------------------------------------------------------------------------
# 1. Calibration coverage — the no-leakage canary
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_coverage_near_target(self, continuous_result):
        """Empirical OOS coverage should sit near 1 - alpha (0.90), not be a
        suspiciously perfect 1.0 (which would smell of leakage)."""
        cov = continuous_result.empirical_coverage
        assert 0.80 <= cov <= 0.985, f"coverage {cov:.3f} far from ~0.90 target"

    def test_refit_and_scored(self, continuous_result):
        assert continuous_result.n_refits > 0, "expected at least one refit"
        assert continuous_result.exposure_scale.notna().all()


# ---------------------------------------------------------------------------
# 2. Continuous sizer behaviour
# ---------------------------------------------------------------------------

class TestContinuousSizer:
    def test_bounds_floor_to_one(self, continuous_result):
        """Continuous exposure is bounded to [floor, 1] and never hits cash."""
        floor = float(load_config().get("ml.conformal.floor", 0.05))
        exp = continuous_result.exposure_scale
        assert exp.min() >= floor - 1e-9, f"exposure below floor: {exp.min()}"
        assert exp.max() <= 1.0 + 1e-9, f"exposure above 1: {exp.max()}"

    def test_is_continuous_not_binary(self, continuous_result):
        """A genuine sizer takes many distinct values, not just on/off."""
        scored = continuous_result.exposure_scale
        scored = scored[scored < 1.0]  # ignore warm-up / fully-confident 1.0s
        assert scored.nunique() > 2, "continuous exposure should vary, not be binary"


# ---------------------------------------------------------------------------
# 3. Binary (legacy) mode
# ---------------------------------------------------------------------------

class TestBinaryMode:
    def test_binary_bounds_and_differs(self, big_panel):
        cfg = load_config()
        cfg.raw["ml"]["conformal"]["mode"] = "binary"
        res = conformal_exposure(big_panel, cfg)
        exp = res.exposure_scale
        assert exp.min() >= -1e-9 and exp.max() <= 1.0 + 1e-9
        assert 0.0 <= res.mean_exposure <= 1.0
        assert "binary on/off gate" in res.status


# ---------------------------------------------------------------------------
# 4. Graceful degradation
# ---------------------------------------------------------------------------

class TestDegradation:
    def test_short_history_returns_unit_exposure(self):
        """Too little history -> safe no-op exposure of 1.0 everywhere, no raise."""
        small = _panel(n=120)
        res = conformal_exposure(small, load_config())
        assert (res.exposure_scale == 1.0).all()
        assert res.n_refits == 0

    def test_bad_mode_degrades(self, big_panel):
        cfg = load_config()
        cfg.raw["ml"]["conformal"]["mode"] = "nonsense"
        res = conformal_exposure(big_panel, cfg)
        assert (res.exposure_scale == 1.0).all()
        assert "invalid conformal mode" in res.status
