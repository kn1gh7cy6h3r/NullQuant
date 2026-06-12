"""
test_regime.py — stationarity of the HMM regime-switching feature pipeline.

The regime controller went comatose (one state ~88% of the time) because its
feature set carried DIRECTIONAL, drifting terms. The rebuilt feature pipeline
(nullquant.ml.regime_switch._build_market_features) must feed the HMM only
scale-free "market texture". These tests pin that contract:

  1. The feature columns are exactly the stationary texture set — no directional
     mean-return / price-level / BTC-trailing-return columns.
  2. The features are invariant to the PRICE LEVEL and the VOLUME SCALE: rescaling
     every close and every volume by constants leaves the features unchanged.
     That is the operational definition of "no price levels / un-normalized SMAs"
     leaking in — only returns/dispersion/relative-volume/skew survive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nullquant.config import load_config
from nullquant.data.loader import Panel
from nullquant.ml.regime_switch import _build_market_features

ASSETS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]
EXPECTED_COLUMNS = {"cs_dispersion", "avg_realized_vol", "vol_to_vol", "ret_skew"}
FORBIDDEN_COLUMNS = {"cs_mean_ret", "btc_trailing_ret", "mean_abs_ret"}


def _panel(n: int = 400, seed: int = 5, price_mult: float = 1.0,
           vol_mult: float = 1.0) -> Panel:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    fields = {k: {} for k in ["Open", "High", "Low", "Close", "Volume"]}
    for i, a in enumerate(ASSETS):
        lr = rng.normal(0.0005, 0.02 + 0.004 * i, n)
        c = 100.0 * np.exp(np.cumsum(lr)) * price_mult
        intra = np.abs(rng.normal(0.0, 0.01, n))
        hi = np.maximum(c * (1.0 + intra), c)
        lo = np.minimum(c * (1.0 - intra), c)
        fields["Open"][a] = c
        fields["High"][a] = hi
        fields["Low"][a] = lo
        fields["Close"][a] = c
        fields["Volume"][a] = np.abs(rng.normal(1e6, 2e5, n)) * vol_mult
    frames = {k: pd.DataFrame(v, index=idx)[ASSETS] for k, v in fields.items()}
    return Panel(fields=frames, assets=ASSETS)


class TestRegimeFeatureContract:
    def test_columns_are_texture_only(self):
        feats = _build_market_features(_panel(), load_config())
        assert set(feats.columns) == EXPECTED_COLUMNS
        assert not (set(feats.columns) & FORBIDDEN_COLUMNS), (
            "directional features must not appear in the HMM input"
        )

    def test_features_finite_after_warmup(self):
        feats = _build_market_features(_panel(), load_config())
        warm = feats.dropna()
        assert len(warm) > 0
        assert np.isfinite(warm.to_numpy()).all(), "features must be finite once warm"


class TestStationarity:
    def test_invariant_to_price_and_volume_scale(self):
        """Rescaling all prices and volumes leaves every feature unchanged — i.e.
        no price LEVEL or absolute-volume term leaks into the HMM."""
        cfg = load_config()
        base = _build_market_features(_panel(seed=9), cfg)
        scaled = _build_market_features(
            _panel(seed=9, price_mult=37.0, vol_mult=1000.0), cfg)
        pd.testing.assert_frame_equal(
            base, scaled, check_names=False, rtol=1e-9, atol=1e-12,
            obj="HMM features must be invariant to price/volume scale",
        )
