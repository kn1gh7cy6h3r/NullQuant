"""
conftest.py — shared synthetic fixtures for Meridian foundation tests.

Builds a deterministic 3-asset OHLCV Panel (~400 daily rows) and a Config
without touching the network or the real data cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.config import load_config
from meridian.data.loader import Panel

ASSETS = ["BTC-USD", "ETH-USD", "LTC-USD"]
N_ROWS = 420  # >200 so SMA200 eventually has values; >365 so vol-targeting warms up
SEED = 42


def _make_synthetic_panel(n: int = N_ROWS, seed: int = SEED) -> Panel:
    """
    Build a Panel with deterministic, coherent OHLCV data.

    Price path: geometric random walk with a modest positive drift so we get
    meaningful signals (trending assets).  High >= Close >= Low on every row.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-01", periods=n, freq="D")

    fields: dict[str, pd.DataFrame] = {}
    opens, highs, lows, closes, vols = {}, {}, {}, {}, {}

    for i, asset in enumerate(ASSETS):
        # Slightly different drift/vol per asset for diversity.
        drift = 0.0008 + i * 0.0002
        sigma = 0.02 + i * 0.005
        log_ret = rng.normal(drift, sigma, size=n)
        log_prices = np.cumsum(log_ret)
        close = 100.0 * np.exp(log_prices)

        # Construct coherent OHLC: High is close * (1 + intraday range),
        # Low is close * (1 - intraday range), Open is between.
        intraday = np.abs(rng.normal(0.0, 0.01, size=n))
        high = close * (1.0 + intraday)
        low = close * (1.0 - intraday)
        open_ = low + rng.random(size=n) * (high - low)

        # Sanity: enforce High >= Close >= Low (may drift slightly from float ops)
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        high = np.maximum(high, open_)
        low = np.minimum(low, open_)

        volume = np.abs(rng.normal(1e6, 2e5, size=n))

        opens[asset] = pd.Series(open_, index=index)
        highs[asset] = pd.Series(high, index=index)
        lows[asset] = pd.Series(low, index=index)
        closes[asset] = pd.Series(close, index=index)
        vols[asset] = pd.Series(volume, index=index)

    fields["Open"] = pd.DataFrame(opens, index=index)
    fields["High"] = pd.DataFrame(highs, index=index)
    fields["Low"] = pd.DataFrame(lows, index=index)
    fields["Close"] = pd.DataFrame(closes, index=index)
    fields["Volume"] = pd.DataFrame(vols, index=index)

    return Panel(fields=fields, assets=ASSETS)


@pytest.fixture(scope="session")
def panel() -> Panel:
    """Synthetic OHLCV Panel — shared across the whole test session."""
    return _make_synthetic_panel()


@pytest.fixture(scope="session")
def cfg():
    """Real project Config, loaded once per session."""
    return load_config()
