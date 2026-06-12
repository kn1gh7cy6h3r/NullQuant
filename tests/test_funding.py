"""
test_funding.py — funding-rate plumbing and graceful degradation.

The perpetual-swap funding signal is OPTIONAL and exchange-sourced. These tests
pin the contract that keeps it from ever breaking an offline / CI run:

  1. Panel.has_funding correctly reports presence of real funding values.
  2. The Binance symbol map covers the configured universe.
  3. load_funding_rates returns None (no exception, no network) when no asset in
     the requested universe maps to a perp.
  4. The LTR model runs identically-shaped output with funding absent vs present
     — the funding feature degrades to a neutral column, never dropping rows.

No test here performs network I/O; the download path is exercised only through
the unmapped-universe short-circuit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nullquant.config import load_config
from nullquant.data.loader import Panel, BINANCE_PERP, load_funding_rates
from nullquant.ml.rank_model import ltr_signal

ASSETS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD",
          "ADA-USD", "SOL-USD", "DOGE-USD", "LTC-USD"]


def _panel(n: int = 500, seed: int = 1, with_funding: bool = False) -> Panel:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    fields = {k: {} for k in ["Open", "High", "Low", "Close", "Volume"]}
    for i, a in enumerate(ASSETS):
        lr = rng.normal(0.0005, 0.02 + 0.003 * i, n)
        c = 100.0 * np.exp(np.cumsum(lr))
        intra = np.abs(rng.normal(0.0, 0.01, n))
        hi = np.maximum(c * (1.0 + intra), c)
        lo = np.minimum(c * (1.0 - intra), c)
        fields["Open"][a] = c
        fields["High"][a] = hi
        fields["Low"][a] = lo
        fields["Close"][a] = c
        fields["Volume"][a] = np.abs(rng.normal(1e6, 2e5, n))
    frames = {k: pd.DataFrame(v, index=idx)[ASSETS] for k, v in fields.items()}
    funding = None
    if with_funding:
        funding = pd.DataFrame(
            rng.normal(0.0, 1e-4, (n, len(ASSETS))), index=idx, columns=ASSETS)
    return Panel(fields=frames, assets=ASSETS, funding=funding)


class TestHasFunding:
    def test_none_is_false(self):
        assert _panel(with_funding=False).has_funding is False

    def test_all_nan_is_false(self):
        p = _panel(with_funding=True)
        p.funding.iloc[:, :] = np.nan
        assert p.has_funding is False

    def test_real_values_is_true(self):
        assert _panel(with_funding=True).has_funding is True


class TestSymbolMap:
    def test_universe_covered(self):
        cfg = load_config()
        for ticker in cfg.data.universe:
            assert ticker in BINANCE_PERP, f"{ticker} missing a perp mapping"


class TestLoaderDegradation:
    def test_unmapped_universe_returns_none(self):
        """No perp mapping -> None, with no network call and no exception."""
        cfg = load_config()
        out = load_funding_rates(cfg, assets=["FAKE-USD", "NOPE-USD"])
        assert out is None


class TestLtrRunsEitherWay:
    def test_ltr_shape_with_and_without_funding(self):
        cfg = load_config()
        r_off = ltr_signal(_panel(with_funding=False), cfg)
        r_on = ltr_signal(_panel(with_funding=True), cfg)
        assert r_off.direction.shape == r_on.direction.shape
        # Direction stays a clean {-1,0,1} matrix in both cases.
        for r in (r_off, r_on):
            vals = set(np.unique(r.direction.to_numpy()))
            assert vals.issubset({-1.0, 0.0, 1.0})
