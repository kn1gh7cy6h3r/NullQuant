"""
test_metrics.py — sanity and invariants for performance statistics.

These guard the headline numbers a reviewer reads: Sharpe sign/scale, the
non-positivity of drawdown, the [0,1] range of probabilistic Sharpe, and the
crucial property that the Deflated Sharpe (more trials) never EXCEEDS the
ordinary probabilistic Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nullquant.metrics import performance as perf


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_sharpe_sign_and_zero():
    # Constant positive returns -> large positive Sharpe.
    pos = pd.Series([0.001] * 500)
    assert perf.sharpe_ratio(pos) > 0
    # All zeros -> undefined std -> defined as 0.
    assert perf.sharpe_ratio(pd.Series([0.0] * 500)) == 0.0


def test_sharpe_scales_with_mean(rng):
    base = pd.Series(rng.normal(0.0005, 0.01, size=1000))
    shifted = base + 0.0005  # strictly higher mean, same vol
    assert perf.sharpe_ratio(shifted) > perf.sharpe_ratio(base)


def test_max_drawdown_non_positive(rng):
    r = pd.Series(rng.normal(0.0, 0.02, size=1000))
    assert perf.max_drawdown(r) <= 0.0
    # Monotonically rising equity -> drawdown ~ 0.
    rising = pd.Series([0.001] * 500)
    assert perf.max_drawdown(rising) == pytest.approx(0.0, abs=1e-9)


def test_psr_in_unit_interval(rng):
    r = pd.Series(rng.normal(0.0008, 0.015, size=800))
    psr = perf.probabilistic_sharpe_ratio(r, 0.0)
    assert 0.0 <= psr <= 1.0


def test_deflated_never_exceeds_psr(rng):
    # Selecting the best of many trials must DEFLATE the Sharpe: DSR <= PSR(0).
    r = pd.Series(rng.normal(0.0008, 0.015, size=800))
    psr0 = perf.probabilistic_sharpe_ratio(r, 0.0)
    dsr = perf.deflated_sharpe_ratio(r, n_trials=50)
    assert dsr <= psr0 + 1e-9
    # More trials -> weakly lower (or equal) deflated Sharpe.
    dsr_more = perf.deflated_sharpe_ratio(r, n_trials=500)
    assert dsr_more <= dsr + 1e-9


def test_summary_keys(rng):
    r = pd.Series(rng.normal(0.0005, 0.01, size=600))
    s = perf.summary(r, n_trials=10)
    for key in ("ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown",
                "calmar", "psr_vs_zero", "deflated_sharpe", "n_obs",
                "sharpe_ci_low", "sharpe_ci_high"):
        assert key in s
    assert s["sharpe_ci_low"] <= s["sharpe_ci_high"]
