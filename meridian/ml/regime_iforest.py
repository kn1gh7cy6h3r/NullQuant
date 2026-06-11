"""
regime_iforest.py — walk-forward Isolation Forest regime / risk filter.

WHAT THIS DOES
--------------
We build a small set of *market-level* (cross-sectional) features per date from
the asset panel and run them through an Isolation Forest to flag abnormal
market regimes (broad crashes, dispersion blow-ups, vol spikes). When the
current day looks abnormal we cut book exposure to ``suppress_scale``;
otherwise exposure runs at 1.0.

WHY WALK-FORWARD (THE NO-LEAKAGE DESIGN)
----------------------------------------
The naive version of this filter fits an Isolation Forest on the *entire*
history — including the very point it then scores — and unsurprisingly flags
the tail of that same distribution. That is leakage: the model has already
"seen" day t when judging day t, so the anomaly call is not something you could
have made in real time.

This implementation is strictly out-of-sample:

  * We maintain a TRAILING window of length ``cfg.ml.regime.lookback``.
  * At each step we fit the IsolationForest (and its StandardScaler) on the
    window ENDING AT t-1, then score the unseen day t.
  * The scored day is NEVER part of the fit window — so every anomaly score is
    a genuine forecast made from past information only.

For speed we refit periodically (every ``_REFIT_EVERY`` days) and reuse the
fitted estimator to score the days in between; the fit window is always strictly
in the past relative to each day being scored, so this introduces no leakage.

All features are backward-looking (rolling/realized), so the feature vector for
date t is itself causal.

The result is an exposure multiplier in [0, 1] indexed by ``panel.close.index``.
Warm-up dates (before the first out-of-sample score is available) default to
1.0 — we never suppress on days we could not actually have scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import Panel
from ..features import indicators as ind
from ..seeds import set_global_seed

# Refit the Isolation Forest at most this often (in days). Between refits the
# previously-fitted estimator scores subsequent (still strictly future) days.
_REFIT_EVERY = 21


@dataclass
class RegimeResult:
    """Output of the regime filter.

    Attributes
    ----------
    exposure_scale:
        pandas Series in [0, 1] indexed by panel dates. 1.0 in a normal regime,
        ``suppress_scale`` when the day was flagged abnormal out-of-sample,
        and 1.0 for warm-up dates that were never scored.
    abnormal_flags:
        pandas Series of bool, True where the day was flagged abnormal.
    scores:
        pandas Series of the raw Isolation Forest anomaly score per date
        (higher = more anomalous; ``decision_function`` is negated so the sign
        is intuitive). NaN on un-scored / warm-up dates.
    n_abnormal:
        Number of dates flagged abnormal.
    frac_abnormal:
        Fraction of *scored* dates flagged abnormal.
    status:
        Human-readable status string (also reports graceful degradation).
    """

    exposure_scale: pd.Series
    abnormal_flags: pd.Series
    scores: pd.Series
    n_abnormal: int
    frac_abnormal: float
    status: str


def _build_market_features(panel: Panel, cfg: Config) -> pd.DataFrame:
    """Build strictly causal market-level (cross-sectional) features per date.

    Every column is backward-looking at date t:
      * cs_mean_ret   — cross-sectional mean of daily log-returns
      * cs_disp_ret   — cross-sectional dispersion (std across assets) of returns
      * avg_realized_vol — average realized vol across assets
      * avg_abs_ret   — cross-sectional mean absolute return (raw stress proxy)
      * btc_vol_z     — z-score of BTC realized vol vs its own trailing history
    """
    rets = panel.log_returns()
    vol_lookback = int(cfg.strategy.vol_lookback)

    # Cross-sectional (across-asset) summaries of the daily return distribution.
    cs_mean_ret = rets.mean(axis=1, skipna=True)
    cs_disp_ret = rets.std(axis=1, skipna=True)
    avg_abs_ret = rets.abs().mean(axis=1, skipna=True)

    # Average realized vol across assets (rolling, hence causal).
    realized = ind.realized_vol(panel.close, vol_lookback, annualize=False)
    avg_realized_vol = realized.mean(axis=1, skipna=True)

    # BTC realized-vol z-score against its own trailing distribution.
    btc_col = next((c for c in panel.close.columns if str(c).upper().startswith("BTC")), None)
    if btc_col is not None:
        btc_vol = realized[btc_col]
        roll = btc_vol.rolling(vol_lookback, min_periods=vol_lookback)
        btc_vol_z = (btc_vol - roll.mean()) / roll.std().replace(0.0, np.nan)
    else:
        btc_vol_z = pd.Series(0.0, index=panel.close.index)

    feats = pd.DataFrame(
        {
            "cs_mean_ret": cs_mean_ret,
            "cs_disp_ret": cs_disp_ret,
            "avg_realized_vol": avg_realized_vol,
            "avg_abs_ret": avg_abs_ret,
            "btc_vol_z": btc_vol_z,
        },
        index=panel.close.index,
    )
    return feats


def compute_regime_filter(panel: Panel, cfg: Config) -> RegimeResult:
    """Walk-forward Isolation Forest regime filter.

    Fits on a trailing window ending at t-1 and scores the unseen day t, so no
    anomaly call ever uses the day it is judging. Returns an exposure multiplier
    in [0, 1] indexed by ``panel.close.index``.

    Degrades gracefully: if scikit-learn is unavailable, returns an all-1.0
    exposure with an explanatory status rather than raising.
    """
    set_global_seed(cfg.seed)

    dates = panel.close.index
    suppress_scale = float(cfg.ml.regime.suppress_scale)
    lookback = int(cfg.ml.regime.lookback)
    contamination = float(cfg.ml.regime.contamination)

    # Default outputs: full exposure everywhere, nothing flagged.
    exposure = pd.Series(1.0, index=dates, dtype=float)
    flags = pd.Series(False, index=dates, dtype=bool)
    scores = pd.Series(np.nan, index=dates, dtype=float)

    # Defensive sklearn import — never raise on a missing optional dependency.
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - exercised only without sklearn
        return RegimeResult(
            exposure_scale=exposure,
            abnormal_flags=flags,
            scores=scores,
            n_abnormal=0,
            frac_abnormal=0.0,
            status=f"sklearn unavailable ({exc}); regime filter disabled, exposure=1.0",
        )

    feats = _build_market_features(panel, cfg)
    valid = feats.dropna()
    if len(valid) <= lookback + 1:
        return RegimeResult(
            exposure_scale=exposure,
            abnormal_flags=flags,
            scores=scores,
            n_abnormal=0,
            frac_abnormal=0.0,
            status=(
                f"insufficient history: {len(valid)} valid feature rows "
                f"<= lookback+1 ({lookback + 1}); exposure=1.0"
            ),
        )

    feat_dates = valid.index
    X = valid.to_numpy(dtype=float)
    n = len(feat_dates)

    scored_positions: list[int] = []
    model: IsolationForest | None = None
    scaler: StandardScaler | None = None
    fit_anchor = -1  # position of the last day included in the current fit window

    # Walk forward: for each scorable position i (>= lookback), the fit window is
    # the `lookback` rows ending at i-1, strictly in the past relative to i.
    for i in range(lookback, n):
        need_refit = model is None or (i - fit_anchor) >= _REFIT_EVERY
        if need_refit:
            train = X[i - lookback:i]  # rows i-lookback .. i-1 (excludes i)
            scaler = StandardScaler().fit(train)
            model = IsolationForest(
                contamination=contamination,
                random_state=int(cfg.seed),
                n_estimators=200,
            ).fit(scaler.transform(train))
            fit_anchor = i - 1

        x_t = scaler.transform(X[i:i + 1])  # the unseen day, never in the fit set
        # decision_function: positive = inlier, negative = outlier. Negate so a
        # larger score means "more anomalous".
        score = -float(model.decision_function(x_t)[0])
        is_abnormal = bool(model.predict(x_t)[0] == -1)

        d = feat_dates[i]
        scores.loc[d] = score
        flags.loc[d] = is_abnormal
        if is_abnormal:
            exposure.loc[d] = suppress_scale
        scored_positions.append(i)

    n_scored = len(scored_positions)
    n_abnormal = int(flags.sum())
    frac_abnormal = float(n_abnormal / n_scored) if n_scored else 0.0
    first_scored = feat_dates[lookback] if n_scored else None

    status = (
        f"ok: walk-forward IsolationForest, lookback={lookback}, "
        f"contamination={contamination}, refit_every={_REFIT_EVERY}d; "
        f"{n_scored} OOS days scored from {first_scored.date() if first_scored is not None else 'n/a'}, "
        f"{n_abnormal} abnormal ({frac_abnormal:.3f}); "
        f"warm-up dates default to exposure=1.0"
    )

    return RegimeResult(
        exposure_scale=exposure,
        abnormal_flags=flags,
        scores=scores,
        n_abnormal=n_abnormal,
        frac_abnormal=frac_abnormal,
        status=status,
    )
