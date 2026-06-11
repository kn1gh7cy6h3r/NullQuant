"""
conformal.py — split-conformal confidence-gated exposure sizing.

WHAT THIS DOES
--------------
Most ML overlays emit a *point* forecast and then someone bets on its sign.
That throws away the single most useful piece of information: how trustworthy is
this particular forecast? Here we replace the point estimate with a
STATISTICALLY CALIBRATED prediction interval on the next-period return and only
let the book take risk when the model is confident enough to exclude zero from
that interval.

The result is an EXPOSURE multiplier in [0, 1], indexed by ``panel.close.index``,
that any strategy can multiply its book by. It does not pick a direction; it
gates *how much* the underlying strategy is allowed to trade.

WHY SPLIT-CONFORMAL (THE NO-LEAKAGE, DISTRIBUTION-FREE DESIGN)
--------------------------------------------------------------
Split-conformal prediction gives finite-sample, distribution-free coverage
guarantees with only one assumption — exchangeability of the calibration and
test residuals — and almost no extra cost on top of a fitted regressor:

  1. Split the trailing window into a FIT set and a held-out CALIBRATION set.
  2. Fit the regressor on the fit set only.
  3. Score nonconformity on calibration points: s_i = |y_i - yhat_i|.
  4. q = empirical (1 - alpha) quantile of {s_i}. The interval for any new
     point is [yhat - q, yhat + q], which covers the truth ~(1 - alpha) of the
     time when residuals are exchangeable.

CAUSALITY / NO REPAINTING
-------------------------
* Features at date d are strictly backward-looking (momentum, realized vol,
  RSI) — see ``_build_features``.
* The target at d is the forward return close[d+h]/close[d]-1, which is only
  *known* at d+h. So when we train/calibrate as of a cutoff T we use only rows
  whose label is already realized: d + h <= T. The calibration set is the most
  recent ``calib_frac`` of those eligible rows.
* We then score the genuinely out-of-sample days that fall after the cutoff and
  before the next refit. Those days were never part of any fit or calibration
  set, so every interval is a real-time forecast.

CONFIDENCE -> EXPOSURE
----------------------
An asset is "confidently directional" on an OOS day when |yhat| > q: the whole
interval [yhat - q, yhat + q] sits on one side of zero, so its sign is trusted.
The per-DATE exposure is the FRACTION of assets that are confidently directional
that day, in [0, 1]. It is forward-filled across each refit window and defaults
to 1.0 during warm-up (before the first calibration exists) — we never gate on
days we could not have scored.

HONEST DIAGNOSTIC
-----------------
``empirical_coverage`` is the truth-teller: the fraction of OOS actuals that
landed inside their own interval. With valid calibration it should sit near
1 - alpha (e.g. ~0.90 at alpha=0.10). If it does not, the conformal assumption
is being violated and the gate should not be trusted — so we report it plainly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import Panel
from ..features import indicators as ind
from ..seeds import set_global_seed

# Momentum feature lookbacks (trading days). A few horizons give the regressor
# both fast and slow trend information without exploding the feature count.
_MOM_LOOKBACKS = (10, 30, 90)
# RSI period (Wilder's classic 14).
_RSI_PERIOD = 14
# Approximate trading days per year (crypto trades 365/yr; the rest of the
# project uses 365, so we stay consistent for window sizing).
_DAYS_PER_YEAR = 365


@dataclass
class ConformalResult:
    """Output of the conformal confidence gate.

    Attributes
    ----------
    exposure_scale:
        pandas Series in [0, 1] indexed by ``panel.close.index``. Per date it is
        the fraction of assets that were confidently directional out-of-sample
        (interval excludes zero), forward-filled across each refit window. 1.0
        for warm-up dates before the first calibration.
    empirical_coverage:
        Fraction of OOS actual forward returns that fell inside their own
        [yhat - q, yhat + q] interval. Should be ~ (1 - alpha). This VALIDATES
        the calibration; NaN-safe and 0.0 if nothing was scored.
    mean_exposure:
        Mean of ``exposure_scale`` over all dates — how often the gate lets the
        book trade.
    n_refits:
        Number of walk-forward refits performed.
    status:
        Human-readable status string (also reports graceful degradation).
    """

    exposure_scale: pd.Series
    empirical_coverage: float
    mean_exposure: float
    n_refits: int
    status: str


# ---------------------------------------------------------------------------
# Feature / target construction (strictly causal)
# ---------------------------------------------------------------------------
def _build_features(panel: Panel, cfg: Config) -> dict[str, pd.DataFrame]:
    """Build backward-looking per-asset features aligned to ``panel.close``.

    Returns a dict ``feature_name -> DataFrame(dates x assets)``. Every feature
    uses only information available at date t (rolling windows, no forward
    shifts), so the feature vector for date t is causal.
    """
    close = panel.close
    vol_lookback = int(cfg.strategy.vol_lookback)

    feats: dict[str, pd.DataFrame] = {}
    for lb in _MOM_LOOKBACKS:
        feats[f"mom_{lb}"] = ind.momentum(close, lb)
    feats["rvol"] = ind.realized_vol(close, vol_lookback, annualize=False)
    feats["rsi14"] = ind.rsi(close, _RSI_PERIOD)
    return feats


def _forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward return over ``horizon`` days: close[t+h]/close[t] - 1.

    NOTE: this is the *label*. Its value at date t is only known at t+h; the
    walk-forward loop below enforces that we only ever train/calibrate on rows
    whose label is already realized as of the cutoff.
    """
    return close.shift(-horizon) / close - 1.0


def _stack_rows(
    feats: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    assets: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Flatten per-asset (date, asset) feature/target panels into row arrays.

    Returns
    -------
    X        : (n_rows, n_features) feature matrix
    y        : (n_rows,) forward-return labels
    row_date : (n_rows,) integer position of the *feature* date d
    label_pos: (n_rows,) integer position of the *label* date d + h
    cols     : ordered feature column names
    """
    cols = list(feats.keys())
    close_index_len = len(target.index)
    horizon_missing = target.isna()  # rows where d+h is out of range -> drop

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    rowpos_parts: list[np.ndarray] = []
    labelpos_parts: list[np.ndarray] = []

    # Per asset, assemble a (dates x features) block, attach the label, and keep
    # only fully-observed rows with a realized forward return.
    for asset in assets:
        block = pd.DataFrame(
            {c: feats[c][asset] for c in cols},
            index=target.index,
        )
        y_asset = target[asset]
        valid = block.notna().all(axis=1) & y_asset.notna()
        if not valid.any():
            continue
        pos = np.flatnonzero(valid.to_numpy())
        X_parts.append(block.to_numpy(dtype=float)[pos])
        y_parts.append(y_asset.to_numpy(dtype=float)[pos])
        rowpos_parts.append(pos)
        labelpos_parts.append(pos)  # placeholder; horizon added by caller

    if not X_parts:
        empty = np.empty((0,), dtype=float)
        return (
            np.empty((0, len(cols)), dtype=float),
            empty,
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
            cols,
        )

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    row_date = np.concatenate(rowpos_parts)
    label_pos = np.concatenate(labelpos_parts)
    _ = (close_index_len, horizon_missing)  # documented, not otherwise needed
    return X, y, row_date, label_pos, cols


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def conformal_exposure(panel: Panel, cfg: Config) -> ConformalResult:
    """Walk-forward split-conformal confidence gate -> exposure multiplier.

    Produces calibrated prediction intervals on next-period (``cfg.ml.horizon``)
    returns and converts model confidence into a per-date exposure scale in
    [0, 1]. Strictly causal: training/calibration only use rows whose forward
    label is already realized at the cutoff, and scored days are out-of-sample.

    Degrades gracefully: any failure (missing sklearn, too little history, etc.)
    returns an all-1.0 exposure with an explanatory status rather than raising.
    """
    set_global_seed(cfg.seed)

    dates = panel.close.index
    n_dates = len(dates)
    # Default output: full exposure everywhere (the safe, no-op gate).
    exposure = pd.Series(1.0, index=dates, dtype=float)

    def _degrade(msg: str) -> ConformalResult:
        return ConformalResult(
            exposure_scale=exposure,
            empirical_coverage=0.0,
            mean_exposure=float(exposure.mean()) if n_dates else 1.0,
            n_refits=0,
            status=msg,
        )

    # --- config / dependency guards -------------------------------------------
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as exc:  # pragma: no cover - exercised only without sklearn
        return _degrade(
            f"sklearn unavailable ({exc}); conformal gate disabled, exposure=1.0"
        )

    try:
        horizon = int(cfg.ml.horizon)
        alpha = float(cfg.ml.conformal.alpha)
        train_years = float(cfg.ml.conformal.train_years)
        calib_frac = float(cfg.ml.conformal.calib_frac)
        refit_every = int(cfg.ml.conformal.refit_every)
    except Exception as exc:
        return _degrade(f"bad conformal config ({exc}); exposure=1.0")

    if not (0.0 < alpha < 1.0) or not (0.0 < calib_frac < 1.0):
        return _degrade(
            f"invalid alpha={alpha} or calib_frac={calib_frac}; exposure=1.0"
        )
    if horizon < 1 or refit_every < 1 or n_dates == 0:
        return _degrade("invalid horizon/refit_every or empty panel; exposure=1.0")

    train_window = int(round(train_years * _DAYS_PER_YEAR))
    if train_window < 2:
        return _degrade("train window too small; exposure=1.0")

    assets = list(panel.assets)
    n_assets = len(assets)
    if n_assets == 0:
        return _degrade("no assets in panel; exposure=1.0")

    # --- causal feature / target panels ---------------------------------------
    try:
        feats = _build_features(panel, cfg)
        target = _forward_return(panel.close, horizon)
        X_all, y_all, rowpos_all, _labelpos, _cols = _stack_rows(feats, target, assets)
        # The label date for a feature row at position p is p + horizon.
        labelpos_all = rowpos_all + horizon
    except Exception as exc:
        return _degrade(f"feature/target build failed ({exc}); exposure=1.0")

    if X_all.shape[0] == 0:
        return _degrade("no fully-observed feature/label rows; exposure=1.0")

    # Per-date exposure accumulators over OOS scoring days.
    date_conf_sum = np.zeros(n_dates, dtype=float)   # sum of confident-directional flags
    date_conf_cnt = np.zeros(n_dates, dtype=float)   # number of (asset) predictions
    scored_any = np.zeros(n_dates, dtype=bool)

    # Coverage accounting across all OOS predictions.
    cover_hits = 0
    cover_total = 0
    n_refits = 0

    # First cutoff: the earliest position at which a full train window of
    # label-realized rows could plausibly exist. We require at least
    # train_window feature dates of history before the cutoff.
    first_cutoff = train_window + horizon
    if first_cutoff >= n_dates:
        return _degrade(
            f"insufficient history: need > {first_cutoff} dates, have {n_dates}; "
            f"exposure=1.0"
        )

    # Walk forward over cutoffs, refitting every ``refit_every`` days. At cutoff
    # T (a position index into ``dates``):
    #   * eligible TRAIN/CALIB rows: feature pos in [T - train_window, T) AND
    #     label pos (= feature pos + horizon) <= T  -> label already realized.
    #   * OOS scoring days: positions (T, min(T + refit_every, n_dates)).
    cutoffs = list(range(first_cutoff, n_dates, refit_every))

    for T in cutoffs:
        train_lo = T - train_window

        # Eligible rows: in window and label realized as of T (no leakage).
        elig = (
            (rowpos_all >= train_lo)
            & (rowpos_all < T)
            & (labelpos_all <= T)
        )
        if elig.sum() < 10:
            continue  # not enough realized labels to fit + calibrate meaningfully

        X_e = X_all[elig]
        y_e = y_all[elig]
        rp_e = rowpos_all[elig]

        # Order eligible rows by feature date so the calibration set is the most
        # RECENT slice — the part of history closest to the OOS days we score.
        order = np.argsort(rp_e, kind="stable")
        X_e = X_e[order]
        y_e = y_e[order]

        n_e = len(y_e)
        n_calib = int(round(calib_frac * n_e))
        n_calib = max(1, min(n_calib, n_e - 1))  # keep >=1 in each split
        n_fit = n_e - n_calib
        if n_fit < 1:
            continue

        X_fit, y_fit = X_e[:n_fit], y_e[:n_fit]
        X_cal, y_cal = X_e[n_fit:], y_e[n_fit:]

        # --- fit regressor on the fit set only --------------------------------
        try:
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=int(cfg.seed),
                n_jobs=-1,
            )
            model.fit(X_fit, y_fit)
        except Exception:
            continue

        # --- split-conformal calibration -------------------------------------
        pred_cal = model.predict(X_cal)
        scores = np.abs(y_cal - pred_cal)            # nonconformity scores
        # Finite-sample conformal quantile level. Using the (1 - alpha) empirical
        # quantile of the calibration scores is the standard split-conformal q.
        q = float(np.quantile(scores, 1.0 - alpha, method="higher"))
        n_refits += 1

        # --- OOS scoring window -----------------------------------------------
        oos_lo = T
        oos_hi = min(T + refit_every, n_dates)

        # For each OOS feature date in the window, predict per asset and gate.
        for asset_idx, asset in enumerate(assets):
            block = pd.DataFrame(
                {c: feats[c][asset] for c in feats.keys()},
                index=dates,
            )
            y_asset = target[asset]
            for pos in range(oos_lo, oos_hi):
                row = block.iloc[pos]
                if row.isna().any():
                    continue
                yhat = float(model.predict(row.to_numpy(dtype=float).reshape(1, -1))[0])

                # Confidence -> directional gate: interval excludes zero.
                confident = abs(yhat) > q
                date_conf_sum[pos] += 1.0 if confident else 0.0
                date_conf_cnt[pos] += 1.0
                scored_any[pos] = True

                # Coverage check uses the realized label (known only ex-post, used
                # purely as a DIAGNOSTIC — it never feeds the live exposure).
                actual = y_asset.iloc[pos]
                if np.isfinite(actual):
                    lo, hi = yhat - q, yhat + q
                    cover_total += 1
                    if lo <= actual <= hi:
                        cover_hits += 1

    if n_refits == 0:
        return _degrade(
            "no refit had enough label-realized rows to calibrate; exposure=1.0"
        )

    # --- assemble per-date exposure -------------------------------------------
    # Where we scored, exposure = fraction of assets confidently directional.
    # Where we did not score (warm-up before first cutoff, or gaps), leave 1.0
    # and forward-fill the gate across each refit window via the running state.
    raw = np.full(n_dates, np.nan, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(date_conf_cnt > 0, date_conf_sum / date_conf_cnt, np.nan)
    raw[scored_any] = frac[scored_any]

    gate = pd.Series(raw, index=dates, dtype=float)
    # Forward-fill the gate across days inside a refit window that had no fully
    # observed feature rows; warm-up (leading NaNs) stays at 1.0.
    gate = gate.ffill()
    first_scored_pos = int(np.argmax(scored_any)) if scored_any.any() else n_dates
    exposure = pd.Series(1.0, index=dates, dtype=float)
    if first_scored_pos < n_dates:
        exposure.iloc[first_scored_pos:] = gate.iloc[first_scored_pos:].fillna(1.0)
    exposure = exposure.clip(0.0, 1.0)

    empirical_coverage = float(cover_hits / cover_total) if cover_total else 0.0
    mean_exposure = float(exposure.mean())

    first_scored_date = dates[first_scored_pos] if first_scored_pos < n_dates else None
    status = (
        f"ok: split-conformal gate, horizon={horizon}d, alpha={alpha} "
        f"(target coverage {1 - alpha:.2f}), train_window={train_window}d, "
        f"calib_frac={calib_frac}, refit_every={refit_every}d; "
        f"{n_refits} refits, {cover_total} OOS predictions, "
        f"empirical_coverage={empirical_coverage:.3f}, "
        f"mean_exposure={mean_exposure:.3f}; "
        f"first scored "
        f"{first_scored_date.date() if first_scored_date is not None else 'n/a'}, "
        f"warm-up exposure=1.0"
    )

    return ConformalResult(
        exposure_scale=exposure,
        empirical_coverage=empirical_coverage,
        mean_exposure=mean_exposure,
        n_refits=n_refits,
        status=status,
    )
