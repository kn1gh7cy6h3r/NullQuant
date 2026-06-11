"""
lstm_forecast.py — an honestly-evaluated, stationary-target LSTM forecaster.

WHY THIS EXISTS (the reframing)
-------------------------------
The original LSTM in this project was a textbook example of how to fool
yourself with deep learning on financial data:

  * It predicted RAW PRICE LEVELS. A network that simply echoes the last close
    ("tomorrow ~= today") earns a tiny MSE on a trending series and looks
    brilliant, while forecasting nothing of value. Price levels are also
    non-stationary, so the learned mapping does not generalize forward.
  * It fit the feature scaler on the FULL series (train + test together), which
    leaks the test-period mean/scale into training — lookahead by construction.
  * It reported IN-SAMPLE error, so there was no honest read on generalization.

This module fixes all three:

  1. STATIONARY TARGET. We predict either the summed log-return over the next
     `horizon` days ("logret") or the realized vol over that horizon ("vol").
     Both are (approximately) stationary, so a low error actually means the
     model learned something, not that prices trend.

  2. STRICT CHRONOLOGICAL SPLIT (70/15/15). Train is the oldest slice, then val,
     then test — the most recent slice. The scaler and every summary statistic
     are fit on TRAIN ONLY and then applied to val/test. We never shuffle across
     the time boundary (we may shuffle *within* train for mini-batching).

  3. HONEST OOS EVALUATION vs A NAIVE BASELINE. We benchmark against the
     random-walk null: for the logret target the naive forecast is 0 (efficient
     market — best guess of future drift is nothing); for the vol target it is
     the last observed realized vol (vol is persistent). We report OOS RMSE and
     directional accuracy for BOTH the LSTM and the baseline, plus a boolean
     `beats_baseline`. On daily crypto, beating the random walk OOS is genuinely
     hard, and reporting "it does not" is a valid, expected result.

CAUSALITY
---------
Every training/eval example is (window of past closes ending at t) -> (target
strictly after t). The input window never overlaps the target horizon, so there
is no leakage from future bars into the features.

The module degrades gracefully: if keras/torch are unavailable it returns a
result with `status` explaining why and `beats_baseline=False`, rather than
raising — the surrounding pipeline must not hard-fail on an optional model.
"""

from __future__ import annotations

# Keras must see its backend selection BEFORE it is imported anywhere.
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import Panel
from ..seeds import set_global_seed


# ===========================================================================
# Result container
# ===========================================================================
@dataclass
class LSTMForecastResult:
    """Outcome of training + honest OOS evaluation for a single asset."""

    asset: str
    target: str                  # "logret" or "vol"
    oos_rmse: float              # LSTM root-mean-squared error on the test slice
    baseline_rmse: float         # naive/random-walk RMSE on the same slice
    oos_dir_acc: float           # LSTM directional accuracy (sign match) on test
    baseline_dir_acc: float      # naive directional accuracy on test
    beats_baseline: bool         # True iff LSTM oos_rmse < baseline_rmse
    latest_forecast: float       # model's prediction for the most recent window
    n_train: int                 # number of training examples
    n_test: int                  # number of test examples
    status: str                  # "ok" or a human-readable failure reason


# ===========================================================================
# Supervised-window construction (strictly causal)
# ===========================================================================
def _build_targets(closes: pd.Series, horizon: int, target: str,
                   vol_window: int) -> pd.Series:
    """
    Compute the stationary target value *anchored at each date t*, where the
    value summarizes the window (t, t+horizon]. Anchoring at t means: "standing
    at the close of day t, what is the target over the next `horizon` days?".

    logret -> sum of the next `horizon` daily log-returns  = log(C[t+h] / C[t]).
    vol    -> realized (daily, non-annualized) vol of the next `horizon`
              log-returns. We use raw daily std so it lives on the same scale as
              a single-day move and is comparable across horizons.

    Dates whose horizon extends past the end of the series are NaN (no peeking).
    """
    logret = np.log(closes / closes.shift(1))            # r[t] = log(C[t]/C[t-1])
    if target == "logret":
        # Sum of r[t+1..t+h] = log(C[t+h]) - log(C[t]); shift(-h) anchors at t.
        fut = np.log(closes).shift(-horizon) - np.log(closes)
        return fut
    elif target == "vol":
        # std of the next `horizon` daily returns, anchored at t.
        # rolling().std() is backward-looking; shift(-horizon) re-anchors so the
        # value at t describes returns r[t+1..t+h].
        fwd_std = logret.rolling(horizon, min_periods=horizon).std().shift(-horizon)
        return fwd_std
    else:
        raise ValueError(f"Unknown target '{target}'; expected 'logret' or 'vol'.")


def _make_windows(closes: pd.Series, window: int, horizon: int, target: str,
                  vol_window: int):
    """
    Build (X, y, dates, last_window) for supervised learning.

    Each sample uses the `window` daily log-returns ending at date t as the
    input feature sequence, and the stationary target anchored at t (covering
    (t, t+horizon]) as the label. Using log-returns (not raw prices) as the
    network input keeps the *inputs* stationary too.

    Returns
    -------
    X : (n, window, 1) float32   — past log-return sequences
    y : (n,) float32             — stationary targets
    dates : DatetimeIndex        — the anchor date t for each sample
    last_window : (1, window, 1) — the most recent fully-observed window, whose
                                   target lies in the (unobservable) future; used
                                   to produce `latest_forecast`.
    """
    logret = np.log(closes / closes.shift(1))
    y_full = _build_targets(closes, horizon, target, vol_window)

    r = logret.to_numpy(dtype="float64")
    y_all = y_full.to_numpy(dtype="float64")
    idx = closes.index

    X_list, y_list, date_list = [], [], []
    # Anchor t at position p; input window is r[p-window+1 .. p] (needs r, so
    # p >= window because r[0] is NaN). Target y_all[p] covers (t, t+horizon].
    n = len(closes)
    for p in range(window, n):
        seq = r[p - window + 1: p + 1]
        if seq.shape[0] != window or not np.isfinite(seq).all():
            continue
        yv = y_all[p]
        if np.isfinite(yv):
            X_list.append(seq)
            y_list.append(yv)
            date_list.append(idx[p])

    # The most recent window for which we have all inputs but NOT the target
    # (target lies in the future) — used for the forward-looking forecast.
    last_window = None
    for p in range(n - 1, window - 1, -1):
        seq = r[p - window + 1: p + 1]
        if seq.shape[0] == window and np.isfinite(seq).all():
            last_window = seq.reshape(1, window, 1).astype("float32")
            break

    X = np.asarray(X_list, dtype="float32").reshape(-1, window, 1)
    y = np.asarray(y_list, dtype="float32")
    dates = pd.DatetimeIndex(date_list)
    return X, y, dates, last_window


# ===========================================================================
# Scaling (fit on TRAIN ONLY)
# ===========================================================================
class _StandardScaler1D:
    """Minimal mean/std standardizer; stats are fit on training data only."""

    def __init__(self) -> None:
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit(self, a: np.ndarray) -> "_StandardScaler1D":
        self.mean_ = float(np.nanmean(a))
        std = float(np.nanstd(a))
        self.std_ = std if std > 1e-12 else 1.0
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mean_) / self.std_

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        return a * self.std_ + self.mean_


# ===========================================================================
# Metrics
# ===========================================================================
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of samples where sign(pred) matches sign(true). Zeros count as up."""
    st = np.sign(y_true)
    sp = np.sign(y_pred)
    st[st == 0] = 1.0
    sp[sp == 0] = 1.0
    return float(np.mean(st == sp))


# ===========================================================================
# Public API — train + honest OOS evaluation
# ===========================================================================
def train_eval_lstm(panel: Panel, cfg: Config, asset: str = "BTC-USD") -> LSTMForecastResult:
    """
    Train an LSTM on the oldest 70% of windows, validate on the next 15%, and
    evaluate honestly out-of-sample on the most recent 15% — against a naive
    random-walk baseline.

    See the module docstring for the full design rationale. Returns an
    `LSTMForecastResult`; on any infrastructure failure (missing keras/torch,
    too little data, training error) it returns a result whose `status`
    explains the issue and `beats_baseline=False`, rather than raising.
    """
    lc = cfg.ml.lstm
    window = int(lc.window)
    horizon = int(lc.horizon)
    units = int(lc.units)
    dropout = float(lc.dropout)
    epochs = int(lc.epochs)
    batch_size = int(lc.batch_size)
    target = str(lc.target)
    vol_window = int(cfg.get("strategy.vol_lookback", horizon))

    def _fail(status: str) -> LSTMForecastResult:
        return LSTMForecastResult(
            asset=asset, target=target,
            oos_rmse=float("nan"), baseline_rmse=float("nan"),
            oos_dir_acc=float("nan"), baseline_dir_acc=float("nan"),
            beats_baseline=False, latest_forecast=float("nan"),
            n_train=0, n_test=0, status=status,
        )

    if asset not in panel.close.columns:
        return _fail(f"asset '{asset}' not in panel (have {list(panel.close.columns)})")

    closes = panel.close[asset].dropna()
    if len(closes) < (window + horizon + 50):
        return _fail(f"insufficient history: {len(closes)} closes for "
                     f"window={window}, horizon={horizon}")

    X, y, dates, last_window = _make_windows(closes, window, horizon, target, vol_window)
    n = len(y)
    if n < 40 or last_window is None:
        return _fail(f"too few usable windows ({n}) after causal construction")

    # --- Strict chronological split: 70 / 15 / 15 (oldest -> newest). ----------
    i_tr = int(n * 0.70)
    i_va = int(n * 0.85)
    if i_tr < 1 or (n - i_va) < 1 or (i_va - i_tr) < 1:
        return _fail(f"split degenerate for n={n} windows")

    X_tr, y_tr = X[:i_tr], y[:i_tr]
    X_va, y_va = X[i_tr:i_va], y[i_tr:i_va]
    X_te, y_te = X[i_va:], y[i_va:]

    # --- Scalers fit on TRAIN ONLY, then applied to val/test (no leakage). -----
    x_scaler = _StandardScaler1D().fit(X_tr.reshape(-1))
    y_scaler = _StandardScaler1D().fit(y_tr.reshape(-1))

    def _sx(a: np.ndarray) -> np.ndarray:
        return x_scaler.transform(a.reshape(-1)).reshape(a.shape).astype("float32")

    Xtr_s, Xva_s, Xte_s = _sx(X_tr), _sx(X_va), _sx(X_te)
    ytr_s = y_scaler.transform(y_tr).astype("float32")
    yva_s = y_scaler.transform(y_va).astype("float32")
    last_s = x_scaler.transform(last_window.reshape(-1)).reshape(last_window.shape).astype("float32")

    # --- Naive / random-walk baseline on the TEST slice. -----------------------
    # logret: best guess of future drift is 0 (efficient-market null).
    # vol   : best guess of next-horizon vol is the last realized vol, which for
    #         each test anchor is the input window's own daily std (causal).
    if target == "vol":
        baseline_pred = X_te.reshape(X_te.shape[0], -1).std(axis=1)
    else:
        baseline_pred = np.zeros_like(y_te)

    baseline_rmse = _rmse(y_te, baseline_pred)
    baseline_dir_acc = _dir_acc(y_te, baseline_pred)

    # --- Build + train the model (defensive on optional heavy imports). --------
    set_global_seed(cfg.seed)
    try:
        import keras
        from keras import layers
    except Exception as exc:  # keras/torch backend unavailable
        return _fail(f"keras/torch unavailable: {type(exc).__name__}: {exc}")

    try:
        model = keras.Sequential([
            layers.Input(shape=(window, 1)),
            layers.LSTM(units, dropout=dropout),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            Xtr_s, ytr_s,
            validation_data=(Xva_s, yva_s),
            epochs=epochs, batch_size=batch_size,
            shuffle=True,        # shuffling WITHIN train is fine; split is fixed.
            verbose=0,
        )

        # Predict on test, invert the target scaling back to target units.
        pred_te_s = model.predict(Xte_s, verbose=0).reshape(-1)
        pred_te = y_scaler.inverse_transform(pred_te_s)

        latest_s = model.predict(last_s, verbose=0).reshape(-1)[0]
        latest_forecast = float(y_scaler.inverse_transform(np.array([latest_s]))[0])
    except Exception as exc:  # training/predict failure — degrade, don't crash
        return _fail(f"training failed: {type(exc).__name__}: {exc}")

    oos_rmse = _rmse(y_te, pred_te)
    oos_dir_acc = _dir_acc(y_te, pred_te)
    beats = bool(oos_rmse < baseline_rmse)

    return LSTMForecastResult(
        asset=asset, target=target,
        oos_rmse=oos_rmse, baseline_rmse=baseline_rmse,
        oos_dir_acc=oos_dir_acc, baseline_dir_acc=baseline_dir_acc,
        beats_baseline=beats, latest_forecast=latest_forecast,
        n_train=int(i_tr), n_test=int(n - i_va), status="ok",
    )


# ===========================================================================
# Public API — directional signal for the ablation harness
# ===========================================================================
def forecast_signal(panel: Panel, cfg: Config, asset: str = "BTC-USD") -> pd.Series:
    """
    Per-date directional view in {-1, 0, +1} = sign of the model's forecast of
    the next-horizon log-return, intended as an OPTIONAL tilt for the ablation
    harness.

    SIMPLIFICATION (documented, not hidden): a true walk-forward would refit the
    LSTM at every rebalance date, which is far too slow for the small epoch
    budget used here. Instead we train ONCE on the chronological train slice
    (the oldest 70% of windows, scalers fit on train only) and then emit signals
    only over the held-out test dates — i.e. the signal series is strictly
    out-of-sample relative to the single fit. This is causal (no test bar
    informs the fit) but is a single-fit approximation of a full walk-forward.

    For the "vol" target a directional log-return view is undefined, so we still
    derive the sign from a logret model: regardless of `cfg.ml.lstm.target`,
    this function trains on the *logret* target so the sign is meaningful.

    On any failure (missing keras/torch, too little data) an EMPTY Series is
    returned so the harness can simply skip the tilt.
    """
    lc = cfg.ml.lstm
    window = int(lc.window)
    horizon = int(lc.horizon)
    units = int(lc.units)
    dropout = float(lc.dropout)
    epochs = int(lc.epochs)
    batch_size = int(lc.batch_size)
    vol_window = int(cfg.get("strategy.vol_lookback", horizon))

    empty = pd.Series(dtype="float64", name=f"{asset}_lstm_dir")

    if asset not in panel.close.columns:
        return empty
    closes = panel.close[asset].dropna()
    if len(closes) < (window + horizon + 50):
        return empty

    # Always train on the logret target here so the sign is interpretable.
    X, y, dates, _ = _make_windows(closes, window, horizon, "logret", vol_window)
    n = len(y)
    if n < 40:
        return empty

    i_tr = int(n * 0.70)
    if i_tr < 1 or (n - i_tr) < 1:
        return empty

    X_tr, y_tr = X[:i_tr], y[:i_tr]
    X_te = X[i_tr:]
    dates_te = dates[i_tr:]

    x_scaler = _StandardScaler1D().fit(X_tr.reshape(-1))
    y_scaler = _StandardScaler1D().fit(y_tr.reshape(-1))

    def _sx(a: np.ndarray) -> np.ndarray:
        return x_scaler.transform(a.reshape(-1)).reshape(a.shape).astype("float32")

    set_global_seed(cfg.seed)
    try:
        import keras
        from keras import layers
    except Exception:
        return empty

    try:
        model = keras.Sequential([
            layers.Input(shape=(window, 1)),
            layers.LSTM(units, dropout=dropout),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            _sx(X_tr), y_scaler.transform(y_tr).astype("float32"),
            epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0,
        )
        pred_s = model.predict(_sx(X_te), verbose=0).reshape(-1)
        pred = y_scaler.inverse_transform(pred_s)
    except Exception:
        return empty

    sign = np.sign(pred)
    # The signal at anchor date t is a view for the position held from t+1 on;
    # the consumer (ablation harness) is responsible for the t+1 shift.
    return pd.Series(sign, index=dates_te, name=f"{asset}_lstm_dir").astype("float64")
