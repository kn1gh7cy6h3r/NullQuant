"""
ml_engine.py — Machine-learning intelligence layer for Meridian.

This module bolts three independent ML models onto the existing rule-based
trading system. None of them replace the SMA-crossover signal or the ATR risk
engine — they *augment* it with a forecast, a confidence score, and a circuit
breaker. Everything here is advisory model output, NOT financial advice.

────────────────────────────────────────────────────────────────────────────
THE THREE MODELS
────────────────────────────────────────────────────────────────────────────
  Model A — LSTM Price Forecast        (deep learning, Keras 3 on torch/MPS)
      A 2-layer LSTM trained on 60-day windows of closing prices that
      projects the next 7 days. Answers: "where might price drift next?"

  Model B — Random Forest Confidence   (scikit-learn)
      Learns, from the risk engine's own realised trade log, which market
      conditions produced *profitable* trades, then scores the live signal
      0–100%. Answers: "how trustworthy is this BUY/SELL right now?"

  Model C — Isolation Forest Anomaly   (scikit-learn)
      An unsupervised outlier detector over price/volume/volatility features.
      Flags candles that look nothing like normal history. Answers: "is the
      market behaving so strangely that we should stand aside?" — it acts as a
      circuit breaker that downgrades a BUY to CAUTION on an anomalous day.

────────────────────────────────────────────────────────────────────────────
DESIGN PRINCIPLES (shared by all three)
────────────────────────────────────────────────────────────────────────────
  • Strictly causal. Every feature at candle t uses only data observable at or
    before t's close (rolling windows are backward-looking; the LSTM only ever
    predicts the *unknown* future from the last 60 *known* closes). No
    look-ahead bias — the same discipline the risk engine already enforces.

  • Non-blocking. The LSTM is the only expensive model; it trains in a daemon
    background thread on a 24-hour cadence and the dashboard never waits on it.
    The Random Forest and Isolation Forest are cheap and train inline.

  • Fail-soft. If a heavy dependency is missing or a model has too little data,
    the corresponding result reports `ready=False` with a human-readable status
    and the dashboard simply hides that piece. The core dashboard keeps working.

  • Warm-up gates (realism):
        LSTM            — no forecast until ≥200 candles of history exist.
        Random Forest   — no confidence until ≥5 completed trades are logged.
        Isolation Forest— trains on full history, refreshes every 30s tick.
"""

from __future__ import annotations

# IMPORTANT: Keras must be told which backend to use *before* it is imported
# anywhere in the process. We use the PyTorch backend because it is the only
# stack that installs on this Python build and still gives real Apple-Silicon
# GPU acceleration (Metal) through torch's MPS device — the same GPU the
# (uninstallable on Py3.14) tensorflow-metal plugin would have targeted.
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
# Let torch silently fall back to CPU for the rare op MPS hasn't implemented,
# instead of hard-crashing the training thread.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── scikit-learn (Models B & C) — guarded so the app survives without it ───────
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    SKLEARN_AVAILABLE = True
    _SKLEARN_ERR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    SKLEARN_AVAILABLE = False
    _SKLEARN_ERR = str(exc)

# Keras + torch (Model A) are imported *lazily* inside the training thread.
# They are heavy (seconds to import) so deferring them keeps dashboard start-up
# snappy and lets the whole app run even if they are absent.


# ── Tunable hyper-parameters (all in one place, easy to audit) ─────────────────

# Model A — LSTM
LSTM_WINDOW = 60            # days of close prices fed in per training sample
LSTM_HORIZON = 7           # days predicted out
LSTM_MIN_CANDLES = 200     # warm-up: no forecast until this much history exists
LSTM_RETRAIN_SECONDS = 24 * 3600   # retrain cadence (24h) in the background
LSTM_EPOCHS = 30           # small net + small data ⇒ converges fast
LSTM_BATCH = 32
LSTM_UNITS = 50            # hidden units per LSTM layer
LSTM_DROPOUT = 0.2         # dropout between layers — regularises the forecast
LSTM_RECENT_ERRORS = 30    # how many recent samples define the ±1σ band

# Model B — Random Forest
RF_MIN_TRADES = 5          # warm-up: no confidence until this many CLOSED trades
RF_WEAK_THRESHOLD = 0.40   # <40% ⇒ WEAK (amber); ≥40% ⇒ STRONG (green/red)
RF_TREES = 200

# Model C — Isolation Forest
IF_LOOKBACK = 30           # rolling window for the z-score / volatility features
IF_CONTAMINATION = 0.03    # expected fraction of candles that are "anomalies"
IF_DISPLAY_DAYS = 30       # how far back the dashboard shades anomaly bands
IF_MIN_CANDLES = 60        # need a little history before outlier detection means much


# ════════════════════════════════════════════════════════════════════════════
#  RESULT CONTAINERS  — plain dataclasses the dashboard renders from.
#  Keeping all presentation-ready numbers in these structs means the dashboard
#  never has to know how any model works.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class LSTMResult:
    available: bool = False          # are the deep-learning libs importable?
    ready: bool = False              # trained model + a usable forecast exist
    training: bool = False           # a background retrain is in flight right now
    device: str = "cpu"             # "mps" (Apple GPU) or "cpu"
    forecast_dates: list = field(default_factory=list)   # next 7 calendar days
    forecast_prices: list = field(default_factory=list)  # predicted closes
    band_upper: list = field(default_factory=list)       # +1σ confidence band
    band_lower: list = field(default_factory=list)       # −1σ confidence band
    direction: str = "—"           # UP / DOWN / FLAT over the 7-day horizon
    target_price: float | None = None      # the day-7 predicted price
    last_close: float | None = None        # today's anchor price
    pct_change: float | None = None        # forecast move over the horizon, %
    status: str = "LSTM idle"        # short human-readable state for the UI


@dataclass
class RFResult:
    available: bool = False
    ready: bool = False
    current_confidence: float | None = None   # 0–100% for the live signal
    current_strength: str | None = None       # "STRONG" / "WEAK"
    current_signal: str | None = None         # the signal the score refers to
    # date(Timestamp) -> {"confidence": float, "strength": str} for each
    # historical BUY/SELL marker, so Panel 2 can label every signal.
    signal_confidences: dict = field(default_factory=dict)
    n_trades: int = 0
    status: str = "Random Forest idle"


@dataclass
class AnomalyResult:
    available: bool = False
    ready: bool = False
    today_anomalous: bool = False
    today_score: float | None = None          # raw outlier score (lower = weirder)
    anomaly_dates: list = field(default_factory=list)   # anomalies in display window
    recent_count: int = 0
    status: str = "Isolation Forest idle"


@dataclass
class MLResults:
    lstm: LSTMResult = field(default_factory=LSTMResult)
    rf: RFResult = field(default_factory=RFResult)
    anomaly: AnomalyResult = field(default_factory=AnomalyResult)

    @staticmethod
    def strength_label(confidence_pct: float) -> str:
        """Map a 0–100 confidence to the WEAK/STRONG bucket used everywhere."""
        return "STRONG" if confidence_pct >= RF_WEAK_THRESHOLD * 100 else "WEAK"


# ════════════════════════════════════════════════════════════════════════════
#  CAUSAL INDICATOR PRIMITIVES
#  Computed locally from raw OHLC so the engine never KeyErrors on whatever
#  columns the caller happens to pass, and stays consistent run-to-run.
# ════════════════════════════════════════════════════════════════════════════

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's 14-period RSI. RSI measures the speed/size of recent up-moves vs
    down-moves on a 0–100 scale; >70 is classically "overbought", <30
    "oversold". It's a momentum feature: the RF uses it to learn whether
    crossovers fired into stretched or fresh conditions. Purely backward-looking.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EMA with alpha = 1/period (matches the ATR convention).
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When there have been no losses at all, RS→∞ ⇒ RSI=100 (and vice-versa).
    return rsi.fillna(100.0).where(avg_loss != 0, 100.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    """
    14-period ATR (same definition as risk_manager.add_atr, recomputed locally
    so ml_engine is self-contained). ATR is the volatility unit; here we feed it
    to the models normalised by price so it is comparable across price regimes.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score: how many standard deviations the current value sits from
    its own trailing `window`-day mean. A regime-relative "how unusual is this?"
    measure, and the backbone of the anomaly features. Backward-looking.
    """
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0.0, np.nan)


# ── Volume: data_manager keeps only OHLC, so we source Volume ourselves ────────

_VOLUME_CACHE = Path(__file__).parent / "data" / "btc_volume.parquet"


def _ensure_volume(df: pd.DataFrame) -> pd.Series:
    """
    Return a Volume series aligned to df.index.

    Several features want trading volume, but Meridian's data_manager
    deliberately keeps only OHLC (the risk engine never needs volume). Rather
    than alter that module, we fetch the *same* BTC-USD series' Volume column
    from yfinance once, cache it locally, and reindex it onto df. This is the
    identical data source the rest of the app already uses — no new dataset.

    If volume is genuinely unavailable (offline, fetch error), we return zeros;
    the dependent z-score then becomes a neutral 0, degrading gracefully instead
    of crashing the dashboard.
    """
    # 1) Already supplied on the frame? Use it.
    if "Volume" in df.columns and df["Volume"].notna().any():
        return df["Volume"].reindex(df.index)

    # 2) Local cache covering today's range?
    try:
        if _VOLUME_CACHE.exists():
            cached = pd.read_parquet(_VOLUME_CACHE)["Volume"]
            cached.index = pd.to_datetime(cached.index)
            if cached.index.max() >= df.index.max() - pd.Timedelta(days=2):
                return cached.reindex(df.index).ffill()
    except Exception:
        pass  # fall through to a fresh fetch

    # 3) Fetch from yfinance (best-effort) and cache.
    try:
        import yfinance as yf

        start = (df.index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end = (df.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download("BTC-USD", start=start, end=end, interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        vol = raw["Volume"].copy()
        vol.index = pd.to_datetime(vol.index).tz_localize(None)
        try:
            _VOLUME_CACHE.parent.mkdir(exist_ok=True)
            vol.to_frame("Volume").to_parquet(_VOLUME_CACHE)
        except Exception:
            pass
        return vol.reindex(df.index).ffill()
    except Exception:
        # 4) Last resort: neutral zeros (volume z-score becomes 0).
        return pd.Series(0.0, index=df.index)


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def build_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model B feature matrix — one row per candle, all strictly causal.

    Why these features? They describe the *trend regime and stretch* a signal
    fires into, which is exactly what separates a crossover that follows through
    from a whipsaw:

      sma_ratio    SMA50/SMA200 — the crossover itself, as a smooth ratio.
                   >1 = bullish structure, <1 = bearish. The headline trend.
      dist_sma50   (Close−SMA50)/SMA50 — how far price has run from its fast
                   mean; large values mean an extended (riskier) entry.
      rsi14        14-day RSI — momentum/overbought-oversold context.
      atr_norm     ATR/Close — current volatility as a fraction of price; high
                   vol crossovers behave differently from calm ones.
      vol7, vol30  7- and 30-day stdev of daily returns — short- vs medium-term
                   realised volatility; their gap captures vol expansion.
      vol_z        30-day z-score of trading volume — is participation unusually
                   high/low? Breakouts on volume tend to stick.
      dow          day of week (0–6) — captures any weekday seasonality in BTC.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"] if "High" in df else close
    low = df["Low"] if "Low" in df else close

    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    rets = close.pct_change()
    volume = _ensure_volume(df)

    out["sma_ratio"] = sma50 / sma200
    out["dist_sma50"] = (close - sma50) / sma50
    out["rsi14"] = _rsi(close, 14)
    out["atr_norm"] = _atr(high, low, close, 14) / close
    out["vol7"] = rets.rolling(7, min_periods=7).std()
    out["vol30"] = rets.rolling(30, min_periods=30).std()
    out["vol_z"] = _rolling_z(volume, IF_LOOKBACK)
    out["dow"] = df.index.dayofweek.astype(float)

    # vol_z is the only feature allowed to be a benign 0 (missing volume);
    # everything else NaN means "still in warm-up" and that row is unusable.
    out["vol_z"] = out["vol_z"].fillna(0.0)
    return out


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model C feature matrix — the dimensions along which a candle can be "weird".

      price_z30    30-day z-score of close — abnormal price level vs recent range.
      vol_z30      30-day z-score of volume — abnormal participation.
      ret1         1-day return — captures violent single-day moves / flash events.
      vol7         7-day realised volatility — turbulence clustering.
      atr_ratio    ATR / 30-day-avg ATR — is volatility itself spiking vs normal?

    An Isolation Forest needs only a handful of well-chosen, scale-comparable
    axes; these five together fingerprint crashes, melt-ups, and liquidity
    shocks — precisely the conditions a trend system should NOT blindly trade.
    All causal.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"] if "High" in df else close
    low = df["Low"] if "Low" in df else close
    volume = _ensure_volume(df)

    atr = _atr(high, low, close, 14)
    out["price_z30"] = _rolling_z(close, IF_LOOKBACK)
    out["vol_z30"] = _rolling_z(volume, IF_LOOKBACK).fillna(0.0)
    out["ret1"] = close.pct_change()
    out["vol7"] = close.pct_change().rolling(7, min_periods=7).std()
    out["atr_ratio"] = atr / atr.rolling(IF_LOOKBACK, min_periods=IF_LOOKBACK).mean()
    return out


# ════════════════════════════════════════════════════════════════════════════
#  THE ENGINE
#  A single long-lived instance (see the `engine` singleton at the bottom) holds
#  all model state across the dashboard's 30-second refresh ticks.
# ════════════════════════════════════════════════════════════════════════════

class MLEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()      # guards all LSTM swap-in state below

        # ── Model A (LSTM) persistent state ──
        self._lstm_model = None            # the trained Keras model
        self._lstm_scaler = None           # MinMaxScaler fit on closes
        self._lstm_resid_std = 0.0         # ±1σ band width (price units)
        self._lstm_last_train = 0.0        # epoch seconds of last successful train
        self._lstm_training = False        # a background retrain is running
        self._lstm_device = "cpu"
        self._lstm_error = ""

        # ── Model B (Random Forest) persistent state ──
        self._rf_model = None
        self._rf_classes = None            # model.classes_ snapshot
        self._rf_trade_count = -1          # last trade count we trained on
        self._rf_error = ""

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC ENTRY POINT — called once per dashboard refresh.
    # ─────────────────────────────────────────────────────────────────────────
    def update(self, df: pd.DataFrame, trades: list[dict],
               current_signal: str | None = None) -> MLResults:
        """
        Run/refresh all three models against the latest data and return a fully
        populated MLResults. Cheap models (RF, IF) run inline; the LSTM only
        kicks off a background retrain when due and otherwise just re-runs fast
        inference with the current weights.

        `current_signal` is the live SMA regime ("BUY"/"SELL"/"NEUTRAL") that the
        RF confidence gauge should describe.
        """
        results = MLResults()
        # Each model is wrapped so one failing never takes down the others or
        # the dashboard. Failures surface as a status string in the UI.
        try:
            results.anomaly = self._run_isolation_forest(df)
        except Exception as exc:  # pragma: no cover
            results.anomaly = AnomalyResult(available=SKLEARN_AVAILABLE,
                                            status=f"Anomaly error: {exc}")
        try:
            results.rf = self._run_random_forest(df, trades, current_signal)
        except Exception as exc:  # pragma: no cover
            results.rf = RFResult(available=SKLEARN_AVAILABLE,
                                  status=f"Confidence error: {exc}")
        try:
            results.lstm = self._run_lstm(df)
        except Exception as exc:  # pragma: no cover
            results.lstm = LSTMResult(status=f"LSTM error: {exc}")

        return results

    # ═════════════════════════════════════════════════════════════════════════
    #  MODEL C — ISOLATION FOREST (anomaly / circuit breaker)
    #  Cheap & unsupervised: retrain on the full history every tick, as specced.
    # ═════════════════════════════════════════════════════════════════════════
    def _run_isolation_forest(self, df: pd.DataFrame) -> AnomalyResult:
        res = AnomalyResult(available=SKLEARN_AVAILABLE)
        if not SKLEARN_AVAILABLE:
            res.status = "scikit-learn not installed"
            return res

        feats = build_anomaly_features(df).replace([np.inf, -np.inf], np.nan)
        clean = feats.dropna()
        if len(clean) < IF_MIN_CANDLES:
            res.status = f"Warming up ({len(clean)}/{IF_MIN_CANDLES} candles)"
            return res

        # Standardise so no single feature dominates the distance geometry,
        # then isolate outliers. contamination sets how aggressive we are.
        X = StandardScaler().fit_transform(clean.values)
        model = IsolationForest(
            n_estimators=200,
            contamination=IF_CONTAMINATION,
            random_state=42,
        )
        labels = model.fit_predict(X)               # -1 = anomaly, +1 = normal
        scores = model.score_samples(X)             # lower = more anomalous

        flags = pd.Series(labels == -1, index=clean.index)
        score_series = pd.Series(scores, index=clean.index)

        # Today = the most recent candle that produced a (non-NaN) feature row.
        last_idx = clean.index[-1]
        res.today_anomalous = bool(flags.loc[last_idx])
        res.today_score = float(score_series.loc[last_idx])

        # Anomalies inside the display window → red bands on Panel 1.
        cutoff = df.index.max() - pd.Timedelta(days=IF_DISPLAY_DAYS)
        recent = flags[flags.index >= cutoff]
        res.anomaly_dates = [d for d in recent.index[recent.values]]
        res.recent_count = len(res.anomaly_dates)

        res.ready = True
        if res.today_anomalous:
            res.status = "⚠ ANOMALY — today's candle is an outlier"
        else:
            res.status = f"Normal · {res.recent_count} anomalies in {IF_DISPLAY_DAYS}d"
        return res

    # ═════════════════════════════════════════════════════════════════════════
    #  MODEL B — RANDOM FOREST (signal confidence)
    #  Supervised on the risk engine's OWN realised outcomes. Retrains only when
    #  the trade count changes (i.e. when new trade data is actually available).
    # ═════════════════════════════════════════════════════════════════════════
    def _run_random_forest(self, df: pd.DataFrame, trades: list[dict],
                           current_signal: str | None) -> RFResult:
        res = RFResult(available=SKLEARN_AVAILABLE)
        if not SKLEARN_AVAILABLE:
            res.status = "scikit-learn not installed"
            return res

        # Label source: only CLOSED trades have a realised P&L to learn from.
        closed = [t for t in trades if t.get("status") == "closed"]
        res.n_trades = len(closed)
        if res.n_trades < RF_MIN_TRADES:
            res.status = f"Warming up ({res.n_trades}/{RF_MIN_TRADES} trades)"
            return res

        feats = build_signal_features(df).replace([np.inf, -np.inf], np.nan)

        # (Re)train only when the realised trade history has grown/changed —
        # "retrain every time new trade data is available".
        if self._rf_model is None or res.n_trades != self._rf_trade_count:
            X, y = [], []
            for t in closed:
                d = t["entry_date"]                 # a BUY/Golden-Cross entry date
                if d in feats.index:
                    row = feats.loc[d]
                    if row.notna().all():
                        X.append(row.values)
                        # Label: did this entry become a profitable trade?
                        y.append(1 if t["pnl_dollars"] > 0 else 0)

            if len(X) < RF_MIN_TRADES:
                res.status = f"Warming up ({len(X)} usable trades)"
                return res

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = RandomForestClassifier(
                    n_estimators=RF_TREES,
                    max_depth=None,
                    min_samples_leaf=2,         # mild regularisation on small data
                    class_weight="balanced",    # robust if wins/losses are skewed
                    random_state=42,
                )
                model.fit(np.array(X), np.array(y))
            self._rf_model = model
            self._rf_classes = list(model.classes_)
            self._rf_trade_count = res.n_trades

        # Confidence for every historical BUY/SELL marker (Panel 2 strength).
        sig_rows = df[df["Signal"].notna()] if "Signal" in df else df.iloc[0:0]
        for d in sig_rows.index:
            if d in feats.index and feats.loc[d].notna().all():
                conf = self._rf_confidence(feats.loc[d].values)
                res.signal_confidences[d] = {
                    "confidence": conf,
                    "strength": MLResults.strength_label(conf),
                }

        # Confidence for the live "now" state → the gauge in Panel 7.
        last_row = feats.iloc[-1]
        if last_row.notna().all():
            conf = self._rf_confidence(last_row.values)
            res.current_confidence = conf
            res.current_strength = MLResults.strength_label(conf)
            res.current_signal = current_signal
            res.ready = True
            res.status = (f"{res.current_strength} · {conf:.0f}% "
                          f"(trained on {res.n_trades} trades)")
        else:
            res.ready = True
            res.status = f"Trained on {res.n_trades} trades"
        return res

    def _rf_confidence(self, feature_row: np.ndarray) -> float:
        """Return P(profitable)·100 for one feature row, robust to single-class fits."""
        proba = self._rf_model.predict_proba(feature_row.reshape(1, -1))[0]
        classes = self._rf_classes
        if 1 in classes:
            return float(proba[classes.index(1)] * 100.0)
        # The model has only ever seen losing trades ⇒ probability of profit ≈ 0.
        return 0.0

    # ═════════════════════════════════════════════════════════════════════════
    #  MODEL A — LSTM PRICE FORECAST
    #  Inference is cheap and runs every tick; (re)training is expensive and runs
    #  in a daemon thread on a 24h cadence, never blocking the dashboard.
    # ═════════════════════════════════════════════════════════════════════════
    def _run_lstm(self, df: pd.DataFrame) -> LSTMResult:
        res = LSTMResult()
        closes = df["Close"].dropna()
        res.last_close = float(closes.iloc[-1]) if len(closes) else None

        # Warm-up gate: refuse to forecast on too little history.
        if len(closes) < LSTM_MIN_CANDLES:
            res.status = f"Warming up ({len(closes)}/{LSTM_MIN_CANDLES} candles)"
            return res

        # Are the deep-learning libraries importable at all?
        if not _keras_importable():
            res.status = "Keras/torch not installed"
            return res
        res.available = True

        # Kick off a background retrain if one is due and none is running.
        now = time.time()
        with self._lock:
            due = (self._lstm_model is None
                   or (now - self._lstm_last_train) > LSTM_RETRAIN_SECONDS)
            if due and not self._lstm_training:
                self._lstm_training = True
                snapshot = closes.to_numpy(dtype=float).copy()
                threading.Thread(
                    target=self._train_lstm_worker,
                    args=(snapshot,),
                    name="meridian-lstm-train",
                    daemon=True,            # never blocks interpreter shutdown
                ).start()
            res.training = self._lstm_training
            res.device = self._lstm_device
            model = self._lstm_model
            scaler = self._lstm_scaler
            resid_std = self._lstm_resid_std
            err = self._lstm_error

        # No trained model yet → we're on the very first training pass.
        if model is None:
            res.status = "LSTM Training…" if res.training else (
                f"LSTM error: {err}" if err else "LSTM initialising…")
            return res

        # ── Fast inference with the current weights (runs every tick) ──────────
        # Predict the next 7 days from the last 60 *known* closes only.
        window = closes.to_numpy(dtype=float)[-LSTM_WINDOW:]
        scaled = scaler.transform(window.reshape(-1, 1)).reshape(1, LSTM_WINDOW, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_scaled = np.asarray(model.predict(scaled, verbose=0)).reshape(-1, 1)
        forecast = scaler.inverse_transform(pred_scaled).flatten()

        # Future business-agnostic calendar days (BTC trades daily incl. weekends).
        last_date = closes.index[-1]
        future_dates = [last_date + timedelta(days=i + 1) for i in range(LSTM_HORIZON)]

        # Confidence band = ±1σ of recent prediction errors (price units).
        upper = forecast + resid_std
        lower = np.clip(forecast - resid_std, 0.0, None)

        target = float(forecast[-1])
        anchor = res.last_close or float(window[-1])
        pct = (target / anchor - 1.0) * 100.0

        res.ready = True
        res.forecast_dates = future_dates
        res.forecast_prices = forecast.tolist()
        res.band_upper = upper.tolist()
        res.band_lower = lower.tolist()
        res.target_price = target
        res.pct_change = pct
        res.direction = "UP" if pct > 0.5 else "DOWN" if pct < -0.5 else "FLAT"
        if res.training:
            res.status = "LSTM Training… (showing previous forecast)"
        else:
            age_h = (now - self._lstm_last_train) / 3600.0
            res.status = (f"{res.direction} {pct:+.1f}% / 7d · {res.device.upper()} "
                          f"· retrained {age_h:.0f}h ago")
        return res

    def _train_lstm_worker(self, closes: np.ndarray) -> None:
        """
        Background thread: build a 2-layer LSTM, train it on 60→7 day windows of
        scaled closes, estimate the ±1σ band from recent prediction errors, then
        atomically swap the new model in. Any failure is captured, never raised
        (this runs detached from the request cycle).
        """
        try:
            # Lazy, thread-local heavy imports. KERAS_BACKEND is already set.
            import keras
            from keras import layers
            import torch

            # Best-effort Apple-Silicon GPU. set_default_device steers where
            # Keras's torch backend allocates tensors; if MPS misbehaves on any
            # op we fall back to CPU and retry once (see except below).
            device = "mps" if torch.backends.mps.is_available() else "cpu"

            def _build_and_fit(dev: str):
                try:
                    torch.set_default_device(dev)
                except Exception:
                    dev = "cpu"
                    torch.set_default_device("cpu")

                # ── Normalise closes to [0,1]; LSTMs train far better on scaled
                #    inputs, and we inverse-transform predictions back to dollars.
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(closes.reshape(-1, 1)).flatten()

                # ── Supervised windows: X = 60 past closes, y = next 7 closes.
                #    Every sample's target lies strictly AFTER its inputs ⇒ no
                #    look-ahead leaks into training.
                X, Y = [], []
                last_start = len(scaled) - LSTM_WINDOW - LSTM_HORIZON
                for i in range(last_start + 1):
                    X.append(scaled[i:i + LSTM_WINDOW])
                    Y.append(scaled[i + LSTM_WINDOW:i + LSTM_WINDOW + LSTM_HORIZON])
                X = np.array(X, dtype="float32").reshape(-1, LSTM_WINDOW, 1)
                Y = np.array(Y, dtype="float32")

                # Keep a chronological copy for the residual-band estimate below
                # (we want the *recent* errors, so order must be preserved there).
                X_ordered, Y_ordered = X, Y

                # Pre-shuffle the training set with NumPy and then fit with
                # shuffle=False. On the MPS (Apple GPU) device, Keras's torch
                # DataLoader otherwise builds a CPU random generator for its
                # shuffle sampler, which clashes with the MPS default device and
                # aborts training. Shuffling here gives the same regularisation
                # benefit without ever touching that GPU/CPU generator mismatch.
                perm = np.random.permutation(len(X))
                X, Y = X[perm], Y[perm]

                # ── Architecture: 2 stacked LSTM layers with dropout, then a
                #    dense head emitting all 7 horizon steps at once.
                model = keras.Sequential([
                    layers.Input(shape=(LSTM_WINDOW, 1)),
                    layers.LSTM(LSTM_UNITS, return_sequences=True),
                    layers.Dropout(LSTM_DROPOUT),
                    layers.LSTM(LSTM_UNITS),
                    layers.Dropout(LSTM_DROPOUT),
                    layers.Dense(LSTM_HORIZON),
                ])
                model.compile(optimizer="adam", loss="mse")
                model.fit(X, Y, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
                          shuffle=False, verbose=0)

                # ── Band width: std of RECENT in-sample prediction errors, in
                #    price units (matches the spec's "±1σ of recent errors").
                #    Use the chronological tail so "recent" really is recent.
                n_recent = min(LSTM_RECENT_ERRORS, len(X_ordered))
                recent_pred = np.asarray(
                    model.predict(X_ordered[-n_recent:], verbose=0))
                pred_price = scaler.inverse_transform(
                    recent_pred.reshape(-1, 1)).reshape(n_recent, LSTM_HORIZON)
                true_price = scaler.inverse_transform(
                    Y_ordered[-n_recent:].reshape(-1, 1)).reshape(n_recent, LSTM_HORIZON)
                resid_std = float(np.std(pred_price - true_price))
                return model, scaler, resid_std, dev

            try:
                model, scaler, resid_std, used = _build_and_fit(device)
            except Exception:
                # MPS path blew up — retry once on CPU before giving up.
                model, scaler, resid_std, used = _build_and_fit("cpu")

            with self._lock:
                self._lstm_model = model
                self._lstm_scaler = scaler
                self._lstm_resid_std = resid_std
                self._lstm_last_train = time.time()
                self._lstm_device = used
                self._lstm_error = ""

        except Exception as exc:  # pragma: no cover - environment dependent
            with self._lock:
                self._lstm_error = str(exc)
        finally:
            with self._lock:
                self._lstm_training = False


# ── Lazy capability probe for the deep-learning stack ──────────────────────────

_KERAS_OK: bool | None = None


def _keras_importable() -> bool:
    """Cheaply (and once) determine whether the LSTM stack can be imported."""
    global _KERAS_OK
    if _KERAS_OK is None:
        try:
            import importlib.util
            _KERAS_OK = (importlib.util.find_spec("keras") is not None
                         and importlib.util.find_spec("torch") is not None)
        except Exception:
            _KERAS_OK = False
    return _KERAS_OK


# ── The process-wide singleton the dashboard talks to ──────────────────────────
engine = MLEngine()


def update(df: pd.DataFrame, trades: list[dict],
           current_signal: str | None = None) -> MLResults:
    """Module-level convenience wrapper around the singleton engine."""
    return engine.update(df, trades, current_signal)
