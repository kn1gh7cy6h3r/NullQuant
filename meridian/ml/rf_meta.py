"""
rf_meta.py — Random Forest META-LABELER evaluated with purged k-fold CV.

The reframing (Lopez de Prado, "Advances in Financial Machine Learning", ch. 3):

  • The base SMA50/200 crossover decides the SIDE of a bet (long on a golden
    cross, short on a death cross). It is a *trend trigger*, nothing more.
  • The Random Forest does NOT predict direction. It answers a strictly narrower,
    learnable question: given that the base rule wants to take this bet, will the
    bet pay off? i.e. P(profitable) under a triple-barrier exit. The target is the
    meta-label built by `build_meta_labels`.

Why this is honest where the old in-sample approach was not:

  • Every feature is CAUSAL — read at the event_date from backward-looking
    indicators only; nothing peeks past the entry bar.
  • Evaluation is strictly out-of-sample via PurgedKFold. A label's outcome
    window [event_date, t1] can overlap a test fold, so plain k-fold leaks; we
    purge overlapping training labels and embargo the buffer after each test
    fold. Every event's predicted probability comes from a fold in which that
    event was held out — so the reported AUC is a genuine OOS number.

If crossovers are rare we may have few events; the count is reported honestly and
the number of CV splits is reduced gracefully (with a note in `status`) rather
than crashing when there are too few events for the configured `n_splits`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import Config
from ..data.loader import Panel
from ..features import indicators as ind
from ..features.labeling import build_meta_labels
from ..seeds import set_global_seed
from ..signals.base import crossover_events
from ..validation.splitters import PurgedKFold

# Window lengths for the two realized-vol features (in trading days). The short
# window captures recent turbulence; the medium one a slower regime estimate.
_VOL_SHORT = 14
_VOL_MEDIUM = 60

# Feature columns, in a fixed order so the model input is reproducible.
FEATURE_COLUMNS: list[str] = [
    "sma_ratio",
    "dist_sma50",
    "rsi14",
    "atr_norm",
    "rv_short",
    "rv_medium",
    "risk_adj_mom",
    "mom_rank",
    "side",
]


@dataclass
class MetaLabelResult:
    """Outcome of the purged-CV meta-labeler evaluation.

    All metrics are out-of-sample (collected from held-out folds).

    event_probs: one row per event with the OOS predicted probability and the
        realized meta-label, columns [event_date, asset, side, prob, label].
    status: "ok", or a human-readable note when something was degraded (e.g.
        too few events, splits reduced, or evaluation skipped entirely).
    """

    n_events: int
    base_rate: float
    oos_auc: float
    oos_accuracy: float
    oos_precision: float
    oos_recall: float
    take_threshold: float
    status: str
    event_probs: pd.DataFrame = field(default_factory=pd.DataFrame)


def _indicator_panels(panel: Panel, cfg: Config) -> dict[str, pd.DataFrame]:
    """Compute every causal indicator panel once (dates x assets)."""
    s = cfg.strategy
    close = panel.close
    high = panel.field("High")
    low = panel.field("Low")

    sma_short = ind.sma(close, s.sma_short)
    sma_long = ind.sma(close, s.sma_long)
    atr = ind.atr(high, low, close, s.atr_period)
    rsi = ind.rsi(close, 14)
    rv_short = ind.realized_vol(close, _VOL_SHORT, annualize=False)
    rv_medium = ind.realized_vol(close, _VOL_MEDIUM, annualize=False)
    ras = ind.risk_adjusted_momentum(close, s.momentum_lookback, s.vol_lookback)
    mom_rank = ind.cross_sectional_rank(ind.momentum(close, s.momentum_lookback))

    return {
        "close": close,
        "sma_short": sma_short,
        "sma_long": sma_long,
        "atr": atr,
        "rsi": rsi,
        "rv_short": rv_short,
        "rv_medium": rv_medium,
        "ras": ras,
        "mom_rank": mom_rank,
    }


def _build_features(events: pd.DataFrame, ip: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the causal feature matrix, one row per event.

    Each feature is read at the event's `event_date` for the event's `asset`
    using only backward-looking indicator panels — no look-ahead. The returned
    frame is aligned positionally to `events` (same order, RangeIndex).
    """
    close = ip["close"]
    sma_short = ip["sma_short"]
    sma_long = ip["sma_long"]
    atr = ip["atr"]
    rsi = ip["rsi"]
    rv_short = ip["rv_short"]
    rv_medium = ip["rv_medium"]
    ras = ip["ras"]
    mom_rank = ip["mom_rank"]

    rows: list[dict[str, float]] = []
    for _, ev in events.iterrows():
        d = ev["event_date"]
        a = ev["asset"]

        c = float(close.at[d, a])
        ss = float(sma_short.at[d, a])
        sl = float(sma_long.at[d, a])
        at = float(atr.at[d, a])

        sma_ratio = ss / sl if sl != 0 else np.nan
        dist_sma50 = (c - ss) / ss if ss != 0 else np.nan
        atr_norm = at / c if c != 0 else np.nan

        rows.append({
            "sma_ratio": sma_ratio,
            "dist_sma50": dist_sma50,
            "rsi14": float(rsi.at[d, a]),
            "atr_norm": atr_norm,
            "rv_short": float(rv_short.at[d, a]),
            "rv_medium": float(rv_medium.at[d, a]),
            "risk_adj_mom": float(ras.at[d, a]),
            "mom_rank": float(mom_rank.at[d, a]),
            "side": float(ev["side"]),
        })

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _make_rf(cfg: Config) -> RandomForestClassifier:
    """Construct the deterministic, class-balanced Random Forest."""
    rf = cfg.ml.rf_meta
    return RandomForestClassifier(
        n_estimators=int(rf.n_estimators),
        max_depth=int(rf.max_depth),
        min_samples_leaf=int(rf.min_samples_leaf),
        class_weight="balanced",
        random_state=cfg.seed,
        n_jobs=-1,
    )


def _empty_result(threshold: float, n_events: int, base_rate: float,
                  status: str) -> MetaLabelResult:
    return MetaLabelResult(
        n_events=n_events,
        base_rate=base_rate,
        oos_auc=float("nan"),
        oos_accuracy=float("nan"),
        oos_precision=float("nan"),
        oos_recall=float("nan"),
        take_threshold=threshold,
        status=status,
        event_probs=pd.DataFrame(
            columns=["event_date", "asset", "side", "prob", "label"]),
    )


def fit_eval_meta(panel: Panel, cfg: Config) -> MetaLabelResult:
    """Full purged-CV evaluation of the RF meta-labeler.

    Steps:
      1. Detect base crossover events and build triple-barrier meta-labels.
      2. Build a causal feature matrix (one row per event, read at event_date).
      3. Run PurgedKFold; for each fold fit the RF on the purged training events
         and predict P(profitable) on the held-out test events.
      4. Collect every event's OOS probability and report OOS AUC / accuracy /
         precision / recall (at the configured take_threshold) plus the base
         rate of label==1.

    Returns a MetaLabelResult; degrades gracefully (status note) when events are
    too few for a meaningful split.
    """
    set_global_seed(cfg.seed)

    rf_cfg = cfg.ml.rf_meta
    threshold = float(rf_cfg.take_threshold)
    s = cfg.strategy

    # --- 1. events + meta-labels -------------------------------------------
    side_events = crossover_events(panel.close, s.sma_short, s.sma_long)
    atr = ind.atr(panel.field("High"), panel.field("Low"), panel.close, s.atr_period)
    labels = build_meta_labels(
        close=panel.close,
        atr=atr,
        side_events=side_events,
        pt_mult=float(rf_cfg.pt_atr_mult),
        sl_mult=float(rf_cfg.sl_atr_mult),
        max_hold_days=int(rf_cfg.max_hold_days),
    )

    if labels.empty:
        return _empty_result(threshold, 0, float("nan"),
                             "no crossover events / labels — nothing to evaluate")

    # Sort events by event_date (stable, asset as tie-break for determinism).
    labels = labels.sort_values(["event_date", "asset"]).reset_index(drop=True)
    n_events = len(labels)
    base_rate = float(labels["label"].mean())

    # --- 2. causal features ------------------------------------------------
    ip = _indicator_panels(panel, cfg)
    feats = _build_features(labels, ip)

    # Drop events with any non-finite feature (warm-up periods etc.). Keep the
    # surviving labels aligned to the feature rows.
    finite = np.isfinite(feats.to_numpy(dtype=float)).all(axis=1)
    if not finite.all():
        labels = labels.loc[finite].reset_index(drop=True)
        feats = feats.loc[finite].reset_index(drop=True)

    n_events = len(labels)
    if n_events == 0:
        return _empty_result(threshold, 0, base_rate,
                             "all events dropped during feature warm-up")
    base_rate = float(labels["label"].mean())

    # Need both classes present to train/score a classifier.
    if labels["label"].nunique() < 2:
        return _empty_result(
            threshold, n_events, base_rate,
            f"only one label class present ({int(labels['label'].iloc[0])}); "
            f"cannot evaluate AUC")

    # --- 3. purged k-fold --------------------------------------------------
    # PurgedKFold derives event START times from t1.index and END times from
    # t1.values, fully positional — so it tolerates event_dates colliding across
    # assets. We therefore index t1 by the event_date timestamps (duplicates are
    # fine) and keep X on a positional index; events are pre-sorted by event_date
    # so the positional order matches the chronological order the splitter needs.
    event_starts = pd.DatetimeIndex(labels["event_date"].to_numpy())
    X = pd.DataFrame(feats.to_numpy(dtype=float), columns=FEATURE_COLUMNS,
                     index=pd.RangeIndex(n_events))
    y = labels["label"].to_numpy(dtype=int)
    t1 = pd.Series(labels["t1"].to_numpy(), index=event_starts)

    requested_splits = int(cfg.validation.purged_cv["n_splits"])
    embargo_frac = float(cfg.validation.purged_cv["embargo_frac"])
    # A fold needs >=1 test event; cap splits at n_events. Require at least 2.
    n_splits = max(2, min(requested_splits, n_events))
    status = "ok"
    if n_splits < requested_splits:
        status = (f"reduced n_splits {requested_splits}->{n_splits} "
                  f"(only {n_events} events)")

    splitter = PurgedKFold(n_splits=n_splits, t1=t1, embargo_frac=embargo_frac)

    oos_prob = np.full(n_events, np.nan, dtype=float)
    folds_scored = 0
    for train_pos, test_pos in splitter.split(X):
        y_train = y[train_pos]
        # A fold with one-class training data cannot fit a useful classifier.
        if np.unique(y_train).size < 2:
            continue
        model = _make_rf(cfg)
        model.fit(X.iloc[train_pos].to_numpy(), y_train)
        proba = model.predict_proba(X.iloc[test_pos].to_numpy())
        # Probability of the positive class (label == 1).
        pos_col = list(model.classes_).index(1)
        oos_prob[test_pos] = proba[:, pos_col]
        folds_scored += 1

    scored = np.isfinite(oos_prob)
    if folds_scored == 0 or scored.sum() == 0:
        return _empty_result(
            threshold, n_events, base_rate,
            "no usable folds (purging/one-class training left nothing to score)")

    if scored.sum() < n_events and status == "ok":
        status = (f"{int(scored.sum())}/{n_events} events scored OOS "
                  f"(rest purged out of every test fold)")
    elif scored.sum() < n_events:
        status += f"; {int(scored.sum())}/{n_events} events scored OOS"

    # --- 4. metrics --------------------------------------------------------
    y_scored = y[scored]
    p_scored = oos_prob[scored]
    preds = (p_scored >= threshold).astype(int)

    if np.unique(y_scored).size < 2:
        oos_auc = float("nan")
        status += "; AUC undefined (one class among scored events)"
    else:
        oos_auc = float(roc_auc_score(y_scored, p_scored))

    oos_accuracy = float(accuracy_score(y_scored, preds))
    oos_precision = float(precision_score(y_scored, preds, zero_division=0))
    oos_recall = float(recall_score(y_scored, preds, zero_division=0))

    event_probs = pd.DataFrame({
        "event_date": labels["event_date"].to_numpy(),
        "asset": labels["asset"].to_numpy(),
        "side": labels["side"].to_numpy(),
        "prob": oos_prob,
        "label": y,
    })

    return MetaLabelResult(
        n_events=n_events,
        base_rate=base_rate,
        oos_auc=oos_auc,
        oos_accuracy=oos_accuracy,
        oos_precision=oos_precision,
        oos_recall=oos_recall,
        take_threshold=threshold,
        status=status,
        event_probs=event_probs,
    )


def exposure_gate(panel: Panel, cfg: Config,
                  result: MetaLabelResult | None = None) -> pd.Series:
    """Per-DATE book-exposure multiplier in [0, 1] driven by the meta-labeler.

    Construction (kept simple and strictly causal):

      • Run (or reuse) the OOS meta-labeler to get, for each crossover event, the
        held-out probability that the bet is profitable. An event is "taken" when
        prob >= take_threshold.
      • A signal is "active" from its event_date until its outcome date t1. On
        any given date the exposure multiplier is the FRACTION of currently
        active signals that the meta-labeler would take:
              exposure(t) = (# active & taken at t) / (# active at t)
      • When no signal is active, default to 1.0 (the meta-labeler has no opinion,
        so it does not scale the book).

    Every event's probability is its OOS value, and a signal only influences dates
    on/after its own event_date, so the gate introduces no look-ahead. The result
    is forward-filled onto the panel calendar and clipped to [0, 1].

    Returns a Series indexed by panel.close.index.
    """
    dates = panel.close.index
    default = pd.Series(1.0, index=dates)

    if result is None:
        result = fit_eval_meta(panel, cfg)

    ep = result.event_probs
    if ep is None or ep.empty:
        return default

    threshold = float(result.take_threshold)

    # Recover each event's outcome date t1 to know when a signal stops being
    # active. event_probs doesn't carry t1, so rebuild the label table once.
    s = cfg.strategy
    rf_cfg = cfg.ml.rf_meta
    side_events = crossover_events(panel.close, s.sma_short, s.sma_long)
    atr = ind.atr(panel.field("High"), panel.field("Low"), panel.close, s.atr_period)
    labels = build_meta_labels(
        close=panel.close,
        atr=atr,
        side_events=side_events,
        pt_mult=float(rf_cfg.pt_atr_mult),
        sl_mult=float(rf_cfg.sl_atr_mult),
        max_hold_days=int(rf_cfg.max_hold_days),
    )
    if labels.empty:
        return default

    # Map (event_date, asset, side) -> t1 for joining onto scored events.
    t1_map = labels.set_index(["event_date", "asset", "side"])["t1"].to_dict()

    # active_count[t] and taken_count[t] accumulated over each event's window.
    active = pd.Series(0, index=dates, dtype=float)
    taken = pd.Series(0, index=dates, dtype=float)

    for _, row in ep.iterrows():
        prob = row["prob"]
        if not np.isfinite(prob):
            continue
        key = (row["event_date"], row["asset"], int(row["side"]))
        t1 = t1_map.get(key)
        if t1 is None:
            continue
        # Window [event_date, t1] intersected with the panel calendar.
        lo = dates.searchsorted(row["event_date"], side="left")
        hi = dates.searchsorted(t1, side="right")
        if hi <= lo:
            continue
        active.iloc[lo:hi] += 1.0
        if prob >= threshold:
            taken.iloc[lo:hi] += 1.0

    # Fraction taken where any signal is active; 1.0 (no opinion) elsewhere.
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = taken / active
    gate = frac.where(active > 0, 1.0)
    gate = gate.clip(0.0, 1.0).fillna(1.0)
    gate.name = "exposure_gate"
    return gate
