"""
labeling.py — triple-barrier labeling for meta-labeling (Lopez de Prado).

The base strategy decides the *side* of a bet (long on a golden cross, short on
a death cross). Meta-labeling asks a different, narrower question that an ML
model can actually answer well: *given that we are about to take this bet,
should we?* The label is whether the bet would have been profitable under a
realistic exit rule — three barriers:

    • profit-take   : entry + side * pt_mult * ATR
    • stop-loss     : entry - side * sl_mult * ATR
    • vertical (time): close out after `max_hold_days`

Each event also records `t1`, the date its outcome is realized. `t1` is what
the purged cross-validator uses to drop training labels whose evaluation window
overlaps the test set — without it, overlapping labels leak information.

Barriers are evaluated on closing prices (no intrabar high/low assumptions), so
the labels are deliberately conservative and reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels_one(
    close: pd.Series,
    atr: pd.Series,
    events: pd.DatetimeIndex,
    side: int,
    pt_mult: float,
    sl_mult: float,
    max_hold_days: int,
) -> pd.DataFrame:
    """
    Label a single asset's events. Returns a frame indexed by event date with:
        side   : +1 long / -1 short
        t1     : date the outcome was realized (barrier touch or time stop)
        ret    : side-adjusted realized return of the bet
        label  : 1 if the bet was profitable (ret > 0), else 0
    """
    close = close.dropna()
    idx = close.index
    pos = {d: i for i, d in enumerate(idx)}
    prices = close.to_numpy(dtype=float)
    atr_arr = atr.reindex(idx).to_numpy(dtype=float)
    n = len(idx)

    rows = []
    for d in events:
        if d not in pos:
            continue
        i = pos[d]
        a = atr_arr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = prices[i]
        pt = entry + side * pt_mult * a
        sl = entry - side * sl_mult * a
        end = min(i + max_hold_days, n - 1)

        t1_pos = end
        for j in range(i + 1, end + 1):
            p = prices[j]
            if side == 1:
                if p >= pt or p <= sl:
                    t1_pos = j
                    break
            else:  # short
                if p <= pt or p >= sl:
                    t1_pos = j
                    break

        exit_price = prices[t1_pos]
        ret = side * (exit_price / entry - 1.0)
        rows.append({
            "asset": close.name,
            "event_date": d,
            "side": side,
            "t1": idx[t1_pos],
            "ret": ret,
            "label": int(ret > 0.0),
        })

    if not rows:
        return pd.DataFrame(
            columns=["asset", "event_date", "side", "t1", "ret", "label"]
        ).set_index("event_date")
    return pd.DataFrame(rows).set_index("event_date")


def build_meta_labels(
    close: pd.DataFrame,
    atr: pd.DataFrame,
    side_events: dict[str, dict[str, pd.DatetimeIndex]],
    pt_mult: float,
    sl_mult: float,
    max_hold_days: int,
) -> pd.DataFrame:
    """
    Build a labeled event table across assets.

    side_events maps asset -> {"long": DatetimeIndex, "short": DatetimeIndex} of
    base-signal entry dates. Returns a long-form frame (one row per event) with
    columns [asset, side, t1, ret, label] and the event_date as a column,
    suitable for joining causal features and feeding the purged-CV meta-labeler.
    """
    frames = []
    for asset, sides in side_events.items():
        if asset not in close.columns:
            continue
        c = close[asset]
        c.name = asset
        a = atr[asset]
        for side_name, events in sides.items():
            side = 1 if side_name == "long" else -1
            lab = triple_barrier_labels_one(
                c, a, events, side, pt_mult, sl_mult, max_hold_days)
            if not lab.empty:
                frames.append(lab.reset_index())
    if not frames:
        return pd.DataFrame(columns=["event_date", "asset", "side", "t1", "ret", "label"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("event_date").reset_index(drop=True)
