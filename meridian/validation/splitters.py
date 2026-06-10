"""
splitters.py — out-of-sample validation machinery.

Two complementary tools:

  • WalkForward — for the *strategy*. Train/calibrate on a trailing window, then
    evaluate on the next untouched block, roll forward, repeat. This mimics how
    the strategy would actually have been deployed and stitches a fully
    out-of-sample equity curve.

  • PurgedKFold — for the *ML models*. Plain k-fold leaks in finance because a
    label's outcome window (event_date -> t1) can overlap the test set. We
    PURGE any training label whose [event_date, t1] interval overlaps a test
    fold, and add an EMBARGO after each test fold so serially-correlated
    information just after the test window can't leak in either
    (Lopez de Prado, "Advances in Financial Machine Learning", ch. 7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


def walk_forward_splits(index: pd.DatetimeIndex, train_years: int,
                        test_months: int, anchored: bool = False) -> list[Split]:
    """
    Generate rolling (or anchored/expanding) walk-forward windows over a date
    index. Train spans `train_years`; each subsequent test block spans
    `test_months`. Train always ends strictly before its test block begins.
    """
    index = pd.DatetimeIndex(index).sort_values()
    if len(index) == 0:
        return []
    start = index[0]
    train_delta = pd.DateOffset(years=train_years)
    test_delta = pd.DateOffset(months=test_months)

    splits: list[Split] = []
    test_start = start + train_delta
    while test_start < index[-1]:
        test_end = test_start + test_delta
        train_lo = start if anchored else test_start - train_delta
        train_mask = (index >= train_lo) & (index < test_start)
        test_mask = (index >= test_start) & (index < test_end)
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            splits.append(Split(train=index[train_mask], test=index[test_mask]))
        test_start = test_end
    return splits


class PurgedKFold:
    """
    Purged k-fold cross-validation with an embargo, for event-based labels.

    Parameters
    ----------
    n_splits : number of contiguous test folds (in time order).
    t1 : Series indexed by event_date with value = the date each label's outcome
         is realized. Drives purging.
    embargo_frac : fraction of the total sample length to embargo after each
         test fold.
    """

    def __init__(self, n_splits: int, t1: pd.Series, embargo_frac: float = 0.01):
        self.n_splits = int(n_splits)
        self.t1 = t1.sort_index()
        self.embargo_frac = float(embargo_frac)

    def split(self, X: pd.DataFrame):
        """Yield (train_positions, test_positions) as integer arrays into X.

        X must be indexed by event_date and aligned with t1's index.
        """
        if not X.index.equals(self.t1.index):
            # Align defensively; both must share event_date ordering.
            t1 = self.t1.reindex(X.index)
        else:
            t1 = self.t1
        n = len(X)
        indices = np.arange(n)
        embargo = int(n * self.embargo_frac)
        fold_bounds = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        event_dates = X.index
        t1_vals = pd.DatetimeIndex(t1.values)

        for start, end in fold_bounds:
            test_pos = indices[start:end]
            test_start_time = event_dates[start]
            test_end_time = t1_vals[test_pos].max()

            # Purge: drop train samples whose [event_date, t1] overlaps the test
            # interval [test_start_time, test_end_time].
            train_mask = np.ones(n, dtype=bool)
            train_mask[start:end] = False
            for j in indices:
                if not train_mask[j]:
                    continue
                ev, ev_t1 = event_dates[j], t1_vals[j]
                if (ev <= test_end_time) and (ev_t1 >= test_start_time):
                    train_mask[j] = False

            # Embargo: drop a buffer of samples immediately after the test fold.
            if embargo > 0:
                emb_hi = min(end + embargo, n)
                train_mask[end:emb_hi] = False

            train_pos = indices[train_mask]
            if len(train_pos) > 0 and len(test_pos) > 0:
                yield train_pos, test_pos
