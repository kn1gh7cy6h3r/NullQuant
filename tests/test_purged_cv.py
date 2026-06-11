"""
test_purged_cv.py — the no-leakage guarantees of the validation splitters.

walk_forward_splits: every train window must end strictly before its test window.
PurgedKFold: no training label's outcome window [event_start, t1] may overlap the
test fold's [min start, max t1] window — this is the property that makes the
cross-validated ML metrics trustworthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.validation.splitters import walk_forward_splits, PurgedKFold


def test_walk_forward_train_precedes_test():
    index = pd.date_range("2019-01-01", periods=5 * 365, freq="D")
    splits = walk_forward_splits(index, train_years=2, test_months=6, anchored=False)
    assert len(splits) >= 2
    for sp in splits:
        assert sp.train.max() < sp.test.min(), "train must end before test begins"
        assert len(sp.train) > 0 and len(sp.test) > 0


def test_walk_forward_anchored_expands():
    index = pd.date_range("2019-01-01", periods=5 * 365, freq="D")
    splits = walk_forward_splits(index, train_years=2, test_months=6, anchored=True)
    # Anchored training always starts at the very beginning.
    for sp in splits:
        assert sp.train.min() == index[0]


def _make_events(n=120, seed=0):
    rng = np.random.default_rng(seed)
    starts = pd.date_range("2019-01-01", periods=n, freq="7D")
    # Each label resolves a random 1-20 days after its start.
    horizons = rng.integers(1, 20, size=n)
    t1_vals = [s + pd.Timedelta(days=int(h)) for s, h in zip(starts, horizons)]
    t1 = pd.Series(pd.to_datetime(t1_vals), index=starts)
    X = pd.DataFrame({"f": rng.normal(size=n)}, index=pd.RangeIndex(n))
    return X, t1, starts


def test_purged_kfold_no_overlap():
    X, t1, starts = _make_events()
    starts = pd.DatetimeIndex(t1.index)
    ends = pd.DatetimeIndex(t1.values)

    pkf = PurgedKFold(n_splits=5, t1=t1, embargo_frac=0.01)
    n_yielded = 0
    for train_pos, test_pos in pkf.split(X):
        n_yielded += 1
        test_lo = starts[test_pos].min()
        test_hi = ends[test_pos].max()
        # No training label's [start, t1] may overlap [test_lo, test_hi].
        tr_starts = starts[train_pos]
        tr_ends = ends[train_pos]
        overlap = (np.asarray(tr_starts <= test_hi) & np.asarray(tr_ends >= test_lo))
        assert not overlap.any(), "purging failed: a train label overlaps the test fold"
        # train and test positions are disjoint.
        assert len(set(train_pos) & set(test_pos)) == 0
    assert n_yielded >= 1


def test_purged_kfold_embargo_drops_samples():
    X, t1, _ = _make_events()
    no_emb = list(PurgedKFold(5, t1, embargo_frac=0.0).split(X))
    big_emb = list(PurgedKFold(5, t1, embargo_frac=0.1).split(X))
    # A larger embargo can only shrink (never grow) the training sets.
    for (tr0, _), (tr1, _) in zip(no_emb, big_emb):
        assert len(tr1) <= len(tr0)
