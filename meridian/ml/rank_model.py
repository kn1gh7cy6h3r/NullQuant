"""
rank_model.py — pointwise Learning-to-Rank cross-sectional selector.

WHAT THIS DOES
--------------
Predicting raw forward returns is a losing game: the level is dominated by a
common market factor and is barely learnable per-asset. Here we change the
question. Instead of "what return will asset a earn?" we ask "where does asset a
RANK against its 8 peers over the next horizon?" — and we trade that ordering:
go LONG the top-ranked coins and SHORT the bottom-ranked ones.

This is *pointwise* learning-to-rank: a single RandomForestRegressor is trained
to predict a per-(asset, date) score, and the cross-sectional ORDER of those
scores is what we act on. To make the score a relative measure we regress on the
CROSS-SECTIONALLY DEMEANED forward return, so the model is explicitly taught
relative out/under-performance rather than market direction.

WHY WALK-FORWARD (THE NO-LEAKAGE DESIGN)
----------------------------------------
Every feature at date d is backward-looking (rolling/realized indicators), so a
feature row is causal. The target, however, is forward-looking: the label at
date d needs close[d + horizon]. That makes training causality the crux:

  * We refit every ``cfg.ml.ltr.refit_every`` days on the trailing
    ``cfg.ml.ltr.train_years`` of rows.
  * CRITICAL: when training at a cutoff T we use ONLY rows whose label is fully
    realized, i.e. d + horizon <= T. A row whose horizon window pokes past T is
    excluded — its label could not have been known at T.
  * The fitted model then scores the next refit window (dates strictly after T),
    producing genuine out-of-sample scores. We stack those OOS scores across the
    whole history.

THE HONEST DIAGNOSTIC
---------------------
``rank_ic`` (mean cross-sectional Spearman correlation between the predicted
score and the realized forward return, over rebalance dates) is the truth-teller.
An IC near 0 means the ranker has no skill — we report it plainly and never tune
to inflate it. ``ic_hit`` (fraction of rebalance dates with positive IC) is a
companion stability read.

OUTPUT
------
A ``LTRResult`` whose ``direction`` is a {-1, 0, +1} matrix aligned to
``panel.close.index`` (all dates) and ``panel.assets`` (same order): on each
weekly rebalance we long the top_k and short the bottom_k predicted scores, and
hold (forward-fill) between rebalances. Warm-up dates (before any OOS score) are
zeros. On any failure we return an all-zeros direction with an explanatory
status — this function never raises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import Panel
from ..features import indicators as ind
from ..seeds import set_global_seed

# Approximate trading days per calendar year (crypto trades 365/yr) — used to
# turn ``train_years`` into a trailing row-window length in days.
_DAYS_PER_YEAR = 365


@dataclass
class LTRResult:
    """Output of the learning-to-rank cross-sectional selector.

    Attributes
    ----------
    direction:
        DataFrame indexed by ``panel.close.index`` (ALL dates), columns
        ``panel.assets`` (same order), values in {-1.0, 0.0, +1.0}. +1 = long,
        -1 = short, 0 = flat. Held (forward-filled) between rebalances; zeros on
        warm-up dates before any out-of-sample score exists.
    rank_ic:
        Mean cross-sectional Spearman correlation between the predicted score and
        the realized forward return, taken over rebalance dates. The honest read
        on whether the ranker has any skill (near 0 ⇒ none). NaN if undefined.
    ic_hit:
        Fraction of rebalance dates whose cross-sectional IC was positive.
    n_refits:
        Number of walk-forward model fits performed.
    status:
        Human-readable status string (also reports graceful degradation).
    """

    direction: pd.DataFrame
    rank_ic: float
    ic_hit: float
    n_refits: int
    status: str


# Fixed feature-column order so the model input is reproducible across refits.
def _feature_columns(lookbacks: list[int]) -> list[str]:
    cols = [f"mom_{lb}" for lb in lookbacks]
    cols += [
        "rvol",        # realized vol over cfg.strategy.vol_lookback
        "rsi14",       # Wilder RSI(14)
        "dist_sma50",  # (Close - SMA_short) / SMA_short
        "trend_sign",  # sign(SMA_short - SMA_long)
        "mom_rank",    # cross-sectional rank of the longest-lookback momentum
        "vol_rank",    # cross-sectional rank of realized vol
    ]
    return cols


def _build_panels(panel: Panel, cfg: Config, lookbacks: list[int]
                  ) -> dict[str, pd.DataFrame]:
    """Compute every strictly causal feature panel once (dates x assets).

    All indicators are backward-looking, so each value at date d uses only
    information available at d.
    """
    close = panel.close
    s = cfg.strategy

    sma_short = ind.sma(close, int(s.sma_short))
    sma_long = ind.sma(close, int(s.sma_long))
    rvol = ind.realized_vol(close, int(s.vol_lookback), annualize=False)
    rsi14 = ind.rsi(close, 14)

    # Distance of price from the short SMA, normalized.
    dist_sma50 = (close - sma_short) / sma_short.replace(0.0, np.nan)
    # Sign of the trend (short vs long SMA): +1 up, -1 down, 0 flat/unknown.
    trend_sign = np.sign(sma_short - sma_long)

    panels: dict[str, pd.DataFrame] = {}
    mom_panels: dict[int, pd.DataFrame] = {}
    for lb in lookbacks:
        m = ind.momentum(close, int(lb))
        panels[f"mom_{lb}"] = m
        mom_panels[lb] = m

    panels["rvol"] = rvol
    panels["rsi14"] = rsi14
    panels["dist_sma50"] = dist_sma50
    panels["trend_sign"] = trend_sign

    # Cross-sectional ranks (within each date) of momentum and vol. Use the
    # longest lookback for the momentum rank (the slowest, most stable signal).
    longest = max(lookbacks)
    panels["mom_rank"] = ind.cross_sectional_rank(mom_panels[longest])
    panels["vol_rank"] = ind.cross_sectional_rank(rvol)
    return panels


def _stack_features(panels: dict[str, pd.DataFrame], cols: list[str],
                    assets: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Stack the per-feature panels into a long (date, asset) feature table.

    Returns a DataFrame with a 2-level index (date, asset) and one column per
    feature, in ``cols`` order. Rows with any non-finite feature are dropped.
    """
    frames = []
    for c in cols:
        p = panels[c].reindex(index=dates, columns=assets)
        s = p.stack(dropna=False)
        s.name = c
        frames.append(s)
    feats = pd.concat(frames, axis=1)
    feats.index = feats.index.set_names(["date", "asset"])
    finite = np.isfinite(feats.to_numpy(dtype=float)).all(axis=1)
    return feats.loc[finite]


def _forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward total return over ``horizon`` days: close[t+h]/close[t] - 1.

    Forward-looking by construction — used ONLY as a training/evaluation target,
    never as a feature, and gated by realized-label causality at fit time.
    """
    return close.shift(-horizon) / close - 1.0


def _empty(direction_shape_dates: pd.DatetimeIndex, assets: list[str],
           status: str) -> LTRResult:
    """All-zeros, never-raises fallback result with an explanatory status."""
    direction = pd.DataFrame(0.0, index=direction_shape_dates, columns=assets)
    return LTRResult(
        direction=direction,
        rank_ic=float("nan"),
        ic_hit=0.0,
        n_refits=0,
        status=status,
    )


def ltr_signal(panel: Panel, cfg: Config) -> LTRResult:
    """Walk-forward pointwise learning-to-rank cross-sectional selector.

    Trains a RandomForestRegressor on cross-sectionally demeaned forward returns,
    refitting every ``cfg.ml.ltr.refit_every`` days on the trailing
    ``cfg.ml.ltr.train_years`` of FULLY-REALIZED rows, and scores the next window
    out-of-sample. On weekly rebalances it longs the top_k / shorts the bottom_k
    predicted scores and holds between rebalances.

    Degrades gracefully: any failure (missing sklearn, too little history, etc.)
    returns an all-zeros direction with an explanatory status. Deterministic via
    ``set_global_seed(cfg.seed)``.
    """
    set_global_seed(cfg.seed)

    dates = panel.close.index
    assets = list(panel.assets)

    # --- defensive sklearn import ------------------------------------------
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as exc:  # pragma: no cover - exercised only without sklearn
        return _empty(dates, assets,
                      f"sklearn unavailable ({exc}); LTR disabled, direction=0")

    # --- config -------------------------------------------------------------
    try:
        ltr = cfg.ml.ltr
        lookbacks = [int(x) for x in ltr.lookbacks]
        train_years = float(ltr.train_years)
        refit_every = int(ltr.refit_every)
        top_k = int(ltr.top_k)
        bottom_k = int(ltr.bottom_k)
        n_estimators = int(ltr.n_estimators)
        horizon = int(cfg.ml.horizon)
        rebalance = str(cfg.strategy.rebalance)
        seed = int(cfg.seed)
    except Exception as exc:
        return _empty(dates, assets, f"config read failed ({exc}); direction=0")

    if not lookbacks:
        return _empty(dates, assets, "no momentum lookbacks configured; direction=0")

    feature_cols = _feature_columns(lookbacks)
    close = panel.close

    # --- 1. causal features + demeaned forward-return target ----------------
    try:
        panels = _build_panels(panel, cfg, lookbacks)
        feats = _stack_features(panels, feature_cols, assets, dates)

        fwd = _forward_return(close, horizon)
        # Cross-sectional demeaning at each date: relative out/under-performance.
        y_demeaned = fwd.sub(fwd.mean(axis=1), axis=0)
    except Exception as exc:
        return _empty(dates, assets, f"feature/target build failed ({exc}); direction=0")

    if feats.empty:
        return _empty(dates, assets, "no finite feature rows (warm-up); direction=0")

    # Align labels onto the feature rows. ``y_label`` is the demeaned forward
    # return (training target); ``y_raw`` is the raw forward return (for IC).
    y_label = y_demeaned.stack(dropna=False).reindex(feats.index)
    y_raw = fwd.stack(dropna=False).reindex(feats.index)

    # The calendar position of each row's date (for the label-realization gate)
    # and of its label-realization date (date_pos + horizon).
    date_index = dates
    pos_of = {d: i for i, d in enumerate(date_index)}
    row_dates = feats.index.get_level_values("date")
    row_pos = np.array([pos_of[d] for d in row_dates], dtype=int)
    label_pos = row_pos + horizon  # position whose close the label needs

    feat_X = feats.to_numpy(dtype=float)
    n_rows = len(feats)

    # --- 2. walk-forward training cutoffs -----------------------------------
    # We refit at a set of cutoff positions spaced ``refit_every`` apart. At each
    # cutoff T (a calendar position) we train on rows whose label is realized
    # (label_pos <= T) within the trailing window, then score rows strictly after
    # the cutoff up to the next cutoff.
    n_dates = len(date_index)
    train_window_days = int(round(train_years * _DAYS_PER_YEAR))

    # First cutoff: the earliest position at which at least a minimal training
    # set could exist. We require at least one realized row, i.e. position
    # >= (first feature row pos + horizon). Start scanning from there.
    first_label_pos = int(label_pos.min()) if n_rows else n_dates
    start_cutoff = max(first_label_pos, 1)

    cutoffs = list(range(start_cutoff, n_dates, refit_every))
    if not cutoffs:
        return _empty(dates, assets, "history too short for any walk-forward refit; direction=0")

    # OOS predicted scores, one per feature row (NaN until scored).
    oos_score = np.full(n_rows, np.nan, dtype=float)
    n_refits = 0

    # Precompute, for speed, a sort of rows by position is unnecessary; we mask.
    for ci, cutoff in enumerate(cutoffs):
        next_cutoff = cutoffs[ci + 1] if ci + 1 < len(cutoffs) else n_dates

        # Training mask: label fully realized at or before the cutoff (no leak),
        # and the feature-row date within the trailing train window.
        train_lo = cutoff - train_window_days
        realized = label_pos <= cutoff
        in_window = (row_pos > train_lo) & (row_pos <= cutoff)
        train_mask = realized & in_window
        y_train = y_label.to_numpy(dtype=float)[train_mask]
        train_finite = np.isfinite(y_train)
        n_train = int(train_finite.sum())

        # Scoring mask: rows strictly AFTER the cutoff, up to the next cutoff.
        # These are genuine out-of-sample dates relative to the fit.
        score_mask = (row_pos > cutoff) & (row_pos <= next_cutoff)

        if n_train < 2 or not score_mask.any():
            continue

        X_tr = feat_X[train_mask][train_finite]
        y_tr = y_train[train_finite]
        try:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            oos_score[score_mask] = model.predict(feat_X[score_mask])
            n_refits += 1
        except Exception:
            # Skip this window on a fit/predict failure; keep going causally.
            continue

    if n_refits == 0:
        return _empty(dates, assets, "no usable walk-forward windows; direction=0")

    # --- 3. assemble the wide OOS score panel (dates x assets) --------------
    score_series = pd.Series(oos_score, index=feats.index)
    score_panel = score_series.unstack("asset").reindex(index=dates, columns=assets)

    # --- 4. rank IC over rebalance dates (the honest diagnostic) ------------
    # Rebalance dates on the panel calendar (e.g. weekly), matching the project
    # convention in signals.base.target_directions.
    rebal_dates = close.resample(rebalance).last().index
    rebal_dates = [d for d in rebal_dates if d in close.index]

    fwd_raw_panel = fwd  # dates x assets raw forward return
    ics: list[float] = []
    for d in rebal_dates:
        s_row = score_panel.loc[d]
        r_row = fwd_raw_panel.loc[d]
        ok = s_row.notna() & r_row.notna()
        # Spearman needs >= 2 points and variation in both vectors.
        if ok.sum() < 2:
            continue
        sr = s_row[ok]
        rr = r_row[ok]
        if sr.nunique() < 2 or rr.nunique() < 2:
            continue
        ic = sr.rank().corr(rr.rank())  # Spearman = Pearson on ranks
        if np.isfinite(ic):
            ics.append(float(ic))

    if ics:
        rank_ic = float(np.mean(ics))
        ic_hit = float(np.mean([1.0 if x > 0 else 0.0 for x in ics]))
    else:
        rank_ic = float("nan")
        ic_hit = 0.0

    # --- 5. build the {-1, 0, +1} direction matrix --------------------------
    direction = pd.DataFrame(0.0, index=dates, columns=assets)
    any_position = False
    for d in rebal_dates:
        row = score_panel.loc[d]
        valid = row.dropna()
        if valid.empty:
            continue
        ranked = valid.sort_values(ascending=False)
        longs = list(ranked.index[:top_k])
        shorts = list(ranked.index[-bottom_k:]) if bottom_k > 0 else []
        # Guard against overlap when the universe is smaller than top_k+bottom_k:
        # a long takes precedence and a short is dropped if it collides.
        shorts = [a for a in shorts if a not in longs]
        if not longs and not shorts:
            continue
        direction.loc[d, :] = 0.0
        if longs:
            direction.loc[d, longs] = 1.0
        if shorts:
            direction.loc[d, shorts] = -1.0
        any_position = True

    # Hold positions between rebalances (forward-fill), zeros during warm-up.
    if any_position:
        # Mark non-rebalance days as NaN first so ffill carries the last set
        # rebalance row; then fill remaining (pre-first-rebalance) with 0.
        rebal_set = set(rebal_dates)
        mask_non_rebal = ~direction.index.isin(rebal_set)
        direction.loc[mask_non_rebal, :] = np.nan
        direction = direction.ffill().fillna(0.0)
    else:
        direction = pd.DataFrame(0.0, index=dates, columns=assets)

    first_scored = None
    scored_dates = score_panel.dropna(how="all").index
    if len(scored_dates):
        first_scored = scored_dates[0]

    status = (
        f"ok: walk-forward LTR (RandomForestRegressor, n_estimators={n_estimators}), "
        f"horizon={horizon}d, train_years={train_years}, refit_every={refit_every}d, "
        f"{n_refits} refits; long top_k={top_k}/short bottom_k={bottom_k} on '{rebalance}' "
        f"rebalances; OOS scores from "
        f"{first_scored.date() if first_scored is not None else 'n/a'}; "
        f"rank_ic={rank_ic:.4f} over {len(ics)} rebalance dates"
    )

    return LTRResult(
        direction=direction,
        rank_ic=rank_ic,
        ic_hit=ic_hit,
        n_refits=n_refits,
        status=status,
    )
