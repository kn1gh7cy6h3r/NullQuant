"""
lead_lag.py — walk-forward lead-lag contagion network.

WHAT THIS DOES
--------------
Some coins consistently LEAD others: when BTC jolts, an altcoin may only react a
day or two later. If that lead-lag relationship is stable, the leader's already-
*observed* move carries information about the follower's *next* move. This module
learns a sparse directed network of such relationships and turns it into a
long/short book that trades the laggards on information that has already arrived.

THE DESIGN (strictly causal, walk-forward)
------------------------------------------
1. ESTIMATE THE NETWORK. On a trailing window of ``cfg.ml.leadlag.lookback`` days
   we score every ordered pair (leader A -> follower B), A != B, at every lag in
   ``1 .. cfg.ml.leadlag.max_lag``: the Pearson correlation between A's return at
   ``t - lag`` and B's return at ``t``. We keep the ``cfg.ml.leadlag.top_edges``
   strongest directed edges overall (by ``|strength|``).

2. WALK FORWARD (NO LEAKAGE). Every ``cfg.ml.leadlag.refit_every`` days we re-fit
   the network on the window ENDING AT t-1. For each day t in the following OOS
   block, the predicted next-day signal for a follower B is

       pred_B(t) = sum over kept edges (A -> B, lag) of
                   strength_{A->B,lag} * ret_A(t + 1 - lag)

   Every leader return used (``t``, ``t-1``, ...) is already observed at the close
   of day t, so ``pred_B(t)`` is a genuine forecast of B's return over (t, t+1]
   made from PAST information only. The fit window never includes the day scored.

3. BUILD THE BOOK. On each weekly rebalance date (``cfg.strategy.rebalance``) we
   rank assets by their predicted signal, LONG the top ``cfg.strategy.top_k`` and
   SHORT the bottom ``cfg.strategy.bottom_k``; directions are forward-filled until
   the next rebalance. Warm-up dates (before the first OOS prediction) are flat.

THE HONEST DIAGNOSTIC
---------------------
``oos_hit_rate`` is the truth-teller: across every (asset, OOS day) it measures
how often ``sign(pred)`` matches ``sign(realized next return)``. Compare it to
0.5. A value near 0.5 means there is no exploitable lead-lag predictability in
this universe over this period — we report it as found, never tuned to inflate.

Degrades gracefully: on any failure we return an all-zero direction matrix with
an explanatory ``status`` rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import Panel
from ..features import indicators as ind
from ..seeds import set_global_seed

# A directed lead-lag edge: (leader, follower, lag, strength).
Edge = tuple[str, str, int, float]


@dataclass
class LeadLagResult:
    """Output of the lead-lag contagion strategy.

    Attributes
    ----------
    direction:
        DataFrame indexed by ``panel.close.index`` with columns ``panel.assets``,
        values in {-1.0, 0.0, +1.0}: the daily long/short book.
    oos_hit_rate:
        Fraction of (asset, OOS day) cells where the sign of the predicted next
        move equals the sign of the realized next return. Compare to 0.5.
    top_edges:
        The kept directed edges (leader, follower, lag, strength) from the LAST
        refit, ordered by descending ``|strength|``.
    n_refits:
        Number of walk-forward network re-estimations performed.
    status:
        Human-readable status string (also reports graceful degradation).
    """

    direction: pd.DataFrame
    oos_hit_rate: float
    top_edges: list[Edge]
    n_refits: int
    status: str = ""


# ---------------------------------------------------------------------------
# Network estimation
# ---------------------------------------------------------------------------
def _estimate_edges(window: np.ndarray, assets: list[str], max_lag: int,
                    top_edges: int) -> list[Edge]:
    """Estimate the strongest directed lead-lag edges on one trailing window.

    ``window`` is a (T, N) array of log returns (rows = days, cols = assets),
    assumed already free of NaNs. For each ordered pair (A -> B), A != B, and
    each lag in ``1..max_lag`` we compute the Pearson correlation between A's
    return at ``t - lag`` and B's return at ``t``. The ``top_edges`` edges with
    the largest ``|strength|`` (across all pairs and lags) are returned.
    """
    n = window.shape[1]
    candidates: list[Edge] = []

    for lag in range(1, max_lag + 1):
        # Leader observations end `lag` days before the follower observation.
        leader = window[:-lag]          # rows 0 .. T-lag-1  (A at t-lag)
        follower = window[lag:]         # rows lag .. T-1    (B at t)
        if leader.shape[0] < 3:
            continue

        # Vectorised Pearson correlation across the aligned rows: corr[a, b] is
        # corr(leader_a, follower_b) = predictive corr of A(t-lag) -> B(t).
        lead_c = leader - leader.mean(axis=0, keepdims=True)
        foll_c = follower - follower.mean(axis=0, keepdims=True)
        lead_norm = np.sqrt((lead_c ** 2).sum(axis=0))
        foll_norm = np.sqrt((foll_c ** 2).sum(axis=0))
        denom = np.outer(lead_norm, foll_norm)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = (lead_c.T @ foll_c) / denom
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                candidates.append((assets[a], assets[b], lag, float(corr[a, b])))

    # Keep the globally strongest directed edges by absolute strength.
    candidates.sort(key=lambda e: abs(e[3]), reverse=True)
    return candidates[:top_edges]


def _predict_from_edges(edges: list[Edge], rets: pd.DataFrame, t_pos: int,
                        asset_pos: dict[str, int]) -> np.ndarray:
    """Predicted next-day signal per asset at integer position ``t_pos``.

    For each kept edge (A -> B, lag, strength) the contribution to follower B is
    ``strength * ret_A(t + 1 - lag)`` — i.e. the leader return observed ``lag-1``
    days ago relative to t. With lag=1 this uses the leader's return at t itself;
    both endpoints are observed by the close of day t, so the prediction of B's
    return over (t, t+1] is strictly causal.
    """
    n = rets.shape[1]
    pred = np.zeros(n, dtype=float)
    values = rets.to_numpy()

    for leader, follower, lag, strength in edges:
        src = t_pos + 1 - lag  # position of the (already observed) leader return
        if src < 0:
            continue
        a = asset_pos[leader]
        b = asset_pos[follower]
        x = values[src, a]
        if np.isfinite(x):
            pred[b] += strength * x
    return pred


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def leadlag_signal(panel: Panel, cfg: Config) -> LeadLagResult:
    """Walk-forward lead-lag contagion signal — see module docstring.

    Returns a :class:`LeadLagResult`. Never raises: on any failure it yields an
    all-zero ``direction`` matrix and an explanatory ``status``.
    """
    set_global_seed(cfg.seed)

    dates = panel.close.index
    assets = list(panel.assets)
    direction = pd.DataFrame(0.0, index=dates, columns=assets, dtype=float)

    # --- read config defensively -------------------------------------------
    try:
        max_lag = int(cfg.ml.leadlag.max_lag)
        lookback = int(cfg.ml.leadlag.lookback)
        refit_every = int(cfg.ml.leadlag.refit_every)
        top_edges = int(cfg.ml.leadlag.top_edges)
        top_k = int(cfg.strategy.top_k)
        bottom_k = int(cfg.strategy.bottom_k)
        rebalance = str(cfg.strategy.rebalance)
        # Weight of the structural funding tilt blended into the book (0 = pure
        # price-action lead-lag). Funding is exogenous to price, so it adds the
        # cross-exchange structural signal that daily price lead-lag lacks.
        funding_weight = float(cfg.get("ml.leadlag.funding_weight", 0.5))
    except Exception as exc:  # pragma: no cover - misconfiguration guard
        return LeadLagResult(
            direction=direction, oos_hit_rate=float("nan"),
            top_edges=[], n_refits=0,
            status=f"config error ({exc}); direction=0",
        )

    if max_lag < 1 or lookback < max_lag + 3 or refit_every < 1 or top_edges < 1:
        return LeadLagResult(
            direction=direction, oos_hit_rate=float("nan"),
            top_edges=[], n_refits=0,
            status=("invalid leadlag params "
                    f"(max_lag={max_lag}, lookback={lookback}, "
                    f"refit_every={refit_every}, top_edges={top_edges}); direction=0"),
        )

    rets = panel.log_returns()
    asset_pos = {a: i for i, a in enumerate(assets)}
    n_dates = len(dates)

    # We can first PREDICT day t once we have `lookback` clean rows ending at t-1
    # available for the fit. Predictions need leader returns up to lag back, all
    # within the trailing window, so the window itself must be NaN-free.
    if n_dates < lookback + 2:
        return LeadLagResult(
            direction=direction, oos_hit_rate=float("nan"),
            top_edges=[], n_refits=0,
            status=(f"insufficient history: {n_dates} dates < lookback+2 "
                    f"({lookback + 2}); direction=0"),
        )

    # --- walk forward -------------------------------------------------------
    # `pred_signal[t]` is the OOS prediction of each asset's return over (t, t+1].
    pred_signal = pd.DataFrame(np.nan, index=dates, columns=assets, dtype=float)
    last_edges: list[Edge] = []
    n_refits = 0
    fit_anchor = -(10 ** 9)  # position of last day in the current fit window

    # The earliest day we can score is `lookback` (its fit window is rows
    # [0, lookback) which all precede it). We stop at n_dates-2 so that a
    # realized NEXT return ret(t+1) exists for the honest hit-rate check.
    try:
        for t in range(lookback, n_dates - 1):
            if not last_edges or (t - 1 - fit_anchor) >= refit_every:
                window = rets.iloc[t - lookback:t]  # rows ending at t-1 (excl. t)
                window = window.dropna(axis=0, how="any")
                if window.shape[0] >= max_lag + 3:
                    last_edges = _estimate_edges(
                        window.to_numpy(dtype=float), assets, max_lag, top_edges,
                    )
                    fit_anchor = t - 1
                    n_refits += 1
                # If the window was too dirty we simply reuse the previous edges
                # (or stay empty until a clean window appears).

            if not last_edges:
                continue

            pred_signal.iloc[t] = _predict_from_edges(last_edges, rets, t, asset_pos)
    except Exception as exc:  # pragma: no cover - numerical safety net
        return LeadLagResult(
            direction=direction, oos_hit_rate=float("nan"),
            top_edges=last_edges, n_refits=n_refits,
            status=f"runtime error during walk-forward ({exc}); direction=0",
        )

    if n_refits == 0 or pred_signal.notna().to_numpy().sum() == 0:
        return LeadLagResult(
            direction=direction, oos_hit_rate=float("nan"),
            top_edges=last_edges, n_refits=n_refits,
            status=("no clean trailing window large enough to estimate the "
                    "network; direction=0"),
        )

    # --- honest OOS hit rate -----------------------------------------------
    # Realized next return aligned to the prediction made at t: ret over (t, t+1].
    realized_next = rets.shift(-1)
    pred_sign = np.sign(pred_signal.to_numpy())
    real_sign = np.sign(realized_next.to_numpy())
    # A cell counts only where we made a non-zero prediction AND a realized next
    # return exists (both finite, both non-zero so sign is defined).
    mask = (
        np.isfinite(pred_signal.to_numpy()) & np.isfinite(realized_next.to_numpy())
        & (pred_sign != 0.0) & (real_sign != 0.0)
    )
    n_eval = int(mask.sum())
    if n_eval > 0:
        hits = int((pred_sign[mask] == real_sign[mask]).sum())
        oos_hit_rate = float(hits / n_eval)
    else:
        oos_hit_rate = float("nan")

    # --- build the long/short book -----------------------------------------
    # The book trades a BLEND of (a) the price-action lead-lag prediction and
    # (b) the structural funding tilt. Both legs are put on a common robust
    # cross-sectional z-scale first so neither swamps the other, then summed:
    #
    #   book_score = z(pred_signal) + funding_weight * funding_rank_signal
    #
    # where funding_rank_signal already encodes "high funding -> bearish". The
    # OOS hit-rate above stays on the PURE lead-lag prediction (the network's own
    # truth-teller); the funding tilt only shapes position sizing/selection.
    book_score = ind.cross_sectional_mad_zscore(pred_signal)
    funding_used = False
    if funding_weight != 0.0 and panel.has_funding:
        funding = panel.funding.reindex(index=dates, columns=assets)
        funding_score = ind.funding_rank_signal(funding)
        book_score = book_score.add(funding_weight * funding_score, fill_value=0.0)
        # Keep a row scorable only where the lead-lag prediction itself existed.
        book_score = book_score.where(pred_signal.notna())
        funding_used = True

    # Rank by the (blended) book score on each rebalance date; hold until next.
    rebal_dates = panel.close.resample(rebalance).last().index
    rebal_dates = [d for d in rebal_dates if d in book_score.index]

    for d in rebal_dates:
        row = book_score.loc[d]
        valid = row.notna()
        if int(valid.sum()) == 0:
            continue
        ranked = row[valid].sort_values(ascending=False)
        longs = list(ranked.index[:top_k])
        shorts = list(ranked.index[::-1][:bottom_k])
        # Avoid a name being both long and short when the cross-section is thin.
        shorts = [s for s in shorts if s not in longs]

        direction.loc[d, :] = 0.0
        if longs:
            direction.loc[d, longs] = 1.0
        if shorts:
            direction.loc[d, shorts] = -1.0

    # Hold each rebalance's directions until the next; warm-up rows stay flat.
    rebal_set = set(rebal_dates)
    direction.loc[[d not in rebal_set for d in direction.index], :] = np.nan
    direction = direction.ffill().fillna(0.0)

    first_oos = dates[lookback].date() if lookback < n_dates else "n/a"
    tilt = (f"funding tilt blended (weight={funding_weight})" if funding_used
            else "funding tilt off (no funding data)")
    status = (
        f"ok: walk-forward lead-lag network, max_lag={max_lag}, "
        f"lookback={lookback}, refit_every={refit_every}d, top_edges={top_edges}; "
        f"{n_refits} refits, {n_eval} (asset, OOS day) cells evaluated from "
        f"{first_oos}; oos_hit_rate={oos_hit_rate:.4f} vs 0.5 baseline; {tilt}"
    )

    return LeadLagResult(
        direction=direction,
        oos_hit_rate=oos_hit_rate,
        top_edges=last_edges,
        n_refits=n_refits,
        status=status,
    )
