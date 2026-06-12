"""
regime_switch.py — Gaussian-HMM regime-switching meta-controller.

WHAT THIS DOES
--------------
A Gaussian HMM discovers a small number of HIDDEN market regimes from
backward-looking, market-level features. A *meta-controller* then decides WHICH
candidate sub-strategy to run inside each regime. The ML here does NOT forecast
price; it learns a regime -> sub-strategy *selection* policy. The output is the
familiar daily DIRECTION matrix in {-1, 0, +1} that the rest of the system
consumes, resampled to the weekly rebalance so turnover matches everything else.

THE THREE CANDIDATE SUB-STRATEGIES (each a full direction matrix)
-----------------------------------------------------------------
  • TREND          — long the strongest / short the weakest by risk-adjusted
                     momentum (the classic cross-sectional momentum tilt).
  • MEAN-REVERSION — the exact reverse (long the weakest / short the strongest).
  • FLAT           — sit out (all zeros).

WHY WALK-FORWARD (THE NO-LEAKAGE DESIGN)
----------------------------------------
The naive version fits the HMM on the whole history, decodes every day, and then
asks — with full hindsight — which sub-strategy each regime *would have* liked.
That double-leaks: the regime labels see the future, and the regime->strategy
mapping is scored on the very days it is later applied to.

This implementation is strictly causal:

  * Every ``cfg.ml.regime.refit_every`` days we (re)fit a fresh GaussianHMM and
    its StandardScaler on the trailing ``cfg.ml.regime.train_years`` of feature
    rows ENDING STRICTLY BEFORE the current block.
  * The regime -> best-sub-strategy mapping is learned ON THE TRAINING WINDOW
    ONLY: for each decoded training state we measure which sub-strategy earned
    the best realized Sharpe over the training days that fell in that state, and
    map the state to that sub-strategy.
  * Out-of-sample, each new day's regime is decoded CAUSALLY — we feed the HMM
    the standardized feature sequence up to and including ``t`` and take the
    LAST decoded state (``predict(...)[-1]``), which uses no future information.
    We then apply that regime's chosen sub-strategy's direction for ``t``.

All features are rolling/realized, so each feature vector is itself causal. The
daily OOS direction is finally resampled to ``cfg.strategy.rebalance`` and
forward-filled; warm-up dates (before any OOS decode is possible) are flat.

Degrades gracefully: if ``hmmlearn`` is unavailable or any fit fails, an
all-zero direction is returned with an explanatory ``status`` — never raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import Panel
from ..features import indicators as ind
from ..seeds import set_global_seed

# Sub-strategy names (also the mapping/occupancy keys).
_TREND = "TREND"
_MEANREV = "MEANREV"
_FLAT = "FLAT"
_SUBSTRATEGIES: list[str] = [_TREND, _MEANREV, _FLAT]

# HMM EM iterations per fit — enough to converge on these short feature series
# while keeping the periodic refits cheap.
_N_ITER = 100

# Trading days per year, used to translate ``train_years`` into a row count.
_DAYS_PER_YEAR = 365


@dataclass
class RegimeSwitchResult:
    """Output of the HMM regime-switching meta-controller.

    Attributes
    ----------
    direction:
        Daily DIRECTION matrix (index=panel.close.index, columns=panel.assets)
        with values in {-1.0, 0.0, +1.0}, resampled to the weekly rebalance and
        forward-filled. Flat (zeros) on warm-up dates.
    n_states:
        Number of hidden HMM states (``cfg.ml.regime.n_states``).
    regime_series:
        Decoded OOS regime (int) per date; only populated on days that were
        actually decoded out-of-sample (NaN elsewhere).
    mapping:
        ``{state -> sub-strategy name}`` learned at the LAST refit (the most
        recent policy). States never observed in that training window are absent.
    occupancy:
        ``{sub-strategy name -> fraction of OOS days}`` it was actually applied.
    status:
        Human-readable status, including graceful-degradation notes and an
        honest read of whether the controller collapsed to a single choice.
    """

    direction: pd.DataFrame
    n_states: int
    regime_series: pd.Series
    mapping: dict[int, str]
    occupancy: dict[str, float]
    status: str = "ok"


# ===========================================================================
# Candidate sub-strategy direction matrices
# ===========================================================================
def _candidate_directions(panel: Panel, cfg: Config) -> dict[str, pd.DataFrame]:
    """Build the daily {-1, 0, +1} direction matrix for each candidate strategy.

    All three share the same calendar/columns as ``panel.close``. TREND and
    MEAN-REVERSION are built DAILY (the meta-controller does the rebalancing
    downstream); FLAT is all zeros. Both directional strategies rank by
    risk-adjusted momentum and take the ``top_k`` / ``bottom_k`` legs:

      • TREND   — long the strongest, short the weakest.
      • MEANREV — long the weakest, short the strongest (the mirror image).
    """
    s = cfg.strategy
    close = panel.close
    top_k = int(s.top_k)
    bottom_k = int(s.bottom_k)

    ras = ind.risk_adjusted_momentum(close, int(s.momentum_lookback), int(s.vol_lookback))

    trend = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    meanrev = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    # For each date, rank the valid assets by risk-adjusted momentum and assign
    # the long/short legs. TREND longs the top, MEANREV longs the bottom.
    for d in close.index:
        row = ras.loc[d]
        valid = row.dropna()
        if len(valid) < 2:
            continue
        ranked = valid.sort_values(ascending=False)  # strongest -> weakest
        strongest = list(ranked.index[:top_k])
        weakest = list(ranked.index[::-1][:bottom_k])
        # Avoid an asset being both long and short when the universe is tiny.
        weakest = [a for a in weakest if a not in strongest]

        if strongest:
            trend.loc[d, strongest] = 1.0
            meanrev.loc[d, strongest] = -1.0
        if weakest:
            trend.loc[d, weakest] = -1.0
            meanrev.loc[d, weakest] = 1.0

    flat = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    return {_TREND: trend, _MEANREV: meanrev, _FLAT: flat}


# ===========================================================================
# Market-level (cross-sectional) HMM features
# ===========================================================================
def _build_market_features(panel: Panel, cfg: Config) -> pd.DataFrame:
    """Build strictly causal, STATIONARY "market-texture" features for the HMM.

    The previous feature set included DIRECTIONAL terms (cross-sectional mean
    return, BTC trailing return). In a market that drifts up most days, those
    terms carry almost all the variance, so the HMM's dominant axis becomes
    "up vs down" and one trend state swallows ~all the mass — the controller
    goes comatose. We therefore strip every price-level / directional / drifting
    term and feed the HMM only scale-free texture that describes the CHARACTER of
    the tape, not its direction. Every column is backward-looking at date t:

      * cs_dispersion    — cross-sectional std of daily log-returns across assets
                           (how much names are pulling apart: trend vs chop)
      * avg_realized_vol — average realized vol across assets (calm vs stressed)
      * vol_to_vol       — mean over assets of relative-volume / realized-vol
                           (participation per unit of risk; a liquidity texture)
      * ret_skew         — mean over assets of each asset's rolling return skew
                           (crash vs melt-up asymmetry)

    Volume is reduced to RELATIVE volume (over its own rolling mean) before use so
    the feature is scale-free and stationary rather than a drifting level.
    """
    rets = panel.log_returns()
    w = int(cfg.strategy.vol_lookback)

    cs_dispersion = rets.std(axis=1, skipna=True)

    realized = ind.realized_vol(panel.close, w, annualize=False)
    avg_realized_vol = realized.mean(axis=1, skipna=True)

    # Relative volume (scale-free) over its own trailing mean, per unit of vol.
    volume = panel.field("Volume")
    rel_volume = volume / volume.rolling(w, min_periods=w).mean()
    vol_to_vol = (rel_volume / realized.replace(0.0, np.nan)).mean(axis=1, skipna=True)

    # Rolling per-asset return skewness, averaged across the universe.
    ret_skew = rets.rolling(w, min_periods=w).skew().mean(axis=1, skipna=True)

    feats = pd.DataFrame(
        {
            "cs_dispersion": cs_dispersion,
            "avg_realized_vol": avg_realized_vol,
            "vol_to_vol": vol_to_vol,
            "ret_skew": ret_skew,
        },
        index=panel.close.index,
    )
    return feats


# ===========================================================================
# Sub-strategy scoring on a training window
# ===========================================================================
def _sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe of a daily return array; 0.0 when undefined."""
    r = returns[np.isfinite(returns)]
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(_DAYS_PER_YEAR))


def _strategy_daily_returns(
    direction: pd.DataFrame, asset_rets: pd.DataFrame
) -> pd.Series:
    """Realized daily PnL of a direction matrix (positions held from t to t+1).

    The direction decided at t earns the asset's return from t to t+1, so we
    shift the direction forward one bar before multiplying — strictly causal and
    consistent with how the live signal is applied.
    """
    pnl = (direction.shift(1) * asset_rets).sum(axis=1, skipna=True)
    return pnl


# ===========================================================================
# Public API — regime-switching meta-controller
# ===========================================================================
def regime_switch_signal(panel: Panel, cfg: Config) -> RegimeSwitchResult:
    """Walk-forward Gaussian-HMM regime-switching meta-controller.

    Pipeline (all causal):
      1. Build market-level features and the TREND / MEANREV / FLAT candidate
         direction matrices.
      2. Walk forward in ``refit_every``-day blocks. At each refit, fit a fresh
         scaler + GaussianHMM on the trailing ``train_years`` of feature rows
         (strictly before the block) and learn a regime -> best-Sharpe
         sub-strategy mapping on those same training days only.
      3. For each OOS day in the block, decode the regime causally (last state of
         the standardized sequence up to and including that day) and apply that
         regime's chosen sub-strategy's daily direction.
      4. Resample the resulting daily direction to ``cfg.strategy.rebalance`` and
         forward-fill; warm-up dates stay flat.

    Returns a :class:`RegimeSwitchResult`. Degrades gracefully: a missing
    ``hmmlearn`` install or any fit failure yields an all-zero direction with an
    explanatory ``status`` rather than raising.
    """
    set_global_seed(cfg.seed)

    dates = panel.close.index
    assets = panel.assets
    n_states = int(cfg.ml.regime.n_states)
    refit_every = int(cfg.ml.regime.refit_every)
    train_rows = int(round(float(cfg.ml.regime.train_years) * _DAYS_PER_YEAR))

    # Default (degraded) outputs: flat everywhere, nothing decoded.
    zero_direction = pd.DataFrame(0.0, index=dates, columns=assets)
    empty_regime = pd.Series(np.nan, index=dates, dtype=float)

    def _degraded(status: str) -> RegimeSwitchResult:
        return RegimeSwitchResult(
            direction=zero_direction.copy(),
            n_states=n_states,
            regime_series=empty_regime.copy(),
            mapping={},
            occupancy={name: 0.0 for name in _SUBSTRATEGIES},
            status=status,
        )

    # --- Defensive optional-dependency imports -----------------------------
    try:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - exercised only without hmmlearn
        return _degraded(
            f"hmmlearn/sklearn unavailable ({exc}); regime controller disabled, direction=0"
        )

    # --- Features + candidate directions -----------------------------------
    candidates = _candidate_directions(panel, cfg)
    asset_rets = panel.log_returns()

    feats = _build_market_features(panel, cfg)
    valid = feats.dropna()
    if len(valid) <= train_rows + 1:
        return _degraded(
            f"insufficient history: {len(valid)} valid feature rows "
            f"<= train_rows+1 ({train_rows + 1}); direction=0"
        )

    feat_dates = valid.index
    X_all = valid.to_numpy(dtype=float)
    n = len(feat_dates)

    # Daily OOS direction we are filling in, plus the decoded-regime record.
    daily_direction = pd.DataFrame(0.0, index=dates, columns=assets)
    regime_record = pd.Series(np.nan, index=dates, dtype=float)
    occupancy_counts = {name: 0 for name in _SUBSTRATEGIES}
    last_mapping: dict[int, str] = {}
    n_oos = 0
    n_fit_failures = 0

    # First scorable position: we need a full training window strictly before it.
    start = train_rows

    # Cached fitted artefacts between refits.
    model: "GaussianHMM | None" = None
    scaler: "StandardScaler | None" = None
    mapping: dict[int, str] = {}
    fit_anchor = -10**9  # position of the last refit

    # --- Walk forward -------------------------------------------------------
    for i in range(start, n):
        need_refit = model is None or (i - fit_anchor) >= refit_every
        if need_refit:
            # Trailing training window: rows [i-train_rows, i-1], strictly past.
            train_slice = slice(i - train_rows, i)
            X_train = X_all[train_slice]
            train_dates = feat_dates[train_slice]
            try:
                scaler = StandardScaler().fit(X_train)
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    random_state=int(cfg.seed),
                    n_iter=_N_ITER,
                )
                model.fit(scaler.transform(X_train))
                # Decode the TRAINING states and learn regime -> best sub-strategy
                # on the training window ONLY.
                train_states = model.predict(scaler.transform(X_train))
                mapping = _learn_mapping(
                    train_states=train_states,
                    train_dates=train_dates,
                    n_states=n_states,
                    candidates=candidates,
                    asset_rets=asset_rets,
                )
                last_mapping = mapping
                fit_anchor = i
            except Exception:
                # A failed fit must not abort the walk-forward; carry the prior
                # model if we have one, otherwise stay flat for this day.
                n_fit_failures += 1
                if model is None:
                    continue

        if model is None or scaler is None:
            continue

        # Causal decode: feed the standardized sequence up to and INCLUDING i,
        # take the LAST state. This uses no information past day i.
        try:
            seq = scaler.transform(X_all[: i + 1])
            state = int(model.predict(seq)[-1])
        except Exception:
            n_fit_failures += 1
            continue

        d = feat_dates[i]
        regime_record.loc[d] = float(state)
        choice = mapping.get(state, _FLAT)  # unseen state -> sit out
        daily_direction.loc[d, :] = candidates[choice].loc[d, :].to_numpy()
        occupancy_counts[choice] += 1
        n_oos += 1

    if n_oos == 0:
        return _degraded(
            f"no OOS days decoded (fit failures={n_fit_failures}); direction=0"
        )

    # --- Resample daily OOS direction to the weekly rebalance ---------------
    # Match the rest of the system's turnover: hold each rebalance's decision
    # until the next rebalance via forward-fill; warm-up rows stay flat.
    rebalance = cfg.strategy.rebalance
    rebal_dates = daily_direction.resample(rebalance).last().index
    rebal_set = {d for d in rebal_dates if d in daily_direction.index}

    resampled = daily_direction.copy()
    resampled.loc[[d not in rebal_set for d in resampled.index], :] = np.nan
    resampled = resampled.ffill().fillna(0.0)

    # --- Occupancy + honest diagnostic -------------------------------------
    occupancy = {name: float(occupancy_counts[name] / n_oos) for name in _SUBSTRATEGIES}

    status = _build_status(
        n_oos=n_oos,
        n_fit_failures=n_fit_failures,
        n_states=n_states,
        train_rows=train_rows,
        refit_every=refit_every,
        rebalance=str(rebalance),
        mapping=last_mapping,
        occupancy=occupancy,
        first_oos=feat_dates[start] if n_oos else None,
    )

    return RegimeSwitchResult(
        direction=resampled,
        n_states=n_states,
        regime_series=regime_record,
        mapping=last_mapping,
        occupancy=occupancy,
        status=status,
    )


# ===========================================================================
# Mapping + status helpers
# ===========================================================================
def _learn_mapping(
    train_states: np.ndarray,
    train_dates: pd.DatetimeIndex,
    n_states: int,
    candidates: dict[str, pd.DataFrame],
    asset_rets: pd.DataFrame,
) -> dict[int, str]:
    """Map each training state to its best-Sharpe sub-strategy (training only).

    For each decoded state we restrict to the training days in that state, score
    every candidate sub-strategy by the realized Sharpe of its daily PnL over
    those days, and assign the state to the best one. A state with too few days
    to score reliably (< 5) defaults to FLAT.
    """
    # Pre-compute each candidate's daily PnL over the whole panel once.
    pnl = {name: _strategy_daily_returns(candidates[name], asset_rets) for name in _SUBSTRATEGIES}

    states_series = pd.Series(train_states, index=train_dates)
    mapping: dict[int, str] = {}
    for st in range(n_states):
        st_dates = states_series.index[states_series.values == st]
        if len(st_dates) < 5:
            mapping[st] = _FLAT
            continue
        best_name = _FLAT
        best_sharpe = -np.inf
        for name in _SUBSTRATEGIES:
            r = pnl[name].reindex(st_dates).to_numpy(dtype=float)
            sh = _sharpe(r)
            # FLAT's Sharpe is 0 by construction; it wins only if both
            # directional strategies are net-negative on this state's days.
            if sh > best_sharpe:
                best_sharpe = sh
                best_name = name
        mapping[st] = best_name
    return mapping


def _build_status(
    n_oos: int,
    n_fit_failures: int,
    n_states: int,
    train_rows: int,
    refit_every: int,
    rebalance: str,
    mapping: dict[int, str],
    occupancy: dict[str, float],
    first_oos,
) -> str:
    """Compose an honest status string, flagging collapse to one sub-strategy."""
    first = first_oos.date() if first_oos is not None else "n/a"
    parts = [
        f"ok: walk-forward GaussianHMM(n_states={n_states}, cov=diag), "
        f"train_rows={train_rows}, refit_every={refit_every}d, rebalance={rebalance}; "
        f"{n_oos} OOS days decoded from {first}"
    ]
    if n_fit_failures:
        parts.append(f"{n_fit_failures} fit/decode failures handled gracefully")

    # Honest read: did the controller actually switch, or collapse?
    used = {name for name, frac in occupancy.items() if frac > 0.0}
    dominant = max(occupancy, key=occupancy.get) if occupancy else _FLAT
    dom_frac = occupancy.get(dominant, 0.0)
    if len(used) <= 1:
        only = next(iter(used)) if used else _FLAT
        parts.append(f"COLLAPSED: controller only ever ran {only} ({dom_frac:.1%} of OOS days)")
    elif dom_frac >= 0.90:
        parts.append(
            f"NEAR-COLLAPSE: {dominant} dominates ({dom_frac:.1%}); little effective switching"
        )
    else:
        active = ", ".join(f"{k}={v:.0%}" for k, v in occupancy.items() if v > 0)
        parts.append(f"switching across {len(used)} sub-strategies ({active})")

    # The last-refit mapping is the most recent policy.
    if mapping:
        map_str = ", ".join(f"{st}->{name}" for st, name in sorted(mapping.items()))
        parts.append(f"last mapping: {map_str}")
    return "; ".join(parts)
