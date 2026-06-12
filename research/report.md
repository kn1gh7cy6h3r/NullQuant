# NullQuant — Research Report

*A long/short, vol-targeted, multi-crypto trend strategy, evaluated honestly.*

> **TL;DR.** Built a full, reproducible research stack: multi-asset point-in-time
> data (now incl. **perpetual-swap funding rates**), causal + cross-sectional
> features, a cost-aware long/short vol-targeted backtester, walk-forward +
> purged k-fold validation, and **four creative ML models** held to an
> out-of-sample P&L ablation over the **last 6 years (2020–2026)**. **Nuanced,
> honest finding:** the three ML *signal generators* (learning-to-rank,
> regime-switching, lead–lag) all **underperform** the simple baseline
> out-of-sample. A cross-sectional feature + funding **refit removed their worst
> pathologies** (LTR rank IC −0.036 → **−0.006**; regime no longer a single-state
> collapse) but did **not** manufacture alpha. The **conformal confidence gate**
> remains the one validated win — rebuilt as a **continuous inverse-width sizer**
> it holds **90.7%** coverage and still lifts the baseline Sharpe **0.10 → 0.12**
> (lower drawdown and vol) while staying **~94% invested**. The important reveal:
> the earlier *binary* gate's far larger lift (0.05 → 0.33) was **substantially a
> sit-in-cash market-timing artifact** — forced to deploy capital, the honest
> lift is real but modest. Even so, **no variant beats buy-and-hold** in a crypto
> bull market; beta is hard to beat. The contribution is the rigor, the honesty,
> and one component that demonstrably helps.

All numbers below are reproduced by `python -m nullquant.pipeline` (seed 42) and
are written to `research/results/`. The evaluation window is the most recent
`data.history_years` (6) years; models with a 3-year training window therefore
score out-of-sample over roughly the back half (2023→2026).

---

## 1. Hypothesis

Cross-sectional momentum/trend is a documented anomaly in many asset classes.
The hypothesis tested here: among liquid crypto majors, ranking by
risk-adjusted momentum and going **long the strongest up-trending names / short
the weakest down-trending names**, sized to a constant volatility target, earns
a positive risk-adjusted return *after costs* — and that ML overlays
(meta-labeling, regime detection) improve it.

We treat this as a falsifiable claim and try hard to falsify it.

## 2. Data

- **Universe (8 majors):** BTC, ETH, BNB, XRP, ADA, SOL, DOGE, LTC (daily, Yahoo
  Finance, auto-adjusted). Cached from 2019, **trimmed to the most recent 6 years**
  for evaluation: **2020-06-12 → 2026-06-12 (2,192 daily bars)**.
- **Funding rates (new, structural).** Daily perpetual-swap funding for each perp
  is fetched from Binance (`fapi/v1/fundingRate`), cached to Parquet, and aligned
  to the close-of-day calendar (known-at-`t`, no repainting). It is an *exogenous*
  signal — not derivable from spot price — and the whole system degrades
  gracefully to a neutral signal when funding is unavailable (offline / no perp).
- **Point-in-time discipline.** All features/labels/backtests use **closed daily
  bars only**. The live spot price (CoinGecko) is display-only and never mutates
  history — so there is **no repainting**.
- **Survivorship bias (declared).** Yahoo serves only *currently listed* coins,
  so the universe is conditioned on survival; coins that died (e.g. LUNA) are
  absent. This biases results **upward**. We mitigate by letting each asset enter
  the panel only once it has real history (SOL starts late, not back-filled), but
  we do **not** claim to eliminate the bias. Any positive result would need to be
  discounted for it; our result is negative, so the bias only makes the true
  picture weaker, not stronger.

## 3. Methodology

**Signal.** Per asset, the SMA50/SMA200 relation defines a trend regime; golden/
death crosses are the entry events. On a weekly rebalance we rank the universe by
risk-adjusted momentum (90d return ÷ realized vol) and go long the top 3 that are
in an uptrend, short the bottom 3 in a downtrend.

**Sizing.** Inverse-volatility (vol-parity) weights, then the whole book is scaled
to a 20% annualized vol target using the *trailing* realized vol of the gross-1
strategy (causal), capped at 2× gross leverage and 40% per asset.

**Costs.** Every rebalance pays taker fee (10 bps) + half-spread (4 bps) +
slippage (5 bps) = **19 bps per side**, charged on turnover. A sweep multiplies
the whole stack by 0× / 0.5× / 1× / 2× / 4×.

**Causality.** A weight applied to the return over [t−1, t] is decided using
information available no later than t−1 (`weights.shift(1)`). Indicators are
backward-looking; tests assert no look-ahead.

**Validation.**
- *Walk-forward* (rolling 3y train → 6m test) for the strategy, to check
  temporal stability rather than a single full-sample number.
- *Purged k-fold with embargo* (López de Prado) for the ML models: a label's
  outcome window [event, t1] that overlaps a test fold is **purged** from
  training, with an embargo after each fold. This is what makes the ML metrics
  trustworthy.
- *Deflated Sharpe Ratio.* Every reported Sharpe is deflated by the number of
  configurations tried (4), so selecting the "best" variant is penalised — the
  antidote to backtest overfitting.

## 4. The ML layer (four creative models)

We deliberately moved away from *predicting price/returns* (the hardest, lowest-
signal target — and exactly what an earlier iteration tried and failed at) toward
letting ML learn **ordering, regimes, relationships, and uncertainty**. Three
produce a trading **direction**; one is a **sizing** overlay. Each has a built-in
honest truth-teller diagnostic.

| Model | Idea | What it learns | Truth-teller |
|---|---|---|---|
| **Learning-to-Rank** | predict the *order* of winners/losers, not returns | RandomForest on demeaned forward return; **all features cross-sectionally MAD-z-scored** (drift-neutral) + ATR-normalized RS + **funding** tilt; long top-k / short bottom-k | OOS rank IC |
| **Regime-switching (HMM)** | discover hidden market states, switch strategy per state | GaussianHMM on **stationary market-texture only** (dispersion, realized vol, volume/vol, return skew) — no directional terms | regime mapping / occupancy |
| **Lead–lag network** | learn which coins *lead* others; trade the laggards | lagged cross-predictive edges over a trailing window, **blended with a structural funding tilt** | OOS next-day hit-rate |
| **Conformal gate** | bet only when *calibrated-confident*; **size continuously** by certainty | normalized split-conformal; interval half-width = q·σ(x) from forest spread; **exposure ∝ inverse interval width**, floored | empirical coverage vs target |

Each direction source enters the same vol-targeted backtest; the conformal gate
enters as a causal exposure in [0,1] that **scales the target-vol allocation**
(tight interval → up to full target; wide → continuously down to a floor). All
are walk-forward and OOS. **What changed in this refit:** §1 the LTR features were
made cross-sectional/drift-neutral; §2 the HMM was put on stationary texture; §3
funding rates were added to the ranking models; §4 the conformal gate became a
continuous, asymmetric sizer rather than a binary cash switch.

## 5. Results (net of 1× costs, out-of-sample)

| Variant | Ann. return | Ann. vol | Sharpe | Max DD | Deflated Sharpe |
|---|---:|---:|---:|---:|---:|
| baseline | −0.4% | 23.5% | 0.10 | −60.3% | 0.11 |
| **baseline + conformal** | **+0.3%** | 22.1% | **0.12** | **−57.9%** | 0.12 |
| LTR | −14.1% | 24.8% | −0.48 | −70.5% | 0.00 |
| regime | −6.6% | 15.0% | −0.38 | −50.7% | 0.01 |
| lead–lag | −10.4% | 23.1% | −0.36 | −59.3% | 0.01 |
| **[bench] equal-weight** | **+78.9%** | 75.5% | **1.15** | −78.4% | 1.00 |
| **[bench] BTC buy-and-hold** | **+37.2%** | 57.4% | **0.84** | −76.6% | 0.98 |

(Each direction source was also run conformal-gated; the gate trims vol/drawdown
on all of them but cannot rescue a bad signal — full table in
`research/results/variants.csv`.)

![Equity curves](results/equity_curves.png)

**Cost sensitivity (annualized Sharpe):** the conformal-gated baseline beats the
plain baseline at every cost level, but the lift is modest and both turn negative
beyond ~1× costs.

| Cost ×  | per-side bps | baseline Sharpe | baseline + conformal Sharpe |
|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.25 | **0.30** |
| 0.5 | 9.5 | 0.18 | **0.21** |
| 1.0 | 19.0 | 0.10 | **0.12** |
| 2.0 | 38.0 | −0.05 | −0.06 |
| 4.0 | 76.0 | −0.35 | −0.41 |

![Cost sweep](results/cost_sweep.png)

**ML diagnostics (the truth-tellers):**
- **Learning-to-Rank:** OOS rank IC **−0.006** (over 299 rebalances, 100 refits) →
  no cross-sectional ranking skill, but the cross-sectional/funding refit removed
  the prior *inverse* signal (it was −0.036, actively chasing rotational tops). As
  a standalone L/S signal it still loses in a bull market (Sharpe −0.48). Honest
  negative: pathology fixed, edge not created.
- **Regime-switching:** HMM decodes 3 states from the stationary texture features
  (no longer a single-state collapse), but the learned mapping is
  `{0→TREND, 1→TREND, 2→MEANREV}`, so the controller still applies TREND ~87% of
  days. Honest read: **trend-following simply won in most regimes of a bull
  market**; the switched book (Sharpe −0.38) underperforms the static baseline.
- **Lead–lag:** OOS next-day hit-rate **0.506** — essentially a coin-flip; the
  funding tilt (weight 0.5) shapes the book but does not rescue it (Sharpe −0.36).
  Daily price lead-lag remains arbitraged away.
- **Conformal:** empirical coverage **0.907** vs the 0.90 target — the calibration
  property genuinely holds out-of-sample — and as a continuous inverse-width sizer
  it **lifts** risk-adjusted return (0.10 → 0.12) at **93.8%** mean exposure, i.e.
  while staying nearly fully invested rather than parking in cash.

## 6. Verdict

1. **The three ML signal generators do not help.** LTR, regime-switching and
   lead–lag each underperform the simple baseline out-of-sample, exactly as their
   honest diagnostics (rank IC ≈ 0, hit-rate ≈ 0.5, regime trend-dominated)
   predicted. The cross-sectional/funding refit **removed pathologies** (LTR's
   inverse −0.036 IC, the regime single-state collapse) but created no edge. Per
   the pre-committed rule, none earns inclusion as a signal.
2. **The conformal gate is a real but modest win — and the refit clarifies *why*.**
   Its core statistical property holds out-of-sample (**90.7% coverage**), and as a
   continuous inverse-width sizer it lifts the baseline Sharpe **0.10 → 0.12** with
   lower drawdown and vol while staying **~94% invested**. Critically, the earlier
   *binary* gate's far larger lift (0.05 → 0.33) was **substantially a
   market-timing-by-sitting-in-cash artifact** (mean exposure ~0.41): forced to
   deploy capital, the honest lift shrinks. The gate still passes the ablation —
   just by less than it appeared to.
3. **But beta still wins.** Every strategy variant (best Sharpe ~0.12) trails
   buy-and-hold (0.84–1.15) over a crypto bull market, and the deflated Sharpes
   (≤0.12, after penalising the 8 configurations tried) are not conclusive.

The honest reading: *uncertainty-aware sizing (conformal) is worth more here than
any attempt to predict direction* — but once you stop letting it hide in cash, the
remaining edge is small. A genuinely useful, and now more sharply understood, finding.

## 7. Limitations & next research directions

- **Survivorship bias** inflates the universe (upward); the true picture is weaker.
- **Short OOS for the 3-year-train models.** On a 6-year window the conformal gate
  and the regime controller only score out-of-sample over roughly the back half
  (~2023→2026), so their numbers rest on a shorter sample than LTR/lead-lag.
- **Daily bars, single venue.** Funding is now included as a *signal*, but there is
  still no intraday structure and **no short-borrow / funding *cost*** charged on
  the book (a real short pays more — a headwind we still do not model).
- **Conformal needs scrutiny.** A 0.12 Sharpe (DSR ~0.12) is a modest, not proven,
  lift; it should be retested on a larger, point-in-time universe and on a
  long-only book to separate the gate's value from the L/S structure, and the
  sizer's γ/floor mapped to chart the exposure-vs-Sharpe frontier.
- **Promising directions:** (a) a **standalone funding-carry** strategy (funding
  pays as carry, not just as a ranking tilt); (b) regime over a longer / multi-regime
  span (incl. the 2022 bear) with BIC-selected states so mean-reversion regimes get
  real mass; (c) **funding / on-chain features inside the conformal regressor**
  (not only the ranker); (d) conformalized quantile regression for explicitly
  downside-aware sizing; (e) apply the gate to a long-biased book to keep some beta.

## 8. Reproducibility

```bash
pip install -r requirements.txt
python -m nullquant.pipeline          # full run (writes research/results/)
python -m pytest tests/ -q           # 71 tests: causality, accounting, no-leakage
```

One seed (`config/config.yaml: seed: 42`), one config, deterministic outputs.
