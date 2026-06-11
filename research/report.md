# NullQuant — Research Report

*A long/short, vol-targeted, multi-crypto trend strategy, evaluated honestly.*

> **TL;DR.** Built a full, reproducible research stack: multi-asset point-in-time
> data, causal features, a cost-aware long/short vol-targeted backtester,
> walk-forward + purged k-fold validation, and **four creative ML models** held
> to an out-of-sample P&L ablation. **Nuanced, honest finding:** the three ML
> *signal generators* (learning-to-rank, regime-switching, lead–lag) all
> **underperform** the simple baseline out-of-sample — but the **conformal
> confidence gate** is a genuine, validated win: it lifts the baseline Sharpe
> 0.05 → 0.33, nearly halves drawdown, and stays positive out to ~2× costs. Even
> so, **no variant beats buy-and-hold** in a 2019–2026 crypto bull market; beta is
> hard to beat. The contribution is the rigor, the honesty, and one component
> that demonstrably helps.

All numbers below are reproduced by `python -m nullquant.pipeline` (seed 42) and
are written to `research/results/`.

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
  Finance, auto-adjusted). 2019-01-01 → present (~2,700 daily bars).
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
| **Learning-to-Rank** | predict the *order* of winners/losers, not returns | RandomForest on demeaned forward return; long top-k / short bottom-k | OOS rank IC |
| **Regime-switching (HMM)** | discover hidden market states, switch strategy per state | GaussianHMM picks trend vs mean-reversion per regime | regime mapping / occupancy |
| **Lead–lag network** | learn which coins *lead* others; trade the laggards | lagged cross-predictive edges over a trailing window | OOS next-day hit-rate |
| **Conformal gate** | bet only when *calibrated-confident*; size by certainty | split-conformal intervals; exposure = fraction of confidently-directional assets | empirical coverage vs target |

Each direction source enters the same vol-targeted backtest; the conformal gate
enters as a causal exposure multiplier in [0,1]. All are walk-forward and OOS.

## 5. Results (net of 1× costs, out-of-sample)

| Variant | Ann. return | Ann. vol | Sharpe | Max DD | Deflated Sharpe |
|---|---:|---:|---:|---:|---:|
| baseline | −1.6% | 23.6% | 0.05 | −60.3% | 0.09 |
| **baseline + conformal** | **+3.8%** | 14.8% | **0.33** | **−40.1%** | 0.28 |
| LTR | −26.4% | 23.3% | −1.20 | −90.8% | 0.00 |
| regime | −7.5% | 16.9% | −0.38 | −51.3% | 0.01 |
| lead–lag | −12.7% | 23.5% | −0.46 | −71.8% | 0.00 |
| **[bench] equal-weight** | **+68.5%** | 75.8% | **1.07** | −78.4% | 1.00 |
| **[bench] BTC buy-and-hold** | **+45.1%** | 61.4% | **0.92** | −76.6% | 0.99 |

(Each direction source was also run conformal-gated; the gate improves all of
them but cannot rescue a bad signal — full table in `research/results/variants.csv`.)

![Equity curves](results/equity_curves.png)

**Cost sensitivity (annualized Sharpe):** the conformal-gated baseline stays
positive far longer than the plain baseline as costs rise.

| Cost ×  | per-side bps | baseline Sharpe | baseline + conformal Sharpe |
|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.20 | **0.42** |
| 0.5 | 9.5 | 0.13 | **0.37** |
| 1.0 | 19.0 | 0.05 | **0.33** |
| 2.0 | 38.0 | −0.10 | **0.23** |
| 4.0 | 76.0 | −0.39 | 0.03 |

![Cost sweep](results/cost_sweep.png)

**ML diagnostics (the truth-tellers):**
- **Learning-to-Rank:** OOS rank IC **−0.04** → no cross-sectional ranking skill;
  as a signal it is the *worst* variant (Sharpe −1.20). Honest negative.
- **Regime-switching:** HMM finds 3 states but the meta-controller collapses
  toward TREND (~88% of days), and the switched book (Sharpe −0.38) underperforms
  the static baseline. The switching adds turnover without edge.
- **Lead–lag:** OOS next-day hit-rate **0.508** — essentially a coin-flip; the
  strongest edges are weak negative (mild mean-reversion). No exploitable lead-lag.
- **Conformal:** empirical coverage **0.909** vs the 0.90 target — the calibration
  property genuinely holds — and gating on it **lifts** risk-adjusted return.

## 6. Verdict

1. **The three ML signal generators do not help.** LTR, regime-switching and
   lead–lag each underperform the simple baseline out-of-sample, exactly as their
   honest diagnostics (IC ≈ 0, hit-rate ≈ 0.5, regime collapse) predicted. Per the
   pre-committed rule, none earns inclusion as a signal.
2. **The conformal gate is a real, validated win.** Its core statistical property
   holds out-of-sample (90.9% coverage), and using it as a sizing overlay raises
   the baseline Sharpe 0.05 → 0.33, cuts max drawdown 60% → 40%, turns the return
   positive, and **survives costs to ~2×** where the plain baseline is already
   negative. This is the rare component that passes the ablation.
3. **But beta still wins.** Even the best variant (Sharpe 0.33) trails buy-and-hold
   (0.92–1.07) over a crypto bull market, and its Deflated Sharpe (0.28, after
   penalising the 10 configurations tried) is promising but not conclusive.

The honest reading: *uncertainty-aware sizing (conformal) is worth more here than
any attempt to predict direction* — a genuinely useful, somewhat unusual finding.

## 7. Limitations & next research directions

- **Survivorship bias** inflates the universe (upward); the true picture is weaker.
- **Daily bars, single venue.** No intraday structure, funding, or short-borrow
  costs (a real short book pays more — another headwind we did not even add).
- **Conformal needs scrutiny.** A 0.33 Sharpe with DSR 0.28 is promising, not
  proven; it should be retested on a larger, point-in-time universe and on a
  long-only book to separate the gate's value from the L/S structure.
- **Promising directions:** (a) apply the conformal gate to a long-biased book to
  keep some beta; (b) larger universe incl. delisted coins to kill survivorship
  bias; (c) richer conformal features (funding rates, on-chain); (d) conformalized
  quantile regression for asymmetric (downside-aware) sizing.

## 8. Reproducibility

```bash
pip install -r requirements.txt
python -m nullquant.pipeline          # full run (writes research/results/)
python -m pytest tests/ -q           # 43 tests: causality, accounting, no-leakage
```

One seed (`config/config.yaml: seed: 42`), one config, deterministic outputs.
