# NullQuant

![CI](https://github.com/kn1gh7cy6h3r/NullQuant/actions/workflows/ci.yml/badge.svg)

A **long/short, volatility-targeted, multi-crypto** trading research system —
built to test a cross-sectional trend hypothesis *honestly*, with the
methodology a quant research desk would expect: realistic costs, strictly
out-of-sample validation, and an intellectual-honesty bar that every model must
clear or be shelved.

> **New here / non-finance background?** Read
> **[`GUIDE.md`](GUIDE.md)** first — it explains the entire project, and every
> term in it, in plain English with no assumed knowledge.

> **Headline finding (nuanced, and that's the point):** four creative ML models
> were built and held to an out-of-sample P&L ablation (last **6 years**,
> 2020–2026). The three *signal generators* (learning-to-rank, HMM
> regime-switching, lead–lag network) all **underperform** a simple baseline —
> the cross-sectional feature + funding refit removed the worst pathologies
> (LTR rank IC **−0.036 → −0.006**) but did not manufacture alpha. The
> **conformal confidence gate** is the one validated win, now a *continuous
> inverse-width sizer*: it holds **90.7%** empirical coverage and still lifts the
> baseline Sharpe (**0.10 → 0.12**, with lower drawdown and vol) while deploying
> **~94%** mean exposure instead of hiding in cash. The honest reveal: the old
> binary gate's much larger lift was **substantially a sit-in-cash market-timing
> artifact**. Even so, **no variant beats buy-and-hold** in a crypto bull market.
> See [`research/report.md`](research/report.md). The deliverable is rigorous
> infrastructure, an honest read, and one component that demonstrably helps — not
> a manufactured edge.

## Why this is built the way it is

Most retail "backtests" are wrong in the same few ways. NullQuant is designed to
avoid each:

| Common failure | What NullQuant does |
|---|---|
| No transaction costs | Fee + spread + slippage on turnover, with a **0×–4× sensitivity sweep** |
| In-sample / overfit | **Walk-forward** + **purged k-fold CV with embargo**; **Deflated Sharpe** |
| Repainting (live price mutates history) | Backtests use **closed bars only**; live price is display-only |
| Survivorship bias hidden | **Declared and discussed**; assets enter only once they have real history |
| ML as decoration | Every model must prove **out-of-sample P&L lift** in an ablation, or it's cut |
| Irreproducible | One **YAML config**, one **seed**, deterministic pipeline, **CI** |

## Architecture

```
nullquant/
  config.py            YAML config loader            seeds.py   deterministic seeding
  data/loader.py       multi-asset OHLCV + perp funding rates (Binance, cached),
                       point-in-time, no repainting, trimmed to data.history_years
  features/
    indicators.py      causal SMA/ATR/RSI/vol/momentum + cross-sectional MAD
                       z-score, ATR-normalized RS, funding rank signal (vectorized)
    labeling.py        triple-barrier meta-labels (+ event end-times for purging)
  signals/base.py      SMA crossover trigger + cross-sectional long/short overlay
  portfolio/
    costs.py           fee + spread + slippage cost model
    backtest.py        long/short, vol-targeted, cost-aware backtest + benchmarks
                       (conformal exposure scales the vol-target allocation)
  metrics/performance.py   Sharpe/Sortino/Calmar + Probabilistic & Deflated Sharpe
  validation/
    splitters.py       walk-forward + purged k-fold (embargo)
    walk_forward.py    out-of-sample strategy evaluation
  ml/
    rank_model.py      learning-to-rank selector — CS-normalized features + funding (OOS rank IC)
    regime_switch.py   HMM meta-controller on stationary market-texture features
    lead_lag.py        lead-lag contagion network + structural funding tilt (OOS hit-rate)
    conformal.py       continuous inverse-width conformal sizer (validated ~90% coverage)
  ablation.py          baseline vs LTR/regime/lead-lag, each × conformal, + cost sweep
  pipeline.py          reproducible end-to-end run -> research/results/
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
./run.sh               # full research pipeline (data → ML → results → artifacts)
./run.sh dashboard     # launch interactive research dashboard at http://localhost:8050
./run.sh tests         # 71 tests: causality, accounting identities, no-leakage
```

**The pipeline fits all four ML models walk-forward on first run (~5 min) and
caches the result.** The dashboard reads from that cache on every subsequent
launch — startup is under 1 second. Force a full retrain with:

```bash
./run.sh pipeline --refresh-signals
```

Light/CI dependencies (no UI or ML) are in `requirements-dev.txt`; CI runs the
foundation test suite on every push.

## Dashboard

`./run.sh dashboard` opens an interactive research UI with six panels:

- **Overview** — headline metrics, honest verdict, key stats
- **Equity** — strategy variants vs benchmarks (log scale, TradingView-style controls)
- **Positions** — current target weights + weight history heatmap
- **Signals** — per-asset trend state and latest signal direction
- **Costs** — Sharpe vs cost-multiplier robustness curve
- **ML Intel** — per-model diagnostics (rank IC, regime occupancy, lead-lag hit-rate + funding tilt, conformal coverage & mean exposure)

Set `NULLQUANT_DEBUG=1` to enable hot-reload when editing `dashboard.py`.

## Configuration

Everything lives in [`config/config.yaml`](config/config.yaml): universe,
`history_years` (evaluation window), `data.funding.enabled` (perp funding fetch),
SMA/ATR/momentum params, vol target, cost assumptions + sweep, validation
windows, and per-model ML settings (incl. the conformal `mode`/`gamma`/`floor`
sizer knobs and the lead-lag `funding_weight`). Change the config, re-run the
pipeline, reproduce exactly.

See [`research/report.md`](research/report.md) for the full methodology, results,
limitations, and next directions.
