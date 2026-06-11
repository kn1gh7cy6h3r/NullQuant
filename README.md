# Meridian

![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)

A **long/short, volatility-targeted, multi-crypto** trading research system —
built to test a cross-sectional trend hypothesis *honestly*, with the
methodology a quant research desk would expect: realistic costs, strictly
out-of-sample validation, and an intellectual-honesty bar that every model must
clear or be shelved.

> **New here / non-finance background?** Read
> **[`GUIDE.md`](GUIDE.md)** first — it explains the entire project, and every
> term in it, in plain English with no assumed knowledge.

> **Headline finding (negative, and that's the point):** a naive cross-sectional
> SMA-crossover long/short book does **not** beat buy-and-hold on a risk-adjusted
> basis over 2019–2026, and none of the three ML overlays adds out-of-sample
> value once costs and proper validation are applied. See
> [`research/report.md`](research/report.md). The deliverable here is the
> rigorous infrastructure and the honest read — not a manufactured edge.

## Why this is built the way it is

Most retail "backtests" are wrong in the same few ways. Meridian is designed to
avoid each:

| Common failure | What Meridian does |
|---|---|
| No transaction costs | Fee + spread + slippage on turnover, with a **0×–4× sensitivity sweep** |
| In-sample / overfit | **Walk-forward** + **purged k-fold CV with embargo**; **Deflated Sharpe** |
| Repainting (live price mutates history) | Backtests use **closed bars only**; live price is display-only |
| Survivorship bias hidden | **Declared and discussed**; assets enter only once they have real history |
| ML as decoration | Every model must prove **out-of-sample P&L lift** in an ablation, or it's cut |
| Irreproducible | One **YAML config**, one **seed**, deterministic pipeline, **CI** |

## Architecture

```
meridian/
  config.py            YAML config loader            seeds.py   deterministic seeding
  data/loader.py       multi-asset OHLCV, point-in-time, no repainting
  features/
    indicators.py      causal SMA/ATR/RSI/vol/momentum (vectorized)
    labeling.py        triple-barrier meta-labels (+ event end-times for purging)
  signals/base.py      SMA crossover trigger + cross-sectional long/short overlay
  portfolio/
    costs.py           fee + spread + slippage cost model
    backtest.py        long/short, vol-targeted, cost-aware backtest + benchmarks
  metrics/performance.py   Sharpe/Sortino/Calmar + Probabilistic & Deflated Sharpe
  validation/
    splitters.py       walk-forward + purged k-fold (embargo)
    walk_forward.py    out-of-sample strategy evaluation
  ml/
    lstm_forecast.py   LSTM on stationary log-returns, vs random-walk baseline
    rf_meta.py         Random Forest meta-labeling, purged-CV OOS AUC
    regime_iforest.py  Isolation Forest regime filter (walk-forward, no leakage)
  ablation.py          baseline vs +regime vs +meta vs both, + cost sweep
  pipeline.py          reproducible end-to-end run -> research/results/
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m meridian.pipeline        # full research run (writes research/results/)
python -m meridian.pipeline --no-lstm   # skip the slow LSTM step
python -m pytest tests/ -q         # 43 tests: causality, accounting, no-leakage
```

Light/CI dependencies (no torch/keras/UI) are in `requirements-dev.txt`; CI runs
the foundation test suite on every push.

## Configuration

Everything lives in [`config/config.yaml`](config/config.yaml): universe,
SMA/ATR/momentum params, vol target, cost assumptions + sweep, validation
windows, and per-model ML settings. Change the config, re-run the pipeline,
reproduce exactly.

## What I'd claim in an interview

- I can build a **causal, cost-aware, multi-asset backtester** and prove its
  accounting identities and no-look-ahead properties with tests.
- I know **why** plain k-fold leaks in finance and implement **purged CV with
  embargo** and **Deflated Sharpe** to defend against overfitting.
- I reframed three ML models from in-sample decoration into **honestly evaluated
  components**, and — finding they don't add OOS value — I **report that** rather
  than hiding it. That's the judgment the work is meant to demonstrate.

See [`research/report.md`](research/report.md) for the full methodology, results,
limitations, and next directions.
