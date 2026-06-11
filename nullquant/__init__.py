"""
NullQuant — a rigorously validated, long/short, vol-targeted multi-crypto
trading research system.

The package is organised by responsibility:
    config       configuration loading
    data         multi-asset data acquisition (point-in-time, no repainting)
    features     causal indicators + triple-barrier labeling
    signals      crossover trigger + cross-sectional ranking overlay
    portfolio    cost model + long/short vol-targeted backtest
    metrics      performance stats incl. Deflated/Probabilistic Sharpe
    validation   walk-forward + purged k-fold CV splitters
    ml           reframed models (LSTM returns/vol, RF meta-label, regime IF)

Design rules enforced throughout:
  • Strictly causal: any value used to trade at t is known by t-1's close.
  • Reproducible: a single seed and a single YAML config drive every run.
  • Honest: every model must prove out-of-sample P&L lift or be shelved.
"""

__version__ = "2.0.0"
