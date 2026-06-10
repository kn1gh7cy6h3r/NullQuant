"""
signals.py — Technical indicator engine for Meridian.

This module implements the core quant logic:
  • 50-day and 200-day Simple Moving Averages
  • Golden Cross / Death Cross crossover detection
  • Historical equity curve simulation ($100k starting capital)
  • Drawdown calculation (how far the portfolio falls from its peak)

These are among the most widely used tools in systematic macro and trend-
following strategies. The SMA crossover is a momentum signal — it tells you
whether short-term price behaviour is trending above or below the long-term
baseline.
"""

import numpy as np
import pandas as pd

INITIAL_CAPITAL = 100_000.0  # Starting portfolio value in USD
SMA_SHORT = 50               # Days for the fast moving average
SMA_LONG = 200               # Days for the slow moving average


# ── Indicators ────────────────────────────────────────────────────────────────

def add_smas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append SMA50 and SMA200 columns to the dataframe.

    A Simple Moving Average smooths out daily noise and reveals the underlying
    trend direction. 50/200 is the canonical pair used by institutional traders
    to spot medium- and long-term trend regimes.
    """
    df = df.copy()
    df["SMA50"] = df["Close"].rolling(window=SMA_SHORT, min_periods=SMA_SHORT).mean()
    df["SMA200"] = df["Close"].rolling(window=SMA_LONG, min_periods=SMA_LONG).mean()
    return df


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Golden Cross (BUY) and Death Cross (SELL) events.

    Golden Cross: SMA50 crosses ABOVE SMA200 → bullish regime change, go long.
    Death Cross:  SMA50 crosses BELOW SMA200 → bearish regime change, go to cash.

    We compare each day's relative SMA position to the previous day's to catch
    the exact candle where the crossover occurred — this avoids look-ahead bias.
    """
    df = df.copy()
    df["Signal"] = None  # Will hold "BUY", "SELL", or None

    valid = df["SMA50"].notna() & df["SMA200"].notna()
    sma50 = df.loc[valid, "SMA50"]
    sma200 = df.loc[valid, "SMA200"]

    prev_sma50 = sma50.shift(1)
    prev_sma200 = sma200.shift(1)

    # Golden Cross: SMA50 was below SMA200 yesterday, is above today
    golden = (prev_sma50 < prev_sma200) & (sma50 > sma200)
    # Death Cross: SMA50 was above SMA200 yesterday, is below today
    death = (prev_sma50 > prev_sma200) & (sma50 < sma200)

    df.loc[golden[golden].index, "Signal"] = "BUY"
    df.loc[death[death].index, "Signal"] = "SELL"

    return df


def add_equity_and_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    LEGACY (kept for reference / comparison only — no longer used by the
    dashboard, which now uses the risk-managed equity curve in risk_manager.py).

    Simulate following every BUY/SELL signal with $100,000 starting capital,
    naively all-in / all-out with no position sizing or stops.

    Rules:
      • Start in cash (no position).
      • On BUY: deploy all cash into BTC at that day's close price.
      • On SELL: liquidate all BTC back to cash at that day's close price.
      • Portfolio value at any day = cash + (shares held × current close).

    Drawdown measures how far the portfolio has fallen from its running peak.
    It exposes the pain a trader would have had to endure — a critical metric
    for evaluating whether a strategy is psychologically survivable.

    drawdown(t) = (portfolio(t) − max(portfolio[0..t])) / max(portfolio[0..t]) × 100
    """
    df = df.copy()

    cash = INITIAL_CAPITAL
    shares = 0.0
    in_position = False
    portfolio_values: list[float] = []

    for _, row in df.iterrows():
        price = row["Close"]
        signal = row["Signal"]

        if signal == "BUY" and not in_position and cash > 0:
            shares = cash / price
            cash = 0.0
            in_position = True
        elif signal == "SELL" and in_position:
            cash = shares * price
            shares = 0.0
            in_position = False

        portfolio_values.append(cash + shares * price)

    df["Portfolio"] = portfolio_values

    # Drawdown: negative percentage below the running peak
    running_peak = df["Portfolio"].cummax()
    df["Drawdown"] = (df["Portfolio"] - running_peak) / running_peak * 100

    return df


# ── Convenience wrappers ──────────────────────────────────────────────────────

def process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the indicator pipeline: SMAs + crossover signals.

    The equity curve and drawdown are NOT computed here anymore — they are
    produced by the risk-managed backtest in risk_manager.run_backtest(), which
    accounts for ATR stops and position sizing. The dashboard calls add_atr()
    and run_backtest() after this to attach the 'Portfolio'/'Drawdown' columns.
    """
    df = add_smas(df)
    df = add_signals(df)
    return df


def current_signal(df: pd.DataFrame) -> str:
    """
    Return the active regime — 'BUY', 'SELL', or 'NEUTRAL' — based on
    the relative position of the two SMAs on the most recent row.
    """
    valid = df.dropna(subset=["SMA50", "SMA200"])
    if valid.empty:
        return "NEUTRAL"
    last = valid.iloc[-1]
    if last["SMA50"] > last["SMA200"]:
        return "BUY"
    if last["SMA50"] < last["SMA200"]:
        return "SELL"
    return "NEUTRAL"


def max_drawdown(df: pd.DataFrame) -> float:
    """Return the worst (most negative) drawdown percentage in the history."""
    if "Drawdown" not in df.columns or df["Drawdown"].isna().all():
        return 0.0
    return float(df["Drawdown"].min())


def total_return(df: pd.DataFrame) -> float:
    """Return the total percentage return of the equity curve."""
    if "Portfolio" not in df.columns or df["Portfolio"].isna().all():
        return 0.0
    first = df["Portfolio"].dropna().iloc[0]
    last = df["Portfolio"].dropna().iloc[-1]
    return (last - first) / first * 100
