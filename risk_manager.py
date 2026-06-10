"""
risk_manager.py — Professional risk-management & position-sizing engine.

This is the heart of what separates a real trading system from a backtest toy.
A signal (Golden/Death Cross) only tells you *direction*. Risk management
answers the questions that actually keep an account alive:

    • How much do I buy?          → position sizing
    • Where do I get out if wrong? → ATR-based stop-loss
    • How do I protect profits?    → ATR trailing stop (Chandelier Exit)
    • When do I stop trading?      → margin-call circuit breaker

Every calculation here is strictly causal: at candle t we only ever use data
that was observable at or before t's close. There is NO look-ahead bias.

────────────────────────────────────────────────────────────────────────────
RISK PARAMETERS  (all in one place so they are easy to audit / tune)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals import INITIAL_CAPITAL  # $100,000 starting capital (single source)

# ── Tunable risk parameters ───────────────────────────────────────────────────

ATR_PERIOD = 14            # Lookback for Average True Range (the volatility unit)
ATR_STOP_MULT = 2.0        # Stop sits this many ATRs away from price

# Fraction of the *current* portfolio risked on a single trade.
#
# Set to the brief's stated "max risk per trade: 5%". (The brief's explicit
# sizing formula used 0.07, which contradicted this 5% cap; we resolve in
# favour of the more conservative 5% rule.) Sizing: a stop-out costs exactly
# this fraction of the *current* portfolio — see the sizing block below.
RISK_PER_TRADE_PCT = 0.05

# Hard floor: if the portfolio ever falls below this, we liquidate and halt.
# A circuit breaker like this is what stops a bad run from becoming a blow-up.
MARGIN_CALL_FLOOR = 15_000.0


# ── Average True Range (ATR) ──────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """
    Append a 14-period ATR column computed on real OHLC data.

    Why ATR and not just "2% of price"? ATR measures *actual* recent
    volatility. A stop placed 2×ATR away adapts automatically: it sits wide in
    turbulent markets (so normal noise doesn't shake you out) and tightens in
    calm markets (so you give back less profit). It is the single most
    important volatility primitive in systematic trading.

    True Range (TR) for each candle is the greatest of:
        • High − Low                  (today's range)
        • |High − PreviousClose|      (gap up from yesterday)
        • |Low  − PreviousClose|      (gap down from yesterday)
    Using the previous close captures overnight gaps that High−Low alone misses.

    ATR is then Wilder's smoothed moving average of TR (an EMA with
    alpha = 1/period) — the original 1978 definition still used industry-wide.

    No look-ahead: TR at t uses High/Low at t and Close at t-1, all known at t.
    """
    df = df.copy()

    # Fall back to Close for any missing OHLC column so the engine never crashes
    # (historical rows always have full OHLC; this only guards the live row).
    high = df["High"] if "High" in df else df["Close"]
    low = df["Low"] if "Low" in df else df["Close"]
    close = df["Close"]
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing == EMA with alpha = 1/period, no bias-correction.
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()

    # Blank out the warm-up window so we never size a trade off an
    # under-smoothed ATR (signals don't fire until 200 candles in anyway).
    atr.iloc[: period - 1] = np.nan

    df["ATR"] = atr
    return df


# ── Full risk-managed backtest ────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict, dict]:
    """
    Walk the price history candle-by-candle and simulate a fully risk-managed
    long-only strategy. Returns:

        df       — same frame plus 'Portfolio', 'Drawdown', 'StopLine' columns
        trades   — list of complete trade records (one dict per trade)
        metrics  — aggregate performance / risk statistics
        state    — the live state at the final candle (for the header & panels)

    ── Strategy rules ──────────────────────────────────────────────────────────
    ENTRY  : on a Golden Cross BUY signal, if currently flat and not halted.
             - Entry price      = that candle's close.
             - Initial stop      = entry − 2×ATR              (volatility stop)
             - Position size     = (portfolio × RISK%) / (entry − stop)
               so that being stopped out costs exactly RISK% of the portfolio.
             - No leverage: if the formula asks for more BTC than our cash can
               buy, we cap at all-in (and real risk ends up below target).

    MANAGE : on every later candle while in the position…
             - First check the stop using the stop level carried in from prior
               candles (we deliberately do NOT let today's high move the stop
               before testing today's low — that would be optimistic / cheating).
                 · gap: if the candle OPENS at/below the stop, exit at the open.
                 · else if the LOW pierces the stop, exit at the stop price.
             - If not stopped, ratchet the trailing stop UP to
               (highest-high-since-entry − 2×ATR). This is the Chandelier Exit:
               it locks in profit as price climbs and never moves down.

    EXIT   : a Death Cross SELL signal closes the position at that close,
             regardless of where the stop is.

    HALT   : if portfolio value drops below MARGIN_CALL_FLOOR, liquidate any
             open position and stop trading permanently (margin-call breaker).
    """
    df = df.copy()

    # Pull raw numpy arrays once — iterating .iloc per row is far slower.
    opens = (df["Open"] if "Open" in df else df["Close"]).to_numpy(dtype=float)
    highs = (df["High"] if "High" in df else df["Close"]).to_numpy(dtype=float)
    lows = (df["Low"] if "Low" in df else df["Close"]).to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    atrs = df["ATR"].to_numpy(dtype=float)
    sigs = df["Signal"].to_numpy(dtype=object)
    dates = df.index

    # ── Portfolio / position state ───────────────────────────────────────────
    cash = INITIAL_CAPITAL     # uninvested USD
    units = 0.0                # BTC currently held
    in_position = False
    halted = False
    margin_call = False

    # Open-trade bookkeeping (valid only while in_position)
    entry_price = 0.0
    entry_date = None
    initial_stop = 0.0         # the stop at entry (entry − 2×ATR)
    stop = 0.0                 # the live trailing stop (ratchets up)
    highest = 0.0              # highest high seen since entry
    entry_dollar_risk = 0.0    # $ that would be lost if the initial stop hit
    entry_risk_pct = 0.0       # that risk as a % of portfolio at entry

    trades: list[dict] = []
    portfolio_series: list[float] = []
    stop_series: list[float] = []

    def _close_trade(exit_idx: int, exit_price: float, reason: str) -> None:
        """Realise the open position into cash and append a finished record."""
        nonlocal cash, units, in_position
        pnl_dollars = units * (exit_price - entry_price)
        pnl_pct = (exit_price / entry_price - 1.0) * 100.0
        cash += units * exit_price          # proceeds returned to cash
        trades.append(dict(
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=dates[exit_idx],
            exit_price=exit_price,
            exit_reason=reason,
            units=units,
            pnl_dollars=pnl_dollars,
            pnl_pct=pnl_pct,
            portfolio_after=cash,            # fully in cash right after the exit
            dollar_risk=entry_dollar_risk,
            risk_pct=entry_risk_pct,
            status="closed",
        ))
        units = 0.0
        in_position = False

    # ── Main causal loop ─────────────────────────────────────────────────────
    for i in range(len(df)):
        price = closes[i]
        o, h, l, a = opens[i], highs[i], lows[i], atrs[i]
        sig = sigs[i]

        # 1) MANAGE an open position — exit checks first, then trail the stop.
        if in_position and not halted:
            if o <= stop:
                # Overnight gap straight through the stop → fill at the open,
                # which is the realistic (worse) price, not the stop level.
                _close_trade(i, o, "Trailing Stop (gap)")
            elif l <= stop:
                # Intraday: price traded down through the stop → fill at stop.
                _close_trade(i, stop, "Trailing Stop")
            else:
                # Survived the candle → raise the trailing stop if price made a
                # new high. Uses the current candle's ATR (Chandelier Exit) and
                # only ever moves the stop UP, never down.
                if h > highest:
                    highest = h
                trailed = highest - ATR_STOP_MULT * a
                if trailed > stop:
                    stop = trailed

        # 2) SIGNAL handling (only if we didn't just get stopped out this candle).
        if in_position and not halted and sig == "SELL":
            # Death Cross overrides the trailing stop — exit at the close.
            _close_trade(i, price, "Death Cross")

        elif (not in_position) and (not halted) and sig == "BUY":
            stop_distance = ATR_STOP_MULT * a
            # Only enter if ATR is valid and gives a sane, positive stop distance.
            if np.isfinite(stop_distance) and stop_distance > 0 and cash > 0:
                portfolio_now = cash            # flat ⇒ portfolio == cash
                entry_price = price
                initial_stop = entry_price - stop_distance

                # Position sizing: risk exactly RISK% of the portfolio.
                risk_dollars = portfolio_now * RISK_PER_TRADE_PCT
                sized_units = risk_dollars / stop_distance
                position_dollars = sized_units * entry_price

                # No-leverage cap: can't deploy more cash than we have.
                if position_dollars > cash:
                    sized_units = cash / entry_price
                    position_dollars = cash

                units = sized_units
                cash -= position_dollars        # leftover stays as cash
                in_position = True

                # Lock in the entry-time risk figures for the trade record.
                entry_dollar_risk = units * stop_distance
                entry_risk_pct = (entry_dollar_risk / portfolio_now) * 100.0
                entry_date = dates[i]
                stop = initial_stop
                highest = entry_price           # initial stop = entry − 2×ATR

        # 3) Mark the portfolio to market at this candle's close.
        portfolio_value = cash + units * price
        portfolio_series.append(portfolio_value)
        stop_series.append(stop if in_position else np.nan)

        # 4) MARGIN-CALL circuit breaker.
        if (not halted) and portfolio_value < MARGIN_CALL_FLOOR:
            margin_call = True
            halted = True
            if in_position:
                _close_trade(i, price, "Margin Call")     # forced liquidation
                portfolio_series[-1] = cash               # now fully in cash
                stop_series[-1] = np.nan

    # ── Attach derived series to the frame ───────────────────────────────────
    df["Portfolio"] = portfolio_series
    running_peak = df["Portfolio"].cummax()
    df["Drawdown"] = (df["Portfolio"] - running_peak) / running_peak * 100.0
    df["StopLine"] = stop_series

    # ── Record the still-open trade (if any) as an unrealised record ──────────
    last_close = float(closes[-1])
    last_date = dates[-1]
    if in_position:
        unreal_pnl = units * (last_close - entry_price)
        trades.append(dict(
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=None,
            exit_price=last_close,            # marked to the latest price
            exit_reason="OPEN",
            units=units,
            pnl_dollars=unreal_pnl,
            pnl_pct=(last_close / entry_price - 1.0) * 100.0,
            portfolio_after=cash + units * last_close,
            dollar_risk=entry_dollar_risk,
            risk_pct=entry_risk_pct,
            status="open",
        ))

    metrics = _compute_metrics(trades, np.asarray(portfolio_series, dtype=float))

    # ── Live state snapshot for the header bar & risk panel ───────────────────
    state = dict(
        in_position=in_position,
        halted=halted,
        margin_call=margin_call,
        atr=float(atrs[-1]) if np.isfinite(atrs[-1]) else None,
        last_close=last_close,
        last_date=last_date,
        # Open-position details (None / 0 when flat)
        entry_price=entry_price if in_position else None,
        entry_date=entry_date if in_position else None,
        units=units if in_position else 0.0,
        position_dollars=(units * last_close) if in_position else 0.0,
        initial_stop=initial_stop if in_position else None,
        trailing_stop=stop if in_position else None,
        unrealized_pnl=(units * (last_close - entry_price)) if in_position else 0.0,
        unrealized_pct=((last_close / entry_price - 1.0) * 100.0) if in_position else 0.0,
    )

    return df, trades, metrics, state


# ── Aggregate performance & risk metrics ──────────────────────────────────────

def _compute_metrics(trades: list[dict], portfolio: np.ndarray) -> dict:
    """
    Summarise the trade history into the numbers a risk desk actually watches.
    Only *closed* trades count toward win/loss statistics; the open trade's
    P&L is unrealised and would distort them.
    """
    closed = [t for t in trades if t["status"] == "closed"]
    n = len(closed)

    wins = [t for t in closed if t["pnl_dollars"] > 0]
    losses = [t for t in closed if t["pnl_dollars"] < 0]

    win_rate = (len(wins) / n * 100.0) if n else 0.0
    avg_win = float(np.mean([t["pnl_dollars"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t["pnl_dollars"] for t in losses])) if losses else 0.0

    # Max consecutive losses — the metric that predicts how much pain you must
    # be able to stomach before the strategy recovers.
    max_consec = run = 0
    for t in closed:
        if t["pnl_dollars"] < 0:
            run += 1
            max_consec = max(max_consec, run)
        else:
            run = 0

    # Profit factor = gross profit / gross loss. >1 means the system makes
    # money; it's a leverage-agnostic quality measure.
    gross_win = sum(t["pnl_dollars"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None

    # Sharpe ratio approximation: mean / std of daily portfolio returns,
    # annualised by √252 trading days. The standard risk-adjusted-return yardstick
    # (here computed gross of the risk-free rate, hence "approximation").
    sharpe = 0.0
    if len(portfolio) > 2:
        rets = np.diff(portfolio) / portfolio[:-1]
        rets = rets[np.isfinite(rets)]
        if rets.size > 1 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252))

    return dict(
        total_trades=n,
        wins=len(wins),
        losses=len(losses),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_consecutive_losses=max_consec,
        profit_factor=profit_factor,
        sharpe=sharpe,
    )
