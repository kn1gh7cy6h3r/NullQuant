"""
data_manager.py — Data acquisition and caching layer for Meridian.

In a real trading system, reliable data is the foundation of everything.
This module handles two jobs:
  1. Historical daily OHLCV data via yfinance, cached locally as Parquet
     so we never redundantly hammer an API on every restart.
  2. Live spot price via CoinGecko's free tier, polled every 30 seconds
     to keep the most recent candle fresh.
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
CACHE_FILE = DATA_DIR / "btc_history.parquet"
TICKER = "BTC-USD"
HISTORY_YEARS = 6

# Full OHLC is required because the risk engine computes ATR (Average True
# Range) from real High/Low/Close — closing prices alone cannot capture
# intraday range or overnight gaps.
OHLC_COLS = ["Open", "High", "Low", "Close"]

# CoinGecko free tier: ~10-30 calls/min — our 30s poll is well within limits.
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def _cache_is_valid(df: pd.DataFrame) -> bool:
    """
    A cache is usable only if it:
      1. Contains the full OHLC columns (invalidates any old close-only cache).
      2. Is recent — the last row is ≤1 day old (data hasn't gone stale).
      3. Covers the full configured HISTORY_YEARS span — a cache built with
         an older, shorter period is silently re-downloaded so ML models and
         the equity curve always have the full depth of history.
    """
    if df.empty or not all(c in df.columns for c in OHLC_COLS):
        return False
    last_date = pd.Timestamp(df.index[-1]).normalize()
    first_date = pd.Timestamp(df.index[0]).normalize()
    today = pd.Timestamp.now().normalize()
    if (today - last_date).days > 1:
        return False
    required_start = today - pd.Timedelta(days=HISTORY_YEARS * 365 + 10)
    return first_date <= required_start


# ── Public API ────────────────────────────────────────────────────────────────

def load_or_download_history() -> pd.DataFrame:
    """
    Load 6 years of BTC-USD daily OHLC bars from the local Parquet cache.
    Re-downloads from yfinance when the cache is missing, stale (>1 day old),
    lacks the OHLC columns the risk engine needs, or doesn't span the full
    configured HISTORY_YEARS period.

    Parquet is used instead of CSV because it preserves the DatetimeIndex dtype
    across save/load cycles without extra parsing, and is ~10x faster to read.
    """
    _ensure_data_dir()

    if CACHE_FILE.exists():
        try:
            cached = pd.read_parquet(CACHE_FILE)
            if _cache_is_valid(cached):
                return cached
        except Exception:
            pass  # Corrupt/old cache — fall through and re-download.

    # Download 3 years of daily bars (+10 buffer days to ensure full coverage)
    start = datetime.now() - timedelta(days=HISTORY_YEARS * 365 + 10)
    end = datetime.now()

    print(f"[Meridian] Downloading {HISTORY_YEARS}y of BTC-USD OHLC history from Yahoo Finance…")
    raw = yf.download(
        TICKER,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=True,
    )

    # yfinance ≥0.2 can return a MultiIndex column level with the ticker name;
    # flatten it so we always have plain column names like "Close".
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    # Keep the full OHLC set (ATR needs High/Low/Close, not just Close).
    df = raw[OHLC_COLS].copy()

    # Normalise to timezone-naive UTC midnight so downstream code never has to
    # deal with mixed-timezone comparisons.
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna()

    df.to_parquet(CACHE_FILE)
    print(f"[Meridian] Cached {len(df)} rows of OHLC to {CACHE_FILE}")
    return df


def fetch_live_price() -> float | None:
    """
    Fetch the current BTC/USD spot price from CoinGecko.
    Returns None on any network or parsing error so callers can degrade
    gracefully rather than crashing the dashboard.
    """
    try:
        resp = requests.get(
            COINGECKO_URL,
            params={"ids": "bitcoin", "vs_currencies": "usd"},
            timeout=5,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        return float(resp.json()["bitcoin"]["usd"])
    except Exception as exc:
        print(f"[Meridian] Live price fetch failed: {exc}")
        return None


def get_full_dataset() -> tuple[pd.DataFrame, float | None]:
    """
    Return (df, live_price) where df is the full historical dataset with
    today's row updated to the current spot price if the API is reachable.

    Injecting the live price into today's candle ensures the SMAs and
    equity curve always reflect the most recent market state.
    """
    df = load_or_download_history()
    live_price = fetch_live_price()

    if live_price is not None:
        today = pd.Timestamp.now().normalize()
        df = df.copy()
        if today in df.index:
            # Update today's forming candle: move the Close to the live spot
            # price and stretch the High/Low to keep the OHLC range coherent.
            df.loc[today, "Close"] = live_price
            df.loc[today, "High"] = max(df.loc[today, "High"], live_price)
            df.loc[today, "Low"] = min(df.loc[today, "Low"], live_price)
        else:
            # No bar for today yet — seed a fresh candle. With a single spot
            # observation, O=H=L=C; its intraday range (and thus ATR impact)
            # fills in as more candles arrive on subsequent days.
            new_row = pd.DataFrame(
                {"Open": [live_price], "High": [live_price],
                 "Low": [live_price], "Close": [live_price]},
                index=[today],
            )
            df = pd.concat([df, new_row])

    return df, live_price
