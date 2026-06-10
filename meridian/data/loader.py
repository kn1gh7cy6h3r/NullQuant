"""
loader.py — multi-asset data acquisition with point-in-time discipline.

Loads daily OHLCV for the configured crypto universe from Yahoo Finance, caches
each asset to Parquet, and assembles a tidy panel aligned on a common calendar.

Two hard rules that keep the research honest:

  1. NO REPAINTING. The history used for every backtest/feature/label is built
     from *closed* daily bars only. The live spot price is returned separately
     and is for display/monitoring — it never mutates the historical panel.

  2. SURVIVORSHIP BIAS IS DOCUMENTED. Yahoo only serves currently-listed coins,
     so the universe is implicitly conditioned on survival. We do not pretend
     otherwise; results in research/report.md are framed with this caveat. We
     mitigate (not eliminate) it by letting each asset enter the panel only once
     it has real history, rather than back-filling.

The canonical object produced here is a `Panel`: a dict of field -> DataFrame
(index = dates, columns = assets) for Open/High/Low/Close/Volume, plus helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import Config, PROJECT_ROOT

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class Panel:
    """
    A multi-asset OHLCV panel.

    fields: dict field-name -> DataFrame (index=DatetimeIndex, columns=assets).
    All frames share the same index and column order. Missing asset-days (before
    an asset listed) are NaN, never forward-filled across the listing boundary.
    """

    fields: dict[str, pd.DataFrame]
    assets: list[str]

    @property
    def close(self) -> pd.DataFrame:
        return self.fields["Close"]

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.fields["Close"].index

    def field(self, name: str) -> pd.DataFrame:
        return self.fields[name]

    def log_returns(self) -> pd.DataFrame:
        """Daily log returns per asset (NaN where price is missing)."""
        import numpy as np
        c = self.close
        return np.log(c / c.shift(1))


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return cache_dir / f"{safe}.parquet"


def _download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for one ticker, returning a clean OHLCV frame."""
    import yfinance as yf

    raw = yf.download(
        ticker, start=start, end=end, interval="1d",
        progress=False, auto_adjust=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=OHLCV)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    cols = [c for c in OHLCV if c in raw.columns]
    df = raw[cols].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(how="all")


def load_history(cfg: Config, *, force_download: bool = False,
                 end: str | None = None) -> Panel:
    """
    Load (or download + cache) the OHLCV panel for the configured universe.

    Each asset is cached individually; a cache is reused when it already extends
    to within two days of `end`. Assets are aligned on the union calendar; an
    asset is NaN before its first real bar (no back-fill across the listing).
    """
    cache_dir = (PROJECT_ROOT / cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start = cfg.data.start
    end = end or pd.Timestamp.now().normalize().strftime("%Y-%m-%d")

    per_asset: dict[str, pd.DataFrame] = {}
    for ticker in cfg.data.universe:
        path = _cache_path(cache_dir, ticker)
        df: pd.DataFrame | None = None
        if path.exists() and not force_download:
            try:
                cached = pd.read_parquet(path)
                last = pd.Timestamp(cached.index[-1]).normalize()
                if (pd.Timestamp(end).normalize() - last).days <= 2:
                    df = cached
            except Exception:
                df = None
        if df is None:
            df = _download_one(ticker, start, end)
            if not df.empty:
                df.to_parquet(path)
        if not df.empty:
            per_asset[ticker] = df

    if not per_asset:
        raise RuntimeError("No data could be loaded for any asset in the universe.")

    assets = [t for t in cfg.data.universe if t in per_asset]
    calendar = sorted(set().union(*[df.index for df in per_asset.values()]))
    calendar = pd.DatetimeIndex(calendar)

    fields: dict[str, pd.DataFrame] = {}
    for fld in OHLCV:
        cols = {}
        for a in assets:
            s = per_asset[a][fld] if fld in per_asset[a].columns else pd.Series(dtype=float)
            cols[a] = s.reindex(calendar)
        fields[fld] = pd.DataFrame(cols, index=calendar)[assets]

    return Panel(fields=fields, assets=assets)


def fetch_live_prices(cfg: Config) -> dict[str, float]:
    """
    Best-effort live spot prices (CoinGecko) for the universe — DISPLAY ONLY.

    Never feeds the historical panel. Returns {ticker: price}; missing/failed
    lookups are simply absent so callers degrade gracefully.
    """
    import requests

    # Map Yahoo-style tickers to CoinGecko ids.
    cg_ids = {
        "BTC-USD": "bitcoin", "ETH-USD": "ethereum", "BNB-USD": "binancecoin",
        "XRP-USD": "ripple", "ADA-USD": "cardano", "SOL-USD": "solana",
        "DOGE-USD": "dogecoin", "LTC-USD": "litecoin",
    }
    ids = [cg_ids[t] for t in cfg.data.universe if t in cg_ids]
    if not ids:
        return {}
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(ids), "vs_currencies": "usd"},
            timeout=5, headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        out: dict[str, float] = {}
        for ticker, cg in cg_ids.items():
            if ticker in cfg.data.universe and cg in data and "usd" in data[cg]:
                out[ticker] = float(data[cg]["usd"])
        return out
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[Meridian] Live price fetch failed: {exc}")
        return {}
