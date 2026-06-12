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

# Yahoo spot ticker -> Binance USDT-margined perpetual symbol. Only assets that
# list a perp can carry a funding signal; the rest stay NaN (neutral) downstream.
BINANCE_PERP = {
    "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "BNB-USD": "BNBUSDT",
    "XRP-USD": "XRPUSDT", "ADA-USD": "ADAUSDT", "SOL-USD": "SOLUSDT",
    "DOGE-USD": "DOGEUSDT", "LTC-USD": "LTCUSDT",
}
_BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


@dataclass
class Panel:
    """
    A multi-asset OHLCV panel.

    fields: dict field-name -> DataFrame (index=DatetimeIndex, columns=assets).
    All frames share the same index and column order. Missing asset-days (before
    an asset listed) are NaN, never forward-filled across the listing boundary.

    funding: OPTIONAL daily perpetual-swap funding-rate frame, same index/columns
    as the OHLCV fields (NaN where an asset has no perp or the fetch failed). It
    is a structural, exchange-sourced signal — kept separate from OHLCV so every
    consumer must opt in and degrade gracefully when it is absent (``None``).
    """

    fields: dict[str, pd.DataFrame]
    assets: list[str]
    funding: pd.DataFrame | None = None

    @property
    def close(self) -> pd.DataFrame:
        return self.fields["Close"]

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.fields["Close"].index

    def field(self, name: str) -> pd.DataFrame:
        return self.fields[name]

    @property
    def has_funding(self) -> bool:
        """True only if a funding frame exists with at least one real value."""
        return self.funding is not None and bool(self.funding.notna().any().any())

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

    If funding is enabled (``data.funding.enabled``, default True) the panel also
    carries a daily perpetual-swap funding-rate frame aligned to the same
    calendar. Funding is best-effort: any failure leaves it NaN/None and the rest
    of the system runs unchanged.
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

    funding = None
    if bool(cfg.get("data.funding.enabled", True)):
        funding = load_funding_rates(
            cfg, assets=assets, calendar=calendar,
            force_download=force_download, end=end,
        )

    panel = Panel(fields=fields, assets=assets, funding=funding)
    # Trim to the most recent ``data.history_years`` so every entry point
    # (pipeline, dashboard) evaluates the same window and shares the signal cache.
    years = int(cfg.get("data.history_years", 0) or 0)
    return limit_to_recent_years(panel, years)


def limit_to_recent_years(panel: Panel, years: int) -> Panel:
    """Return ``panel`` trimmed to its most recent ``years`` of history (funding
    frame included). A non-positive ``years`` is a no-op. The on-disk cache keeps
    its full span — this only narrows the in-memory evaluation window."""
    if not years or years <= 0:
        return panel
    cutoff = panel.index.max() - pd.DateOffset(years=int(years))
    mask = panel.index >= cutoff
    fields = {k: v.loc[mask] for k, v in panel.fields.items()}
    funding = panel.funding.loc[mask] if panel.funding is not None else None
    return Panel(fields=fields, assets=panel.assets, funding=funding)


# ---------------------------------------------------------------------------
# Perpetual-swap funding rates (structural signal) — best-effort, cached
# ---------------------------------------------------------------------------
def _funding_cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"funding_{symbol}.parquet"


def _download_funding_one(symbol: str, start: str, end: str) -> pd.Series:
    """Download daily funding for one Binance perp symbol as a Series of daily
    summed funding (the day's total carry). Empty Series on any failure.

    Binance returns 8-hourly settlements (``fundingRate``); summing the three
    daily prints gives the daily funding paid by the crowded side. Paginates
    forward by ``startTime`` until ``end``. Network/parse errors degrade to an
    empty Series so the caller can simply skip this asset.
    """
    try:
        import requests
    except Exception:
        return pd.Series(dtype=float)

    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    rows: list[tuple[int, float]] = []
    cursor = start_ms
    # Each call returns <=1000 settlements (~333 days). Cap iterations defensively.
    for _ in range(64):
        try:
            resp = requests.get(
                _BINANCE_FUNDING_URL,
                params={"symbol": symbol, "startTime": cursor,
                        "endTime": end_ms, "limit": 1000},
                timeout=10, headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            batch = resp.json()
        except Exception:
            break
        if not isinstance(batch, list) or not batch:
            break
        for item in batch:
            try:
                rows.append((int(item["fundingTime"]), float(item["fundingRate"])))
            except (KeyError, TypeError, ValueError):
                continue
        last_time = int(batch[-1]["fundingTime"])
        if len(batch) < 1000 or last_time >= end_ms:
            break
        cursor = last_time + 1  # advance past the last settlement to avoid dupes

    if not rows:
        return pd.Series(dtype=float)

    ts = pd.to_datetime([r[0] for r in rows], unit="ms")
    s = pd.Series([r[1] for r in rows], index=ts).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    # Sum intraday settlements to a daily total; normalize the stamp to midnight
    # so it aligns with the daily OHLCV calendar (close-of-day, known at t).
    daily = s.groupby(s.index.normalize()).sum()
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None)
    return daily


def load_funding_rates(cfg: Config, *, assets: list[str] | None = None,
                       calendar: pd.DatetimeIndex | None = None,
                       force_download: bool = False,
                       end: str | None = None) -> pd.DataFrame | None:
    """Daily perpetual funding-rate panel (dates x assets), or None if nothing
    could be loaded. Cached per symbol to Parquet exactly like the OHLCV cache;
    a cache within two days of ``end`` is reused. Assets without a perp mapping,
    or whose fetch fails, are simply absent — the caller treats missing funding
    as a neutral (0) signal, so the whole feature is optional.
    """
    cache_dir = (PROJECT_ROOT / cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    universe = list(assets) if assets is not None else list(cfg.data.universe)
    start = cfg.data.start
    end = end or pd.Timestamp.now().normalize().strftime("%Y-%m-%d")

    cols: dict[str, pd.Series] = {}
    for ticker in universe:
        symbol = BINANCE_PERP.get(ticker)
        if symbol is None:
            continue
        path = _funding_cache_path(cache_dir, symbol)
        s: pd.Series | None = None
        if path.exists() and not force_download:
            try:
                cached = pd.read_parquet(path)
                series = cached.iloc[:, 0]
                last = pd.Timestamp(series.index[-1]).normalize()
                if (pd.Timestamp(end).normalize() - last).days <= 2:
                    s = series
            except Exception:
                s = None
        if s is None:
            s = _download_funding_one(symbol, start, end)
            if not s.empty:
                try:
                    s.to_frame("funding").to_parquet(path)
                except Exception:
                    pass
        if s is not None and not s.empty:
            cols[ticker] = s

    if not cols:
        return None

    funding = pd.DataFrame(cols)
    if calendar is not None:
        # Reindex onto the OHLCV calendar. Do NOT forward-fill across gaps: a day
        # with no settlement stays NaN (neutral), preventing stale carry leaking.
        funding = funding.reindex(calendar)
    funding = funding.reindex(columns=universe)
    return funding


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
        print(f"[NullQuant] Live price fetch failed: {exc}")
        return {}
