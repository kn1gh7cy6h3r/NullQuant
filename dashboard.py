"""
dashboard.py — Visual layer for Meridian.

Linear.app-inspired UI: flat near-black surfaces, frosted-glass cards, a fixed
left sidebar, an always-visible header, and one content panel at a time. No
neon, no glow, no shadows, no pulse animations.

Architecture
------------
  • Header bar (48px, full width) — wordmark, live BTC/USD price, signal pill
    and inline Portfolio / Total Return / Max Drawdown stats.
  • Fixed 200px sidebar — Lucide-icon navigation, never scrolls.
  • Main content — every panel lives in the DOM; the sidebar shows exactly one
    at a time (client-side, see assets/meridian.js). Default: Price & SMAs.

The data pipeline (data_manager -> signals -> risk_manager -> ml_engine) and the
single interval-driven callback are preserved exactly; only presentation changed.
Custom CSS lives in assets/meridian.css.
"""

from __future__ import annotations

import traceback

import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

from data_manager import get_full_dataset
from signals import process, current_signal, max_drawdown, total_return
from risk_manager import (
    add_atr, run_backtest, MARGIN_CALL_FLOOR, RISK_PER_TRADE_PCT,
    ATR_PERIOD, ATR_STOP_MULT,
)
# ML intelligence layer (Models A/B/C). Imported defensively: if the ML stack
# can't load for any reason the rest of the dashboard must keep working, so we
# fall back to reporting every model as unavailable (ml stays None).
try:
    import ml_engine
    from ml_engine import MLResults, RF_WEAK_THRESHOLD
    ML_IMPORT_OK = True
except Exception as _ml_exc:  # pragma: no cover - defensive
    ML_IMPORT_OK = False
    _ML_IMPORT_ERR = str(_ml_exc)

# ── Palette (Linear.app inspired — no neon, no glow) ──────────────────────────

BG       = "#0a0a0a"   # Page / chart background
SURFACE  = "#0f0f0f"   # Sidebar / header
TEXT     = "#ededed"   # Primary text
TEXT2    = "#737373"   # Secondary text
TEXT3    = "#404040"   # Tertiary text
ACCENT   = "#ffffff"   # Accent
POS      = "#22c55e"   # Positive / BUY
NEG      = "#ef4444"   # Negative / SELL
WARN     = "#f59e0b"   # Warning
BORDER   = "rgba(255,255,255,0.06)"
GRID     = "rgba(255,255,255,0.04)"

REFRESH_MS = 30_000    # 30-second live refresh
APP_VERSION = "v2.0"
LSTM_HORIZON_DAYS = 7
ANOMALY_WINDOW_DAYS = 30


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb hex string to an rgba(…) CSS string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Shared chart layout ───────────────────────────────────────────────────────

_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family="'Inter','system-ui',sans-serif", size=11),
    xaxis=dict(
        gridcolor=GRID, showgrid=True, zeroline=False,
        tickfont=dict(color=TEXT2, size=10),
        linecolor=BORDER,
        showspikes=True, spikecolor=TEXT2, spikethickness=1,
        spikedash="solid", spikemode="across", spikesnap="cursor",
    ),
    yaxis=dict(
        gridcolor=GRID, showgrid=True, zeroline=False,
        tickfont=dict(color=TEXT2, size=10),
        linecolor=BORDER,
        showspikes=True, spikecolor=TEXT2, spikethickness=1,
        spikedash="solid", spikemode="across",
    ),
    margin=dict(l=60, r=20, t=20, b=28),
    showlegend=False,
    hovermode="x",
    hoverlabel=dict(
        bgcolor="#111111",
        bordercolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=11, family="'Inter','system-ui',sans-serif"),
    ),
    transition=dict(duration=300, easing="cubic-in-out"),
    dragmode="pan",
)


def _L(**overrides) -> dict:
    """Merge _BASE with per-chart overrides (nested dicts shallow-merged)."""
    out = dict(_BASE)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


_GRAPH_CONFIG = dict(
    displayModeBar=False,
    scrollZoom=True,
    displaylogo=False,
    doubleClick="reset",
)


# ── Regime-band helper ────────────────────────────────────────────────────────

def _add_regime_bands(fig: go.Figure, df: pd.DataFrame) -> None:
    """Shade translucent vertical bands per signal regime (green BUY, red SELL)."""
    signal_rows = df[df["Signal"].notna()]
    if signal_rows.empty:
        return
    signals = list(zip(signal_rows.index, signal_rows["Signal"]))
    for i, (start, sig) in enumerate(signals):
        end = signals[i + 1][0] if i + 1 < len(signals) else df.index[-1]
        fill = _rgba(POS, 0.06) if sig == "BUY" else _rgba(NEG, 0.06)
        fig.add_vrect(x0=start, x1=end, fillcolor=fill, line_width=0, layer="below")


# ── Panel: Price & SMAs ───────────────────────────────────────────────────────

def _price_chart(df: pd.DataFrame, ml: "MLResults | None" = None) -> go.Figure:
    """
    Candlestick price with SMA50/SMA200 overlays, ATR trailing stop, optional
    LSTM forecast + confidence band, BUY/SELL markers, anomaly bands, and a
    floating info pill (annotation) in the top-left corner. Edge to edge.
    """
    fig = go.Figure()

    # Model C — barely-visible anomaly bands, drawn first so they sit beneath.
    if ml is not None and ml.anomaly.ready and ml.anomaly.anomaly_dates:
        for d in ml.anomaly.anomaly_dates:
            fig.add_vrect(
                x0=d - pd.Timedelta(hours=12), x1=d + pd.Timedelta(hours=12),
                fillcolor=_rgba(NEG, 0.04), line_width=0, layer="below",
            )

    # Candlesticks — green up, red down, grey #737373 wicks & body outlines.
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="BTC/USD",
        increasing=dict(line=dict(color=TEXT2, width=1), fillcolor=POS),
        decreasing=dict(line=dict(color=TEXT2, width=1), fillcolor=NEG),
        whiskerwidth=0.4,
        hoverinfo="x+y",
    ))

    # SMA50 — white at 60% opacity, thin. SMA200 — grey, thin dashed.
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA50"], mode="lines", name="SMA50",
        line=dict(color=_rgba(ACCENT, 0.6), width=1),
        hovertemplate="SMA50 %{y:$,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA200"], mode="lines", name="SMA200",
        line=dict(color=TEXT2, width=1, dash="dash"),
        hovertemplate="SMA200 %{y:$,.0f}<extra></extra>",
    ))

    # Trailing stop — amber, dotted, only where a position was open.
    if "StopLine" in df.columns and df["StopLine"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["StopLine"], mode="lines", name="Trailing stop",
            line=dict(color=WARN, width=1.5, dash="dot"),
            connectgaps=False,
            hovertemplate="stop %{y:$,.0f}<extra></extra>",
        ))

    # BUY markers below the candle, SELL markers above.
    buys = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Low"] * 0.985, mode="markers", name="BUY",
            marker=dict(color=POS, size=9, symbol="triangle-up"),
            hovertemplate="BUY<extra></extra>",
        ))
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["High"] * 1.015, mode="markers", name="SELL",
            marker=dict(color=NEG, size=9, symbol="triangle-down"),
            hovertemplate="SELL<extra></extra>",
        ))

    # Model A — 7-day LSTM forecast (grey dotted extension) + confidence band.
    if ml is not None and ml.lstm.ready and ml.lstm.forecast_prices:
        last_date = df.index[-1]
        last_close = float(df["Close"].iloc[-1])
        fx = [last_date] + list(ml.lstm.forecast_dates)
        fy = [last_close] + list(ml.lstm.forecast_prices)
        up = [last_close] + list(ml.lstm.band_upper)
        lo = [last_close] + list(ml.lstm.band_lower)
        fig.add_trace(go.Scatter(
            x=fx, y=up, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fx, y=lo, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_rgba(ACCENT, 0.03),
            name="LSTM band", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="lines", name="LSTM forecast",
            line=dict(color=TEXT2, width=1.5, dash="dot"),
            hovertemplate="forecast %{y:$,.0f}<extra></extra>",
        ))

    # Floating info pill — current price + SMA values, top-left corner.
    fig.add_annotation(
        xref="paper", yref="paper", x=0.0, y=1.0,
        xanchor="left", yanchor="top",
        text=_price_overlay_text(df),
        showarrow=False, align="left",
        font=dict(color=TEXT, size=11, family="'Inter','system-ui',sans-serif"),
        bgcolor=_rgba(ACCENT, 0.03), bordercolor=BORDER, borderwidth=1,
        borderpad=8,
    )

    fig.update_layout(**_L(
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06, bgcolor=SURFACE),
            showspikes=True, spikecolor=TEXT2, spikethickness=1,
            spikedash="solid", spikemode="across", spikesnap="cursor",
        ),
        margin=dict(l=60, r=20, t=20, b=10),
    ))
    return fig


def _price_overlay_text(df: pd.DataFrame) -> str:
    """Build the multi-line text for the price chart's floating info pill."""
    last = df.iloc[-1]
    def fmt(v):
        return f"${v:,.0f}" if pd.notna(v) else "—"
    return (
        f"<b>BTC/USD</b>  {fmt(last['Close'])}<br>"
        f"SMA50 {fmt(last['SMA50'])}   SMA200 {fmt(last['SMA200'])}"
    )


# ── Panel: Signals ────────────────────────────────────────────────────────────

def _signal_chart(df: pd.DataFrame, ml: "MLResults | None" = None) -> go.Figure:
    """
    Regime bands (green = long, red = cash) over a dim price line, with clean
    BUY/SELL markers and STRONG/WEAK strength labels from the Random Forest.
    """
    fig = go.Figure()
    _add_regime_bands(fig, df)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines", name="BTC/USD",
        line=dict(color=_rgba(TEXT, 0.20), width=1),
        hovertemplate="%{y:$,.0f}<extra></extra>",
    ))

    buys = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Close"], mode="markers", name="BUY",
            marker=dict(color=POS, size=10, symbol="triangle-up"),
            hovertemplate="BUY %{y:$,.0f}<extra></extra>",
        ))
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["Close"], mode="markers", name="SELL",
            marker=dict(color=NEG, size=10, symbol="triangle-down"),
            hovertemplate="SELL %{y:$,.0f}<extra></extra>",
        ))

    # Model B — STRONG/WEAK confidence labels above each historical signal.
    if ml is not None and ml.rf.ready and ml.rf.signal_confidences:
        strong_x, strong_y, strong_t = [], [], []
        weak_x, weak_y, weak_t = [], [], []
        for d, info in ml.rf.signal_confidences.items():
            if d not in df.index:
                continue
            y = df.loc[d, "Close"]
            label = f"{info['strength']} {info['confidence']:.0f}%"
            if info["strength"] == "STRONG":
                strong_x.append(d); strong_y.append(y); strong_t.append(label)
            else:
                weak_x.append(d); weak_y.append(y); weak_t.append(label)
        if strong_x:
            fig.add_trace(go.Scatter(
                x=strong_x, y=strong_y, mode="text", text=strong_t,
                textposition="top center", textfont=dict(color=POS, size=9),
                hoverinfo="skip", showlegend=False,
            ))
        if weak_x:
            fig.add_trace(go.Scatter(
                x=weak_x, y=weak_y, mode="text", text=weak_t,
                textposition="top center", textfont=dict(color=WARN, size=9),
                hoverinfo="skip", showlegend=False,
            ))

    fig.update_layout(**_L(
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06, bgcolor=SURFACE),
            showspikes=True, spikecolor=TEXT2, spikethickness=1,
            spikedash="solid", spikemode="across", spikesnap="cursor",
        ),
        margin=dict(l=60, r=20, t=20, b=10),
    ))
    return fig


# ── Panel: Drawdown ───────────────────────────────────────────────────────────

def _drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """Area drawdown curve (red), worst point annotated. Edge to edge."""
    fig = go.Figure()
    dd = df["Drawdown"].dropna()
    if not dd.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Drawdown"], mode="lines", name="Drawdown",
            line=dict(color=NEG, width=1.5),
            fill="tozeroy", fillcolor=_rgba(NEG, 0.08),
            hovertemplate="%{y:.2f}%<extra></extra>",
        ))
        fig.add_hline(y=0, line=dict(color=BORDER, width=1))
        worst_idx, worst_val = dd.idxmin(), dd.min()
        fig.add_annotation(
            x=worst_idx, y=worst_val, text=f"Worst {worst_val:.1f}%",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            arrowcolor=NEG, ax=46, ay=-26,
            font=dict(color=NEG, size=10),
            bgcolor=_rgba(BG, 0.9), bordercolor=BORDER, borderwidth=1, borderpad=5,
        )
    fig.update_layout(**_L(yaxis=dict(ticksuffix="%")))
    return fig


# ── Panel: Equity ─────────────────────────────────────────────────────────────

def _equity_chart(df: pd.DataFrame, trades: list[dict] | None = None) -> go.Figure:
    """Area equity curve (green), $100k baseline, trade entry/exit markers."""
    fig = go.Figure()
    eq = df["Portfolio"].dropna()
    if not eq.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Portfolio"], mode="lines", name="Portfolio",
            line=dict(color=POS, width=2),
            fill="tozeroy", fillcolor=_rgba(POS, 0.06),
            hovertemplate="$%{y:,.0f}<extra></extra>",
        ))
        fig.add_hline(
            y=100_000, line=dict(color=_rgba(TEXT, 0.18), width=1, dash="dash"),
            annotation_text="$100k", annotation_position="top left",
            annotation_font=dict(color=TEXT2, size=10),
        )
        if trades:
            entry_x, entry_y, exit_x, exit_y = [], [], [], []
            for t in trades:
                ed = t["entry_date"]
                if ed in df.index:
                    entry_x.append(ed); entry_y.append(df.loc[ed, "Portfolio"])
                xd = t["exit_date"]
                if xd is not None and xd in df.index:
                    exit_x.append(xd); exit_y.append(df.loc[xd, "Portfolio"])
            if entry_x:
                fig.add_trace(go.Scatter(
                    x=entry_x, y=entry_y, mode="markers", name="Entry",
                    marker=dict(color=POS, size=8, symbol="triangle-up"),
                    hovertemplate="entry $%{y:,.0f}<extra></extra>",
                ))
            if exit_x:
                fig.add_trace(go.Scatter(
                    x=exit_x, y=exit_y, mode="markers", name="Exit",
                    marker=dict(color=NEG, size=8, symbol="triangle-down"),
                    hovertemplate="exit $%{y:,.0f}<extra></extra>",
                ))
    fig.update_layout(**_L(yaxis=dict(tickprefix="$", tickformat=",.0f")))
    return fig


# ── Empty / error placeholder ─────────────────────────────────────────────────

def _empty_figure(message: str = "Loading…") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(color=TEXT2, size=14),
    )
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


# ── Panel: Trade Log ──────────────────────────────────────────────────────────

# Static column definitions for the trade-log DataTable. All columns always
# shown (no toggle). The hidden numeric `pnl_num` drives row colouring.
TRADE_LOG_COLUMNS = [
    {"name": "Entry Date",  "id": "entry_date"},
    {"name": "Entry",       "id": "entry_price"},
    {"name": "Exit Date",   "id": "exit_date"},
    {"name": "Exit",        "id": "exit_price"},
    {"name": "Exit Reason", "id": "reason"},
    {"name": "BTC Units",   "id": "units"},
    {"name": "Risk $",      "id": "risk"},
    {"name": "P&L $",       "id": "pnl_dollars"},
    {"name": "P&L %",       "id": "pnl_pct"},
    {"name": "Portfolio",   "id": "portfolio_after"},
    {"name": "pnl_num",     "id": "pnl_num"},  # hidden — drives row colouring
]


def _trade_log_data(trades: list[dict]) -> list[dict]:
    """Turn raw trade dicts into display-ready rows, newest first."""
    rows = []
    for t in reversed(trades):
        is_open = t["status"] == "open"
        rows.append({
            "entry_date": t["entry_date"].strftime("%Y-%m-%d"),
            "entry_price": f"${t['entry_price']:,.0f}",
            "exit_date": "open" if is_open else t["exit_date"].strftime("%Y-%m-%d"),
            "exit_price": f"${t['exit_price']:,.0f}",
            "reason": t["exit_reason"],
            "units": f"{t['units']:.4f}",
            "risk": f"${t['dollar_risk']:,.0f} ({t['risk_pct']:.1f}%)",
            "pnl_dollars": f"${t['pnl_dollars']:+,.0f}" + (" *" if is_open else ""),
            "pnl_pct": f"{t['pnl_pct']:+.1f}%",
            "portfolio_after": f"${t['portfolio_after']:,.0f}",
            "pnl_num": round(t["pnl_dollars"], 2),
        })
    return rows


# ── Panel: Risk Metrics ───────────────────────────────────────────────────────

def _metric_card(label: str, value: str, color: str = TEXT, sub: str = "") -> html.Div:
    """A single compact metric tile."""
    return html.Div(className="metric", children=[
        html.Div(label, className="lbl"),
        html.Div(value, className="val", style=dict(color=color)),
        html.Div(sub, className="sub") if sub else None,
    ])


def _risk_metrics_panel(metrics: dict, state: dict) -> html.Div:
    """Live position/volatility tiles on top, realised performance below."""
    atr = state.get("atr")
    in_pos = state.get("in_position")

    atr_str = f"${atr:,.0f}" if atr else "—"
    if in_pos:
        init_stop_str = f"${state['initial_stop']:,.0f}"
        trail_stop_str = f"${state['trailing_stop']:,.0f}"
        pos_btc_str = f"{state['units']:.4f} BTC"
        pos_usd_str = f"${state['position_dollars']:,.0f}"
        upnl = state["unrealized_pnl"]
        upnl_str = f"${upnl:+,.0f} ({state['unrealized_pct']:+.1f}%)"
        stop_color, pos_color = NEG, TEXT
        upnl_color = POS if upnl >= 0 else NEG
    else:
        init_stop_str = trail_stop_str = pos_btc_str = pos_usd_str = "—"
        upnl_str = "flat"
        stop_color = pos_color = upnl_color = TEXT2

    wr = metrics["win_rate"]
    pf = metrics["profit_factor"]
    pf_str = f"{pf:.2f}" if pf is not None else "∞"
    sharpe = metrics["sharpe"]

    live_tiles = [
        _metric_card(f"ATR ({ATR_PERIOD})", atr_str, WARN, "volatility unit"),
        _metric_card("Initial Stop", init_stop_str, stop_color,
                     f"entry − {ATR_STOP_MULT:g}×ATR"),
        _metric_card("Trailing Stop", trail_stop_str, stop_color, "ratchets up only"),
        _metric_card("Position (BTC)", pos_btc_str, pos_color),
        _metric_card("Position ($)", pos_usd_str, pos_color),
        _metric_card("Unrealised P&L", upnl_str, upnl_color),
    ]
    perf_tiles = [
        _metric_card("Win Rate", f"{wr:.0f}%", POS if wr >= 50 else NEG,
                     f"{metrics['wins']}W / {metrics['losses']}L"),
        _metric_card("Avg Win", f"${metrics['avg_win']:,.0f}", POS),
        _metric_card("Avg Loss", f"${metrics['avg_loss']:,.0f}", NEG),
        _metric_card("Max Consec. Loss", f"{metrics['max_consecutive_losses']}", NEG,
                     "worst losing streak"),
        _metric_card("Profit Factor", pf_str, POS if (pf or 0) >= 1 else NEG,
                     "gross win / loss"),
        _metric_card("Sharpe (approx)", f"{sharpe:.2f}", POS if sharpe >= 0 else NEG,
                     "risk-adj. return"),
        _metric_card("Total Trades", f"{metrics['total_trades']}", TEXT, "closed"),
        _metric_card("Risk / Trade", f"{RISK_PER_TRADE_PCT * 100:.0f}%", WARN,
                     "target sizing"),
    ]
    return html.Div(children=[
        html.Div("Live Risk", className="metric-section"),
        html.Div(className="metric-grid", children=live_tiles),
        html.Div("Performance", className="metric-section"),
        html.Div(className="metric-grid", children=perf_tiles),
    ])


# ── Banners (circuit breakers) ────────────────────────────────────────────────

def _margin_banner(state: dict) -> tuple:
    """(children, style) for the margin-call banner. Hidden unless tripped."""
    hidden = dict(display="none")
    if not state.get("margin_call"):
        return "", hidden
    msg = (f"MARGIN CALL · PORTFOLIO BREACHED ${MARGIN_CALL_FLOOR:,.0f} FLOOR · "
           f"ALL POSITIONS LIQUIDATED · TRADING HALTED")
    return msg, dict(display="block")


def _anomaly_banner(ml: "MLResults | None") -> tuple:
    """(children, style) for the anomaly-alert banner. Hidden unless tripped."""
    hidden = dict(display="none")
    if ml is None or not ml.anomaly.ready or not ml.anomaly.today_anomalous:
        return "", hidden
    msg = ("ANOMALY ALERT · TODAY'S CANDLE IS A STATISTICAL OUTLIER · "
           "ISOLATION-FOREST CIRCUIT BREAKER ACTIVE · BUY SIGNALS DOWNGRADED TO CAUTION")
    return msg, dict(display="block")


# ── Panel: ML Intel ───────────────────────────────────────────────────────────

def _confidence_gauge(conf: float, strength: str) -> go.Figure:
    """Clean arc gauge for the Random Forest live signal confidence (0–100%)."""
    num_color = POS if strength == "STRONG" else WARN
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf,
        number=dict(suffix="%", font=dict(color=num_color, size=30)),
        gauge=dict(
            shape="angular",
            axis=dict(range=[0, 100], tickcolor=TEXT3,
                      tickfont=dict(color=TEXT3, size=8)),
            bar=dict(color=num_color, thickness=0.30),
            bgcolor=_rgba(ACCENT, 0.03),
            borderwidth=0,
            steps=[
                dict(range=[0, RF_WEAK_THRESHOLD * 100], color=_rgba(ACCENT, 0.03)),
                dict(range=[RF_WEAK_THRESHOLD * 100, 100], color=_rgba(ACCENT, 0.06)),
            ],
            threshold=dict(line=dict(color=TEXT2, width=1),
                           thickness=0.85, value=RF_WEAK_THRESHOLD * 100),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=10, b=0), height=150,
        font=dict(color=TEXT, family="'Inter','system-ui',sans-serif"),
    )
    return fig


def _ml_card(title: str, subtitle: str, body: list) -> html.Div:
    return html.Div(className="mlcard", children=[
        html.Div(title, className="title"),
        html.Div(subtitle, className="subtitle"),
        html.Div(body),
    ])


def _ml_panel(ml: "MLResults | None") -> html.Div:
    """Three equal cards (LSTM / RF / Anomaly) + a not-financial-advice label."""
    if ml is None:
        return html.Div("ML engine unavailable.",
                        style=dict(color=TEXT2, padding="8px"))

    # Card A — LSTM Price Forecast.
    lstm = ml.lstm
    if lstm.ready:
        dir_color = {"UP": POS, "DOWN": NEG}.get(lstm.direction, TEXT2)
        arrow = {"UP": "↑", "DOWN": "↓", "FLAT": "→"}.get(lstm.direction, "")
        lstm_body = [
            html.Div([
                html.Span(f"{arrow} {lstm.direction}",
                          style=dict(color=dir_color, fontWeight="600", fontSize="24px")),
                html.Span(f"  {lstm.pct_change:+.1f}% / {LSTM_HORIZON_DAYS}d",
                          style=dict(color=dir_color, fontSize="13px")),
            ]),
            html.Div(f"{LSTM_HORIZON_DAYS}-day target  ${lstm.target_price:,.0f}",
                     style=dict(color=TEXT, fontSize="13px", marginTop="10px")),
            html.Div("± confidence band shown on the price chart",
                     style=dict(color=TEXT2, fontSize="11px", marginTop="6px")),
        ]
        if lstm.training:
            lstm_body.append(html.Div("LSTM training… (showing previous forecast)",
                                      style=dict(color=WARN, fontSize="11px",
                                                 marginTop="10px")))
    elif lstm.training:
        lstm_body = [html.Div("LSTM training…", style=dict(color=WARN, fontSize="13px")),
                     html.Div("First model fit in progress.",
                              style=dict(color=TEXT2, fontSize="11px", marginTop="6px"))]
    else:
        lstm_body = [html.Div(lstm.status, style=dict(color=TEXT2, fontSize="12px"))]
    card_a = _ml_card("LSTM Price Forecast",
                      f"2-layer LSTM · 60→{LSTM_HORIZON_DAYS}d · model output, not advice",
                      lstm_body)

    # Card B — Random Forest Signal Confidence.
    rf = ml.rf
    if rf.ready and rf.current_confidence is not None:
        rf_body = [
            dcc.Graph(figure=_confidence_gauge(rf.current_confidence, rf.current_strength),
                      config=dict(displayModeBar=False), style=dict(height="150px")),
            html.Div(f"{rf.current_strength} · {rf.current_signal or '—'} signal",
                     style=dict(color=POS if rf.current_strength == "STRONG" else WARN,
                                fontSize="12px", fontWeight="600", textAlign="center")),
            html.Div(f"trained on {rf.n_trades} realised trades",
                     style=dict(color=TEXT2, fontSize="11px", textAlign="center",
                                marginTop="4px")),
        ]
    else:
        rf_body = [html.Div(rf.status, style=dict(color=TEXT2, fontSize="12px",
                                                  padding="28px 0", textAlign="center"))]
    card_b = _ml_card("Signal Confidence",
                      "Random Forest on realised trades · not advice", rf_body)

    # Card C — Isolation Forest Anomaly Detection.
    anom = ml.anomaly
    if anom.ready:
        status_txt, status_color = (("ANOMALOUS", NEG) if anom.today_anomalous
                                    else ("NORMAL", POS))
        anom_body = [
            html.Div(status_txt, style=dict(color=status_color, fontWeight="600",
                                            fontSize="24px")),
            html.Div("today's candle", style=dict(color=TEXT2, fontSize="11px",
                                                  marginBottom="10px")),
            html.Div(f"{anom.recent_count} anomalies in last {ANOMALY_WINDOW_DAYS}d",
                     style=dict(color=TEXT, fontSize="13px")),
            html.Div("acts as a BUY → CAUTION circuit breaker",
                     style=dict(color=TEXT2, fontSize="11px", marginTop="6px")),
        ]
    else:
        anom_body = [html.Div(anom.status, style=dict(color=TEXT2, fontSize="12px"))]
    card_c = _ml_card("Anomaly Detection",
                      "Isolation Forest · circuit breaker · not advice", anom_body)

    return html.Div(children=[
        html.Div(className="ml-grid", children=[card_a, card_b, card_c]),
        html.Div("Model outputs — not financial advice", className="ml-disclaimer"),
    ])


# ── Panel: Overview ───────────────────────────────────────────────────────────

def _overview_stat(label: str, value: str, color: str = TEXT, sub: str = "") -> html.Div:
    return html.Div(className="statcard", children=[
        html.Div(label, className="lbl"),
        html.Div(value, className="val tnum", style=dict(color=color)),
        html.Div(sub, className="sub") if sub else None,
    ])


def _overview_panel(df: pd.DataFrame, live_price, latest_portfolio: float,
                    ret: float, mdd: float, ml: "MLResults | None") -> html.Div:
    """4 stat cards + a 2-column grid: last signal (left) / ML status (right)."""
    price_str = f"${live_price:,.0f}" if live_price else "—"

    stats = html.Div(className="stat-row", children=[
        _overview_stat("Live Price", price_str, TEXT),
        _overview_stat("Portfolio Value", f"${latest_portfolio:,.0f}", TEXT),
        _overview_stat("Total Return", f"{ret:+.1f}%", POS if ret >= 0 else NEG),
        _overview_stat("Max Drawdown", f"{mdd:.1f}%", NEG),
    ])

    # Last signal card.
    sig_rows = df[df["Signal"].notna()]
    if not sig_rows.empty:
        last_d = sig_rows.index[-1]
        last_sig = sig_rows["Signal"].iloc[-1]
        sig_color = POS if last_sig == "BUY" else NEG
        strength_kv = None
        if ml is not None and ml.rf.ready and last_d in ml.rf.signal_confidences:
            info = ml.rf.signal_confidences[last_d]
            strength_kv = html.Div(className="kv", children=[
                html.Span("Strength", className="k"),
                html.Span(f"{info['strength']} · {info['confidence']:.0f}%",
                          className="v",
                          style=dict(color=POS if info["strength"] == "STRONG" else WARN)),
            ])
        last_signal_card = html.Div(className="card", style=dict(padding="20px"), children=[
            html.Div("Last Signal", className="panel-title"),
            html.Div(className="kv", children=[
                html.Span("Type", className="k"),
                html.Span(last_sig, className="v", style=dict(color=sig_color)),
            ]),
            html.Div(className="kv", children=[
                html.Span("Date", className="k"),
                html.Span(last_d.strftime("%Y-%m-%d"), className="v"),
            ]),
            html.Div(className="kv", children=[
                html.Span("Price", className="k"),
                html.Span(f"${df.loc[last_d, 'Close']:,.0f}", className="v"),
            ]),
            strength_kv,
        ])
    else:
        last_signal_card = html.Div(className="card", style=dict(padding="20px"), children=[
            html.Div("Last Signal", className="panel-title"),
            html.Div("No crossover signals yet.", style=dict(color=TEXT2)),
        ])

    # ML status card.
    if ml is None:
        ml_rows = [html.Div("ML engine unavailable.", style=dict(color=TEXT2))]
    else:
        lstm_v = (f"{ml.lstm.direction} {ml.lstm.pct_change:+.1f}%"
                  if ml.lstm.ready else ml.lstm.status)
        rf_v = (f"{ml.rf.current_strength} · {ml.rf.current_confidence:.0f}%"
                if ml.rf.ready and ml.rf.current_confidence is not None else ml.rf.status)
        anom_v = (("ANOMALOUS" if ml.anomaly.today_anomalous else "NORMAL")
                  if ml.anomaly.ready else ml.anomaly.status)
        ml_rows = [
            html.Div(className="kv", children=[
                html.Span("LSTM Forecast", className="k"),
                html.Span(lstm_v, className="v")]),
            html.Div(className="kv", children=[
                html.Span("Signal Confidence", className="k"),
                html.Span(rf_v, className="v")]),
            html.Div(className="kv", children=[
                html.Span("Anomaly Status", className="k"),
                html.Span(anom_v, className="v",
                          style=dict(color=NEG if (ml.anomaly.ready and
                                     ml.anomaly.today_anomalous) else TEXT))]),
        ]
    ml_status_card = html.Div(className="card", style=dict(padding="20px"), children=[
        html.Div("ML Status", className="panel-title"), *ml_rows,
    ])

    return html.Div(children=[
        stats,
        html.Div(className="two-col", children=[last_signal_card, ml_status_card]),
    ])


# ── Header builder ────────────────────────────────────────────────────────────

def _hdr_stat(label: str, value: str, color: str = TEXT) -> html.Div:
    return html.Div(className="hdr-stat", children=[
        html.Div(label, className="lbl"),
        html.Div(value, className="val tnum", style=dict(color=color)),
    ])


def _build_header(live_price, signal: str, badge_text: str, badge_class: str,
                  latest_portfolio: float, ret: float, mdd: float) -> html.Div:
    """Center: BTC/USD live price. Right: signal pill + inline stats."""
    price_display = f"${live_price:,.0f}" if live_price else "—"
    return html.Div(children=[
        html.Div(className="hdr-spacer"),
        html.Div(className="hdr-center", children=[
            html.Span("BTC/USD", className="hdr-pair-label"),
            html.Span(price_display, className="hdr-price tnum"),
        ]),
        html.Div(className="hdr-stats", children=[
            html.Span(badge_text, className=f"pill {badge_class}"),
            _hdr_stat("Portfolio", f"${latest_portfolio:,.0f}", TEXT),
            _hdr_stat("Total Return", f"{ret:+.1f}%", POS if ret >= 0 else NEG),
            _hdr_stat("Max Drawdown", f"{mdd:.1f}%", NEG),
        ]),
    ])


# ── Dash application ──────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="Meridian · BTC Dashboard",
    update_title=None,
    external_scripts=[
        {"src": "https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"},
    ],
)
server = app.server


def _nav_item(panel: str, icon: str, label: str, active: bool = False) -> html.Div:
    """One sidebar navigation row: a Lucide icon + label, tagged with data-panel."""
    cls = "nav-item active" if active else "nav-item"
    return html.Div(
        className=cls,
        **{"data-panel": panel},
        children=[
            html.I(**{"data-lucide": icon}),
            html.Span(label),
        ],
    )


def _panel(panel: str, extra_class: str, children, active: bool = False) -> html.Div:
    """A content panel; only the active one is displayed (see meridian.js)."""
    cls = f"panel {extra_class}" + (" active" if active else "")
    return html.Div(className=cls, **{"data-panel": panel}, children=children)


def _chart_panel(panel: str, graph_id: str, active: bool = False) -> html.Div:
    """A full-bleed chart panel."""
    return _panel(panel, "", html.Div(className="chart-fill", children=[
        dcc.Graph(id=graph_id, className="chart-fill",
                  style=dict(height="100%", width="100%"),
                  config=_GRAPH_CONFIG),
    ]), active=active)


app.layout = html.Div(className="app", children=[

    # ── Header (always visible, full width) ───────────────────────────────────
    html.Div(className="header", children=[
        html.Div(className="hdr-left", children=[
            html.Div("MERIDIAN", className="wordmark"),
        ]),
        html.Div(id="header-stats", className="hdr-right"),
    ]),

    # ── Circuit-breaker banners (hidden until tripped) ────────────────────────
    html.Div(id="margin-banner", className="banner banner-neg",
             style=dict(display="none")),
    html.Div(id="anomaly-banner", className="banner banner-warn",
             style=dict(display="none")),

    # ── Body: sidebar + main content ──────────────────────────────────────────
    html.Div(className="body-row", children=[

        # Sidebar (fixed, never scrolls). The wordmark lives only in the header.
        html.Div(className="sidebar", children=[
            html.Div(className="nav", children=[
                _nav_item("price",    "activity",          "Price & SMAs", active=True),
                _nav_item("signals",  "bar-chart-2",       "Signals"),
                _nav_item("drawdown", "trending-down",     "Drawdown"),
                _nav_item("equity",   "dollar-sign",       "Equity"),
                _nav_item("trades",   "clipboard-list",    "Trade Log"),
                _nav_item("ml",       "brain",             "ML Intel"),
                _nav_item("overview", "layout-dashboard",  "Overview"),
            ]),
            html.Div(APP_VERSION, className="sidebar-footer"),
        ]),

        # Main content — every panel in the DOM; one shown at a time.
        html.Div(className="main", children=[

            _chart_panel("price", "price-chart", active=True),
            _chart_panel("signals", "signal-chart"),
            _chart_panel("drawdown", "drawdown-chart"),
            _chart_panel("equity", "equity-chart"),

            # Trade Log — table (60%) + risk metrics (40%).
            _panel("trades", "split", [
                html.Div(className="col-trades", children=[
                    html.Div("Trade Log", className="panel-title"),
                    dash_table.DataTable(
                        id="trade-log",
                        columns=TRADE_LOG_COLUMNS,
                        data=[],
                        hidden_columns=["pnl_num"],
                        page_action="none",
                        sort_action="native",
                        style_as_list_view=True,
                        style_table=dict(overflowX="auto"),
                        style_header=dict(
                            backgroundColor="transparent", color=TEXT2,
                            fontWeight="600", fontSize="11px",
                            letterSpacing="0.08em", border="none",
                            borderBottom=f"1px solid {BORDER}",
                            textTransform="uppercase",
                            fontFamily="'Inter',system-ui,sans-serif",
                            padding="10px 12px",
                        ),
                        style_cell=dict(
                            backgroundColor="transparent", color=TEXT,
                            fontFamily="'Inter',system-ui,sans-serif",
                            fontSize="12px", border="none",
                            padding="9px 12px", textAlign="right",
                            whiteSpace="nowrap",
                        ),
                        style_cell_conditional=[
                            {"if": {"column_id": c}, "textAlign": "left"}
                            for c in ("entry_date", "exit_date", "reason")
                        ],
                        style_data=dict(borderBottom=f"1px solid {BORDER}"),
                        style_data_conditional=[
                            {"if": {"row_index": "odd"},
                             "backgroundColor": "rgba(255,255,255,0.02)"},
                            {"if": {"filter_query": "{pnl_num} > 0",
                                    "column_id": "pnl_dollars"},
                             "color": POS, "fontWeight": "600"},
                            {"if": {"filter_query": "{pnl_num} > 0",
                                    "column_id": "pnl_pct"},
                             "color": POS, "fontWeight": "600"},
                            {"if": {"filter_query": "{pnl_num} > 0",
                                    "column_id": "entry_date"},
                             "borderLeft": f"2px solid {POS}"},
                            {"if": {"filter_query": "{pnl_num} < 0",
                                    "column_id": "pnl_dollars"},
                             "color": NEG, "fontWeight": "600"},
                            {"if": {"filter_query": "{pnl_num} < 0",
                                    "column_id": "pnl_pct"},
                             "color": NEG, "fontWeight": "600"},
                            {"if": {"filter_query": "{pnl_num} < 0",
                                    "column_id": "entry_date"},
                             "borderLeft": f"2px solid {NEG}"},
                            {"if": {"filter_query": '{reason} = "OPEN"'},
                             "backgroundColor": "rgba(255,255,255,0.03)"},
                            {"if": {"filter_query": '{reason} contains "Stop"',
                                    "column_id": "reason"}, "color": WARN},
                            {"if": {"filter_query": '{reason} = "Margin Call"',
                                    "column_id": "reason"},
                             "color": NEG, "fontWeight": "600"},
                        ],
                    ),
                ]),
                html.Div(className="col-risk", children=[
                    html.Div("Risk Metrics", className="panel-title"),
                    html.Div(id="risk-metrics"),
                ]),
            ]),

            # ML Intel.
            _panel("ml", "scroll pad", [
                html.Div("ML Intel", className="panel-title"),
                html.Div(id="ml-panel"),
            ]),

            # Overview.
            _panel("overview", "scroll pad", [
                html.Div("Overview", className="panel-title"),
                html.Div(id="overview-content"),
            ]),
        ]),
    ]),

    # ── Footer ────────────────────────────────────────────────────────────────
    html.Div(id="footer", className="footer"),

    # ── Hidden loading sentinel (drives dcc.Loading on first fetch) ───────────
    dcc.Loading(id="page-loading", type="circle", color=ACCENT,
                children=html.Div(id="loading-trigger", style=dict(display="none"))),

    dcc.Interval(id="refresh-interval", interval=REFRESH_MS, n_intervals=0),
])


# ── Single data callback (fires on load and every 30s) ────────────────────────

@app.callback(
    Output("price-chart",     "figure"),
    Output("signal-chart",    "figure"),
    Output("drawdown-chart",  "figure"),
    Output("equity-chart",    "figure"),
    Output("trade-log",       "data"),
    Output("risk-metrics",    "children"),
    Output("margin-banner",   "children"),
    Output("margin-banner",   "style"),
    Output("anomaly-banner",  "children"),
    Output("anomaly-banner",  "style"),
    Output("ml-panel",        "children"),
    Output("overview-content","children"),
    Output("header-stats",    "children"),
    Output("footer",          "children"),
    Output("loading-trigger", "children"),
    Input("refresh-interval", "n_intervals"),
)
def refresh_dashboard(n_intervals: int):
    """
    Master callback: fires on page load (n=0) and every 30 seconds thereafter.
    Fetches fresh data, runs the indicator pipeline AND the risk-managed
    backtest, then returns every panel in one round-trip.
    """
    loading_msg = "Fetching data…" if n_intervals == 0 else ""

    try:
        # ── Data → indicators → ATR → risk-managed backtest ──────────────────
        df_raw, live_price = get_full_dataset()
        df = process(df_raw)                          # SMAs + crossover signals
        df = add_atr(df)                              # 14-period ATR on real OHLC
        df, trades, metrics, state = run_backtest(df) # sizing, stops, P&L

        # ── ML intelligence layer (Models A/B/C) ─────────────────────────────
        signal = current_signal(df)
        ml = None
        if ML_IMPORT_OK:
            try:
                ml = ml_engine.update(df, trades, signal)
            except Exception:
                traceback.print_exc()
                ml = None

        # ── Charts ───────────────────────────────────────────────────────────
        price_fig    = _price_chart(df, ml)
        signal_fig   = _signal_chart(df, ml)
        drawdown_fig = _drawdown_chart(df)
        equity_fig   = _equity_chart(df, trades)

        # ── Panels ─────────────────────────────────────────────────────────--
        trade_data   = _trade_log_data(trades)
        risk_panel   = _risk_metrics_panel(metrics, state)
        banner_msg, banner_style = _margin_banner(state)
        anom_msg, anom_style = _anomaly_banner(ml)
        ml_panel     = _ml_panel(ml)

        # ── Header figures ─────────────────────────────────────────────────--
        mdd = max_drawdown(df)
        ret = total_return(df)
        latest_portfolio = (
            df["Portfolio"].dropna().iloc[-1]
            if not df["Portfolio"].isna().all() else 100_000
        )

        # Circuit breaker: an anomalous candle downgrades a live BUY to CAUTION.
        anomaly_today = bool(ml and ml.anomaly.ready and ml.anomaly.today_anomalous)
        if state["halted"]:
            badge_text, badge_class = "HALTED", "pill-sell"
        elif anomaly_today and signal == "BUY":
            badge_text, badge_class = "CAUTION", "pill-warn"
        else:
            badge_text = signal
            badge_class = {"BUY": "pill-buy", "SELL": "pill-sell"}.get(signal, "pill-neutral")

        header = _build_header(live_price, signal, badge_text, badge_class,
                               latest_portfolio, ret, mdd)
        overview = _overview_panel(df, live_price, latest_portfolio, ret, mdd, ml)

        # ── Footer ───────────────────────────────────────────────────────────
        buy_count  = int((df["Signal"] == "BUY").sum())
        sell_count = int((df["Signal"] == "SELL").sum())
        last_date  = df.index[-1].strftime("%Y-%m-%d") if not df.empty else "—"
        footer = (
            f"MERIDIAN · data through {last_date} · "
            f"{buy_count} golden crosses · {sell_count} death crosses · "
            f"{metrics['total_trades']} closed trades · live refresh every 30s"
        )

        return (price_fig, signal_fig, drawdown_fig, equity_fig,
                trade_data, risk_panel, banner_msg, banner_style,
                anom_msg, anom_style, ml_panel, overview,
                header, footer, loading_msg)

    except Exception:
        traceback.print_exc()
        empty  = _empty_figure("Data unavailable — retrying in 30s…")
        hidden = dict(display="none")
        err    = html.Div("Data error — retrying…", style=dict(color=NEG))
        return (empty, empty, empty, empty,
                [], "", "", hidden, "", hidden, "", "",
                err, "", loading_msg)
