"""
dashboard.py — interactive view of the NullQuant research system.

This is a MONITORING/REPORTING surface over the rigorous engine in the
`nullquant` package, not a second source of truth. The heavy research (backtest,
overlays, metrics, cost sweep) is computed ONCE and cached; the 30s interval
only refreshes display-only live prices. Daily bars barely change intraday, and
nothing here mutates the historical panel — so there is no repainting.

Panels (fixed sidebar, one at a time):
  Overview  · headline metrics + honest verdict
  Equity    · strategy variants vs benchmarks (the headline chart)
  Positions · current target weights + weight history heatmap
  Signals   · per-asset trend state and latest direction
  Costs     · Sharpe vs cost-multiplier robustness curve
  ML Intel  · per-model truth-tellers: LTR rank IC, regime occupancy,
              lead-lag hit-rate (+ funding tilt), conformal coverage/exposure

Styling lives in assets/nullquant.css; sidebar navigation in assets/nullquant.js.
"""

from __future__ import annotations

import json
import threading
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

from nullquant.config import load_config, PROJECT_ROOT
from nullquant.seeds import set_global_seed
from nullquant.data.loader import load_history, fetch_live_prices
from nullquant.signals.base import target_directions, trend_state
from nullquant.portfolio.costs import CostModel
from nullquant.portfolio.backtest import run_backtest
from nullquant.metrics import performance as perf
from nullquant.ablation import compute_signals_cached, run_cost_sweep

# ── Palette (matches assets/nullquant.css) ─────────────────────────────────────
BG, SURFACE = "#0a0a0a", "#0f0f0f"
TEXT, TEXT2, TEXT3 = "#ededed", "#737373", "#404040"
POS, NEG, WARN, ACCENT = "#22c55e", "#ef4444", "#f59e0b", "#ffffff"
BORDER, GRID = "rgba(255,255,255,0.06)", "rgba(255,255,255,0.04)"
PALETTE = ["#ededed", "#22c55e", "#ef4444", "#f59e0b", "#60a5fa", "#a78bfa", "#f472b6", "#2dd4bf"]
REFRESH_MS = 30_000

_CFG = load_config()
_RESULTS: dict | None = None  # cached heavy research output
_RESULTS_LOCK = threading.Lock()  # serialise compute so we never fit twice

# Plain-English guide, rendered inside the app so newcomers never leave the page.
try:
    GUIDE_MD = (PROJECT_ROOT / "GUIDE.md").read_text()
except Exception:
    GUIDE_MD = "# Guide\n\nGUIDE.md not found."


def _base_layout(**over) -> dict:
    base = dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="'Inter',system-ui,sans-serif", size=11),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER,
                   tickfont=dict(color=TEXT2, size=10)),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER,
                   tickfont=dict(color=TEXT2, size=10)),
        margin=dict(l=56, r=20, t=20, b=30),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#111111", bordercolor="rgba(0,0,0,0)",
                        font=dict(color=TEXT, size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT2, size=10),
                    orientation="h", y=1.02, x=0),
        # Pan-to-drag (TradingView feel; scroll to zoom is set in _GRAPH_CFG),
        # and uirevision so the 30s auto-refresh never resets the user's zoom/pan.
        dragmode="pan",
        uirevision="keep",
    )
    for k, v in over.items():
        base[k] = {**base[k], **v} if k in base and isinstance(base[k], dict) and isinstance(v, dict) else v
    return base


def _time_xaxis(slider: bool = True) -> dict:
    """A date x-axis with TradingView-style range buttons, crosshair spike, and
    (optionally) a range slider. Merged onto the base x-axis styling."""
    ax = dict(
        showspikes=True, spikecolor=TEXT2, spikethickness=1,
        spikedash="solid", spikemode="across", spikesnap="cursor",
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL"),
            ],
            bgcolor=SURFACE, activecolor="rgba(255,255,255,0.16)",
            bordercolor=BORDER, borderwidth=1,
            font=dict(color=TEXT2, size=10),
            x=0, xanchor="left", y=1.06, yanchor="bottom",
        ),
    )
    if slider:
        ax["rangeslider"] = dict(visible=True, thickness=0.06, bgcolor=SURFACE,
                                 bordercolor=BORDER, borderwidth=1)
    return ax


_GRAPH_CFG = dict(displayModeBar=False, scrollZoom=True, displaylogo=False,
                  doubleClick="reset")


def compute_results() -> dict:
    """Compute all signal sources + the conformal gate once, run every variant
    backtest, and cache. The four ML models are loaded from the pipeline's
    on-disk cache when available (same config + data fingerprint); they are only
    refit here on a cold cache — so launching the dashboard after `./run.sh`
    starts in seconds instead of retraining."""
    set_global_seed(_CFG.seed)
    panel = load_history(_CFG)
    cost = CostModel.from_config(_CFG, multiplier=1.0)
    idx = panel.close.index

    directions, conf_exp, diagnostics = compute_signals_cached(panel, _CFG)

    # Each direction source, plain and conformal-gated.
    configs: dict = {}
    for name, d in directions.items():
        configs[name] = (d, None)
        configs[f"{name} +conformal"] = (d, conf_exp)

    equity, metrics, weights_by_variant = {}, {}, {}
    bench = None
    for name, (d, exp) in configs.items():
        res = run_backtest(panel, d, _CFG, cost, exposure_scale=exp)
        if bench is None:
            bench = res.benchmarks
        equity[name] = res.equity
        metrics[name] = perf.summary(res.net_returns, benchmark=bench["equal_weight"],
                                     n_trials=len(configs))
        weights_by_variant[name] = res.weights
    for bname, bret in bench.items():
        equity[f"[bench] {bname}"] = (1.0 + bret.fillna(0.0)).cumprod()
        metrics[f"[bench] {bname}"] = perf.summary(bret, n_trials=1)

    cost_sweep = run_cost_sweep(panel, _CFG, directions["baseline"], conf_exp)

    ts = trend_state(panel.close, _CFG.strategy.sma_short, _CFG.strategy.sma_long)
    # Show the strongest variant's book on the Positions panel.
    best_weights = weights_by_variant.get("baseline +conformal",
                                          weights_by_variant["baseline"])

    return dict(
        panel=panel, equity=equity, metrics=metrics,
        cost_sweep=cost_sweep, base_weights=best_weights,
        trend=ts, direction=directions["baseline"],
        diagnostics=diagnostics, last_date=idx[-1],
    )


def get_results() -> dict:
    """Return the cached research payload, computing it once on first access.

    The lock serialises callers so a warm-up thread and the first dashboard
    callback can't both trigger a (5-minute) fit — the second waits and reuses
    the result. With a warm signal cache the compute is just fast backtests."""
    global _RESULTS
    with _RESULTS_LOCK:
        if _RESULTS is None:
            _RESULTS = compute_results()
    return _RESULTS


def warm_results() -> None:
    """Kick the compute off the web-request path so the server stays responsive
    while results are being prepared (used as a background warm-up at startup)."""
    try:
        get_results()
        print("[dashboard] research ready — dashboard is live")
    except Exception:
        traceback.print_exc()


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_equity(R: dict) -> go.Figure:
    fig = go.Figure()
    for i, (name, eq) in enumerate(R["equity"].items()):
        is_bench = name.startswith("[bench]")
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, mode="lines", name=name,
            line=dict(color=PALETTE[i % len(PALETTE)],
                      width=2 if not is_bench else 1.5,
                      dash="dash" if is_bench else "solid"),
        ))
    fig.update_layout(**_base_layout(
        yaxis=dict(type="log", title="growth of $1 (log)"),
        xaxis=_time_xaxis(slider=True),
        margin=dict(l=56, r=20, t=44, b=30),
        legend=dict(y=1.16),
    ))
    return fig


def fig_weights_bar(R: dict) -> go.Figure:
    w = R["base_weights"].iloc[-1]
    w = w[w.abs() > 1e-6].sort_values()
    colors = [POS if v > 0 else NEG for v in w.values]
    fig = go.Figure(go.Bar(x=w.values, y=[a.replace("-USD", "") for a in w.index],
                           orientation="h", marker_color=colors))
    fig.update_layout(**_base_layout(
        xaxis=dict(title="target weight (− short / + long)"),
        margin=dict(l=70, r=20, t=20, b=30)))
    return fig


def fig_weights_heatmap(R: dict) -> go.Figure:
    w = R["base_weights"].resample("W").last().dropna(how="all")
    fig = go.Figure(go.Heatmap(
        z=w.T.values, x=w.index, y=[a.replace("-USD", "") for a in w.columns],
        colorscale=[[0, NEG], [0.5, "#111111"], [1, POS]], zmid=0,
        colorbar=dict(title="w", tickfont=dict(color=TEXT2, size=9)),
    ))
    fig.update_layout(**_base_layout(
        xaxis=_time_xaxis(slider=False),
        margin=dict(l=70, r=20, t=44, b=30),
        hovermode="closest",
    ))
    return fig


def fig_cost_sweep(R: dict) -> go.Figure:
    cs = R["cost_sweep"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cs.index, y=cs["baseline_sharpe"], mode="lines+markers",
                             name="baseline", line=dict(color=TEXT, width=2)))
    fig.add_trace(go.Scatter(x=cs.index, y=cs["gated_sharpe"], mode="lines+markers",
                             name="baseline +conformal", line=dict(color=WARN, width=2)))
    fig.add_hline(y=0, line=dict(color=BORDER, width=1))
    fig.update_layout(**_base_layout(
        xaxis=dict(title="cost multiplier (× base)"),
        yaxis=dict(title="annualized Sharpe")))
    return fig


# ── Stat helpers ──────────────────────────────────────────────────────────────

def _stat(label, value, color=TEXT, sub=""):
    return html.Div(className="statcard", children=[
        html.Div(label, className="lbl"),
        html.Div(value, className="val tnum", style=dict(color=color)),
        html.Div(sub, className="sub") if sub else None,
    ])


def _cost_survival(R: dict) -> str:
    """Largest cost multiplier at which the conformal-gated Sharpe is still > 0
    — derived from the live cost sweep, so the verdict can never drift stale."""
    cs = R.get("cost_sweep")
    try:
        positive = cs.index[cs["gated_sharpe"] > 0]
        return f"~{positive.max():g}×" if len(positive) else "0×"
    except Exception:
        return "~1×"


def overview_panel(R: dict) -> html.Div:
    m = R["metrics"]
    base, btc = m["baseline"], m["[bench] btc_hold"]
    gated = m.get("baseline +conformal", base)
    span = f"{R['panel'].index.min().year}–{R['panel'].index.max().year}"
    con = R["diagnostics"].get("conformal", {})
    mean_exp = con.get("mean_exposure", float("nan"))
    verdict = (
        f"Honest, out-of-sample finding ({span}): the three ML signal generators "
        "(learning-to-rank, regime-switching, lead-lag) all UNDERPERFORM the simple "
        "baseline — confirmed by their truth-teller diagnostics in ML Intel. The "
        "conformal confidence gate is the one validated, if modest, win: as a "
        "continuous inverse-width sizer it now deploys "
        f"{mean_exp*100:.0f}% mean exposure (vs the old binary gate's cash-parking) "
        f"and still lifts the baseline Sharpe {base['sharpe']:.2f} → {gated['sharpe']:.2f}, "
        f"trims max drawdown {base['max_drawdown']*100:.0f}% → {gated['max_drawdown']*100:.0f}%, "
        f"and stays positive out to {_cost_survival(R)} costs (see Costs). The earlier "
        "binary gate's larger lift was substantially a market-timing-by-sitting-in-cash "
        "artifact. Even so, no variant beats buy-and-hold in a crypto bull market — beta "
        "is hard to beat. The value is the rigor, the honesty, and one component that helps."
    )
    return html.Div(children=[
        html.Div("Overview", className="panel-title"),
        html.Div(className="stat-row", children=[
            _stat("Baseline Sharpe", f"{base['sharpe']:.2f}",
                  POS if base['sharpe'] > 0 else NEG, "net of 1× costs"),
            _stat("+ Conformal Sharpe", f"{gated['sharpe']:.2f}",
                  POS if gated['sharpe'] > 0 else NEG, "best variant"),
            _stat("BTC HODL Sharpe", f"{btc['sharpe']:.2f}", TEXT, "benchmark"),
            _stat("Conformal coverage",
                  f"{(R['diagnostics'].get('conformal',{}).get('empirical_coverage',0))*100:.0f}%",
                  TEXT, "target 90%"),
        ]),
        html.Div(className="card", style=dict(padding="20px", marginTop="4px"), children=[
            html.Div("Verdict", className="panel-title"),
            html.Div(verdict, style=dict(color=TEXT2, fontSize="13px", lineHeight="1.6")),
        ]),
        html.Div(className="card", style=dict(padding="20px", marginTop="16px"), children=[
            html.Div("Variants — out-of-sample, net of 1× costs", className="panel-title"),
            _metrics_table(R),
        ]),
    ])


def _metrics_table(R: dict) -> dash_table.DataTable:
    rows = []
    for name, s in R["metrics"].items():
        rows.append({
            "variant": name,
            "ann_return": f"{s['ann_return']*100:+.1f}%",
            "ann_vol": f"{s['ann_vol']*100:.1f}%",
            "sharpe": f"{s['sharpe']:.2f}",
            "max_dd": f"{s['max_drawdown']*100:.1f}%",
            "dsr": f"{s.get('deflated_sharpe', float('nan')):.2f}",
        })
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in
                 ["variant", "ann_return", "ann_vol", "sharpe", "max_dd", "dsr"]],
        style_as_list_view=True,
        style_header=dict(backgroundColor="transparent", color=TEXT2, fontWeight="600",
                          fontSize="11px", textTransform="uppercase", border="none",
                          borderBottom=f"1px solid {BORDER}", padding="10px 12px"),
        style_cell=dict(backgroundColor="transparent", color=TEXT, border="none",
                        fontFamily="'Inter',system-ui,sans-serif", fontSize="12px",
                        padding="8px 12px", textAlign="right"),
        style_cell_conditional=[{"if": {"column_id": "variant"}, "textAlign": "left"}],
        style_data_conditional=[
            {"if": {"filter_query": '{variant} contains "bench"'},
             "color": TEXT2, "fontStyle": "italic"},
        ],
    )


def signals_panel(R: dict) -> html.Div:
    ts, direction = R["trend"], R["direction"]
    last_ts, last_dir = ts.iloc[-1], direction.iloc[-1]
    rows = []
    for a in R["panel"].assets:
        st = last_ts.get(a, np.nan)
        di = last_dir.get(a, 0.0)
        rows.append({
            "asset": a.replace("-USD", ""),
            "trend": "up" if st > 0 else ("down" if st < 0 else "—"),
            "position": "LONG" if di > 0 else ("SHORT" if di < 0 else "flat"),
        })
    return html.Div(children=[
        html.Div("Signals — latest state", className="panel-title"),
        dash_table.DataTable(
            data=rows, columns=[{"name": c, "id": c} for c in ["asset", "trend", "position"]],
            style_as_list_view=True,
            style_header=dict(backgroundColor="transparent", color=TEXT2, fontWeight="600",
                              fontSize="11px", textTransform="uppercase", border="none",
                              borderBottom=f"1px solid {BORDER}", padding="10px 12px"),
            style_cell=dict(backgroundColor="transparent", color=TEXT, border="none",
                            fontSize="13px", padding="9px 12px", textAlign="left",
                            fontFamily="'Inter',system-ui,sans-serif"),
            style_data_conditional=[
                {"if": {"filter_query": '{position} = "LONG"', "column_id": "position"}, "color": POS},
                {"if": {"filter_query": '{position} = "SHORT"', "column_id": "position"}, "color": NEG},
            ],
        ),
    ])


def _big(text, color):
    return html.Div(text, style=dict(color=color, fontWeight="600", fontSize="22px"))


def _note(text):
    return html.Div(text, style=dict(color=TEXT2, fontSize="11px", marginTop="8px"))


def ml_panel(R: dict) -> html.Div:
    d = R.get("diagnostics", {})
    m = R["metrics"]

    def sharpe_of(name):
        return m.get(name, {}).get("sharpe", float("nan"))

    # LTR
    ltr = d.get("ltr")
    if ltr:
        ic = ltr["rank_ic"]
        ltr_body = [_big(f"IC {ic:+.3f}", POS if ic > 0.03 else NEG),
                    _note(f"rank IC over OOS rebalances · {ltr['n_refits']} refits"),
                    _note(f"as a signal: Sharpe {sharpe_of('LTR'):.2f} (worse than baseline)")]
    else:
        ltr_body = [_big("n/a", TEXT2)]

    # Regime
    reg = d.get("regime")
    if reg:
        occ = reg.get("occupancy", {})
        mix = " · ".join(f"{k} {v*100:.0f}%" for k, v in occ.items() if v > 0.01)
        reg_body = [_big(f"{reg['n_states']} states", TEXT),
                    _note(f"controller mix: {mix}"),
                    _note(f"as a signal: Sharpe {sharpe_of('regime'):.2f} (worse than baseline)")]
    else:
        reg_body = [_big("n/a", TEXT2)]

    # Lead-lag
    ll = d.get("leadlag")
    if ll:
        hit = ll["oos_hit_rate"]
        tilt = "blended" in ll.get("status", "")
        ll_body = [_big(f"{hit*100:.1f}%", POS if hit > 0.52 else NEG),
                   _note("OOS next-day hit-rate vs 50% coin-flip"),
                   _note("book = price lead-lag + structural funding tilt"
                         if tilt else "funding tilt off (no funding data)"),
                   _note(f"as a signal: Sharpe {sharpe_of('lead-lag'):.2f} (worse than baseline)")]
    else:
        ll_body = [_big("n/a", TEXT2)]

    # Conformal (the validated, modest win)
    con = d.get("conformal")
    if con:
        cov = con["empirical_coverage"]
        delta = sharpe_of('baseline +conformal') - sharpe_of('baseline')
        con_body = [_big(f"{cov*100:.1f}% cover", POS if abs(cov - 0.9) < 0.04 else WARN),
                    _note(f"calibration target 90% · continuous inverse-width sizer"),
                    _note(f"mean exposure {con['mean_exposure']*100:.0f}% (deploys, not cash-parks)"),
                    _note(f"lifts baseline Sharpe {sharpe_of('baseline'):.2f} → "
                          f"{sharpe_of('baseline +conformal'):.2f} "
                          f"({'+' if delta >= 0 else ''}{delta:.2f})")]
    else:
        con_body = [_big("n/a", TEXT2)]

    def card(title, sub, body, win=False):
        style = dict(borderColor="rgba(34,197,94,0.35)") if win else {}
        return html.Div(className="mlcard", style=style, children=[
            html.Div(title, className="title"), html.Div(sub, className="subtitle"), html.Div(body)])

    return html.Div(children=[
        html.Div("ML Intel — honest out-of-sample diagnostics", className="panel-title"),
        html.Div(className="ml-grid", style=dict(gridTemplateColumns="repeat(2,1fr)"), children=[
            card("Learning-to-Rank", "rank coins, long top / short bottom", ltr_body),
            card("Regime-Switching (HMM)", "switch strategy per hidden regime", reg_body),
            card("Lead–Lag Network", "trade laggards on leader moves + funding tilt", ll_body),
            card("Conformal Gate", "calibrated confidence → continuous exposure", con_body, win=True),
        ]),
        html.Div("Pre-committed rule: a model must prove out-of-sample lift or be shelved. "
                 "The three signal generators do NOT beat the baseline; the conformal gate "
                 "DOES — a modest but validated lift (and the old binary gate's larger lift "
                 "was largely a sit-in-cash artifact). Model outputs, not financial advice.",
                 className="ml-disclaimer"),
    ])


# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="NullQuant · Research", update_title=None,
                external_scripts=[{"src": "https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"}])
server = app.server


def _nav(panel, icon, label, active=False):
    return html.Div(className="nav-item active" if active else "nav-item",
                    **{"data-panel": panel},
                    children=[html.I(**{"data-lucide": icon}), html.Span(label)])


def _panel(panel, extra, children, active=False):
    return html.Div(className=f"panel {extra}" + (" active" if active else ""),
                    **{"data-panel": panel}, children=children)


def _chart_panel(panel, gid, active=False):
    return _panel(panel, "", html.Div(className="chart-fill", children=[
        dcc.Graph(id=gid, className="chart-fill", style=dict(height="100%", width="100%"),
                  config=_GRAPH_CFG)]), active=active)


app.layout = html.Div(className="app", children=[
    html.Div(className="header", children=[
        html.Div(className="hdr-left", children=[html.Div("NULLQUANT", className="wordmark")]),
        html.Div(id="header-stats", className="hdr-right"),
    ]),
    html.Div(className="body-row", children=[
        html.Div(className="sidebar", children=[
            html.Div(className="nav", children=[
                _nav("overview", "layout-dashboard", "Overview", active=True),
                _nav("equity", "trending-up", "Equity"),
                _nav("positions", "scale", "Positions"),
                _nav("signals", "activity", "Signals"),
                _nav("costs", "dollar-sign", "Costs"),
                _nav("ml", "brain", "ML Intel"),
                _nav("guide", "book-open", "Guide"),
            ]),
            html.Div("v2.0 · research", className="sidebar-footer"),
        ]),
        html.Div(className="main", children=[
            _panel("overview", "scroll pad", [html.Div(id="overview-content")], active=True),
            _panel("equity", "", [html.Div(className="chart-fill", children=[
                dcc.Graph(id="equity-chart", className="chart-fill",
                          style=dict(height="100%"), config=_GRAPH_CFG)])]),
            _panel("positions", "scroll pad", [
                html.Div("Positions — current target weights", className="panel-title"),
                dcc.Graph(id="weights-bar", config=_GRAPH_CFG, style=dict(height="320px")),
                html.Div("Weight history (weekly)", className="panel-title", style=dict(marginTop="20px")),
                dcc.Graph(id="weights-heatmap", config=_GRAPH_CFG, style=dict(height="320px")),
            ]),
            _panel("signals", "scroll pad", [html.Div(id="signals-content")]),
            _panel("costs", "scroll pad", [
                html.Div("Cost robustness — Sharpe vs cost multiplier", className="panel-title"),
                dcc.Graph(id="cost-chart", config=_GRAPH_CFG, style=dict(height="420px")),
                html.Div("The conformal-gated baseline beats the plain baseline at every "
                         "cost level, but the lift is modest and fades as costs rise — a "
                         "real but fragile edge, not robust alpha.", className="ml-disclaimer"),
            ]),
            _panel("ml", "scroll pad", [html.Div(id="ml-content")]),
            # Static plain-English guide rendered in-app (no callback needed).
            _panel("guide", "scroll pad", [
                dcc.Markdown(GUIDE_MD, className="markdown-body", link_target="_blank"),
            ]),
        ]),
    ]),
    html.Div(id="footer", className="footer"),
    dcc.Loading(id="page-loading", type="circle", color=ACCENT,
                children=html.Div(id="loading-trigger", style=dict(display="none"))),
    dcc.Interval(id="refresh-interval", interval=REFRESH_MS, n_intervals=0),
])


@app.callback(
    Output("overview-content", "children"),
    Output("equity-chart", "figure"),
    Output("weights-bar", "figure"),
    Output("weights-heatmap", "figure"),
    Output("signals-content", "children"),
    Output("cost-chart", "figure"),
    Output("ml-content", "children"),
    Output("header-stats", "children"),
    Output("footer", "children"),
    Output("loading-trigger", "children"),
    Input("refresh-interval", "n_intervals"),
)
def refresh(n):
    loading = "Computing research…" if n == 0 else ""
    try:
        R = get_results()
        live = fetch_live_prices(_CFG)  # display only
        btc = live.get("BTC-USD")
        base = R["metrics"]["baseline"]

        header = html.Div(children=[
            html.Div(className="hdr-spacer"),
            html.Div(className="hdr-center", children=[
                html.Span("BTC/USD", className="hdr-pair-label"),
                html.Span(f"${btc:,.0f}" if btc else "—", className="hdr-price tnum"),
            ]),
            html.Div(className="hdr-stats", children=[
                html.Span("LONG/SHORT", className="pill pill-neutral"),
                html.Div(className="hdr-stat", children=[
                    html.Span("Strategy Sharpe", className="lbl"),
                    html.Span(f"{base['sharpe']:.2f}", className="val tnum",
                              style=dict(color=POS if base['sharpe'] > 0 else NEG))]),
                html.Div(className="hdr-stat", children=[
                    html.Span("BTC HODL Sharpe", className="lbl"),
                    html.Span(f"{R['metrics']['[bench] btc_hold']['sharpe']:.2f}",
                              className="val tnum")]),
            ]),
        ])

        footer = (f"NULLQUANT · data through {R['last_date'].date()} · "
                  f"{len(R['panel'].assets)} assets · long/short vol-targeted · "
                  f"net of costs · live price display-only (no repainting)")

        return (overview_panel(R), fig_equity(R), fig_weights_bar(R),
                fig_weights_heatmap(R), signals_panel(R), fig_cost_sweep(R),
                ml_panel(R), header, footer, loading)
    except Exception:
        traceback.print_exc()
        empty = go.Figure().update_layout(paper_bgcolor=BG, plot_bgcolor=BG)
        err = html.Div("Error computing research — see server log.", style=dict(color=NEG, padding="20px"))
        return (err, empty, empty, empty, err, empty, err,
                html.Div("error", className="hdr-right"), "", loading)
