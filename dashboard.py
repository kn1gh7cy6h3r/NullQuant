"""
dashboard.py — interactive view of the Meridian research system.

This is a MONITORING/REPORTING surface over the rigorous engine in the
`meridian` package, not a second source of truth. The heavy research (backtest,
overlays, metrics, cost sweep) is computed ONCE and cached; the 30s interval
only refreshes display-only live prices. Daily bars barely change intraday, and
nothing here mutates the historical panel — so there is no repainting.

Panels (fixed sidebar, one at a time):
  Overview  · headline metrics + honest verdict
  Equity    · strategy variants vs benchmarks (the headline chart)
  Positions · current target weights + weight history heatmap
  Signals   · per-asset trend state and latest direction
  Costs     · Sharpe vs cost-multiplier robustness curve
  ML Intel  · RF meta AUC, regime stats, LSTM-vs-random-walk (honest)

Styling lives in assets/meridian.css; sidebar navigation in assets/meridian.js.
"""

from __future__ import annotations

import json
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

from meridian.config import load_config, PROJECT_ROOT
from meridian.seeds import set_global_seed
from meridian.data.loader import load_history, fetch_live_prices
from meridian.signals.base import target_directions, trend_state
from meridian.portfolio.costs import CostModel
from meridian.portfolio.backtest import run_backtest
from meridian.metrics import performance as perf
from meridian.ablation import build_overlays, run_cost_sweep, _exposure_product

# ── Palette (matches assets/meridian.css) ─────────────────────────────────────
BG, SURFACE = "#0a0a0a", "#0f0f0f"
TEXT, TEXT2, TEXT3 = "#ededed", "#737373", "#404040"
POS, NEG, WARN, ACCENT = "#22c55e", "#ef4444", "#f59e0b", "#ffffff"
BORDER, GRID = "rgba(255,255,255,0.06)", "rgba(255,255,255,0.04)"
PALETTE = ["#ededed", "#22c55e", "#ef4444", "#f59e0b", "#60a5fa", "#a78bfa", "#f472b6", "#2dd4bf"]
REFRESH_MS = 30_000

_CFG = load_config()
_RESULTS: dict | None = None  # cached heavy research output

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
    )
    for k, v in over.items():
        base[k] = {**base[k], **v} if k in base and isinstance(base[k], dict) and isinstance(v, dict) else v
    return base


_GRAPH_CFG = dict(displayModeBar=False, scrollZoom=True, displaylogo=False)


def compute_results() -> dict:
    """Run the backtest variants + overlays + cost sweep once; cache the result."""
    set_global_seed(_CFG.seed)
    panel = load_history(_CFG)
    direction = target_directions(panel, _CFG)
    cost = CostModel.from_config(_CFG, multiplier=1.0)

    overlays = build_overlays(panel, _CFG)
    regime, meta = overlays.get("regime"), overlays.get("meta")
    meta_res = overlays.get("_meta_result")
    idx = panel.close.index

    variants = {
        "baseline": None,
        "+regime": regime,
        "+meta": meta,
        "+regime+meta": _exposure_product(regime, meta, index=idx),
    }
    equity, metrics, weights_by_variant = {}, {}, {}
    bench = None
    for name, exp in variants.items():
        res = run_backtest(panel, direction, _CFG, cost, exposure_scale=exp)
        if bench is None:
            bench = res.benchmarks
        equity[name] = res.equity
        metrics[name] = perf.summary(res.net_returns, benchmark=bench["equal_weight"],
                                     n_trials=len(variants))
        weights_by_variant[name] = res.weights
    for bname, bret in bench.items():
        equity[f"[bench] {bname}"] = (1.0 + bret.fillna(0.0)).cumprod()
        metrics[f"[bench] {bname}"] = perf.summary(bret, n_trials=1)

    cost_sweep = run_cost_sweep(panel, _CFG, direction,
                                best_exposure=variants["+regime+meta"])

    # Read ML diagnostics from the last pipeline run if available (LSTM is slow).
    summary_path = PROJECT_ROOT / "research" / "results" / "summary.json"
    ml_summary = {}
    if summary_path.exists():
        try:
            ml_summary = json.loads(summary_path.read_text())
        except Exception:
            ml_summary = {}

    ts = trend_state(panel.close, _CFG.strategy.sma_short, _CFG.strategy.sma_long)
    base_weights = weights_by_variant["baseline"]

    return dict(
        panel=panel, equity=equity, metrics=metrics,
        cost_sweep=cost_sweep, base_weights=base_weights,
        trend=ts, direction=direction, meta_res=meta_res,
        regime=regime, ml_summary=ml_summary,
        last_date=idx[-1],
    )


def get_results() -> dict:
    global _RESULTS
    if _RESULTS is None:
        _RESULTS = compute_results()
    return _RESULTS


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
    fig.update_layout(**_base_layout(yaxis=dict(type="log", title="growth of $1 (log)")))
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
    fig.update_layout(**_base_layout(margin=dict(l=70, r=20, t=20, b=30)))
    return fig


def fig_cost_sweep(R: dict) -> go.Figure:
    cs = R["cost_sweep"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cs.index, y=cs["baseline_sharpe"], mode="lines+markers",
                             name="baseline", line=dict(color=TEXT, width=2)))
    fig.add_trace(go.Scatter(x=cs.index, y=cs["overlaid_sharpe"], mode="lines+markers",
                             name="+regime+meta", line=dict(color=WARN, width=2)))
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


def overview_panel(R: dict) -> html.Div:
    m = R["metrics"]
    base, ew, btc = m["baseline"], m["[bench] equal_weight"], m["[bench] btc_hold"]
    verdict = (
        "Across 2019–2026, the cross-sectional long/short crossover book is NOT "
        "competitive with buy-and-hold on a risk-adjusted basis, and the ML "
        "overlays add no out-of-sample lift (see ML Intel). This is the honest, "
        "rigorously-validated finding — the value here is the methodology, not a "
        "manufactured edge."
    )
    return html.Div(children=[
        html.Div("Overview", className="panel-title"),
        html.Div(className="stat-row", children=[
            _stat("Strategy Sharpe", f"{base['sharpe']:.2f}",
                  POS if base['sharpe'] > 0 else NEG, "net of 1× costs"),
            _stat("Strategy Ann. Return", f"{base['ann_return']*100:+.1f}%",
                  POS if base['ann_return'] > 0 else NEG),
            _stat("BTC HODL Sharpe", f"{btc['sharpe']:.2f}", TEXT, "benchmark"),
            _stat("Equal-Weight Sharpe", f"{ew['sharpe']:.2f}", TEXT, "benchmark"),
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


def ml_panel(R: dict) -> html.Div:
    ml = R["ml_summary"]
    meta = R.get("meta_res")
    regime = R.get("regime")
    # Meta card
    if meta is not None:
        auc = meta.oos_auc
        meta_body = [
            html.Div(f"AUC {auc:.2f}", style=dict(color=POS if auc > 0.55 else NEG,
                     fontWeight="600", fontSize="24px")),
            html.Div(f"{meta.n_events} events · base rate {meta.base_rate*100:.0f}%",
                     style=dict(color=TEXT2, fontSize="11px", marginTop="8px")),
            html.Div("OOS via purged k-fold. AUC ≤ 0.5 ⇒ no usable edge.",
                     style=dict(color=TEXT2, fontSize="11px", marginTop="6px")),
        ]
    else:
        meta_body = [html.Div("unavailable", style=dict(color=TEXT2))]
    # Regime card
    if regime is not None:
        frac = float((regime < 1.0).mean())
        regime_body = [
            html.Div(f"{frac*100:.1f}%", style=dict(color=WARN, fontWeight="600", fontSize="24px")),
            html.Div("of days flagged abnormal", style=dict(color=TEXT2, fontSize="11px", marginTop="8px")),
            html.Div("Walk-forward IsolationForest; gating hurt OOS P&L here.",
                     style=dict(color=TEXT2, fontSize="11px", marginTop="6px")),
        ]
    else:
        regime_body = [html.Div("unavailable", style=dict(color=TEXT2))]
    # LSTM card (from pipeline summary.json)
    lstm = ml.get("lstm") if ml else None
    if lstm:
        beats = lstm["beats_baseline"]
        lstm_body = [
            html.Div("beats RW" if beats else "no skill",
                     style=dict(color=POS if beats else NEG, fontWeight="600", fontSize="24px")),
            html.Div(f"RMSE {lstm['oos_rmse']:.4f} vs RW {lstm['baseline_rmse']:.4f}",
                     style=dict(color=TEXT2, fontSize="11px", marginTop="8px")),
            html.Div(f"dir acc {lstm['oos_dir_acc']*100:.0f}% vs {lstm['baseline_dir_acc']*100:.0f}%",
                     style=dict(color=TEXT2, fontSize="11px", marginTop="4px")),
        ]
    else:
        lstm_body = [html.Div("run pipeline", style=dict(color=TEXT2, fontSize="13px")),
                     html.Div("python -m meridian.pipeline", style=dict(color=TEXT3, fontSize="11px", marginTop="6px"))]

    def card(title, sub, body):
        return html.Div(className="mlcard", children=[
            html.Div(title, className="title"), html.Div(sub, className="subtitle"), html.Div(body)])

    return html.Div(children=[
        html.Div("ML Intel — honest out-of-sample diagnostics", className="panel-title"),
        html.Div(className="ml-grid", children=[
            card("RF Meta-Label", "take/skip a signal · purged CV", meta_body),
            card("Regime Filter", "IsolationForest · walk-forward", regime_body),
            card("LSTM Forecast", "log-returns vs random walk", lstm_body),
        ]),
        html.Div("Per the pre-committed rule, a model must prove OOS P&L lift or be "
                 "shelved. None clears the bar — reported honestly. Model outputs, "
                 "not financial advice.", className="ml-disclaimer"),
    ])


# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="Meridian · Research", update_title=None,
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
        html.Div(className="hdr-left", children=[html.Div("MERIDIAN", className="wordmark")]),
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
                html.Div("Even at zero cost the edge is marginal; it does not survive "
                         "realistic friction.", className="ml-disclaimer"),
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

        footer = (f"MERIDIAN · data through {R['last_date'].date()} · "
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
