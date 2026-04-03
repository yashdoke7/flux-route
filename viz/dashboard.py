"""
FluxRoute – Interactive Plotly Dash dashboard.

Optional interactive dashboard with:
- Scenario selector (task + seed)
- Baseline selector
- Live metric cards
- Topology graph placeholder
- Download report button

Run: python -m viz.dashboard
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table

logger = logging.getLogger("fluxroute.dashboard")

RESULTS_DIR = Path("results")


def _load_data() -> pd.DataFrame:
    path = RESULTS_DIR / "eval_results.json"
    if not path.exists():
        logger.warning("No eval results found. Returning empty DataFrame.")
        return pd.DataFrame()
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def create_app() -> Dash:
    """Build and return the Dash app (does NOT start the server)."""
    app = Dash(__name__, title="FluxRoute Dashboard")

    df = _load_data()
    tasks = sorted(df["task_id"].unique().tolist()) if not df.empty else []
    agents = sorted(df["agent"].unique().tolist()) if not df.empty else []
    seeds = sorted(df["seed"].unique().tolist()) if not df.empty else []

    app.layout = html.Div(
        style={
            "fontFamily": "'Inter', sans-serif",
            "backgroundColor": "#0f172a",
            "color": "#e2e8f0",
            "minHeight": "100vh",
            "padding": "24px",
        },
        children=[
            # Header
            html.Div(
                style={"textAlign": "center", "marginBottom": "24px"},
                children=[
                    html.H1(
                        "⚡ FluxRoute Dashboard",
                        style={
                            "background": "linear-gradient(90deg, #38bdf8, #a78bfa)",
                            "WebkitBackgroundClip": "text",
                            "WebkitTextFillColor": "transparent",
                            "fontSize": "2.5rem",
                            "marginBottom": "8px",
                        },
                    ),
                    html.P(
                        "Adaptive RL Routing · Evaluation Explorer",
                        style={"color": "#94a3b8", "fontSize": "1.1rem"},
                    ),
                ],
            ),
            # Controls row
            html.Div(
                style={
                    "display": "flex",
                    "gap": "16px",
                    "marginBottom": "24px",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Div([
                        html.Label("Task", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="task-select",
                            options=[{"label": t, "value": t} for t in tasks],
                            value=tasks[0] if tasks else None,
                            style={"width": "250px", "color": "#0f172a"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Agent", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="agent-select",
                            options=[{"label": a, "value": a} for a in agents],
                            value=agents,
                            multi=True,
                            style={"width": "350px", "color": "#0f172a"},
                        ),
                    ]),
                ],
            ),
            # Metric cards
            html.Div(id="metric-cards", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                "gap": "12px",
                "marginBottom": "24px",
            }),
            # Charts
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
                },
                children=[
                    dcc.Graph(id="grade-bar"),
                    dcc.Graph(id="latency-box"),
                ],
            ),
            # Table
            html.Div(
                style={"marginTop": "24px"},
                children=[
                    html.H3("Detailed Results"),
                    dash_table.DataTable(
                        id="results-table",
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#1e293b",
                            "color": "#e2e8f0",
                            "fontWeight": "bold",
                        },
                        style_cell={
                            "backgroundColor": "#1e293b",
                            "color": "#cbd5e1",
                            "border": "1px solid #334155",
                            "padding": "8px",
                        },
                        page_size=15,
                    ),
                ],
            ),
            # Download
            html.Div(
                style={"marginTop": "20px", "textAlign": "center"},
                children=[
                    html.A(
                        "📥 Download Report (JSON)",
                        href="/download-report",
                        style={
                            "color": "#38bdf8",
                            "fontSize": "1rem",
                            "textDecoration": "underline",
                        },
                    ),
                ],
            ),
        ],
    )

    # ------ callbacks ------

    @app.callback(
        [
            Output("metric-cards", "children"),
            Output("grade-bar", "figure"),
            Output("latency-box", "figure"),
            Output("results-table", "data"),
            Output("results-table", "columns"),
        ],
        [
            Input("task-select", "value"),
            Input("agent-select", "value"),
        ],
    )
    def update(task, selected_agents):
        if df.empty or not task or not selected_agents:
            empty_fig = go.Figure()
            return [], empty_fig, empty_fig, [], []

        sub = df[(df["task_id"] == task) & (df["agent"].isin(selected_agents))]

        # metric cards
        avg_grade = sub["grade"].mean()
        avg_lat = sub["mean_latency_ms"].mean() if "mean_latency_ms" in sub else 0
        avg_loss = sub["loss_rate"].mean() if "loss_rate" in sub else 0
        avg_tp = sub["throughput"].mean() if "throughput" in sub else 0

        cards = [
            _card("Avg Grade", f"{avg_grade:.3f}"),
            _card("Avg Latency", f"{avg_lat:.1f} ms"),
            _card("Avg Loss", f"{avg_loss:.3f}"),
            _card("Avg Throughput", f"{avg_tp:.0f}"),
        ]

        # grade bar
        grade_fig = px.bar(
            sub.groupby("agent")["grade"].mean().reset_index(),
            x="agent", y="grade", color="agent",
            title="Grade by Agent",
            template="plotly_dark",
        )
        grade_fig.update_layout(yaxis_range=[0, 1])

        # latency box
        lat_fig = px.box(
            sub, x="agent", y="mean_latency_ms", color="agent",
            title="Latency Distribution",
            template="plotly_dark",
        )

        # table
        cols_show = ["agent", "seed", "grade", "mean_latency_ms",
                     "p95_latency_ms", "loss_rate", "throughput"]
        cols_exist = [c for c in cols_show if c in sub.columns]
        table_data = sub[cols_exist].round(4).to_dict("records")
        table_cols = [{"name": c, "id": c} for c in cols_exist]

        return cards, grade_fig, lat_fig, table_data, table_cols

    return app


def _card(title: str, value: str) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": "#1e293b",
            "borderRadius": "10px",
            "padding": "16px",
            "textAlign": "center",
            "border": "1px solid #334155",
        },
        children=[
            html.Div(title, style={"color": "#94a3b8", "fontSize": "0.85rem"}),
            html.Div(value, style={
                "fontSize": "1.6rem",
                "fontWeight": "bold",
                "color": "#38bdf8",
                "marginTop": "4px",
            }),
        ],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    logger.info("Starting FluxRoute Dashboard on http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
