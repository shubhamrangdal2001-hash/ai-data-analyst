"""Plotly chart builders for the Streamlit frontend."""
from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = px.colors.qualitative.Bold
PRIMARY = "#4f46e5"
DANGER = "#e11d48"
SUCCESS = "#10b981"

LAYOUT_BASE = dict(
    font_family="'DM Sans', sans-serif",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
)


def fig_distribution(data: dict[str, Any]) -> go.Figure:
    values = data["values"]
    col = data["column"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, name=col, marker_color=PRIMARY, opacity=0.8, nbinsx=40))
    stats = data.get("stats", {})
    if stats.get("mean") is not None:
        fig.add_vline(x=stats["mean"], line_dash="dash", line_color=DANGER, annotation_text=f"Mean: {stats['mean']:.2f}")
        fig.add_vline(x=stats["median"], line_dash="dot", line_color=SUCCESS, annotation_text=f"Median: {stats['median']:.2f}")
    fig.update_layout(title=f"Distribution of {col}", xaxis_title=col, yaxis_title="Count", **LAYOUT_BASE)
    return fig


def fig_bar(data: dict[str, Any]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=data["values"], y=data["labels"], orientation="h",
        marker_color=px.colors.sequential.Viridis[:len(data["labels"])],
    ))
    fig.update_layout(title=f"Top Values — {data['column']}", xaxis_title="Count", **LAYOUT_BASE)
    return fig


def fig_correlation_heatmap(matrix: dict[str, dict[str, float]]) -> go.Figure:
    import pandas as pd
    df = pd.DataFrame(matrix)
    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=df.round(2).values,
        texttemplate="%{text}",
        colorbar_title="r",
    ))
    fig.update_layout(title="Pearson Correlation Matrix", **LAYOUT_BASE)
    return fig


def fig_missing_values(missing: list[dict]) -> go.Figure:
    if not missing:
        return None
    labels = [m["column"] for m in missing]
    pcts = [m["missing_pct"] for m in missing]
    fig = go.Figure(go.Bar(
        x=pcts, y=labels, orientation="h",
        marker_color=[DANGER if p > 20 else "#f59e0b" if p > 5 else "#6366f1" for p in pcts],
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
    ))
    fig.update_layout(title="Missing Values by Column (%)", xaxis_title="Missing %", **LAYOUT_BASE)
    return fig


def fig_model_comparison(leaderboard: list[dict], task_type: str) -> go.Figure:
    import pandas as pd
    df = pd.DataFrame(leaderboard)
    metric = "roc_auc" if task_type == "classification" else "r2"
    if metric not in df.columns:
        metric = df.columns[2] if len(df.columns) > 2 else df.columns[-1]
    fig = go.Figure(go.Bar(
        x=df["model"],
        y=df[metric],
        marker_color=[PRIMARY if i == 0 else "#a5b4fc" for i in range(len(df))],
        text=df[metric].round(4),
        textposition="outside",
    ))
    fig.update_layout(title=f"Model Comparison — {metric.upper()}", xaxis_title="Model", yaxis_title=metric, **LAYOUT_BASE)
    return fig


def fig_feature_importance(importance: dict[str, float], title: str = "Feature Importance") -> go.Figure:
    items = sorted(importance.items(), key=lambda x: x[1])[-15:]
    fig = go.Figure(go.Bar(
        x=[v for _, v in items],
        y=[k for k, _ in items],
        orientation="h",
        marker_color=PRIMARY,
    ))
    fig.update_layout(title=title, xaxis_title="Importance", **LAYOUT_BASE)
    return fig


def fig_forecast(data: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["historical_dates"], y=data["historical_values"],
        mode="lines", name="Historical", line=dict(color=PRIMARY, width=2),
    ))
    yhat = data.get("yhat", [])
    dates = data.get("dates", [])
    lower = data.get("yhat_lower")
    upper = data.get("yhat_upper")

    split_idx = len(data["historical_dates"])
    fig.add_trace(go.Scatter(
        x=dates[split_idx:], y=yhat[split_idx:],
        mode="lines", name="Forecast", line=dict(color=DANGER, width=2, dash="dash"),
    ))

    if lower and upper:
        fig.add_trace(go.Scatter(
            x=dates[split_idx:] + dates[split_idx:][::-1],
            y=upper[split_idx:] + lower[split_idx:][::-1],
            fill="toself", fillcolor="rgba(225,29,72,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence Interval",
        ))

    fig.update_layout(title="Time-Series Forecast", xaxis_title="Date", yaxis_title="Value", **LAYOUT_BASE)
    return fig


def fig_scatter(df_dict: list[dict], x_col: str, y_col: str, color_col: str | None = None) -> go.Figure:
    import pandas as pd
    df = pd.DataFrame(df_dict)
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, color_discrete_sequence=PALETTE, opacity=0.7)
    fig.update_layout(title=f"{x_col} vs {y_col}", **LAYOUT_BASE)
    return fig
