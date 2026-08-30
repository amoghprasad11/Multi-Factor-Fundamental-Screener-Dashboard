"""
Professional interactive dashboard for the Multi-Factor Fundamental Screener.

Usage in Google Colab:
    from professional_dashboard import build_professional_dashboard
    dashboard = build_professional_dashboard(results)
    dashboard.show()

The function accepts the existing pipeline `results` dictionary, so no
changes to the scoring/data-collection code are required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots


def build_professional_dashboard(results: dict) -> go.Figure:
    """Build the polished 4x2 interactive dashboard from pipeline results."""

    if "data" not in results:
        raise ValueError("results must contain a 'data' DataFrame.")

    df = results["data"].copy()
    metrics = results.get("metrics", {}) or {}

    required = [
        "ticker", "name", "sector", "composite_score",
        "value_score", "quality_score", "growth_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in results['data']: {missing}")

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    n_stocks = len(df)
    top = df.iloc[0]
    top_ticker = str(top["ticker"])
    top_score = float(top["composite_score"])
    avg_score = float(df["composite_score"].mean())

    factor_avgs = {
        "Value": float(df["value_score"].mean()),
        "Quality": float(df["quality_score"].mean()),
        "Growth": float(df["growth_score"].mean()),
    }

    ic = metrics.get("information_coefficient", np.nan)
    try:
        ic = float(ic)
    except (TypeError, ValueError):
        ic = np.nan

    min_score = float(df["composite_score"].min())
    max_score = float(df["composite_score"].max())
    score_colors = sample_colorscale("RdYlGn", np.linspace(0, 1, 101))

    def score_color(value: float) -> str:
        if max_score == min_score:
            idx = 50
        else:
            normalized = (float(value) - min_score) / (max_score - min_score)
            idx = int(max(0, min(1, normalized)) * 100)
        return score_colors[idx]

    sector_avg = df.groupby("sector")["composite_score"].mean().sort_values()
    sector_counts = df["sector"].value_counts().sort_values()
    top15 = df.head(15).sort_values("composite_score")
    top10 = df.head(10).sort_values("composite_score")

    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[
            "📊 Value vs. Quality",
            "🏆 Top 15 Ranked Stocks",
            "🏢 Average Score by Sector",
            "📈 Composite Score Distribution",
            "⚖️ Factor Comparison",
            "⭐ Top 10 Stocks",
            "🌎 Stocks by Sector",
            "📈 Score vs. 12M Return",
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # 1 — Value vs Quality
    scatter = df.dropna(subset=["value_score", "quality_score"]).copy()
    market_cap = pd.to_numeric(
        scatter.get("market_cap", pd.Series(1, index=scatter.index)),
        errors="coerce",
    ).fillna(1)
    max_cap = float(market_cap.max()) if len(market_cap) else 1.0
    sizes = (np.sqrt(market_cap / max_cap) * 38).clip(5, 38) if max_cap > 0 else np.full(len(scatter), 8)

    fig.add_trace(
        go.Scatter(
            x=scatter["value_score"],
            y=scatter["quality_score"],
            mode="markers",
            text=scatter["ticker"],
            marker=dict(
                size=sizes,
                color=scatter["composite_score"],
                colorscale="RdYlGn",
                cmin=min_score,
                cmax=max_score,
                showscale=True,
                colorbar=dict(title="Score", thickness=12, len=0.7),
                opacity=0.82,
                line=dict(width=0.5, color="#FFFFFF"),
            ),
            customdata=np.column_stack([
                scatter["name"].fillna(""),
                scatter["sector"].fillna(""),
                scatter["composite_score"],
            ]),
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[0]}<br><br>"
                "Sector: %{customdata[1]}<br>"
                "Value: %{x:.3f}<br>Quality: %{y:.3f}<br>"
                "Composite: %{customdata[2]:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#64748B", row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#64748B", row=1, col=1)

    # 2 — Top 15
    fig.add_trace(
        go.Bar(
            x=top15["composite_score"],
            y=top15["ticker"],
            orientation="h",
            text=[f"{x:.3f}" for x in top15["composite_score"]],
            textposition="outside",
            marker=dict(color=[score_color(x) for x in top15["composite_score"]]),
            customdata=np.column_stack([
                top15["name"].fillna(""),
                top15["sector"].fillna(""),
                top15["value_score"],
                top15["quality_score"],
                top15["growth_score"],
            ]),
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata[0]}<br><br>"
                "Sector: %{customdata[1]}<br>Composite: %{x:.3f}<br>"
                "Value: %{customdata[2]:.3f}<br>"
                "Quality: %{customdata[3]:.3f}<br>"
                "Growth: %{customdata[4]:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=2,
    )

    # 3 — Sector averages
    fig.add_trace(
        go.Bar(
            x=sector_avg.values,
            y=sector_avg.index,
            orientation="h",
            text=[f"{x:.3f}" for x in sector_avg.values],
            textposition="outside",
            marker=dict(color=[score_color(x) for x in sector_avg.values]),
            hovertemplate="<b>%{y}</b><br>Average Score: %{x:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # 4 — Score distribution
    fig.add_trace(
        go.Histogram(
            x=df["composite_score"],
            nbinsx=25,
            marker=dict(color="#8B5CF6"),
            hovertemplate="Score: %{x:.3f}<br>Stocks: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=2,
    )
    fig.add_vline(x=avg_score, line_dash="dash", line_color="#F5C542", line_width=2, row=2, col=2)

    # 5 — Factor comparison
    fig.add_trace(
        go.Bar(
            x=list(factor_avgs.keys()),
            y=list(factor_avgs.values()),
            text=[f"{x:.3f}" for x in factor_avgs.values()],
            textposition="outside",
            marker=dict(color="#F59E0B"),
            hovertemplate="<b>%{x}</b><br>Average Score: %{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=3, col=1,
    )

    # 6 — Top 10
    fig.add_trace(
        go.Bar(
            x=top10["composite_score"],
            y=top10["ticker"],
            orientation="h",
            text=[f"{x:.3f}" for x in top10["composite_score"]],
            textposition="outside",
            marker=dict(color=[score_color(x) for x in top10["composite_score"]]),
            customdata=np.column_stack([top10["name"].fillna(""), top10["sector"].fillna("")]),
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata[0]}<br>"
                "Sector: %{customdata[1]}<br>Composite: %{x:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=3, col=2,
    )

    # 7 — Sector distribution
    fig.add_trace(
        go.Bar(
            x=sector_counts.values,
            y=sector_counts.index,
            orientation="h",
            text=sector_counts.values,
            textposition="outside",
            marker=dict(color="#38BDF8"),
            hovertemplate="<b>%{y}</b><br>Stocks: %{x}<extra></extra>",
            showlegend=False,
        ),
        row=4, col=1,
    )

    # 8 — Score vs trailing 12M return
    if "trailing_return_12m" in df.columns:
        returns = df.dropna(subset=["composite_score", "trailing_return_12m"]).copy()
        returns["return_display"] = (returns["trailing_return_12m"] * 100).clip(-100, 300)
        fig.add_trace(
            go.Scatter(
                x=returns["composite_score"],
                y=returns["return_display"],
                mode="markers",
                text=returns["ticker"],
                marker=dict(
                    size=7,
                    color=returns["composite_score"],
                    colorscale="RdYlGn",
                    cmin=min_score,
                    cmax=max_score,
                    opacity=0.7,
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>Composite: %{x:.3f}<br>"
                    "12M Return: %{y:.2f}%<extra></extra>"
                ),
                showlegend=False,
            ),
            row=4, col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#64748B", row=4, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="#64748B", row=4, col=2)

    # Axes
    fig.update_xaxes(title_text="Value Score", row=1, col=1)
    fig.update_yaxes(title_text="Quality Score", row=1, col=1)
    fig.update_xaxes(title_text="Composite Score", row=1, col=2)
    fig.update_xaxes(title_text="Average Composite", row=2, col=1)
    fig.update_xaxes(title_text="Composite Score", row=2, col=2)
    fig.update_yaxes(title_text="Number of Stocks", row=2, col=2)
    fig.update_yaxes(title_text="Average Score", row=3, col=1)
    fig.update_xaxes(title_text="Composite Score", row=3, col=2)
    fig.update_xaxes(title_text="Number of Stocks", row=4, col=1)
    fig.update_xaxes(title_text="Composite Score", row=4, col=2)
    fig.update_yaxes(title_text="12M Return (%)", row=4, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#07101F",
        plot_bgcolor="#0B1628",
        height=1900,
        margin=dict(l=80, r=80, t=230, b=270),
        font=dict(family="Arial", color="#E8EEF7"),
        title=dict(
            text=(
                "<b>Multi-Factor Fundamental Screener</b><br>"
                "<span style='font-size:16px;'>"
                "Full S&P 500 • Value + Quality + Growth</span>"
            ),
            x=0.03,
            xanchor="left",
            y=0.985,
            font=dict(size=30, color="#F8FAFC"),
        ),
        hovermode="closest",
        showlegend=False,
    )

    fig.update_xaxes(
        gridcolor="#1D3049",
        zerolinecolor="#3A506B",
        tickfont=dict(color="#AFC0D8"),
        title_font=dict(color="#BFD0E5"),
    )
    fig.update_yaxes(
        gridcolor="#1D3049",
        zerolinecolor="#3A506B",
        tickfont=dict(color="#AFC0D8"),
        title_font=dict(color="#BFD0E5"),
    )

    for annotation in fig.layout.annotations:
        annotation.font = dict(size=16, color="#F1F5F9")

    ic_text = "—" if pd.isna(ic) else f"{ic:.3f}"
    kpis = [
        (f"<b>{n_stocks}</b><br><span style='font-size:12px'>Stocks Screened</span>", 0.10),
        (f"<b>{top_ticker}</b><br><span style='font-size:12px'>Top Ranked Stock</span>", 0.30),
        (f"<b>{top_score:.3f}</b><br><span style='font-size:12px'>Top Composite Score</span>", 0.50),
        (f"<b>{avg_score:.3f}</b><br><span style='font-size:12px'>Average Composite</span>", 0.70),
        (f"<b>{ic_text}</b><br><span style='font-size:12px'>Information Coefficient</span>", 0.90),
    ]

    for text, x in kpis:
        fig.add_annotation(
            x=x, y=1.10, xref="paper", yref="paper",
            text=text, showarrow=False, align="center",
            bgcolor="#101D33", bordercolor="#29405F",
            borderwidth=1, borderpad=12,
            font=dict(size=22, color="#F8FAFC"),
        )

    explanation = (
        "<b>💡 How to Read Your Screener</b><br><br>"
        "<b>Composite Score:</b> A higher score means the stock ranks more favorably "
        "according to the model.<br><br>"
        "<b>Value vs. Quality:</b> Stocks toward the upper-right score positively on "
        "both dimensions. Larger bubbles represent larger companies.<br><br>"
        "<b>Top Ranked Stocks:</b> Companies with the highest overall composite scores.<br><br>"
        "<b>Factor Comparison:</b> Compares average Value, Quality, and Growth scores.<br><br>"
        "<b>Score vs. 12M Return:</b> Helps investigate whether higher scores have been "
        "associated with stronger trailing returns. Extreme returns are clipped for display "
        "only so the main cluster remains readable.<br><br>"
        "<span style='color:#94A3B8'>Important: this dashboard is a quantitative research "
        "tool and does not guarantee future investment performance.</span>"
    )

    fig.add_annotation(
        x=0.5, y=-0.10, xref="paper", yref="paper",
        text=explanation, showarrow=False, align="left",
        xanchor="center", bgcolor="#0D1A2D",
        bordercolor="#29405F", borderwidth=1, borderpad=18,
        font=dict(size=13, color="#DCE6F5"), width=1100,
    )

    return fig
