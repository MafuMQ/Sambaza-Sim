"""
Chart builders for the Sambaza-Sim dashboard.
All charts use a shared dark-theme palette.
"""

import plotly.graph_objects as go
import numpy as np

# ── Design tokens (must match layout.py) ──────────────────────────────────
_BG_PLOT   = '#16192a'
_BG_PAPER  = '#252836'
_TEXT      = '#e2e8f0'
_MUTED     = '#64748b'
_GRID      = '#2d3142'
_ACCENT    = '#4f8ef7'
_GREEN     = '#22c55e'
_RED       = '#ef4444'
_FONT      = 'Inter, system-ui, sans-serif'

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG_PAPER,
    plot_bgcolor=_BG_PLOT,
    font=dict(color=_TEXT, family=_FONT, size=12),
    margin=dict(l=16, r=16, t=44, b=16),
)


def build_waterfall_chart(before_total: float, after_total: float) -> go.Figure:
    """
    Waterfall chart: Total Output Before → Δ → After.

    Parameters
    ----------
    before_total : float
        Sum of sector outputs in the baseline scenario.
    after_total : float
        Sum of sector outputs in the policy/shock scenario.
    """
    delta = after_total - before_total
    sign  = '+' if delta >= 0 else ''

    fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=['absolute', 'relative', 'total'],
        x=['Before', f'Δ  {sign}${delta:,.0f}', 'After'],
        y=[before_total, delta, 0],
        connector={'line': {'color': _GRID, 'width': 1}},
        increasing={'marker': {'color': _GREEN, 'line': {'color': _GREEN, 'width': 0}}},
        decreasing={'marker': {'color': _RED,   'line': {'color': _RED,   'width': 0}}},
        totals    ={'marker': {'color': _ACCENT, 'line': {'color': _ACCENT,'width': 0}}},
        # Show value on absolute/relative bars; total bar gets an annotation below
        text=[f'${before_total:,.0f}', f'{sign}${delta:,.0f}', f'${after_total:,.0f}'],
        textposition='outside',
        textfont=dict(size=12, color=_TEXT),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text='Total Output — Before vs After', font=dict(size=13, color=_TEXT)),
        showlegend=False,
        height=300,
        xaxis=dict(showgrid=False, linecolor=_GRID, tickfont=dict(color=_TEXT)),
        yaxis=dict(
            showgrid=True, gridcolor=_GRID, zeroline=False,
            tickprefix='$', tickformat=',.0f', tickfont=dict(color=_MUTED),
        ),
    )
    return fig


def build_delta_bar_chart(df) -> go.Figure:
    """
    Horizontal bar chart showing Δ Output per sector, sorted by magnitude.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: 'Sector', 'Output_Delta'.
    """
    df_sorted = df.sort_values('Output_Delta', ascending=True)
    values    = df_sorted['Output_Delta'].tolist()
    labels    = df_sorted['Sector'].tolist()
    colors    = [_GREEN if v >= 0 else _RED for v in values]
    text      = [f'+${v:,.0f}' if v >= 0 else f'−${abs(v):,.0f}' for v in values]

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=text,
        textposition='outside',
        textfont=dict(size=11, color=_TEXT),
        cliponaxis=False,
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text='Output Change by Sector (Δ)', font=dict(size=13, color=_TEXT)),
        height=max(240, 68 * len(df_sorted)),
        xaxis=dict(
            showgrid=True, gridcolor=_GRID,
            zeroline=True, zerolinecolor='#4a4f6a', zerolinewidth=1,
            tickprefix='$', tickformat=',.0f', tickfont=dict(color=_MUTED),
        ),
        yaxis=dict(showgrid=False, linecolor=_GRID, tickfont=dict(color=_TEXT, size=11)),
    )
    return fig
