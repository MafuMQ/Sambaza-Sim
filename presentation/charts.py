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


def build_ledger_trajectory_chart(
    iterations: list,
    balances: list,
    deposits: list = None,
    reserved_amount: float = 0.0,
) -> go.Figure:
    """
    Line/area chart tracking Savings Ledger balance progression across iterations.
    """
    fig = go.Figure()

    # Area + line for running balance
    fig.add_trace(go.Scatter(
        x=iterations,
        y=balances,
        mode='lines+markers',
        name='Ledger Balance',
        line=dict(color=_ACCENT, width=3),
        marker=dict(size=7, color=_ACCENT, line=dict(width=1.5, color='#ffffff')),
        fill='tozeroy',
        fillcolor='rgba(79, 142, 247, 0.12)',
        hovertemplate='<b>%{x}</b><br>Balance: $%{y:,.2f}<extra></extra>',
    ))

    # If deposits provided, add as a bar trace on secondary or subtle trace
    if deposits and any(d > 0 for d in deposits):
        fig.add_trace(go.Bar(
            x=iterations,
            y=deposits,
            name='Period Savings Inflow',
            marker=dict(color='rgba(34, 197, 94, 0.45)', line=dict(color=_GREEN, width=1)),
            hovertemplate='<b>%{x}</b><br>Savings Inflow: +$%{y:,.2f}<extra></extra>',
        ))

    # Annotation if capital was reserved
    if reserved_amount > 0 and len(balances) > 0:
        fig.add_annotation(
            x=iterations[0],
            y=balances[0],
            text=f"Reserved: -${reserved_amount:,.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=_RED,
            arrowsize=1,
            arrowwidth=1.5,
            ax=45,
            ay=-35,
            font=dict(size=10, color=_RED),
            bgcolor=_BG_PAPER,
            bordercolor=_RED,
            borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text='Savings Ledger Balance Trajectory', font=dict(size=13, color=_TEXT)),
        height=320,
        barmode='relative',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(size=11, color=_MUTED),
            bgcolor='rgba(0,0,0,0)',
        ),
        xaxis=dict(showgrid=True, gridcolor=_GRID, linecolor=_GRID, tickfont=dict(color=_TEXT)),
        yaxis=dict(
            showgrid=True, gridcolor=_GRID, zeroline=True, zerolinecolor='#4a4f6a',
            tickprefix='$', tickformat=',.0f', tickfont=dict(color=_MUTED),
        ),
    )
    return fig


def build_savings_breakdown_chart(
    initial_injection: float,
    wage_savings: float,
    surplus_savings: float,
    reserved_capital: float,
    ending_balance: float,
) -> go.Figure:
    """
    Waterfall / breakdown chart comparing initial injection, period savings inflows,
    capital reservations, and final balance.
    """
    labels = ['Initial Balance', 'Wage Savings', 'Surplus Savings', 'Capital Reserved', 'Ending Balance']
    values = [initial_injection, wage_savings, surplus_savings, -reserved_capital, ending_balance]
    measures = ['absolute', 'relative', 'relative', 'relative', 'total']

    fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=measures,
        x=labels,
        y=[initial_injection, wage_savings, surplus_savings, -reserved_capital, 0],
        connector={'line': {'color': _GRID, 'width': 1}},
        increasing={'marker': {'color': _GREEN}},
        decreasing={'marker': {'color': _RED}},
        totals={'marker': {'color': _ACCENT}},
        text=[f'${v:,.2f}' if v >= 0 else f'-${abs(v):,.2f}' for v in values],
        textposition='outside',
        textfont=dict(size=11, color=_TEXT),
        cliponaxis=False,
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text='Savings Ledger Cash Flow Breakdown', font=dict(size=13, color=_TEXT)),
        showlegend=False,
        height=320,
        xaxis=dict(showgrid=False, linecolor=_GRID, tickfont=dict(color=_TEXT, size=11)),
        yaxis=dict(
            showgrid=True, gridcolor=_GRID, zeroline=False,
            tickprefix='$', tickformat=',.0f', tickfont=dict(color=_MUTED),
        ),
    )
    return fig

