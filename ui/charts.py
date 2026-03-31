import plotly.graph_objects as go
import pandas as pd

def build_output_chart(df: pd.DataFrame, actual_iterations: int) -> go.Figure:
    top_df_out = df.head(15).sort_values(by='Output_Before', ascending=True)
    fig_out = go.Figure()
    # After added first so Before (last) renders on top in grouped horizontal bars
    fig_out.add_trace(go.Bar(
        y=top_df_out['Sector'],
        x=top_df_out['Output_After'],
        name='After',
        orientation='h',
        marker_color='#2ecc71',
        legendrank=2
    ))
    fig_out.add_trace(go.Bar(
        y=top_df_out['Sector'],
        x=top_df_out['Output_Before'],
        name='Before',
        orientation='h',
        marker_color='#3498db',
        legendrank=1
    ))
    fig_out.update_layout(
        title=f'Gross Output Comparison - Top 15 Sectors ({actual_iterations} iteration{"s" if actual_iterations != 1 else ""})',
        barmode='group',
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig_out

def build_va_chart(df: pd.DataFrame, actual_iterations: int) -> go.Figure:
    top_df_va = df.head(15).sort_values(by='VA_Before', ascending=True)
    fig_va = go.Figure()
    # After added first so Before (last) renders on top in grouped horizontal bars
    fig_va.add_trace(go.Bar(
        y=top_df_va['Sector'],
        x=top_df_va['VA_After'],
        name='After',
        orientation='h',
        marker_color='#e67e22',
        legendrank=2
    ))
    fig_va.add_trace(go.Bar(
        y=top_df_va['Sector'],
        x=top_df_va['VA_Before'],
        name='Before',
        orientation='h',
        marker_color='#9b59b6',
        legendrank=1
    ))
    fig_va.update_layout(
        title=f'Value Added Comparison - Top 15 Sectors ({actual_iterations} iteration{"s" if actual_iterations != 1 else ""})',
        barmode='group',
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig_va
