import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, State, callback, html, no_update, ctx
import dash_ag_grid as dag

from presentation.layout import examples_config, sector_options
from presentation.charts import build_output_chart, build_va_chart



# ---------------------------------------------------------------------------
# Tech-change builder: show/hide fields based on change type
# ---------------------------------------------------------------------------

@callback(
    Output('tc-sector-container', 'style'),
    Output('tc-input-sector-container', 'style'),
    Input('tc-change-type', 'value'),
)
def toggle_tc_fields(change_type):
    """Show/hide sector dropdowns depending on the selected change type."""
    show = {'display': 'block'}
    hide = {'display': 'none'}
    if change_type == 'add_sector_change':
        return show, hide
    elif change_type == 'add_input_change':
        return hide, show
    else:  # add_coefficient_change
        return show, show


# ---------------------------------------------------------------------------
# Tech-change builder: add / clear operations in the store
# ---------------------------------------------------------------------------

@callback(
    Output('store-tech-changes', 'data'),
    Input('tc-add-button', 'n_clicks'),
    Input('tc-clear-button', 'n_clicks'),
    State('store-tech-changes', 'data'),
    State('tc-change-type', 'value'),
    State('tc-sector', 'value'),
    State('tc-input-sector', 'value'),
    State('tc-operation', 'value'),
    State('tc-value', 'value'),
    prevent_initial_call=True,
)
def manage_tech_changes(add_clicks, clear_clicks, current_changes, change_type, sector, input_sector, operation, value):
    triggered = ctx.triggered_id
    if triggered == 'tc-clear-button':
        return []

    # Validate inputs for the add action
    if value is None:
        return current_changes or []

    entry = {
        'method': change_type,
        'params': {
            'change_type': operation,
            'value': float(value),
        }
    }

    if change_type == 'add_sector_change':
        if not sector:
            return current_changes or []
        entry['params']['sector_idx'] = sector
    elif change_type == 'add_input_change':
        if not input_sector:
            return current_changes or []
        entry['params']['input_sector_idx'] = input_sector
    elif change_type == 'add_coefficient_change':
        if not sector or not input_sector:
            return current_changes or []
        entry['params']['sector_idx'] = sector
        entry['params']['input_sector_idx'] = input_sector

    changes = list(current_changes or [])
    changes.append(entry)
    return changes


# ---------------------------------------------------------------------------
# Tech-change builder: render the list of pending changes
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    'add_sector_change': 'All inputs of',
    'add_input_change': 'Usage of',
    'add_coefficient_change': 'Coefficient',
}

@callback(
    Output('tc-changes-display', 'children'),
    Input('store-tech-changes', 'data'),
)
def render_tc_changes(changes):
    if not changes:
        return html.P('No changes added yet.', style={'fontSize': '0.85em', 'color': '#95a5a6'})

    items = []
    for i, ch in enumerate(changes):
        method = ch['method']
        p = ch['params']
        op = p.get('change_type', '?')
        val = p.get('value', '?')

        if method == 'add_sector_change':
            desc = f"{METHOD_LABELS[method]} sector {p.get('sector_idx', '?')}: {op} {val}"
        elif method == 'add_input_change':
            desc = f"{METHOD_LABELS[method]} input {p.get('input_sector_idx', '?')} everywhere: {op} {val}"
        else:
            desc = f"{METHOD_LABELS[method]} [{p.get('input_sector_idx', '?')} → {p.get('sector_idx', '?')}]: {op} {val}"

        items.append(
            html.Div(
                f"#{i+1}  {desc}",
                style={
                    'fontSize': '0.8em', 'padding': '4px 8px',
                    'backgroundColor': '#eaf4fb', 'borderRadius': '4px',
                    'marginBottom': '4px', 'border': '1px solid #aed6f1',
                }
            )
        )
    return html.Div(items)


@callback(
    Output('example-description', 'children'),
    Output('input-total-fd', 'value'),
    Output('input-total-fd', 'disabled'),
    Output('input-total-fd-hint', 'children'),
    Output('input-solver', 'value'),
    Output('input-income-tax-before', 'value'),
    Output('input-income-tax-after', 'value'),
    Output('input-corp-tax-before', 'value'),
    Output('input-corp-tax-after', 'value'),
    Input('example-selector', 'value')
)
def update_controls(example_id):
    if not example_id or example_id not in examples_config:
        return "", None, True, "", 'supply_curves', 0.0, 0.0, 0.0, 0.0
    
    config = examples_config[example_id]
    desc_lines = config.get('description', [])
    if isinstance(desc_lines, list):
        desc_elements = []
        for line in desc_lines:
            desc_elements.append(line)
            desc_elements.append(html.Br())
        desc = html.P(desc_elements[:-1]) if desc_elements else html.P()
    else:
        desc = desc_lines
    
    params = config.get('params', {})
    solver = params.get('solver_type', 'supply_curves')
    inc_before = params.get('income_tax_rate_before', 0.0)
    inc_after = params.get('income_tax_rate_after', inc_before)
    corp_before = params.get('corporate_tax_rate_before', 0.0)
    corp_after = params.get('corporate_tax_rate_after', corp_before)
    
    fd_raw = params.get('final_demand', None)
    if fd_raw is not None:
        fd_total = round(float(np.array(fd_raw, dtype=float).sum()), 2)
        fd_disabled = False
        fd_hint = "Sector demands will be scaled proportionally to match this total."
    else:
        fd_total = None
        fd_disabled = True
        fd_hint = "FD customization is available for vector-based demand scenarios only."
    
    return desc, fd_total, fd_disabled, fd_hint, solver, inc_before, inc_after, corp_before, corp_after

@callback(
    Output('summary-output', 'children'),
    Output('summary-output', 'style'),
    Output('summary-va', 'children'),
    Output('summary-va', 'style'),
    Output('summary-fd', 'children'),
    Output('summary-fd', 'style'),
    Output('summary-tax', 'children'),
    Output('summary-tax', 'style'),
    Output('graph-va-comparison', 'figure'),
    Output('graph-output-comparison', 'figure'),
    Output('table-results', 'rowData'),
    Output('table-va-components', 'rowData'),
    Output('table-fd-components', 'rowData'),
    Output('table-output-proportions', 'rowData'),
    Output('iteration-comparison-div', 'children'),
    Output('table-demand-vector', 'rowData'),
    Output('table-va-coefficients', 'rowData'),
    Output('store-matrices', 'data'),
    Input('run-button', 'n_clicks'),
    State('example-selector', 'value'),
    State('input-iterations', 'value'),
    State('input-solver', 'value'),
    State('input-income-tax-before', 'value'),
    State('input-income-tax-after', 'value'),
    State('input-corp-tax-before', 'value'),
    State('input-corp-tax-after', 'value'),
    State('store-tech-changes', 'data'),
    State('input-total-fd', 'value'),
)
def execute_simulation(n_clicks, example_id, iterations, solver, inc_before, inc_after, corp_before, corp_after, ui_tech_changes, total_fd_override):
    empty_matrix_store = {}
    if not example_id or example_id not in examples_config:
        return "-", {}, "-", {}, "-", {}, "-", {}, go.Figure(), go.Figure(), [], [], [], [], html.Div(), [], [], empty_matrix_store
        
    config = examples_config[example_id]
    params = config['params'].copy()
    
    # 1. Pipeline Gatekeeper: Get Pre-Calibrated Model
    from pipeline.setup_data import get_calibrated_model
    try:
        model, isic_map = get_calibrated_model(demoDB=False, loggingLevel=logging.WARNING)
    except Exception as e:
        print(f"Calibration error: {e}")
        err_fig = go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
        return "Error", {'color': 'red'}, "-", {}, "-", {}, "-", {}, err_fig, go.Figure(), [], [], [], [], html.Div(), [], [], empty_matrix_store

    # 2. Setup Baseline Demand (Payload)
    if 'final_demand' in params and params['final_demand']:
        base_demand = np.array(params['final_demand'], dtype=float)
    else:
        # Fallback to uniform demand for testing
        base_demand = np.full(model.n, params.get('uniform_demand', 1000.0))

    # Apply total FD override
    if total_fd_override is not None and total_fd_override > 0:
        old_total = base_demand.sum()
        if old_total > 0:
            base_demand = base_demand * (total_fd_override / old_total)

    # 3. Simulate Baseline Scenario (Phase 2)
    X_before = model.simulate(base_demand)
    
    # 4. Setup Shock Demand (Payload)
    # For now, we simulate a simple demand shock if provided, otherwise keep it same.
    # We will "add the iterative circular flow scenarios later" as per user feedback.
    shock_demand = base_demand.copy()
    if 'demand_shock' in params and params['demand_shock']:
        for isic_str, shock_val in params['demand_shock'].items():
            if isic_str in isic_map:
                shock_demand[isic_map[isic_str]] += shock_val

    # 5. Simulate Shock Scenario (Phase 2)
    X_after = model.simulate(shock_demand)
    
    # 6. Basic Formatting for UI (Detached from raw DB data)
    d_X = (X_after - X_before).sum()
    d_FD = shock_demand.sum() - base_demand.sum()
    
    # VA calculations using simple coefficients from the model
    VA_before_vec = model.VA_coeffs * X_before
    VA_after_vec = model.VA_coeffs * X_after
    d_VA = VA_after_vec.sum() - VA_before_vec.sum()
    
    def format_summary(val):
        color = '#27ae60' if val > 0 else ('#e74c3c' if val < 0 else '#7f8c8d')
        sign = '+' if val > 0 else ''
        return f"{sign}${val:,.2f}", {'color': color, 'margin': 0}
        
    out_X, style_X = format_summary(d_X)
    out_VA, style_VA = format_summary(d_VA)
    out_FD, style_FD = format_summary(d_FD)
    out_Tax, style_Tax = format_summary(0) # Tax deferred
    
    # Sector names for tables
    idx_map = {idx: isic for isic, idx in isic_map.items()}
    sectors = [idx_map.get(i, f"Sector {i}") for i in range(model.n)]
    
    df = pd.DataFrame({
        'Sector': sectors,
        'Output_Before': X_before,
        'Output_After': X_after,
        'Output_Delta': X_after - X_before,
        'VA_Before': VA_before_vec,
        'VA_After': VA_after_vec,
        'VA_Delta': VA_after_vec - VA_before_vec,
        'FD_Before': base_demand,
        'FD_After': shock_demand,
        'FD_Delta': shock_demand - base_demand,
        # Placeholders for deferred complex variables
        'VA_Output_Before': np.zeros(model.n),
        'VA_Output_After': np.zeros(model.n),
        'VA_Output_Delta': np.zeros(model.n),
        'Int_Output_Before': np.zeros(model.n),
        'Int_Output_After': np.zeros(model.n),
        'Int_Output_Delta': np.zeros(model.n)
    })
    
    df = df.sort_values(by='Output_Before', ascending=False)
    row_data = df.to_dict('records')
    
    fig_out = build_output_chart(df, 1)
    fig_va = build_va_chart(df, 1)

    # Deferred component tables returned as empty
    va_components_data = []
    fd_components_data = []
    output_proportions_data = []
    iteration_comparison_div = html.Div("Iterative circular flow tracking deferred.")

    demand_vector_data = [{'Sector': s, 'Demand': round(float(d), 4)} for s, d in zip(sectors, base_demand)]
    
    va_coeff_data = []
    for i, s in enumerate(sectors):
        va_coeff_data.append({
            'Sector': s,
            'VA_Before': round(float(model.VA_coeffs[i]), 4),
            'VA_After':  round(float(model.VA_coeffs[i]), 4),
            'VA_Delta':  0.0,
        })

    def _mat_to_list(m):
        return m.tolist() if m is not None else None

    # Z = A * diag(X)
    Z_before_mat = model.A * X_before[np.newaxis, :]
    Z_after_mat  = model.A * X_after[np.newaxis, :]

    matrix_store = {
        'sectors': sectors,
        'A_before': _mat_to_list(model.A),
        'A_after':  _mat_to_list(model.A), # Tech change deferred
        'L_before': _mat_to_list(model.L),
        'L_after':  _mat_to_list(model.L),
        'Z_before': _mat_to_list(Z_before_mat),
        'Z_after':  _mat_to_list(Z_after_mat),
    }

    return (out_X, style_X, out_VA, style_VA, out_FD, style_FD, out_Tax, style_Tax,
            fig_va, fig_out,
            row_data, va_components_data, fd_components_data, output_proportions_data,
            iteration_comparison_div,
            demand_vector_data, va_coeff_data, matrix_store)


# ---------------------------------------------------------------------------
# Matrix visualisation callback
# ---------------------------------------------------------------------------

@callback(
    Output('graph-matrix-heatmap', 'figure'),
    Output('table-matrix', 'rowData'),
    Output('table-matrix', 'columnDefs'),
    Output('matrix-table-title', 'children'),
    Input('matrix-type-selector', 'value'),
    State('store-matrices', 'data'),
)
def update_matrix_display(matrix_key, store):
    if not store or not store.get('sectors'):
        return go.Figure(), [], [], "No simulation data"

    sectors = store['sectors']
    n = len(sectors)

    # Retrieve chosen matrix
    A_before = np.array(store['A_before']) if store.get('A_before') else None
    A_after  = np.array(store['A_after'])  if store.get('A_after')  else None
    L_before = np.array(store['L_before']) if store.get('L_before') else None
    L_after  = np.array(store['L_after'])  if store.get('L_after')  else None
    Z_before = np.array(store['Z_before']) if store.get('Z_before') else None
    Z_after  = np.array(store['Z_after'])  if store.get('Z_after')  else None

    label_map = {
        'A_before': ('Technical Coefficients A — Before', A_before),
        'A_after':  ('Technical Coefficients A — After',  A_after),
        'delta_A':  ('Change in Coefficients ΔA',
                     (A_after - A_before) if (A_before is not None and A_after is not None) else None),
        'L_before': ('Leontief Inverse L — Before', L_before),
        'L_after':  ('Leontief Inverse L — After',  L_after),
        'delta_L':  ('Change in Leontief ΔL',
                     (L_after - L_before) if (L_before is not None and L_after is not None) else None),
        'Z_before': ('Flow Table Z — Before ($)',   Z_before),
        'Z_after':  ('Flow Table Z — After ($)',    Z_after),
        'delta_Z':  ('Change in Flow Table ΔZ ($)',
                     (Z_after - Z_before) if (Z_before is not None and Z_after is not None) else None),
    }

    title, matrix = label_map.get(matrix_key, ('Unknown', None))

    if matrix is None:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Matrix not available", showarrow=False, font=dict(size=16))
        return empty_fig, [], [], title

    # Use short labels (strip ISIC prefix noise if long)
    short_labels = [s[:15] if len(s) > 15 else s for s in sectors]

    # ---- Heatmap ----
    is_delta = matrix_key.startswith('delta_')
    colorscale = 'RdBu_r' if is_delta else 'Blues'
    zmid = 0 if is_delta else None

    heatmap_fig = go.Figure(go.Heatmap(
        z=matrix,
        x=short_labels,
        y=short_labels,
        colorscale=colorscale,
        zmid=zmid,
        colorbar=dict(title='Value'),
        hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z:.4f}<extra></extra>',
        text=[[f'{v:.3f}' for v in row] for row in matrix],
        texttemplate='%{text}',
        textfont=dict(size=9),
    ))
    heatmap_fig.update_layout(
        title=title,
        xaxis=dict(title='Column (buying sector)', tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(title='Row (selling sector)', tickfont=dict(size=10), autorange='reversed'),
        height=500,
        margin=dict(l=100, r=40, t=60, b=120),
    )

    # ---- AgGrid table ----
    col_defs = [{"field": "Sector", "pinned": "left", "width": 160, "sortable": False}]
    delta_cell_style = {
        "function": (
            "params.value < -0.0001 ? {'color':'#e74c3c','fontWeight':'bold'} : "
            "params.value > 0.0001 ? {'color':'#2980b9'} : ({})"
        )
    }
    for lbl in short_labels:
        col = {
            "field": lbl,
            "width": 90,
            "sortable": False,
            "valueFormatter": {"function": "d3.format(',.4f')(params.value)"},
        }
        if is_delta:
            col["cellStyle"] = delta_cell_style
        col_defs.append(col)

    row_data = []
    for i, row_label in enumerate(short_labels):
        row = {"Sector": sectors[i]}
        for j, col_label in enumerate(short_labels):
            row[col_label] = round(float(matrix[i][j]), 4)
        row_data.append(row)

    return heatmap_fig, row_data, col_defs, title


