import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, State, callback, html, no_update, ctx
import dash_ag_grid as dag

from presentation.layout import get_examples_config
from presentation.charts import build_waterfall_chart, build_delta_bar_chart



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
        return html.P('No changes added yet.', style={'fontSize': '11px', 'color': '#64748b'})

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
                    'fontSize': '11px', 'padding': '6px 10px',
                    'backgroundColor': '#252836', 'borderRadius': '6px',
                    'marginBottom': '6px', 'border': '1px solid #272b3d',
                    'color': '#e2e8f0', 'lineHeight': '1.4'
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
    examples_config = get_examples_config()
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
    Output('graph-waterfall', 'figure'),
    Output('graph-delta-bar', 'figure'),
    Output('table-results', 'rowData'),
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
    State('input-wage-spend', 'value'),
    State('input-surplus-spend', 'value'),
    State('input-tax-spend', 'value'),
    State('input-economy-type', 'value'),
)
def execute_simulation(n_clicks, example_id, iterations, solver, inc_before, inc_after, corp_before, corp_after, ui_tech_changes, total_fd_override, wage_spend, surplus_spend, tax_spend, economy_type):
    iterations = int(iterations) if iterations else 5
    wage_spend = float(wage_spend) if wage_spend is not None else 1.0
    surplus_spend = float(surplus_spend) if surplus_spend is not None else 1.0
    tax_spend = float(tax_spend) if tax_spend is not None else 1.0

    empty_matrix_store = {}
    examples_config = get_examples_config()
    if not example_id or example_id not in examples_config:
        return "-", {}, "-", {}, "-", {}, "-", {}, go.Figure(), go.Figure(), [], [], [], empty_matrix_store
        
    config = examples_config[example_id]
    params = config['params'].copy()
    
    # 1. Pipeline Gatekeeper: Get Pre-Calibrated Model
    from pipeline.setup_data import get_calibrated_model
    from core.io_matrix import IOModel
    from simulators.tech_change import TechnologicalChange
    from simulators.tech_change_loader import build_tech_change_from_spec
    try:
        model, isic_map = get_calibrated_model(demoDB=False, loggingLevel=logging.WARNING)
    except Exception as e:
        print(f"Calibration error: {e}")
        err_fig = go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
        return "Error", {'color': 'red'}, "-", {}, "-", {}, "-", {}, err_fig, go.Figure(), [], [], [], empty_matrix_store

    # 2. Setup Baseline Demand (Payload)
    if 'final_demand' in params and params['final_demand']:
        base_demand = np.array(params['final_demand'], dtype=float)
    else:
        # Fallback to uniform demand for testing
        base_demand = np.full(model.n, params.get('uniform_demand', 1000.0))

    # Parse Proportions Vectors
    def parse_proportions(key):
        if key in params and params[key]:
            val = params[key]
            try:
                if isinstance(val, list):
                    return np.array([float(x) for x in val])
                elif isinstance(val, str):
                    return np.array([float(x) for x in val.split(';')])
            except:
                pass
        return None

    c_props = parse_proportions('wage_proportions')
    i_props = parse_proportions('surplus_proportions')
    g_props = parse_proportions('government_proportions')

    # Apply total FD override
    if total_fd_override is not None and total_fd_override > 0:
        old_total = base_demand.sum()
        if old_total > 0:
            base_demand = base_demand * (total_fd_override / old_total)

    # 3. Setup Initial Demand (Payload)
    shock_demand = base_demand.copy()
    if 'demand_shock' in params and params['demand_shock']:
        for isic_str, shock_val in params['demand_shock'].items():
            if isic_str in isic_map:
                shock_demand[isic_map[isic_str]] += shock_val

    # 4. Simulate Tax Policy (Iterative) - Baseline
    from simulators.tax_simulator import run_tax_simulation
    
    history_before = run_tax_simulation(
        model=model,
        isic_map=isic_map,
        base_demand=shock_demand,
        iterations=iterations,
        income_tax_rate=inc_before,
        corporate_tax_rate=corp_before,
        income_tax_applies_to="wages",
        wage_spend_rate=wage_spend,
        surplus_spend_rate=surplus_spend,
        tax_spend_rate=tax_spend,
        economy_type=economy_type,
        wage_proportions=c_props,
        surplus_proportions=i_props,
        government_proportions=g_props
    )

    # 4b. Apply UI tech changes to build the "after" model
    model_after = model  # default: same model if no tech changes
    A_after_mat = model.A.copy()
    if ui_tech_changes:
        try:
            spec = {
                "name": "UI Tech Change",
                "description": "Changes applied via the GUI",
                "changes": ui_tech_changes,
            }
            tech_change = build_tech_change_from_spec(spec, isic_map)
            A_new, VA_new = tech_change.apply(
                A_matrix=model.A,
                VA_vector=model.VA_coeffs,
                isic_map=isic_map,
            )
            model_after = IOModel(A=A_new, VA_coeffs=VA_new)
            A_after_mat = A_new
        except Exception as e:
            print(f"Tech change application error: {e}")

    # 4c. Simulate Tax Policy (Iterative) - New Policy (using modified model)
    history_after = run_tax_simulation(
        model=model_after,
        isic_map=isic_map,
        base_demand=shock_demand,
        iterations=iterations,
        income_tax_rate=inc_after,
        corporate_tax_rate=corp_after,
        income_tax_applies_to="wages",
        wage_spend_rate=wage_spend,
        surplus_spend_rate=surplus_spend,
        tax_spend_rate=tax_spend,
        economy_type=economy_type,
        wage_proportions=c_props,
        surplus_proportions=i_props,
        government_proportions=g_props
    )
    
    # 5. Extract Before (Equilibrium Baseline) and After (Equilibrium New Policy) states
    first_iter = history_before[-1]
    last_iter = history_after[-1]
    
    X_before = first_iter["X"]
    X_after = last_iter["X"]
    
    # 6. Basic Formatting for UI
    d_X = X_after.sum() - X_before.sum()
    d_FD = last_iter["total_demand"] - first_iter["total_demand"]
    
    # VA calculations from the tax simulator results
    VA_before_vec = sum(first_iter["gross_income"].values())
    VA_after_vec = sum(last_iter["gross_income"].values())
    d_VA = VA_after_vec.sum() - VA_before_vec.sum()
    
    def format_summary(val):
        color = '#22c55e' if val > 0 else ('#ef4444' if val < 0 else '#64748b')
        sign = '+' if val > 0 else ''
        return f"{sign}${val:,.2f}", {
            'color': color,
            'margin': 0,
            'fontSize': '24px',
            'fontWeight': '700',
            'lineHeight': '1',
        }
        
    out_X, style_X = format_summary(d_X)
    out_VA, style_VA = format_summary(d_VA)
    out_FD, style_FD = format_summary(d_FD)
    
    tax_before = first_iter["tax_results"]["total_tax"].sum()
    tax_after = last_iter["tax_results"]["total_tax"].sum()
    out_Tax, style_Tax = format_summary(tax_after - tax_before)
    
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
        'FD_Before': first_iter["Y"],
        'FD_After': last_iter["Y"],
        'FD_Delta': last_iter["Y"] - first_iter["Y"],
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
    
    fig_waterfall = build_waterfall_chart(X_before.sum(), X_after.sum())
    fig_delta     = build_delta_bar_chart(df)

    # (placeholder tables removed — they were always empty)

    demand_vector_data = [{'Sector': s, 'Demand': round(float(d), 4)} for s, d in zip(sectors, first_iter["Y"])]
    
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
    Z_after_mat  = A_after_mat * X_after[np.newaxis, :]

    matrix_store = {
        'sectors': sectors,
        'A_before': _mat_to_list(model.A),
        'A_after':  _mat_to_list(A_after_mat),
        'L_before': _mat_to_list(model.L),
        'L_after':  _mat_to_list(model_after.L),
        'Z_before': _mat_to_list(Z_before_mat),
        'Z_after':  _mat_to_list(Z_after_mat),
    }

    return (out_X, style_X, out_VA, style_VA, out_FD, style_FD, out_Tax, style_Tax,
            fig_waterfall, fig_delta,
            row_data, demand_vector_data, va_coeff_data, matrix_store)


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
    _DARK_BG    = '#16192a'
    _PAPER_BG   = '#1e2130'
    _TEXT_COLOR = '#e2e8f0'
    _MUTED      = '#64748b'
    _GRID_C     = '#272b3d'
    _ACCENT     = '#4f8ef7'

    if not store or not store.get('sectors'):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor=_PAPER_BG, plot_bgcolor=_DARK_BG,
            font=dict(color=_TEXT_COLOR),
            xaxis=dict(visible=False), yaxis=dict(visible=False)
        )
        return empty_fig, [], [], "No simulation data"

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
    
    if is_delta:
        # Diverging colorscale: Red (negative) -> Dark Background (zero) -> Blue (positive)
        colorscale = [[0.0, '#ef4444'], [0.5, _DARK_BG], [1.0, _ACCENT]]
    else:
        # Sequential colorscale: Dark Background -> Blue (positive)
        colorscale = [[0.0, _DARK_BG], [1.0, _ACCENT]]
        
    zmid = 0 if is_delta else None

    heatmap_fig = go.Figure(go.Heatmap(
        z=matrix,
        x=short_labels,
        y=short_labels,
        colorscale=colorscale,
        zmid=zmid,
        colorbar=dict(title=dict(text='Value', font=dict(color=_TEXT_COLOR)), tickfont=dict(color=_TEXT_COLOR)),
        hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z:.4f}<extra></extra>',
        text=[[f'{v:.3f}' for v in row] for row in matrix],
        texttemplate='%{text}',
        textfont=dict(size=9, color=_TEXT_COLOR),
    ))
    heatmap_fig.update_layout(
        title=dict(text=title, font=dict(color=_TEXT_COLOR, size=13)),
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT_COLOR, family='Inter, system-ui, sans-serif'),
        xaxis=dict(title=dict(text='Column (buying sector)', font=dict(color=_MUTED)), tickangle=-45, tickfont=dict(size=10, color=_MUTED),
                   gridcolor=_GRID_C, linecolor=_GRID_C),
        yaxis=dict(title=dict(text='Row (selling sector)', font=dict(color=_MUTED)), tickfont=dict(size=10, color=_MUTED),
                   autorange='reversed', gridcolor=_GRID_C, linecolor=_GRID_C),
        height=480,
        margin=dict(l=100, r=40, t=60, b=120),
    )

    # ---- AgGrid table ----
    col_defs = [{"field": "Sector", "pinned": "left", "width": 160, "sortable": False}]
    delta_cell_style = {
        "function": (
            "params.value < -0.0001 ? {'color':'#ef4444','fontWeight':'600'} : "
            "params.value > 0.0001 ? {'color':'#4f8ef7'} : ({})"
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



from pipeline.setup_data import setup_data

@callback(
    Output('url', 'href'),
    Output('load-source-status', 'children'),
    Input('load-source-btn', 'n_clicks'),
    State('data-source-selector', 'value'),
    prevent_initial_call=True
)
def load_new_data_source(n_clicks, source_folder):
    if not source_folder:
        return no_update, "Please select a folder."
    try:
        setup_data(source=source_folder, overwrite_existing_data=True)
        return "/", ""
    except Exception as e:
        return no_update, f"Error: {e}"

