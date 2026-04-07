import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, State, callback, html, no_update, ctx
import dash_ag_grid as dag

from ui.layout import examples_config, sector_options
from ui.charts import build_output_chart, build_va_chart
from core.simulation import run_simulation


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
    
    # Apply total FD override — rescale the demand vector proportionally
    if total_fd_override is not None and total_fd_override > 0 and 'final_demand' in params:
        old_fd = np.array(params['final_demand'], dtype=float)
        old_total = old_fd.sum()
        if old_total > 0:
            params['final_demand'] = (old_fd * (total_fd_override / old_total)).tolist()
    
    # Check if this is a tech comparison
    is_tech_comparison = params.get('is_tech_comparison', False)
    
    # Override params with UI inputs
    params['iterations'] = iterations
    params['solver_type'] = solver
    params['income_tax_rate_before'] = inc_before
    params['income_tax_rate_after'] = inc_after
    params['corporate_tax_rate_before'] = corp_before
    params['corporate_tax_rate_after'] = corp_after
    
    # Ensure some demand parameter exists to prevent simulation error
    demand_keys = ['final_demand', 'demand_vector', 'target_isic', 'demand_shock', 'uniform_demand', 'total_demand']
    if not any(k in params for k in demand_keys):
        params['uniform_demand'] = 1000.0
        
    # Map consumption_distribution to demand_distribution if present
    if 'consumption_distribution' in params:
        params.setdefault('demand_distribution', params.pop('consumption_distribution'))
        
    params.setdefault('before_name', 'Before Change')
    params.setdefault('after_name', 'After Change')
    
    # Build TechnologicalChange from UI-defined operations (if any)
    ui_tech_change = None
    if ui_tech_changes:
        from pipeline.load_tech_changes import build_tech_change_from_spec
        spec = {
            "name": "Custom UI Tech Change",
            "description": "User-defined technology changes from the dashboard",
            "changes": ui_tech_changes,
        }
        # isic_map not needed yet; build_tech_change_from_spec resolves ISIC codes
        # via the method calls which accept string sector_idx directly
        ui_tech_change = build_tech_change_from_spec(spec, {})
    
    # Build inputs for simulation
    # If tech change comparison, we need to extract tech_change_builder
    try:
        if ui_tech_change is not None:
            # User built custom tech changes in the UI — apply them
            from core.io_matrix import build_io_matrix
            A_baseline, VA_baseline, isic_map_curr = build_io_matrix(demoDB=False, loggingLevel=logging.WARNING)

            # If the selected example also has a tech change, apply it first
            combined_tech = ui_tech_change
            if is_tech_comparison:
                tech_config = params['tech_change_config']
                example_tc = tech_config['tech_change_builder'](isic_map_curr)
                # Merge: apply example changes first, then UI changes on top
                combined_tech.matrix_changes = example_tc.matrix_changes + combined_tech.matrix_changes

            A_changed, VA_changed = combined_tech.apply(A_baseline, VA_baseline, isic_map_curr)

            final_demand = np.array(params['final_demand'], dtype=float) if params.get('final_demand') else None

            res = run_simulation(
                final_demand=final_demand,
                uniform_demand=params.get('uniform_demand') if final_demand is None else None,
                A_before=A_baseline,
                A_after=A_changed,
                VA_before=VA_baseline,
                VA_after=VA_changed,
                isic_map=isic_map_curr,
                before_name="Baseline",
                after_name="After Tech Change",
                iterations=iterations,
                demand_distribution=params.get('demand_distribution', 'proportional'),
                income_tax_rate_before=inc_before,
                income_tax_rate_after=inc_after,
                corporate_tax_rate_before=corp_before,
                corporate_tax_rate_after=corp_after,
                income_tax_applies_to=params.get('income_tax_applies_to', 'bonusWages'),
                consumption_proportions=params.get('consumption_proportions', None),
                investment_proportions=params.get('investment_proportions', None),
                government_proportions=params.get('government_proportions', None),
                consumption_rate=params.get('consumption_rate', 1.0),
                solver_type=solver,
            )
        elif is_tech_comparison:
            final_demand = np.array(params['final_demand'], dtype=float)
            tech_config = params['tech_change_config']
            use_multi_level = params.get('use_multi_level', False)
            
            tech_change = tech_config['tech_change_builder']({}) # Need to re-instantiate or run with multi-level handler
            
            if use_multi_level:
                res = run_simulation(
                    final_demand=final_demand,
                    tech_change=tech_change,
                    before_name="Baseline Technology",
                    after_name=tech_config['name'],
                    iterations=iterations,
                    demand_distribution=params.get('demand_distribution', 'proportional'),
                    income_tax_rate_before=inc_before,
                    income_tax_rate_after=inc_after,
                    corporate_tax_rate_before=corp_before,
                    corporate_tax_rate_after=corp_after,
                    income_tax_applies_to=params.get('income_tax_applies_to', 'bonusWages'),
                    consumption_proportions=params.get('consumption_proportions', None),
                    investment_proportions=params.get('investment_proportions', None),
                    government_proportions=params.get('government_proportions', None),
                    consumption_rate=params.get('consumption_rate', 1.0),
                    solver_type=solver,
                )
            else:
                from core.io_matrix import build_io_matrix
                A_baseline, VA_baseline, isic_map_curr = build_io_matrix(demoDB=False, loggingLevel=logging.WARNING)
                tech_change = tech_config['tech_change_builder'](isic_map_curr)
                A_changed, VA_changed = tech_change.apply(A_baseline, VA_baseline, isic_map_curr)
                
                res = run_simulation(
                    final_demand=final_demand,
                    A_before=A_baseline,
                    A_after=A_changed,
                    VA_before=VA_baseline,
                    VA_after=VA_changed,
                    isic_map=isic_map_curr,
                    before_name="Baseline Technology",
                    after_name=tech_config['name'],
                    iterations=iterations,
                    demand_distribution=params.get('demand_distribution', 'proportional'),
                    income_tax_rate_before=inc_before,
                    income_tax_rate_after=inc_after,
                    corporate_tax_rate_before=corp_before,
                    corporate_tax_rate_after=corp_after,
                    income_tax_applies_to=params.get('income_tax_applies_to', 'bonusWages'),
                    consumption_proportions=params.get('consumption_proportions', None),
                    investment_proportions=params.get('investment_proportions', None),
                    government_proportions=params.get('government_proportions', None),
                    consumption_rate=params.get('consumption_rate', 1.0),
                    solver_type=solver,
                )
        else:
            res = run_simulation(**params)
            
    except Exception as e:
        print(f"Simulation error: {e}")
        err_fig = go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
        return f"Error", {'color': 'red'}, f"Error", {'color': 'red'}, f"Error", {'color': 'red'}, f"Error", {'color': 'red'}, err_fig, go.Figure(), [], [], [], [], html.Div(), [], [], empty_matrix_store

    if not res:
        return "No Data", {}, "No Data", {}, "No Data", {}, "No Data", {}, go.Figure(), go.Figure(), [], [], [], [], html.Div(), [], [], empty_matrix_store

    # Process Results
    deltas = res['deltas']
    before = res['before']
    after = res['after']
    isic_map = res['isic_map']
    before_history = res.get('before_history', [])
    after_history = res.get('after_history', [])
    A_before_mat = res.get('A_before')
    A_after_mat  = res.get('A_after')
    VA_before_vec = res.get('VA_before')
    VA_after_vec  = res.get('VA_after')
    L_before_mat = res.get('L_before')
    L_after_mat  = res.get('L_after')
    demand_vec   = res.get('demand_vector')

    # Determine actual iterations run
    actual_iterations = len(before_history)
    
    # 1. Summaries
    d_X = deltas['X'].sum()
    d_VA = deltas['VA']
    
    # Final Demand - extract from history since it's the same in both scenarios
    last_b = before_history[-1] if before_history else {}
    last_a = after_history[-1] if after_history else {}
    fd_before_vec = last_b.get('demand', np.zeros(len(isic_map)))
    fd_after_vec = last_a.get('demand', np.zeros(len(isic_map)))
    d_FD = fd_after_vec.sum() - fd_before_vec.sum()
    
    # Taxes might not be present if 0
    t_before = before['after_tax'].get('total_tax', np.zeros(1)).sum() if before.get('after_tax') else 0
    t_after = after['after_tax'].get('total_tax', np.zeros(1)).sum() if after.get('after_tax') else 0
    d_Tax = t_after - t_before
    
    def format_summary(val):
        color = '#27ae60' if val > 0 else ('#e74c3c' if val < 0 else '#7f8c8d')
        sign = '+' if val > 0 else ''
        return f"{sign}${val:,.2f}", {'color': color, 'margin': 0}
        
    out_X, style_X = format_summary(d_X)
    out_VA, style_VA = format_summary(d_VA)
    out_FD, style_FD = format_summary(d_FD)
    out_Tax, style_Tax = format_summary(d_Tax)
    
    # 2. Extract sector names using the isic map
    sectors = []
    idx_map = {idx: isic for isic, idx in isic_map.items()}
    for i in range(len(isic_map)):
        isic = idx_map.get(i, f"Sector {i}")
        sectors.append(isic)
    
    # Extract output proportions from history
    va_output_ratio_before = last_b.get('va_output_ratio', np.zeros(len(isic_map)))
    va_output_ratio_after = last_a.get('va_output_ratio', np.zeros(len(isic_map)))
    int_output_ratio_before = last_b.get('intermediate_output_ratio', np.zeros(len(isic_map)))
    int_output_ratio_after = last_a.get('intermediate_output_ratio', np.zeros(len(isic_map)))
        
    df = pd.DataFrame({
        'Sector': sectors,
        'Output_Before': before['X'],
        'Output_After': after['X'],
        'Output_Delta': deltas['X'],
        'VA_Before': before['VA_by_sector'],
        'VA_After': after['VA_by_sector'],
        'VA_Delta': deltas['VA_by_sector'],
        'FD_Before': fd_before_vec,
        'FD_After': fd_after_vec,
        'FD_Delta': fd_after_vec - fd_before_vec,
        'VA_Output_Before': va_output_ratio_before,
        'VA_Output_After': va_output_ratio_after,
        'VA_Output_Delta': va_output_ratio_after - va_output_ratio_before,
        'Int_Output_Before': int_output_ratio_before,
        'Int_Output_After': int_output_ratio_after,
        'Int_Output_Delta': int_output_ratio_after - int_output_ratio_before
    })
    
    # Sort for better presentation
    df = df.sort_values(by='Output_Before', ascending=False)
    
    row_data = df.to_dict('records')
    
    # 3. Figures
    fig_out = build_output_chart(df, actual_iterations)
    fig_va = build_va_chart(df, actual_iterations)

    # Build VA Components Table
    va_components_data = []
    income_before = before.get('income', {})
    income_after = after.get('income', {})
    
    # Create reverse mapping from index to ISIC for sorting
    idx_to_isic = {idx: isic for isic, idx in isic_map.items()}
    for idx in sorted(idx_to_isic.keys()):
        isic = idx_to_isic[idx]
        sector_name = isic
        
        va_components_data.append({
            'Sector': sector_name,
            'minWages_Before': round(income_before.get('minWages', np.zeros(1))[idx], 2),
            'minWages_After': round(income_after.get('minWages', np.zeros(1))[idx], 2),
            'bonusWages_Before': round(income_before.get('bonusWages', np.zeros(1))[idx], 2),
            'bonusWages_After': round(income_after.get('bonusWages', np.zeros(1))[idx], 2),
            'wages_Before': round(income_before.get('wages', np.zeros(1))[idx], 2),
            'wages_After': round(income_after.get('wages', np.zeros(1))[idx], 2),
            'surplus_Before': round(income_before.get('surplus', np.zeros(1))[idx], 2),
            'surplus_After': round(income_after.get('surplus', np.zeros(1))[idx], 2),
        })

    # Build FD Components Table - get demand vectors from history
    fd_components_data = []
    last_b = before_history[-1] if before_history else {}
    last_a = after_history[-1] if after_history else {}
    
    demand_C_before = last_b.get('demand_C', np.zeros(len(isic_map)))
    demand_C_after = last_a.get('demand_C', np.zeros(len(isic_map)))
    demand_I_before = last_b.get('demand_I', np.zeros(len(isic_map)))
    demand_I_after = last_a.get('demand_I', np.zeros(len(isic_map)))
    demand_G_before = last_b.get('demand_G', np.zeros(len(isic_map)))
    demand_G_after = last_a.get('demand_G', np.zeros(len(isic_map)))
    FD_before = last_b.get('demand', np.zeros(len(isic_map)))
    FD_after = last_a.get('demand', np.zeros(len(isic_map)))
    
    for idx in sorted(idx_to_isic.keys()):
        isic = idx_to_isic[idx]
        sector_name = isic
        
        fd_components_data.append({
            'Sector': sector_name,
            'C_Before': round(demand_C_before[idx] if demand_C_before is not None else 0, 2),
            'C_After': round(demand_C_after[idx] if demand_C_after is not None else 0, 2),
            'I_Before': round(demand_I_before[idx] if demand_I_before is not None else 0, 2),
            'I_After': round(demand_I_after[idx] if demand_I_after is not None else 0, 2),
            'G_Before': round(demand_G_before[idx] if demand_G_before is not None else 0, 2),
            'G_After': round(demand_G_after[idx] if demand_G_after is not None else 0, 2),
            'FD_Before': round(FD_before[idx], 2),
            'FD_After': round(FD_after[idx], 2),
        })

    # Build Output Proportions Table
    output_proportions_data = []
    for idx in sorted(idx_to_isic.keys()):
        isic = idx_to_isic[idx]
        sector_name = isic
        
        output_proportions_data.append({
            'Sector': sector_name,
            'VA_Output_Before': round(va_output_ratio_before[idx], 4),
            'VA_Output_After': round(va_output_ratio_after[idx], 4),
            'VA_Output_Delta': round(va_output_ratio_after[idx] - va_output_ratio_before[idx], 4),
            'Int_Output_Before': round(int_output_ratio_before[idx], 4),
            'Int_Output_After': round(int_output_ratio_after[idx], 4),
            'Int_Output_Delta': round(int_output_ratio_after[idx] - int_output_ratio_before[idx], 4),
        })

    # Build Iteration Comparison Display
    iteration_comparison_rows = []
    for i in range(actual_iterations):
        if i < len(before_history) and i < len(after_history):
            b_state = before_history[i]
            a_state = after_history[i]
            
            # Extract output (X is a numpy array)
            output_before = b_state.get('X', np.array([]))
            output_after = a_state.get('X', np.array([]))
            
            # Extract VA total (pre-calculated)
            va_before = b_state.get('VA_total', 0)
            va_after = a_state.get('VA_total', 0)
            
            iteration_comparison_rows.append({
                'Iteration': i + 1,
                'Output_Before': round(output_before.sum(), 2),
                'Output_After': round(output_after.sum(), 2),
                'VA_Before': round(va_before, 2),
                'VA_After': round(va_after, 2),
            })
    
    iteration_comparison_div = dag.AgGrid(
        id='iteration-comparison-table',
        rowData=iteration_comparison_rows,
        columnDefs=[
            {'field': 'Iteration', 'headerName': 'Iteration'},
            {'field': 'Output_Before', 'headerName': 'Output Before'},
            {'field': 'Output_After', 'headerName': 'Output After'},
            {'field': 'VA_Before', 'headerName': 'VA Before'},
            {'field': 'VA_After', 'headerName': 'VA After'},
        ],
        defaultColDef={'resizable': True, 'sortable': True, 'filter': True},
        style={'height': '400px', 'width': '100%'},
    )

    # -------------------------------------------------------------------
    # Build Demand Vector Table
    # -------------------------------------------------------------------
    demand_vector_data = []
    if demand_vec is not None:
        for i, isic in sorted([(idx, isc) for isc, idx in isic_map.items()]):
            demand_vector_data.append({'Sector': isic, 'Demand': round(float(demand_vec[i]), 4)})

    # -------------------------------------------------------------------
    # Build VA Coefficients Table
    # -------------------------------------------------------------------
    va_coeff_data = []
    if VA_before_vec is not None and VA_after_vec is not None:
        for i, isic in sorted([(idx, isc) for isc, idx in isic_map.items()]):
            va_coeff_data.append({
                'Sector': isic,
                'VA_Before': round(float(VA_before_vec[i]), 4),
                'VA_After':  round(float(VA_after_vec[i]),  4),
                'VA_Delta':  round(float(VA_after_vec[i] - VA_before_vec[i]), 4),
            })

    # -------------------------------------------------------------------
    # Build matrix store (serialise numpy arrays as nested lists)
    # -------------------------------------------------------------------
    sector_labels = [isc for _, isc in sorted([(idx, isc) for isc, idx in isic_map.items()])]

    def _mat_to_list(m):
        return m.tolist() if m is not None else None

    # Flow (transactions) matrix  Z = A * diag(X)
    X_before_vec = before['X']
    X_after_vec  = after['X']
    Z_before_mat = A_before_mat * X_before_vec[np.newaxis, :] if A_before_mat is not None else None
    Z_after_mat  = A_after_mat  * X_after_vec[np.newaxis, :]  if A_after_mat  is not None else None

    matrix_store = {
        'sectors': sector_labels,
        'A_before': _mat_to_list(A_before_mat),
        'A_after':  _mat_to_list(A_after_mat),
        'L_before': _mat_to_list(L_before_mat),
        'L_after':  _mat_to_list(L_after_mat),
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
