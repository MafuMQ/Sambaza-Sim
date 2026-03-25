import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html
import dash_ag_grid as dag

from ui.layout import examples_config
from ui.charts import build_output_chart, build_va_chart
from core.simulation import run_simulation


@callback(
    Output('example-description', 'children'),
    Output('example-fd-info', 'children'),
    Output('input-solver', 'value'),
    Output('input-income-tax-before', 'value'),
    Output('input-income-tax-after', 'value'),
    Output('input-corp-tax-before', 'value'),
    Output('input-corp-tax-after', 'value'),
    Input('example-selector', 'value')
)
def update_controls(example_id):
    if not example_id or example_id not in examples_config:
        return "", html.Div(), 'supply_curves', 0.0, 0.0, 0.0, 0.0
    
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
    
    # Compute total final demand for the baseline
    fd_raw = params.get('final_demand', None)
    if fd_raw is not None:
        fd_total = np.array(fd_raw, dtype=float).sum()
        fd_info = html.Div([
            html.Span('Total Final Demand: ', style={'color': '#7f8c8d'}),
            html.Strong(f"${fd_total:,.2f}", style={'color': '#2c3e50'})
        ], style={'padding': '8px 12px', 'backgroundColor': '#eaf4fb', 'borderRadius': '4px', 'border': '1px solid #aed6f1'})
    else:
        fd_info = html.Div()
    
    return desc, fd_info, solver, inc_before, inc_after, corp_before, corp_after

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
    Input('run-button', 'n_clicks'),
    State('example-selector', 'value'),
    State('input-iterations', 'value'),
    State('input-solver', 'value'),
    State('input-income-tax-before', 'value'),
    State('input-income-tax-after', 'value'),
    State('input-corp-tax-before', 'value'),
    State('input-corp-tax-after', 'value')
)
def execute_simulation(n_clicks, example_id, iterations, solver, inc_before, inc_after, corp_before, corp_after):
    if not example_id or example_id not in examples_config:
        return "-", {}, "-", {}, "-", {}, "-", {}, go.Figure(), go.Figure(), [], [], [], [], html.Div()
        
    config = examples_config[example_id]
    params = config['params'].copy()
    
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
    
    # Build inputs for simulation
    # If tech change comparison, we need to extract tech_change_builder
    try:
        if is_tech_comparison:
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
        return f"Error", {'color': 'red'}, f"Error", {'color': 'red'}, f"Error", {'color': 'red'}, f"Error", {'color': 'red'}, go.Figure().add_annotation(text=f"Error: {e}", showarrow=False), go.Figure(), [], [], [], [], html.Div()

    if not res:
        return "No Data", {}, "No Data", {}, "No Data", {}, "No Data", {}, go.Figure(), go.Figure(), [], [], [], [], html.Div()

    # Process Results
    deltas = res['deltas']
    before = res['before']
    after = res['after']
    isic_map = res['isic_map']
    before_history = res.get('before_history', [])
    after_history = res.get('after_history', [])
    
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

    return out_X, style_X, out_VA, style_VA, out_FD, style_FD, out_Tax, style_Tax, fig_va, fig_out, row_data, va_components_data, fd_components_data, output_proportions_data, iteration_comparison_div
