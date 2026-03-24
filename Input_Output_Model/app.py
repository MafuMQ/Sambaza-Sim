import os
import sys
import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import logging

# Ensure root path is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Input_Output_Model.demos.demo import load_examples_from_database
from Input_Output_Model.demos.util.simulation import run_simulation

# Database URL
db_path = os.path.join(root_dir, 'data.db')
db_url = f"sqlite:///{db_path}"

# Load examples
try:
    examples_config = load_examples_from_database(db_url)
    example_options = [{'label': f"Example {k}: {v['title']}", 'value': k} for k, v in examples_config.items()]
    default_example = list(examples_config.keys())[0] if examples_config else None
except Exception as e:
    print(f"Error loading examples: {e}")
    examples_config = {}
    example_options = []
    default_example = None

# Initialize the app
app = Dash(__name__)

# App styling
CARD_STYLE = {
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'borderRadius': '8px',
    'padding': '20px',
    'backgroundColor': 'white',
    'marginBottom': '20px'
}

# App layout
app.layout = html.Div(
    style={'fontFamily': 'system-ui, -apple-system, sans-serif', 'padding': '20px', 'backgroundColor': '#f5f7fa', 'minHeight': '100vh'},
    children=[
        html.H1('Sambaza-Sim Input-Output Model Dashboard', style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        
        html.Div(
            style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'},
            children=[
                # Sidebar / Controls
                html.Div(
                    style={'flex': '1 1 300px', 'maxWidth': '400px', **CARD_STYLE},
                    children=[
                        html.H3('Configuration', style={'marginTop': '0', 'color': '#34495e'}),
                        
                        html.Label('Select Dataset / Example Scenario:'),
                        dcc.Dropdown(
                            id='example-selector',
                            options=example_options,
                            value=default_example,
                            clearable=False,
                            style={'marginBottom': '20px'}
                        ),
                        
                        html.Div(id='example-description', style={'marginBottom': '10px', 'fontSize': '0.9em', 'color': '#7f8c8d'}),
                        
                        html.Div(id='example-fd-info', style={'marginBottom': '20px', 'fontSize': '0.9em'}),
                        
                        html.Hr(),
                        html.H4('Parameters', style={'color': '#34495e'}),
                        
                        html.Label('Iterations:'),
                        dcc.Input(id='input-iterations', type='number', value=1, min=1, step=1, style={'width': '100%', 'marginBottom': '10px'}),
                        
                        html.Label('Solver Type:'),
                        dcc.Dropdown(
                            id='input-solver',
                            options=[
                                {'label': 'Supply Curves', 'value': 'supply_curves'},
                                {'label': 'Leontief Inverse', 'value': 'leontief'}
                            ],
                            value='supply_curves',
                            clearable=False,
                            style={'marginBottom': '10px'}
                        ),
                        
                        html.Label('Income Tax Rate (Before):'),
                        dcc.Slider(id='input-income-tax-before', min=0, max=1, step=0.01, value=0.0, tooltip={"placement": "bottom", "always_visible": False}),
                        
                        html.Label('Income Tax Rate (After):'),
                        dcc.Slider(id='input-income-tax-after', min=0, max=1, step=0.01, value=0.0, tooltip={"placement": "bottom", "always_visible": False}),
                        
                        html.Label('Corporate Tax Rate (Before):'),
                        dcc.Slider(id='input-corp-tax-before', min=0, max=1, step=0.01, value=0.0, tooltip={"placement": "bottom", "always_visible": False}),
                        
                        html.Label('Corporate Tax Rate (After):'),
                        dcc.Slider(id='input-corp-tax-after', min=0, max=1, step=0.01, value=0.0, tooltip={"placement": "bottom", "always_visible": False}),
                        
                        html.Button(
                            'Run Simulation', 
                            id='run-button', 
                            n_clicks=0,
                            style={
                                'width': '100%', 'padding': '12px', 'backgroundColor': '#3498db', 
                                'color': 'white', 'border': 'none', 'borderRadius': '4px', 
                                'fontSize': '16px', 'cursor': 'pointer', 'marginTop': '20px',
                                'fontWeight': 'bold'
                            }
                        )
                    ]
                ),
                
                # Main Content Area
                html.Div(
                    style={'flex': '3 1 600px', 'display': 'flex', 'flexDirection': 'column', 'gap': '20px'},
                    children=[
                        # Summary Cards
                        html.Div(
                            style={'display': 'flex', 'gap': '20px'},
                            children=[
                                html.Div(style={'flex': 1, **CARD_STYLE}, children=[
                                    html.H4('Gross Output Change', style={'margin': '0 0 10px 0', 'color': '#7f8c8d'}),
                                    html.H2(id='summary-output', style={'margin': 0, 'color': '#2c3e50'})
                                ]),
                                html.Div(style={'flex': 1, **CARD_STYLE}, children=[
                                    html.H4('Value Added Change', style={'margin': '0 0 10px 0', 'color': '#7f8c8d'}),
                                    html.H2(id='summary-va', style={'margin': 0, 'color': '#2c3e50'})
                                ]),
                                html.Div(style={'flex': 1, **CARD_STYLE}, children=[
                                    html.H4('Final Demand Change', style={'margin': '0 0 10px 0', 'color': '#7f8c8d'}),
                                    html.H2(id='summary-fd', style={'margin': 0, 'color': '#2c3e50'})
                                ]),
                                html.Div(style={'flex': 1, **CARD_STYLE}, children=[
                                    html.H4('Tax Revenue Change', style={'margin': '0 0 10px 0', 'color': '#7f8c8d'}),
                                    html.H2(id='summary-tax', style={'margin': 0, 'color': '#2c3e50'})
                                ])
                            ]
                        ),
                        
                        # Tabs for Results
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                dcc.Loading(
                                    id="loading-results",
                                    type="circle",
                                    children=[
                                        dcc.Tabs(id="tabs-results", value='tab-graphs', children=[
                                            dcc.Tab(label='Visualizations', value='tab-graphs', children=[
                                                html.Div(style={'padding': '20px'}, children=[
                                                    dcc.Graph(id='graph-va-comparison', style={'marginBottom': '30px'}),
                                                    dcc.Graph(id='graph-output-comparison')
                                                ])
                                            ]),
                                            dcc.Tab(label='Data Tables', value='tab-tables', children=[
                                                html.Div(style={'padding': '20px'}, children=[
                                                    html.H3("Sector-by-Sector Breakdown"),
                                                    dag.AgGrid(
                                                        id="table-results",
                                                        columnDefs=[
                                                            {"field": "Sector", "width": 250},
                                                            {"field": "Output_Before", "headerName": "Output (Before)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "Output_After", "headerName": "Output (After)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "Output_Delta", "headerName": "Δ Output", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "VA_Before", "headerName": "VA (Before)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "VA_After", "headerName": "VA (After)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "VA_Delta", "headerName": "Δ VA", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "FD_Before", "headerName": "FD (Before)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "FD_After", "headerName": "FD (After)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "FD_Delta", "headerName": "Δ FD", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}}
                                                        ],
                                                        rowData=[],
                                                        defaultColDef={"sortable": True, "filter": True, "resizable": True},
                                                        dashGridOptions={"pagination": True, "paginationPageSize": 15},
                                                        style={"height": 500, "width": "100%"}
                                                    )
                                                ])
                                            ]),
                                            dcc.Tab(label='Detailed Tables', value='tab-detailed', children=[
                                                html.Div(style={'padding': '20px'}, children=[
                                                    html.H3("Value Added Components by Sector"),
                                                    dag.AgGrid(
                                                        id="table-va-components",
                                                        columnDefs=[
                                                            {"field": "Sector", "width": 200, "pinned": "left"},
                                                            {"field": "minWages_Before", "headerName": "Min Wages (Before)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "minWages_After", "headerName": "Min Wages (After)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "bonusWages_Before", "headerName": "Bonus Wages (Before)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "bonusWages_After", "headerName": "Bonus Wages (After)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "wages_Before", "headerName": "Total Wages (Before)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "wages_After", "headerName": "Total Wages (After)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "surplus_Before", "headerName": "Surplus (Before)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "surplus_After", "headerName": "Surplus (After)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                        ],
                                                        rowData=[],
                                                        defaultColDef={"sortable": True, "filter": True, "resizable": True},
                                                        dashGridOptions={"pagination": True, "paginationPageSize": 15},
                                                        style={"height": 400, "width": "100%"}
                                                    ),
                                                    
                                                    html.H3("Final Demand Components by Sector", style={'marginTop': '30px'}),
                                                    dag.AgGrid(
                                                        id="table-fd-components",
                                                        columnDefs=[
                                                            {"field": "Sector", "width": 200, "pinned": "left"},
                                                            {"field": "C_Before", "headerName": "Consumption (Before)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "C_After", "headerName": "Consumption (After)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "I_Before", "headerName": "Investment (Before)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "I_After", "headerName": "Investment (After)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "G_Before", "headerName": "Government (Before)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "G_After", "headerName": "Government (After)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "FD_Before", "headerName": "Total FD (Before)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "FD_After", "headerName": "Total FD (After)", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                        ],
                                                        rowData=[],
                                                        defaultColDef={"sortable": True, "filter": True, "resizable": True},
                                                        dashGridOptions={"pagination": True, "paginationPageSize": 15},
                                                        style={"height": 400, "width": "100%"}
                                                    ),
                                                    
                                                    html.H3("Iteration-by-Iteration Comparison", style={'marginTop': '30px'}),
                                                    html.Div(id='iteration-comparison-div')
                                                ])
                                            ])
                                        ])
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# Callback to update inputs when a new example is selected
# Note: iterations is NOT updated here - it remains user-controlled
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

# Callback to run simulation and update outputs
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
        return "-", {}, "-", {}, "-", {}, go.Figure(), go.Figure(), [], [], [], html.Div()
        
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
                from Input_Output_Model.util.Evaluators import build_io_matrix
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
        return f"Error", {'color': 'red'}, f"Error", {'color': 'red'}, f"Error", {'color': 'red'}, f"Error", {'color': 'red'}, go.Figure().add_annotation(text=f"Error: {e}", showarrow=False), go.Figure(), [], [], [], html.Div()

    if not res:
        return "No Data", {}, "No Data", {}, "No Data", {}, "No Data", {}, go.Figure(), go.Figure(), [], [], [], html.Div()

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
        'FD_Delta': fd_after_vec - fd_before_vec
    })
    
    # Sort for better presentation
    df = df.sort_values(by='Output_Before', ascending=False)
    
    row_data = df.to_dict('records')
    
    # 3. Figures
    # Top 15 sectors for Output
    top_df_out = df.head(15).sort_values(by='Output_Before', ascending=True)
    fig_out = go.Figure()
    fig_out.add_trace(go.Bar(
        y=top_df_out['Sector'],
        x=top_df_out['Output_Before'],
        name='Before',
        orientation='h',
        marker_color='#3498db'
    ))
    fig_out.add_trace(go.Bar(
        y=top_df_out['Sector'],
        x=top_df_out['Output_After'],
        name='After',
        orientation='h',
        marker_color='#2ecc71'
    ))
    fig_out.update_layout(
        title=f'Gross Output Comparison - Top 15 Sectors ({actual_iterations} iteration{"s" if actual_iterations != 1 else ""})',
        barmode='group',
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Top 15 for VA
    top_df_va = df.head(15).sort_values(by='VA_Before', ascending=True)
    fig_va = go.Figure()
    fig_va.add_trace(go.Bar(
        y=top_df_va['Sector'],
        x=top_df_va['VA_Before'],
        name='Before',
        orientation='h',
        marker_color='#9b59b6'
    ))
    fig_va.add_trace(go.Bar(
        y=top_df_va['Sector'],
        x=top_df_va['VA_After'],
        name='After',
        orientation='h',
        marker_color='#e67e22'
    ))
    fig_va.update_layout(
        title=f'Value Added Comparison - Top 15 Sectors ({actual_iterations} iteration{"s" if actual_iterations != 1 else ""})',
        barmode='group',
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

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

    return out_X, style_X, out_VA, style_VA, out_FD, style_FD, out_Tax, style_Tax, fig_va, fig_out, row_data, va_components_data, fd_components_data, iteration_comparison_div

if __name__ == '__main__':
    app.run(debug=True, port=8050)
