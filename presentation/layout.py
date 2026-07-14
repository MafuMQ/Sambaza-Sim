import os
import sys
from dash import Dash, html, dcc
import dash_ag_grid as dag

# Ensure root path is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from simulators.tech_change_loader import rebuild_examples_dict_from_db
from pipeline.db.repositories.good_repo import GoodsDatabase

# Database URL
db_path = os.path.join(root_dir, 'data.db')
db_url = f"sqlite:///{db_path}"

# Helper functions to get fresh data from the DB for callbacks
def get_examples_config():
    try:
        return rebuild_examples_dict_from_db(db_url)
    except Exception as e:
        print(f"Error loading examples: {e}")
        return {}

def get_sector_options():
    try:
        goods_db = GoodsDatabase(database_url=db_url)
        all_goods = goods_db.get_all_goods()
        return [{'label': f"{g.name} ({g.isic})", 'value': g.isic} for g in all_goods]
    except Exception as e:
        print(f"Error loading sector options: {e}")
        return []

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
def serve_layout():
    # Freshly load options
    examples_config = get_examples_config()
    example_options = [{'label': f"Example {k}: {v['title']}", 'value': k} for k, v in examples_config.items()]
    default_example = list(examples_config.keys())[0] if examples_config else None
    sector_options = get_sector_options()
    
    # Scan data/ directory for valid source folders
    data_dir = os.path.join(root_dir, 'data')
    data_sources = []
    if os.path.exists(data_dir):
        for entry in os.listdir(data_dir):
            entry_path = os.path.join(data_dir, entry)
            if os.path.isdir(entry_path):
                if os.path.exists(os.path.join(entry_path, 'goods.csv')) and os.path.exists(os.path.join(entry_path, 'productions.csv')):
                    data_sources.append({'label': f"data/{entry}", 'value': f"data/{entry}"})
    
    return html.Div(
        style={'fontFamily': 'system-ui, -apple-system, sans-serif', 'padding': '20px', 'backgroundColor': '#f5f7fa', 'minHeight': '100vh'},
        children=[
            dcc.Location(id='url', refresh=True),
            # Client-side store for matrix data
            dcc.Store(id='store-matrices'),
            # Store for user-defined tech change operations
            dcc.Store(id='store-tech-changes', data=[]),
            
            html.H1('Sambaza-Sim Input-Output Model Dashboard', style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
            
            html.Div(
                style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'},
                children=[
                    # Sidebar / Controls
                    html.Div(
                        style={'flex': '1 1 300px', 'maxWidth': '400px', **CARD_STYLE},
                        children=[
                            html.H3('Data Source', style={'marginTop': '0', 'color': '#34495e'}),
                            html.Label('Select Data Source Folder:'),
                            dcc.Dropdown(
                                id='data-source-selector',
                                options=data_sources,
                                placeholder="Select a folder in data/",
                                style={'marginBottom': '10px'}
                            ),
                            html.Button('Load Data Source', id='load-source-btn', n_clicks=0, style={'width': '100%', 'padding': '8px', 'backgroundColor': '#f39c12', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'marginBottom': '20px'}),
                            html.Div(id='load-source-status', style={'marginBottom': '10px', 'fontSize': '0.9em', 'color': 'red'}),
                            html.Hr(),
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
                        
                        html.Label('Total Final Demand ($):'),
                        dcc.Input(
                            id='input-total-fd',
                            type='number',
                            min=0,
                            step=1.0,
                            placeholder='Enter total final demand...',
                            style={'width': '100%', 'marginBottom': '4px'}
                        ),
                        html.Div(id='input-total-fd-hint', style={'fontSize': '0.8em', 'color': '#7f8c8d', 'marginBottom': '15px'}),
                        
                        html.Hr(),
                        html.H4('Parameters', style={'color': '#34495e'}),
                        
                        html.Label('Iterations:'),
                        dcc.Slider(id='input-iterations', min=1, max=20, step=1, value=5, marks={i: str(i) for i in range(1, 21, 2)}, tooltip={"placement": "bottom", "always_visible": False}),
                        
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
                        
                        html.Label('Wage Spend Rate:'),
                        dcc.Slider(id='input-wage-spend', min=0, max=1, step=0.01, value=1.0, tooltip={"placement": "bottom", "always_visible": False}),
                        
                        html.Label('Surplus Spend Rate:'),
                        dcc.Slider(id='input-surplus-spend', min=0, max=1, step=0.01, value=1.0, tooltip={"placement": "bottom", "always_visible": False}),
                        
                        html.Label('Government Spend Rate:'),
                        dcc.Slider(id='input-tax-spend', min=0, max=1, step=0.01, value=1.0, tooltip={"placement": "bottom", "always_visible": False}),
                        
                        html.Label('Economy Type (Import Leakage):'),
                        dcc.Dropdown(
                            id='input-economy-type',
                            options=[
                                {'label': 'Open Economy (Imports Leak)', 'value': 'open'},
                                {'label': 'Closed Economy (Imports Recycled)', 'value': 'closed'}
                            ],
                            value='open',
                            clearable=False,
                            style={'marginBottom': '10px'}
                        ),
                        
                        html.Hr(),
                        html.H4('Technology Changes', style={'color': '#34495e'}),
                        html.P('Build custom tech changes to apply on top of the selected scenario.',
                               style={'fontSize': '0.85em', 'color': '#7f8c8d', 'marginBottom': '10px'}),
                        
                        # Change type selector
                        html.Label('Change Type:'),
                        dcc.Dropdown(
                            id='tc-change-type',
                            options=[
                                {'label': 'Change all inputs of a sector', 'value': 'add_sector_change'},
                                {'label': 'Change usage of an input across all sectors', 'value': 'add_input_change'},
                                {'label': 'Change a specific coefficient', 'value': 'add_coefficient_change'},
                            ],
                            value='add_sector_change',
                            clearable=False,
                            style={'marginBottom': '10px'}
                        ),
                        
                        # Sector selector (producing sector / column)
                        html.Div(id='tc-sector-container', children=[
                            html.Label('Sector (column in A matrix):'),
                            dcc.Dropdown(
                                id='tc-sector',
                                options=sector_options,
                                placeholder='Select sector...',
                                style={'marginBottom': '10px'}
                            ),
                        ]),
                        
                        # Input sector selector (input / row) — shown for coefficient_change and input_change
                        html.Div(id='tc-input-sector-container', children=[
                            html.Label('Input Sector (row in A matrix):'),
                            dcc.Dropdown(
                                id='tc-input-sector',
                                options=sector_options,
                                placeholder='Select input sector...',
                                style={'marginBottom': '10px'}
                            ),
                        ]),
                        
                        # Operation type
                        html.Label('Operation:'),
                        dcc.Dropdown(
                            id='tc-operation',
                            options=[
                                {'label': 'Multiply (e.g. 0.8 = 20% reduction)', 'value': 'multiply'},
                                {'label': 'Add (e.g. -0.05)', 'value': 'add'},
                                {'label': 'Set (absolute value)', 'value': 'set'},
                            ],
                            value='multiply',
                            clearable=False,
                            style={'marginBottom': '10px'}
                        ),
                        
                        # Value
                        html.Label('Value:'),
                        dcc.Input(id='tc-value', type='number', value=0.8, step=0.01, style={'width': '100%', 'marginBottom': '10px'}),
                        
                        # Add button
                        html.Button(
                            '+ Add Change',
                            id='tc-add-button',
                            n_clicks=0,
                            style={
                                'width': '100%', 'padding': '8px', 'backgroundColor': '#27ae60',
                                'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                'fontSize': '14px', 'cursor': 'pointer', 'marginBottom': '10px'
                            }
                        ),
                        
                        # Clear all button
                        html.Button(
                            'Clear All Changes',
                            id='tc-clear-button',
                            n_clicks=0,
                            style={
                                'width': '100%', 'padding': '6px', 'backgroundColor': '#e74c3c',
                                'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                'fontSize': '12px', 'cursor': 'pointer', 'marginBottom': '10px'
                            }
                        ),
                        
                        # Display of current changes
                        html.Div(id='tc-changes-display', style={'marginBottom': '10px'}),
                        
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
                                                            {"field": "FD_Delta", "headerName": "Δ FD", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                                            {"field": "VA_Output_Before", "headerName": "VA/Output (Before)", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                            {"field": "VA_Output_After", "headerName": "VA/Output (After)", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                            {"field": "VA_Output_Delta", "headerName": "Δ VA/Output", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                            {"field": "Int_Output_Before", "headerName": "Int/Output (Before)", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                            {"field": "Int_Output_After", "headerName": "Int/Output (After)", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                            {"field": "Int_Output_Delta", "headerName": "Δ Int/Output", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}}
                                                        ],
                                                        rowData=[],
                                                        defaultColDef={"sortable": True, "filter": True, "resizable": True},
                                                        dashGridOptions={"pagination": True, "paginationPageSize": 15},
                                                        style={"height": 500, "width": "100%"}
                                                    ),

                                                    html.H3("Value Added Components by Sector", style={'marginTop': '30px'}),
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

                                                    html.H3("Output Composition Ratios by Sector", style={'marginTop': '30px'}),
                                                    dag.AgGrid(
                                                        id="table-output-proportions",
                                                        columnDefs=[
                                                            {"field": "Sector", "width": 200, "pinned": "left"},
                                                            {"field": "VA_Output_Before", "headerName": "VA/Output (Before)", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                            {"field": "VA_Output_After", "headerName": "VA/Output (After)", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                            {"field": "VA_Output_Delta", "headerName": "Δ VA/Output", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                            {"field": "Int_Output_Before", "headerName": "Int/Output (Before)", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                            {"field": "Int_Output_After", "headerName": "Int/Output (After)", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                            {"field": "Int_Output_Delta", "headerName": "Δ Int/Output", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                        ],
                                                        rowData=[],
                                                        defaultColDef={"sortable": True, "filter": True, "resizable": True},
                                                        dashGridOptions={"pagination": True, "paginationPageSize": 15},
                                                        style={"height": 400, "width": "100%"}
                                                    ),

                                                    html.H3("Iteration-by-Iteration Comparison", style={'marginTop': '30px'}),
                                                    html.Div(id='iteration-comparison-div')
                                                ])
                                            ]),
                                            dcc.Tab(label='IO Matrices', value='tab-matrices', children=[
                                                html.Div(style={'padding': '20px'}, children=[
                                                    # Demand vector & VA coefficients side by side
                                                    html.Div(
                                                        style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px', 'flexWrap': 'wrap'},
                                                        children=[
                                                            html.Div(style={'flex': '1 1 300px'}, children=[
                                                                html.H3("Final Demand Vector"),
                                                                dag.AgGrid(
                                                                    id="table-demand-vector",
                                                                    columnDefs=[
                                                                        {"field": "Sector", "width": 200},
                                                                        {"field": "Demand", "headerName": "Final Demand ($)", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                                    ],
                                                                    rowData=[],
                                                                    defaultColDef={"sortable": True, "filter": True, "resizable": True},
                                                                    style={"height": 350, "width": "100%"}
                                                                )
                                                            ]),
                                                            html.Div(style={'flex': '1 1 500px'}, children=[
                                                                html.H3("Value Added Coefficients"),
                                                                dag.AgGrid(
                                                                    id="table-va-coefficients",
                                                                    columnDefs=[
                                                                        {"field": "Sector", "width": 200},
                                                                        {"field": "VA_Before", "headerName": "VA Coeff (Before)", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                                        {"field": "VA_After", "headerName": "VA Coeff (After)", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"}},
                                                                        {"field": "VA_Delta", "headerName": "Δ VA Coeff", "valueFormatter": {"function": "d3.format(',.4f')(params.value)"},
                                                                         "cellStyle": {"function": "params.value < 0 ? {'color': '#e74c3c'} : params.value > 0 ? {'color': '#27ae60'} : {}"}},
                                                                    ],
                                                                    rowData=[],
                                                                    defaultColDef={"sortable": True, "filter": True, "resizable": True},
                                                                    style={"height": 350, "width": "100%"}
                                                                )
                                                            ]),
                                                        ]
                                                    ),

                                                    # Matrix selector
                                                    html.Div(
                                                        style={'display': 'flex', 'gap': '20px', 'alignItems': 'center', 'marginBottom': '15px', 'flexWrap': 'wrap'},
                                                        children=[
                                                            html.Div([
                                                                html.Label('Matrix to display:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                                                                dcc.Dropdown(
                                                                    id='matrix-type-selector',
                                                                    options=[
                                                                        {'label': 'Technical Coefficients  A  (Before)', 'value': 'A_before'},
                                                                        {'label': 'Technical Coefficients  A  (After)',  'value': 'A_after'},
                                                                        {'label': 'Change in Coefficients  ΔA',          'value': 'delta_A'},
                                                                        {'label': 'Leontief Inverse  L  (Before)',        'value': 'L_before'},
                                                                        {'label': 'Leontief Inverse  L  (After)',         'value': 'L_after'},
                                                                        {'label': 'Change in Leontief  ΔL',               'value': 'delta_L'},
                                                                        {'label': 'Flow Table  Z  (Before)  — $',         'value': 'Z_before'},
                                                                        {'label': 'Flow Table  Z  (After)  — $',          'value': 'Z_after'},
                                                                        {'label': 'Change in Flow Table  ΔZ  — $',        'value': 'delta_Z'},
                                                                    ],
                                                                    value='A_before',
                                                                    clearable=False,
                                                                    style={'width': '380px'}
                                                                ),
                                                            ]),
                                                        ]
                                                    ),

                                                    # Matrix heatmap
                                                    dcc.Graph(id='graph-matrix-heatmap', style={'marginBottom': '20px'}),

                                                    # Matrix as data table
                                                    html.H4(id='matrix-table-title', children="Technical Coefficient Matrix (A) — Before"),
                                                    dag.AgGrid(
                                                        id="table-matrix",
                                                        columnDefs=[],
                                                        rowData=[],
                                                        defaultColDef={"sortable": False, "filter": False, "resizable": True, "width": 100},
                                                        dashGridOptions={"pagination": False},
                                                        style={"height": 500, "width": "100%"}
                                                    ),
                                                ])
                                            ]),
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

app.layout = serve_layout



