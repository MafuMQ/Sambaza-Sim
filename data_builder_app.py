import os
import sys
import json
import pandas as pd
import dash
from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

# Parse ISIC Codes
def load_isic_data():
    try:
        df = pd.read_csv('data/isic/ISIC_Rev_5_english_structure.csv', encoding='latin1')
        options = []
        hierarchy = {'sections': [], 'divisions': {}, 'groups': {}, 'classes': {}}
        
        for _, row in df.iterrows():
            code = str(row['ISIC Rev 5 Code']).strip()
            title = str(row['ISIC Rev 5 Title']).strip()
            
            padded_code = code.ljust(4, '0')[:4]
            if code.isalpha():
                formatted_isic = f"{code}000_000_000"
            else:
                formatted_isic = f"A{padded_code}_000_000"
                
            val_json = json.dumps({'title': title, 'formatted': formatted_isic, 'raw': code})
            options.append({'label': f"{code} - {title}", 'value': val_json})
            
            # Build Hierarchy
            if len(code) == 1 and code.isalpha():
                hierarchy['sections'].append({'label': f"{code} - {title}", 'value': code})
            elif len(code) == 2 and code.isdigit():
                # We don't have direct mapping of division to section in the code itself, 
                # but usually sections span ranges of divisions.
                # Actually, the CSV might not explicitly link Division -> Section on the row. 
                # Wait, we need to infer parent from previous row or something. 
                pass # Will refine this inside the callback if needed, or just flatten.
            
        return options, df
    except Exception as e:
        print(f"Failed to load ISIC: {e}")
        return [], None

ISIC_OPTIONS, ISIC_DF = load_isic_data()

# Process hierarchy more robustly
def build_hierarchy(df):
    hierarchy = {'sections': [], 'divisions': {}, 'groups': {}, 'classes': {}}
    if df is None: return hierarchy
    
    current_section = None
    for _, row in df.iterrows():
        code = str(row['ISIC Rev 5 Code']).strip()
        title = str(row['ISIC Rev 5 Title']).strip()
        
        padded_code = code.ljust(4, '0')[:4]
        formatted_isic = f"{code}000_000_000" if code.isalpha() else f"A{padded_code}_000_000"
        val_json = json.dumps({'title': title, 'formatted': formatted_isic, 'raw': code})
        
        opt = {'label': f"{code} - {title}", 'value': val_json}
        
        if len(code) == 1 and code.isalpha():
            current_section = code
            hierarchy['sections'].append({'label': f"{code} - {title}", 'value': code})
            hierarchy['divisions'][current_section] = []
        elif len(code) == 2 and code.isdigit():
            if current_section:
                hierarchy['divisions'][current_section].append({'label': f"{code} - {title}", 'value': code})
            hierarchy['groups'][code] = []
        elif len(code) == 3 and code.isdigit():
            parent = code[:2]
            if parent not in hierarchy['groups']: hierarchy['groups'][parent] = []
            hierarchy['groups'][parent].append({'label': f"{code} - {title}", 'value': code})
            hierarchy['classes'][code] = []
        elif len(code) == 4 and code.isdigit():
            parent = code[:3]
            if parent not in hierarchy['classes']: hierarchy['classes'][parent] = []
            hierarchy['classes'][parent].append(opt)
            
    return hierarchy

ISIC_HIERARCHY = build_hierarchy(ISIC_DF)

def parse_tech_changes_df(tech_df):
    """Parse flat tech_changes.csv into scenario records with per-investment capital requirements."""
    if tech_df is None or tech_df.empty or 'tech_change_id' not in tech_df.columns:
        return []
    tech_df = tech_df.fillna('')
    scenarios = []

    grouped = []
    current_key = None
    current_group = []
    for _, row in tech_df.iterrows():
        key = (row.get('example_id', ''), row.get('tech_change_id', ''))
        if key != current_key:
            if current_group:
                grouped.append(current_group)
            current_key = key
            current_group = [row]
        else:
            current_group.append(row)
    if current_group:
        grouped.append(current_group)

    for group in grouped:
        first_row = group[0]
        ex_id = first_row.get('example_id', len(scenarios) + 1)
        try:
            ex_id = int(float(ex_id))
        except (ValueError, TypeError):
            ex_id = len(scenarios) + 1

        tc_id = str(first_row.get('tech_change_id', ''))
        title = str(first_row.get('title', ''))
        desc = str(first_row.get('description', ''))
        iters = first_row.get('iterations', '')
        try:
            iters = int(float(iters)) if iters != '' else 5
        except (ValueError, TypeError):
            iters = 5

        investments = []
        for row in group:
            method = str(row.get('method', '')).strip()
            if method == 'set_capital_requirements':
                req_raw = row.get('requirements', '')
                req_cost = 0.0
                if req_raw:
                    try:
                        req_dict = json.loads(req_raw) if isinstance(req_raw, str) else req_raw
                        if isinstance(req_dict, dict):
                            req_cost = sum(float(v) for v in req_dict.values())
                        else:
                            req_cost = float(req_dict)
                    except Exception:
                        pass
                dur_raw = row.get('investment_duration', '')
                inv_dur = 1
                if dur_raw != '':
                    try:
                        inv_dur = int(float(dur_raw))
                    except Exception:
                        inv_dur = 1

                if investments:
                    investments[-1]['capital_cost'] = req_cost
                    investments[-1]['investment_duration'] = inv_dur
                else:
                    investments.append({
                        'method': 'set_capital_requirements',
                        'capital_cost': req_cost,
                        'investment_duration': inv_dur
                    })
            elif method:
                val = row.get('value', '')
                try:
                    val_float = float(val) if val != '' else 1.0
                except (ValueError, TypeError):
                    val_float = 1.0

                inv = {
                    'method': method,
                    'sector_idx': row.get('sector_idx', ''),
                    'input_sector_idx': row.get('input_sector_idx', ''),
                    'change_type': row.get('change_type_param', 'multiply'),
                    'value': val_float,
                    'capital_cost': 0.0,
                    'investment_duration': 1,
                    'production_id': row.get('production_id', ''),
                    'isic': row.get('isic', ''),
                    'tier_index': row.get('tier_index', ''),
                    'field': row.get('field', ''),
                    'cap': row.get('cap', ''),
                    'price': row.get('price', '')
                }
                investments.append(inv)

        scenarios.append({
            'example_id': ex_id,
            'tech_change_id': tc_id,
            'change_type': first_row.get('change_type', 'tech_change'),
            'title': title,
            'description': desc,
            'iterations': iters,
            'investments_count': len(investments),
            'total_capital_cost': sum(inv.get('capital_cost', 0.0) for inv in investments),
            'investments': investments
        })

    return scenarios

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

def serve_layout():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    import_options = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, f)) and not f.startswith('.'):
                if os.path.exists(os.path.join(data_dir, f, 'goods.csv')) and os.path.exists(os.path.join(data_dir, f, 'productions.csv')):
                    import_options.append({'label': f, 'value': f})

    return dbc.Container([
        dcc.Store(id='goods-store', data=[]),
        dcc.Store(id='productions-store', data=[]),
        dcc.Store(id='tax-policies-store', data=[]),
        dcc.Store(id='tech-changes-store', data=[]),
        dcc.Store(id='pending-investments-store', data=[]),
        dcc.ConfirmDialog(id='import-confirm-dialog', message='Importing will overwrite your current unsaved session data. Continue?'),
        dcc.ConfirmDialog(id='generate-confirm-dialog', message='This dataset folder already exists. Generating will overwrite it. Continue?'),
        
        dbc.NavbarSimple(
            brand="Sambaza-Sim Data Builder Utility",
            brand_href="#",
            color="primary",
            dark=True,
            className="mb-4 shadow-sm"
        ),
        
        # Setup Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("0. Import Existing Dataset", className="font-weight-bold"),
                    dbc.CardBody([
                        html.Label("Select Dataset:", className="mb-1"),
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(id='import-dropdown', options=import_options, placeholder="Select..."),
                            ], width=8),
                            dbc.Col([
                                dbc.Button("Import", id='import-btn', color="warning", n_clicks=0, className="w-100"),
                            ], width=4),
                        ]),
                        html.Div(id='import-status', className="mt-2")
                    ])
                ], className="mb-4 shadow-sm h-100")
            ], width=12, lg=6, className="mb-4 mb-lg-0"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("1. Output Configuration", className="font-weight-bold"),
                    dbc.CardBody([
                        html.Label("Target Folder Name (inside data/):", className="mb-1"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(id='target-folder-input', type='text', placeholder='e.g. custom_scenario', className="form-control"),
                            ], width=8),
                            dbc.Col([
                                dbc.Button("GENERATE", id='generate-btn', color="success", n_clicks=0, className="w-100"),
                            ], width=4),
                        ]),
                        html.Div(id='folder-status', className="text-danger mt-1"),
                        html.Div(id='generate-status', className="mt-2 font-weight-bold")
                    ])
                ], className="mb-4 shadow-sm h-100")
            ], width=12, lg=6),
        ], className="align-items-stretch mb-4"),
        
        # Main Split
        dbc.Row([
            # GOODS COLUMN
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("2. Add Goods", className="font-weight-bold"),
                    dbc.CardBody([
                        dcc.Tabs([
                            dcc.Tab(label='Direct Search', children=[
                                html.Div([
                                    html.Label("Search ISIC Sector:", className="fw-bold mb-2"),
                                    dcc.Dropdown(id='isic-dropdown-direct', options=ISIC_OPTIONS, placeholder="Search ISIC sector..."),
                                ], className="py-3")
                            ]),
                            dcc.Tab(label='Hierarchical', children=[
                                html.Div([
                                    dbc.Row([
                                        dbc.Col([html.Label("1. Section:", className="mb-1"), dcc.Dropdown(id='isic-section', options=ISIC_HIERARCHY['sections'], placeholder="Section...")]),
                                        dbc.Col([html.Label("2. Division:", className="mb-1"), dcc.Dropdown(id='isic-division', placeholder="Division...")]),
                                    ], className="mb-2"),
                                    dbc.Row([
                                        dbc.Col([html.Label("3. Group:", className="mb-1"), dcc.Dropdown(id='isic-group', placeholder="Group...")]),
                                        dbc.Col([html.Label("4. Class:", className="mb-1"), dcc.Dropdown(id='isic-class', placeholder="Class...")]),
                                    ])
                                ], className="py-3")
                            ])
                        ]),
                        
                        dcc.Store(id='selected-isic-store', data=None),
                        html.Div(id='selected-isic-display', className="font-weight-bold text-primary mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Custom ISIC Sub-Class 1:", className="mb-1"),
                                dbc.Input(id='isic-sub1-input', type='text', value='000', className="form-control"),
                            ]),
                            dbc.Col([
                                html.Label("Custom ISIC Sub-Class 2:", className="mb-1"),
                                dbc.Input(id='isic-sub2-input', type='text', value='000', className="form-control")
                            ])
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Good Name:", className="mb-1"),
                                dbc.Input(id='good-name-input', type='text', placeholder="e.g., Steel", className="form-control"),
                            ]),
                            dbc.Col([
                                html.Label("Description:", className="mb-1"),
                                dbc.Input(id='good-desc-input', type='text', placeholder="e.g., High-grade", className="form-control"),
                            ])
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Add Good", id='add-good-btn', color="primary", n_clicks=0, className="me-2"),
                                dbc.Button("Remove Selected", id='remove-goods-btn', color="danger", outline=True, n_clicks=0),
                            ])
                        ], className="mb-3"),
                        html.Div(id='add-good-status', className="text-danger mb-2"),
                        
                        html.H5("Current Goods", className="mb-0 mt-4"),
                        html.Small("Edit Descriptive Name Directly", className="text-muted d-block mb-2"),
                        dag.AgGrid(
                            id='goods-grid',
                            columnDefs=[
                                {'field': 'id', 'headerName': 'ID', 'editable': False, 'checkboxSelection': True, 'width': 80},
                                {'field': 'name', 'editable': False},
                                {'field': 'isic', 'headerName': 'Formatted ISIC', 'editable': False},
                                {'field': 'descriptive_name', 'editable': True}
                            ],
                            rowData=[],
                            dashGridOptions={'rowSelection': 'multiple'},
                            style={'height': 300, 'width': '100%'}
                        )
                    ])
                ], className="shadow-sm mb-4 h-100")
            ], width=12, lg=6, className="mb-4 mb-lg-0"),
            
            # PRODUCTIONS COLUMN
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("3. Add Production", className="font-weight-bold"),
                    dbc.CardBody([
                        html.Small("Note: Goods used as inputs must be defined in the Goods section.", className="text-muted d-block mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Good to Produce:", className="mb-1"),
                                dcc.Dropdown(id='produce-dropdown', placeholder="Select good..."),
                            ], width=8),
                            dbc.Col([
                                html.Label("Producer ID:", className="mb-1"),
                                dbc.Input(id='producer-id-input', type='number', value=1001, className="form-control"),
                            ], width=4)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Rate / Cap:", className="mb-1"),
                                dbc.Input(id='production-rate-input', type='number', value=100, className="form-control"),
                            ]),
                            dbc.Col([
                                html.Label("Quantity:", className="mb-1"),
                                dbc.Input(id='production-qty-input', type='number', value=50, className="form-control"),
                            ]),
                            dbc.Col([
                                html.Label("Price:", className="mb-1"),
                                dbc.Input(id='price-input', type='number', value=0, disabled=True, className="form-control"),
                            ])
                        ], className="mb-2"),
                        
                        dbc.Checklist(id='auto-price-checkbox', options=[{'label': ' Auto-calc Price (Inputs + VA)', 'value': 'auto'}], value=['auto'], className="mb-4"),
                        
                        html.H5("Inputs (Requires Goods)"),
                        dbc.Row([
                            dbc.Col([dcc.Dropdown(id='input-good-dropdown', placeholder="Select input good...")], width=6),
                            dbc.Col([dbc.Input(id='input-qty', type='number', placeholder="Cost / Qty", className="form-control")], width=3),
                            dbc.Col([dbc.Button("Add Input", id='add-input-btn', color="secondary", outline=True, n_clicks=0, className="w-100")], width=3)
                        ], className="mb-2"),
                        html.Ul(id='current-inputs-list', className="text-muted small"),
                        dcc.Store(id='current-inputs-store', data={}),
                        
                        html.Hr(),
                        html.H5("Value Added Components"),
                        dcc.RadioItems(id='va-mode', options=[
                            {'label': ' Absolute Values ($)', 'value': 'absolute'},
                            {'label': ' Total VA + Percentages (%)', 'value': 'percentage'}
                        ], value='absolute', className="mb-3", inline=True, inputClassName="me-2", labelClassName="me-3"),
                        
                        html.Div([
                            dbc.Row([
                                dbc.Col([html.Label("Total Value Added ($):", className="mb-1")], width=4),
                                dbc.Col([dbc.Input(id='total-va-input', type='number', value=0, className="form-control")], width=8)
                            ])
                        ], id='total-va-container', style={'display': 'none'}, className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Wages ($):", id='va-wages-label', className="mb-1"),
                                dbc.Input(id='va-wages-input', type='number', value=10, className="form-control"),
                            ], width=4),
                            dbc.Col([
                                html.Label("Surplus ($):", id='va-surplus-label', className="mb-1"),
                                dbc.Input(id='va-surplus-input', type='number', value=5, className="form-control"),
                            ], width=4)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Add Production", id='add-prod-btn', color="primary", n_clicks=0, className="me-2"),
                                dbc.Button("Remove Selected", id='remove-productions-btn', color="danger", outline=True, n_clicks=0),
                            ])
                        ], className="mb-3"),
                        html.Div(id='add-prod-status', className="text-danger mb-2"),
                        
                        html.H5("Current Productions", className="mb-0 mt-4"),
                        dag.AgGrid(
                            id='productions-grid',
                            columnDefs=[
                                {'field': 'produce_name', 'headerName': 'Produces', 'checkboxSelection': True},
                                {'field': 'producer', 'headerName': 'Producer ID'},
                                {'field': 'production_inputs', 'headerName': 'Inputs'},
                                {'field': 'production_added_values', 'headerName': 'Value Added'}
                            ],
                            rowData=[],
                            dashGridOptions={'rowSelection': 'multiple'},
                            style={'height': 300, 'width': '100%'}
                        )
                    ])
                ], className="shadow-sm mb-4 h-100")
            ], width=12, lg=6)
        ], className="align-items-stretch mb-4"),
        
        # Tax Policy / Scenario Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("4. Tax Policies & Spending Profiles", className="font-weight-bold bg-info text-white"),
                    dbc.CardBody([
                        html.Small("Define the baseline final demand and distribution profiles.", className="text-muted d-block mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Scenario Title:", className="mb-1"),
                                dbc.Input(id='tax-title-input', type='text', placeholder="e.g. Baseline Tax Policy", className="form-control"),
                            ], width=6),
                            dbc.Col([
                                html.Label("Description:", className="mb-1"),
                                dbc.Input(id='tax-desc-input', type='text', placeholder="Optional details...", className="form-control"),
                            ], width=6)
                        ], className="mb-3"),

                        html.H6("Spending Profiles by Sector", className="mt-4"),
                        html.Small("Base Demand ($) defines the baseline values. The % columns dictate how dynamic income is spent (should sum to 1.0).", className="text-muted d-block mb-2"),
                        
                        dbc.Button("Load Current Goods", id='load-tax-goods-btn', color="secondary", size="sm", className="mb-2"),
                        
                        dag.AgGrid(
                            id='tax-profiles-grid',
                            columnDefs=[
                                {'field': 'isic', 'headerName': 'ISIC', 'editable': False, 'width': 120},
                                {'field': 'name', 'headerName': 'Good', 'editable': False},
                                {'field': 'base_demand', 'headerName': 'Base Demand ($)', 'editable': True, 'type': 'numericColumn'},
                                {'field': 'wage_percent', 'headerName': 'Wage Spend %', 'editable': True, 'type': 'numericColumn'},
                                {'field': 'surplus_percent', 'headerName': 'Surplus Spend %', 'editable': True, 'type': 'numericColumn'},
                                {'field': 'gov_percent', 'headerName': 'Gov Spend %', 'editable': True, 'type': 'numericColumn'}
                            ],
                            rowData=[],
                            dashGridOptions={'singleClickEdit': True},
                            style={'height': 300, 'width': '100%'}
                        ),

                        dbc.Row([
                            dbc.Col([
                                html.Label("Income Tax Rate:", className="mt-3 mb-1"),
                                dbc.Input(id='tax-income-input', type='number', value=0.1, step=0.01, min=0, max=1, className="form-control"),
                            ], width=4),
                            dbc.Col([
                                html.Label("Corporate Tax Rate:", className="mt-3 mb-1"),
                                dbc.Input(id='tax-corp-input', type='number', value=0.25, step=0.01, min=0, max=1, className="form-control"),
                            ], width=4),
                            dbc.Col([
                                html.Label("Iterations:", className="mt-3 mb-1"),
                                dbc.Input(id='tax-iterations-input', type='number', value=5, step=1, min=1, className="form-control"),
                            ], width=4)
                        ], className="mb-3"),

                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Add Scenario", id='add-tax-btn', color="primary", n_clicks=0, className="me-2"),
                                dbc.Button("Remove Selected", id='remove-tax-btn', color="danger", outline=True, n_clicks=0),
                            ])
                        ], className="mt-3 mb-3"),
                        html.Div(id='add-tax-status', className="text-danger mb-2"),

                        dag.AgGrid(
                            id='tax-policies-grid',
                            columnDefs=[
                                {'field': 'title', 'headerName': 'Title', 'checkboxSelection': True},
                                {'field': 'iterations', 'headerName': 'Iterations', 'width': 100},
                                {'field': 'income_tax_rate_after', 'headerName': 'Inc Tax', 'width': 100},
                                {'field': 'corporate_tax_rate_after', 'headerName': 'Corp Tax', 'width': 100}
                            ],
                            rowData=[],
                            dashGridOptions={'rowSelection': 'multiple'},
                            style={'height': 200, 'width': '100%'}
                        )
                    ])
                ], className="shadow-sm mb-4 h-100")
            ], width=12)
        ], className="align-items-stretch mb-4"),

        # Section 5: Technological Changes & Capital Investments
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("5. Technological Changes & Capital Investments", className="font-weight-bold"),
                    dbc.CardBody([
                        html.P(
                            "Configure technological changes (productivity shifts, efficiency changes, or input substitutions) "
                            "and specify per-investment capital requirements ($ monetary cost and build duration in iterations).",
                            className="text-muted small mb-3"
                        ),
                        # Scenario metadata
                        dbc.Row([
                            dbc.Col([
                                html.Label("Scenario Title:", className="mb-1 font-weight-bold"),
                                dbc.Input(id='tc-title-input', type='text', placeholder='e.g. Energy Efficiency with Capital Investment', className="form-control"),
                            ], width=12, md=5),
                            dbc.Col([
                                html.Label("Scenario ID (slug):", className="mb-1 font-weight-bold"),
                                dbc.Input(id='tc-id-input', type='text', placeholder='e.g. energy_efficiency_capital', className="form-control"),
                            ], width=12, md=4),
                            dbc.Col([
                                html.Label("Iterations:", className="mb-1 font-weight-bold"),
                                dbc.Input(id='tc-iterations-input', type='number', value=5, min=1, step=1, className="form-control"),
                            ], width=12, md=3),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Description:", className="mb-1"),
                                dbc.Input(id='tc-desc-input', type='text', placeholder='Describe what this tech change accomplishes...', className="form-control"),
                            ], width=12),
                        ], className="mb-3"),

                        # Investment Builder Sub-Card
                        dbc.Card([
                            dbc.CardHeader("Compose Investments for this Scenario", className="font-weight-bold bg-white text-primary"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Change Method:", className="mb-1"),
                                        dcc.Dropdown(
                                            id='tc-method-dropdown',
                                            options=[
                                                {'label': 'Single Coefficient A[i, j]', 'value': 'add_coefficient_change'},
                                                {'label': 'Input Row Across All Sectors (i)', 'value': 'add_input_change'},
                                                {'label': 'Sector Column Across All Inputs (j)', 'value': 'add_sector_change'},
                                            ],
                                            value='add_coefficient_change',
                                            clearable=False
                                        ),
                                    ], width=12, md=4),
                                    dbc.Col([
                                        html.Div(id='tc-sector-container', children=[
                                            html.Label("Sector (j - consumer/output):", className="mb-1"),
                                            dcc.Dropdown(id='tc-sector-dropdown', placeholder="Select producing sector..."),
                                        ]),
                                    ], width=12, md=4),
                                    dbc.Col([
                                        html.Div(id='tc-input-sector-container', children=[
                                            html.Label("Input Good (i - input/resource):", className="mb-1"),
                                            dcc.Dropdown(id='tc-input-sector-dropdown', placeholder="Select input good..."),
                                        ]),
                                    ], width=12, md=4),
                                ], className="mb-2"),

                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Change Type:", className="mb-1"),
                                        dcc.Dropdown(
                                            id='tc-type-dropdown',
                                            options=[
                                                {'label': 'Multiply (e.g. 0.8 = -20% inputs needed)', 'value': 'multiply'},
                                                {'label': 'Add (e.g. +0.05)', 'value': 'add'},
                                                {'label': 'Replace with value', 'value': 'replace'},
                                            ],
                                            value='multiply',
                                            clearable=False
                                        ),
                                    ], width=12, md=4),
                                    dbc.Col([
                                        html.Label("Value:", className="mb-1"),
                                        dbc.Input(id='tc-value-input', type='number', value=0.8, step=0.01, className="form-control"),
                                    ], width=12, md=2),
                                    dbc.Col([
                                        html.Label("Capital Cost ($):", className="mb-1 font-weight-bold text-success"),
                                        dbc.Input(id='tc-cost-input', type='number', min=0, value=0.0, step=25.0, placeholder="0 = free", className="form-control"),
                                    ], width=12, md=3),
                                    dbc.Col([
                                        html.Label("Build Duration (iters):", className="mb-1 font-weight-bold text-info"),
                                        dbc.Input(id='tc-duration-input', type='number', min=1, value=1, step=1, className="form-control"),
                                    ], width=12, md=3),
                                ], className="mb-3"),

                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button("+ Add Investment to Scenario", id='add-investment-btn', color="info", outline=False, n_clicks=0, className="me-2"),
                                        dbc.Button("Clear Pending Investments", id='clear-investments-btn', color="secondary", outline=True, n_clicks=0),
                                    ])
                                ], className="mb-2"),
                                html.Div(id='add-investment-status', className="text-danger small mb-2"),

                                html.Label("Staged Investments for this Scenario:", className="font-weight-bold small text-muted mt-2"),
                                html.Div(id='pending-investments-display', className="p-2 border rounded bg-white mb-2", style={'minHeight': '50px'}, children=[
                                    html.Span("No investments added to this scenario yet.", className="text-muted fst-italic")
                                ])
                            ])
                        ], className="mb-3 border-info"),

                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Save Scenario", id='add-tc-scenario-btn', color="primary", n_clicks=0, className="me-2"),
                                dbc.Button("Remove Selected Scenario", id='remove-tc-scenario-btn', color="danger", outline=True, n_clicks=0),
                            ])
                        ], className="mt-2 mb-3"),
                        html.Div(id='add-tc-scenario-status', className="text-danger mb-2"),

                        dag.AgGrid(
                            id='tech-changes-grid',
                            columnDefs=[
                                {'field': 'example_id', 'headerName': 'ID', 'width': 70, 'checkboxSelection': True},
                                {'field': 'tech_change_id', 'headerName': 'Slug', 'width': 180},
                                {'field': 'title', 'headerName': 'Title', 'width': 260},
                                {'field': 'investments_count', 'headerName': '# Investments', 'width': 130},
                                {'field': 'total_capital_cost', 'headerName': 'Total Cost ($)', 'width': 130},
                                {'field': 'iterations', 'headerName': 'Iterations', 'width': 100},
                                {'field': 'description', 'headerName': 'Description', 'flex': 1},
                            ],
                            rowData=[],
                            dashGridOptions={'rowSelection': 'multiple'},
                            style={'height': 220, 'width': '100%'}
                        )
                    ])
                ], className="shadow-sm mb-4 h-100")
            ], width=12)
        ], className="align-items-stretch")
    ], fluid=True, className="p-4 bg-light")

app.layout = serve_layout

# --- Callbacks ---

@app.callback(
    Output('isic-division', 'options'),
    Output('isic-division', 'value'),
    Input('isic-section', 'value')
)
def update_divisions(section):
    if not section: return [], None
    return ISIC_HIERARCHY['divisions'].get(section, []), None

@app.callback(
    Output('isic-group', 'options'),
    Output('isic-group', 'value'),
    Input('isic-division', 'value')
)
def update_groups(division):
    if not division: return [], None
    return ISIC_HIERARCHY['groups'].get(division, []), None

@app.callback(
    Output('isic-class', 'options'),
    Output('isic-class', 'value'),
    Input('isic-group', 'value')
)
def update_classes(group):
    if not group: return [], None
    return ISIC_HIERARCHY['classes'].get(group, []), None

@app.callback(
    Output('import-confirm-dialog', 'displayed'),
    Output('goods-store', 'data', allow_duplicate=True),
    Output('productions-store', 'data', allow_duplicate=True),
    Output('tax-policies-store', 'data', allow_duplicate=True),
    Output('tech-changes-store', 'data', allow_duplicate=True),
    Output('goods-grid', 'rowData', allow_duplicate=True),
    Output('productions-grid', 'rowData', allow_duplicate=True),
    Output('target-folder-input', 'value', allow_duplicate=True),
    Output('import-status', 'children'),
    Input('import-btn', 'n_clicks'),
    Input('import-confirm-dialog', 'submit_n_clicks'),
    State('import-dropdown', 'value'),
    State('goods-store', 'data'),
    prevent_initial_call=True
)
def handle_import(import_btn, confirm_btn, import_folder, current_goods):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'import-btn':
        if not import_folder:
            return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, html.Span("Please select a dataset to import.", style={'color': 'red'})
        
        # If there is existing unsaved data, prompt
        if current_goods and len(current_goods) > 0:
            return True, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        # Otherwise, proceed to import naturally
        return perform_import(import_folder)
        
    if trigger_id == 'import-confirm-dialog':
        return perform_import(import_folder)
        
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

def perform_import(folder_name):
    base_dir = os.path.join(os.path.dirname(__file__), 'data', folder_name)
    try:
        goods_df = pd.read_csv(os.path.join(base_dir, 'goods.csv')).fillna('')
        goods_data = goods_df.to_dict('records')
        
        prods_df = pd.read_csv(os.path.join(base_dir, 'productions.csv')).fillna('')
        prods_data = prods_df.to_dict('records')

        tax_data = []
        tax_path = os.path.join(base_dir, 'tax_policies.csv')
        if os.path.exists(tax_path):
            tax_df = pd.read_csv(tax_path).fillna('')
            tax_data = tax_df.to_dict('records')

        tech_data = []
        tech_path = os.path.join(base_dir, 'tech_changes.csv')
        if os.path.exists(tech_path):
            tech_df = pd.read_csv(tech_path).fillna('')
            tech_data = parse_tech_changes_df(tech_df)
        
        msg = f"Imported '{folder_name}' successfully ({len(goods_data)} goods, {len(prods_data)} prods, {len(tax_data)} tax scenarios, {len(tech_data)} tech scenarios)."
        return False, goods_data, prods_data, tax_data, tech_data, goods_data, prods_data, folder_name, html.Span(msg, style={'color': 'green'})
    except Exception as e:
        return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, html.Span(f"Error importing dataset: {e}", style={'color': 'red'})

@app.callback(
    Output('selected-isic-store', 'data'),
    Output('selected-isic-display', 'children'),
    Input('isic-dropdown-direct', 'value'),
    Input('isic-class', 'value'),
    prevent_initial_call=True
)
def sync_selected_isic(direct_val, class_val):
    ctx = callback_context
    if not ctx.triggered:
        return None, "No ISIC selected."
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'isic-dropdown-direct' and direct_val:
        data = json.loads(direct_val)
        return direct_val, f"Selected: {data['raw']} - {data['title']} (from Direct Search)"
    elif triggered_id == 'isic-class' and class_val:
        data = json.loads(class_val)
        return class_val, f"Selected: {data['raw']} - {data['title']} (from Hierarchy)"
        
    return None, "No ISIC selected."

@app.callback(
    Output('goods-store', 'data'),
    Output('goods-grid', 'rowData'),
    Output('add-good-status', 'children'),
    Output('good-name-input', 'value'),
    Output('good-desc-input', 'value'),
    Input('add-good-btn', 'n_clicks'),
    Input('remove-goods-btn', 'n_clicks'),
    Input('goods-grid', 'cellValueChanged'),
    State('goods-grid', 'selectedRows'),
    State('selected-isic-store', 'data'),
    State('good-name-input', 'value'),
    State('good-desc-input', 'value'),
    State('isic-sub1-input', 'value'),
    State('isic-sub2-input', 'value'),
    State('goods-store', 'data')
)
def manage_goods(add_n, remove_n, cell_change, selected_rows, isic_val, name, desc, sub1, sub2, goods):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, "", dash.no_update, dash.no_update
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'remove-goods-btn':
        if not selected_rows: return dash.no_update, dash.no_update, "No goods selected for removal.", dash.no_update, dash.no_update
        ids_to_remove = [r['id'] for r in selected_rows]
        updated_goods = [g for g in goods if g['id'] not in ids_to_remove]
        return updated_goods, updated_goods, f"Removed {len(ids_to_remove)} good(s).", dash.no_update, dash.no_update
        
    if trigger_id == 'goods-grid':
        # Cell value changed, cell_change contains the updated row
        for updated_row in cell_change:
            for i, g in enumerate(goods):
                if g['id'] == updated_row['data']['id']:
                    goods[i] = updated_row['data']
        return goods, goods, "Good updated.", dash.no_update, dash.no_update

    if trigger_id == 'add-good-btn':
        if not isic_val or not name:
            return dash.no_update, dash.no_update, "ISIC and Name are required.", dash.no_update, dash.no_update
        
        isic_data = json.loads(isic_val)
        base_isic = isic_data['formatted'] # e.g. A0111_000_000
        # Modify the base ISIC with custom sub parts
        parts = base_isic.split('_')
        s1 = sub1 if sub1 else '000'
        s2 = sub2 if sub2 else '000'
        formatted_isic = f"{parts[0]}_{s1}_{s2}"
    
        if any(g['isic'] == formatted_isic for g in goods):
            return dash.no_update, dash.no_update, "Good with this ISIC already exists.", dash.no_update, dash.no_update
        
        new_id = len(goods) + 1 if not goods else max(g['id'] for g in goods) + 1
        new_good = {
            'id': new_id,
            'name': name,
            'descriptive_name': desc or "",
            'id_number': new_id * 1000,
            'isic': formatted_isic,
            'isic_section': formatted_isic[0],
            'isic_division': formatted_isic[1:3] if len(formatted_isic) >= 3 else '00',
            'isic_group': formatted_isic[3:4] if len(formatted_isic) >= 4 else '0',
            'isic_class': formatted_isic[4:5] if len(formatted_isic) >= 5 else '0',
            'sub_class_a': '0', 'sub_class_b': '0', 'sub_class_c': '0', 'sub_class_nf': '000'
        }
        goods.append(new_good)
        return goods, goods, "Good added successfully.", "", ""

@app.callback(
    Output('produce-dropdown', 'options'),
    Output('input-good-dropdown', 'options'),
    Output('tc-sector-dropdown', 'options'),
    Output('tc-input-sector-dropdown', 'options'),
    Input('goods-store', 'data')
)
def update_good_dropdowns(goods):
    goods = goods or []
    produce_opts = [{'label': g.get('name', 'Unknown'), 'value': g.get('isic', '')} for g in goods]
    detailed_opts = [{'label': f"{g.get('name', 'Unknown')} ({g.get('isic', '')})", 'value': g.get('isic', '')} for g in goods]
    return produce_opts, produce_opts, detailed_opts, detailed_opts

@app.callback(
    Output('current-inputs-store', 'data'),
    Output('current-inputs-list', 'children'),
    Input('add-input-btn', 'n_clicks'),
    Input('add-prod-btn', 'n_clicks'),
    State('input-good-dropdown', 'value'),
    State('input-qty', 'value'),
    State('current-inputs-store', 'data'),
    State('goods-store', 'data')
)
def manage_inputs(add_in_n, add_prod_n, good_isic, qty, current_inputs, goods):
    ctx = callback_context
    if not ctx.triggered:
        return current_inputs, [html.Li(f"{next((g['name'] for g in goods if g['isic'] == k), k)}: {v}") for k, v in current_inputs.items()]
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'add-prod-btn':
        return {}, []
    
    if triggered_id == 'add-input-btn':
        if good_isic and qty is not None:
            current_inputs[good_isic] = qty
        
        list_items = []
        for k, v in current_inputs.items():
            g_name = next((g['name'] for g in goods if g['isic'] == k), k)
            list_items.append(html.Li(f"{g_name} ({k}): {v}"))
            
        return current_inputs, list_items
    
    return current_inputs, []

@app.callback(
    Output('total-va-container', 'style'),
    Output('va-wages-label', 'children'),
    Output('va-surplus-label', 'children'),
    Input('va-mode', 'value')
)
def toggle_va_mode(mode):
    if mode == 'percentage':
        return {'display': 'block', 'marginBottom': '10px'}, "Wages (%):", "Surplus (%):"
    return {'display': 'none', 'marginBottom': '10px'}, "Wages ($):", "Surplus ($):"

@app.callback(
    Output('price-input', 'value'),
    Output('price-input', 'disabled'),
    Input('auto-price-checkbox', 'value'),
    Input('current-inputs-store', 'data'),
    Input('va-mode', 'value'),
    Input('total-va-input', 'value'),
    Input('va-wages-input', 'value'),
    Input('va-surplus-input', 'value'),
    State('price-input', 'value')
)
def live_update_price(auto_price, current_inputs, va_mode, total_va, wages, surplus, current_price):
    is_auto = 'auto' in (auto_price or [])
    
    if not is_auto:
        return dash.no_update, False
        
    inputs_sum = sum(current_inputs.values()) if current_inputs else 0.0
    w = wages or 0.0
    s = surplus or 0.0
    
    if va_mode == 'absolute':
        va_sum = w + s
    else:
        va_sum = total_va or 0.0
        
    calculated_price = inputs_sum + va_sum
    return calculated_price, True

@app.callback(
    Output('productions-store', 'data'),
    Output('productions-grid', 'rowData'),
    Output('add-prod-status', 'children'),
    Input('add-prod-btn', 'n_clicks'),
    Input('remove-productions-btn', 'n_clicks'),
    State('productions-grid', 'selectedRows'),
    State('produce-dropdown', 'value'),
    State('producer-id-input', 'value'),
    State('production-rate-input', 'value'),
    State('production-qty-input', 'value'),
    State('price-input', 'value'),
    State('current-inputs-store', 'data'),
    State('va-mode', 'value'),
    State('total-va-input', 'value'),
    State('va-wages-input', 'value'),
    State('va-surplus-input', 'value'),
    State('goods-store', 'data'),
    State('productions-store', 'data')
)
def manage_productions(add_n, remove_n, selected_rows, produce_isic, producer_id, p_rate, p_qty, price, inputs_dict, va_mode, total_va, wages, surplus, goods, productions):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, ""
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'remove-productions-btn':
        if not selected_rows: return dash.no_update, dash.no_update, "No productions selected for removal."
        ids_to_remove = [r['id'] for r in selected_rows]
        updated_prods = [p for p in productions if p['id'] not in ids_to_remove]
        return updated_prods, updated_prods, f"Removed {len(ids_to_remove)} production(s)."

    if trigger_id == 'add-prod-btn':
        if not produce_isic:
            return dash.no_update, dash.no_update, "Select a good to produce."
        
        target_good = next((g for g in goods if g['isic'] == produce_isic), None)
        if not target_good:
            return dash.no_update, dash.no_update, "Invalid good selected."
            
        new_id = len(productions) + 1 if not productions else max(p['id'] for p in productions) + 1
    
    w = wages or 0.0
    s = surplus or 0.0
    
    if va_mode == 'percentage':
        if abs(w + s - 100.0) > 0.001:
            return dash.no_update, dash.no_update, f"Error: Percentages sum to {w+s}% instead of 100%."
            
        tot = total_va or 0.0
        w = tot * (w / 100.0)
        s = tot * (s / 100.0)
    
    va = {
        "wages": w,
        "surplus": s
    }
    
    prod = {
        'id': new_id,
        'name': f"{target_good['name']} Production",
        'descriptive_name': "",
        'id_number': new_id * 2000,
        'isic': produce_isic,
        'producer': producer_id or 1001,
        'produce': target_good['id_number'],
        'produce_name': target_good['name'],
        'production_inputs': json.dumps(inputs_dict),
        'production_added_values': json.dumps(va),
        'production_rate': p_rate or 0,
        'production_quantity': p_qty or 0,
        'price': price or 0
    }
    
    productions.append(prod)
    
    return productions, productions, "Production added successfully."

@app.callback(
    Output('generate-confirm-dialog', 'displayed'),
    Output('generate-status', 'children'),
    Output('generate-status', 'style'),
    Input('generate-btn', 'n_clicks'),
    State('target-folder-input', 'value'),
    State('goods-store', 'data'),
    State('productions-store', 'data'),
    State('tax-policies-store', 'data'),
    State('tech-changes-store', 'data')
)
def generate_files_check(n, folder_name, goods, productions, tax_policies, tech_changes):
    if not n:
        return dash.no_update, dash.no_update, dash.no_update
    if not folder_name:
        return False, "Please specify a Target Folder Name.", {'color': 'red'}
    if not goods:
        return False, "Please add at least one Good.", {'color': 'red'}
    
    # Validation: Ensure all goods have at least 1 production
    produced_isics = set(p['isic'] for p in productions)
    missing_prods = [g['name'] for g in goods if g['isic'] not in produced_isics]
    if missing_prods:
        return False, f"Validation Error: The following goods have no production defined: {', '.join(missing_prods)}", {'color': 'red'}
        
    # Validation: Ensure no production references a non-existent good (as input or produced)
    valid_isics = set(g['isic'] for g in goods)
    for p in productions:
        if p['isic'] not in valid_isics:
            return False, f"Validation Error: Production '{p['name']}' produces a good that no longer exists.", {'color': 'red'}
        p_inputs = json.loads(p['production_inputs'])
        for in_isic in p_inputs.keys():
            if in_isic not in valid_isics:
                return False, f"Validation Error: Production '{p['name']}' uses a non-existent input good ({in_isic}).", {'color': 'red'}
    
    base_dir = os.path.join(os.path.dirname(__file__), 'data', folder_name)
    if os.path.exists(base_dir) and len(os.listdir(base_dir)) > 0:
        return True, dash.no_update, dash.no_update
        
    msg, style = perform_generation(folder_name, goods, productions, tax_policies, tech_changes)
    return False, msg, style

@app.callback(
    Output('generate-status', 'children', allow_duplicate=True),
    Output('generate-status', 'style', allow_duplicate=True),
    Input('generate-confirm-dialog', 'submit_n_clicks'),
    State('target-folder-input', 'value'),
    State('goods-store', 'data'),
    State('productions-store', 'data'),
    State('tax-policies-store', 'data'),
    State('tech-changes-store', 'data'),
    prevent_initial_call=True
)
def generate_files_confirmed(confirm_n, folder_name, goods, productions, tax_policies, tech_changes):
    if not confirm_n:
        return dash.no_update, dash.no_update
    return perform_generation(folder_name, goods, productions, tax_policies, tech_changes)

def perform_generation(folder_name, goods, productions, tax_policies, tech_changes):
    base_dir = os.path.join(os.path.dirname(__file__), 'data', folder_name)
    try:
        os.makedirs(base_dir, exist_ok=True)
    except Exception as e:
        return f"Error creating directory: {e}", {'color': 'red'}
        
    goods_df = pd.DataFrame(goods)
    expected_good_cols = ['id','name','descriptive_name','id_number','isic','isic_section','isic_division','isic_group','isic_class','sub_class_a','sub_class_b','sub_class_c','sub_class_nf']
    for c in expected_good_cols:
        if c not in goods_df.columns:
            goods_df[c] = ''
    goods_df[expected_good_cols].to_csv(os.path.join(base_dir, 'goods.csv'), index=False)
    
    prod_df = pd.DataFrame(productions)
    expected_prod_cols = ['id','name','descriptive_name','id_number','isic','producer','produce','produce_name','production_inputs','production_added_values','production_rate','production_quantity','price']
    for c in expected_prod_cols:
        if c not in prod_df.columns:
            prod_df[c] = ''
            
    dummy_cols = ['production_material_efficiency', 'production_labour_efficiency', 'production_energy_efficiency', 'contact_name', 'contact_email', 'contact_phone', 'contact_phone2', 'contact_website', 'address', 'address_street', 'address_city', 'address_country', 'address_postal_code', 'total_inputs_cost', 'total_value_added']
    for d in dummy_cols:
        prod_df[d] = ''
        
    full_prod_cols = expected_prod_cols[:12] + dummy_cols + ['price']
    prod_df[full_prod_cols].to_csv(os.path.join(base_dir, 'productions.csv'), index=False)
    
    if tax_policies and len(tax_policies) > 0:
        tax_df = pd.DataFrame(tax_policies)
        expected_tax_cols = ['example_id','change_type','title','description','final_demand','income_tax_rate_before','income_tax_rate_after','corporate_tax_rate_before','corporate_tax_rate_after','income_tax_applies_to','iterations','wage_proportions','surplus_proportions','government_proportions','tech_change_function_name','tech_change_params']
        for c in expected_tax_cols:
            if c not in tax_df.columns:
                tax_df[c] = ''
        tax_df[expected_tax_cols].to_csv(os.path.join(base_dir, 'tax_policies.csv'), index=False)
    
    if tech_changes and len(tech_changes) > 0:
        tc_rows = []
        for s in tech_changes:
            seq = 1
            invs = s.get('investments', [])
            has_multi = any('curve' in inv.get('method', '') or 'production' in inv.get('method', '') for inv in invs)
            for inv in invs:
                method = inv.get('method', 'add_coefficient_change')
                row = {
                    'example_id': s.get('example_id', 1),
                    'tech_change_id': s.get('tech_change_id', ''),
                    'change_type': s.get('change_type', 'tech_change'),
                    'title': s.get('title', ''),
                    'description': s.get('description', ''),
                    'final_demand': s.get('final_demand', ''),
                    'income_tax_rate_before': s.get('income_tax_rate_before', ''),
                    'income_tax_rate_after': s.get('income_tax_rate_after', ''),
                    'corporate_tax_rate_before': s.get('corporate_tax_rate_before', ''),
                    'corporate_tax_rate_after': s.get('corporate_tax_rate_after', ''),
                    'income_tax_applies_to': s.get('income_tax_applies_to', ''),
                    'iterations': s.get('iterations', 5),
                    'wage_proportions': s.get('wage_proportions', ''),
                    'surplus_proportions': s.get('surplus_proportions', ''),
                    'government_proportions': s.get('government_proportions', ''),
                    'use_multi_level': True if has_multi else '',
                    'sequence_number': seq,
                    'method': method,
                    'sector_idx': inv.get('sector_idx', ''),
                    'input_sector_idx': inv.get('input_sector_idx', ''),
                    'production_id': inv.get('production_id', ''),
                    'isic': inv.get('isic', ''),
                    'tier_index': inv.get('tier_index', ''),
                    'field': inv.get('field', ''),
                    'change_type_param': inv.get('change_type', 'multiply'),
                    'value': inv.get('value', ''),
                    'efficiency_type': inv.get('efficiency_type', ''),
                    'va_component': inv.get('va_component', ''),
                    'input_isic': inv.get('input_isic', ''),
                    'exclude_list': inv.get('exclude_list', ''),
                    'position': inv.get('position', ''),
                    'cap': inv.get('cap', ''),
                    'price': inv.get('price', ''),
                    'va_components': inv.get('va_components', ''),
                    'requirements': '',
                    'investment_duration': ''
                }
                tc_rows.append(row)
                seq += 1

                cost = float(inv.get('capital_cost', 0.0) or 0.0)
                dur = int(inv.get('investment_duration', 1) or 1)
                if cost > 0 or dur > 1:
                    req_row = dict(row)
                    req_row['sequence_number'] = seq
                    req_row['method'] = 'set_capital_requirements'
                    req_row['sector_idx'] = ''
                    req_row['input_sector_idx'] = ''
                    req_row['change_type_param'] = ''
                    req_row['value'] = ''
                    req_row['requirements'] = json.dumps({"_total": cost})
                    req_row['investment_duration'] = dur
                    tc_rows.append(req_row)
                    seq += 1

        expected_tc_cols = [
            'example_id', 'tech_change_id', 'change_type', 'title', 'description',
            'final_demand', 'income_tax_rate_before', 'income_tax_rate_after',
            'corporate_tax_rate_before', 'corporate_tax_rate_after', 'income_tax_applies_to',
            'iterations', 'wage_proportions', 'surplus_proportions', 'government_proportions',
            'use_multi_level', 'sequence_number', 'method', 'sector_idx', 'input_sector_idx',
            'production_id', 'isic', 'tier_index', 'field', 'change_type_param', 'value',
            'efficiency_type', 'va_component', 'input_isic', 'exclude_list',
            'position', 'cap', 'price', 'va_components', 'requirements', 'investment_duration'
        ]
        tc_df = pd.DataFrame(tc_rows)
        for c in expected_tc_cols:
            if c not in tc_df.columns:
                tc_df[c] = ''
        tc_df[expected_tc_cols].to_csv(os.path.join(base_dir, 'tech_changes.csv'), index=False)

    return f"Files generated successfully in data/{folder_name}!", {'color': 'green'}

# --- Tax Policies Callbacks ---

@app.callback(
    Output('tax-profiles-grid', 'rowData'),
    Input('load-tax-goods-btn', 'n_clicks'),
    State('goods-store', 'data'),
    prevent_initial_call=True
)
def load_tax_goods(n, goods):
    if not goods: return []
    return [{
        'isic': g['isic'], 
        'name': g['name'], 
        'base_demand': 100, 
        'wage_percent': round(1.0/len(goods), 4),
        'surplus_percent': round(1.0/len(goods), 4),
        'gov_percent': round(1.0/len(goods), 4)
    } for g in goods]

@app.callback(
    Output('tax-policies-store', 'data', allow_duplicate=True),
    Output('add-tax-status', 'children'),
    Input('add-tax-btn', 'n_clicks'),
    State('tax-title-input', 'value'),
    State('tax-desc-input', 'value'),
    State('tax-profiles-grid', 'rowData'),
    State('tax-income-input', 'value'),
    State('tax-corp-input', 'value'),
    State('tax-iterations-input', 'value'),
    State('tax-policies-store', 'data'),
    prevent_initial_call=True
)
def add_tax_policy(n, title, desc, rowData, inc_tax, corp_tax, iters, policies):
    if not title or not rowData:
        return dash.no_update, "Title and loaded goods are required."
    
    try:
        base_d = ";".join(str(float(r.get('base_demand', 0))) for r in rowData)
        wage_p = ";".join(str(float(r.get('wage_percent', 0))) for r in rowData)
        surplus_p = ";".join(str(float(r.get('surplus_percent', 0))) for r in rowData)
        gov_p = ";".join(str(float(r.get('gov_percent', 0))) for r in rowData)
    except Exception as e:
        return dash.no_update, f"Error parsing numbers: {e}"

    new_id = len(policies) + 1
    policy = {
        'example_id': new_id,
        'change_type': 'tax_policy',
        'title': title,
        'description': desc or "",
        'final_demand': base_d,
        'income_tax_rate_before': inc_tax or 0.0,
        'income_tax_rate_after': inc_tax or 0.0,
        'corporate_tax_rate_before': corp_tax or 0.0,
        'corporate_tax_rate_after': corp_tax or 0.0,
        'income_tax_applies_to': 'wages',
        'iterations': iters or 5,
        'wage_proportions': wage_p,
        'surplus_proportions': surplus_p,
        'government_proportions': gov_p,
        'tech_change_function_name': '',
        'tech_change_params': ''
    }
    
    policies.append(policy)
    return policies, "Scenario added."

@app.callback(
    Output('tax-policies-grid', 'rowData'),
    Input('tax-policies-store', 'data')
)
def update_tax_grid(policies):
    return policies or []

@app.callback(
    Output('tax-policies-store', 'data', allow_duplicate=True),
    Input('remove-tax-btn', 'n_clicks'),
    State('tax-policies-grid', 'selectedRows'),
    State('tax-policies-store', 'data'),
    prevent_initial_call=True
)
def remove_tax_policy(n, selected, policies):
    if not selected: return dash.no_update
    titles_to_remove = {s['title'] for s in selected}
    new_policies = [p for p in policies if p['title'] not in titles_to_remove]
    for i, p in enumerate(new_policies):
        p['example_id'] = i + 1
    return new_policies

# --- Technological Changes Callbacks ---

@app.callback(
    Output('tc-sector-container', 'style'),
    Output('tc-input-sector-container', 'style'),
    Input('tc-method-dropdown', 'value')
)
def toggle_tc_method_fields(method):
    if method == 'add_input_change':
        return {'display': 'none'}, {'display': 'block'}
    elif method == 'add_sector_change':
        return {'display': 'block'}, {'display': 'none'}
    else:  # add_coefficient_change
        return {'display': 'block'}, {'display': 'block'}

@app.callback(
    Output('pending-investments-store', 'data'),
    Output('pending-investments-display', 'children'),
    Output('add-investment-status', 'children'),
    Input('add-investment-btn', 'n_clicks'),
    Input('clear-investments-btn', 'n_clicks'),
    State('tc-method-dropdown', 'value'),
    State('tc-sector-dropdown', 'value'),
    State('tc-input-sector-dropdown', 'value'),
    State('tc-type-dropdown', 'value'),
    State('tc-value-input', 'value'),
    State('tc-cost-input', 'value'),
    State('tc-duration-input', 'value'),
    State('pending-investments-store', 'data'),
    State('goods-store', 'data'),
    prevent_initial_call=True
)
def manage_pending_investments(add_btn, clear_btn, method, sector, input_sector,
                               change_type, value, cost, duration,
                               pending, goods):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    pending = list(pending or [])

    if trigger_id == 'clear-investments-btn':
        return [], html.Span("No investments added to this scenario yet.", className="text-muted fst-italic"), ""

    if trigger_id == 'add-investment-btn':
        if value is None:
            return dash.no_update, dash.no_update, "Value is required."

        if method in ('add_coefficient_change', 'add_sector_change') and not sector:
            return dash.no_update, dash.no_update, "Sector (j) is required for this method."

        if method in ('add_coefficient_change', 'add_input_change') and not input_sector:
            return dash.no_update, dash.no_update, "Input good (i) is required for this method."

        cost_val = float(cost or 0.0)
        dur_val = int(duration or 1)

        inv = {
            'method': method,
            'sector_idx': sector if method in ('add_coefficient_change', 'add_sector_change') else '',
            'input_sector_idx': input_sector if method in ('add_coefficient_change', 'add_input_change') else '',
            'change_type': change_type or 'multiply',
            'value': float(value),
            'capital_cost': cost_val,
            'investment_duration': dur_val
        }
        pending.append(inv)

    # Render pending list
    good_map = {g.get('isic'): g.get('name') for g in (goods or [])}
    items = []
    for idx, inv in enumerate(pending, 1):
        m = inv['method']
        sec_name = good_map.get(inv['sector_idx'], inv['sector_idx']) if inv['sector_idx'] else ''
        in_name = good_map.get(inv['input_sector_idx'], inv['input_sector_idx']) if inv['input_sector_idx'] else ''
        
        if m == 'add_coefficient_change':
            desc = f"A[{in_name}, {sec_name}]: {inv['change_type']} {inv['value']}"
        elif m == 'add_input_change':
            desc = f"Input row [{in_name}] everywhere: {inv['change_type']} {inv['value']}"
        elif m == 'add_sector_change':
            desc = f"Sector [{sec_name}] all inputs: {inv['change_type']} {inv['value']}"
        else:
            desc = f"{m}: {inv['change_type']} {inv['value']}"

        cost_txt = f"${inv['capital_cost']:,.2f}" if inv['capital_cost'] > 0 else "Free ($0)"
        dur_txt = f"{inv['investment_duration']} iters" if inv['investment_duration'] > 1 else "1 iter"

        items.append(
            dbc.Badge(
                f"#{idx} {desc} | Req: {cost_txt}, {dur_txt}",
                color="secondary" if inv['capital_cost'] == 0 else "success",
                className="me-2 mb-2 p-2",
                style={'fontSize': '0.85rem'}
            )
        )

    return pending, items if items else html.Span("No investments added to this scenario yet.", className="text-muted fst-italic"), ""

@app.callback(
    Output('tech-changes-store', 'data', allow_duplicate=True),
    Output('pending-investments-store', 'data', allow_duplicate=True),
    Output('pending-investments-display', 'children', allow_duplicate=True),
    Output('add-tc-scenario-status', 'children'),
    Output('tc-title-input', 'value'),
    Output('tc-id-input', 'value'),
    Output('tc-desc-input', 'value'),
    Input('add-tc-scenario-btn', 'n_clicks'),
    State('tc-title-input', 'value'),
    State('tc-id-input', 'value'),
    State('tc-desc-input', 'value'),
    State('tc-iterations-input', 'value'),
    State('pending-investments-store', 'data'),
    State('tech-changes-store', 'data'),
    prevent_initial_call=True
)
def add_tc_scenario(n, title, tc_id, desc, iters, pending, scenarios):
    if not title:
        return dash.no_update, dash.no_update, dash.no_update, "Scenario title is required.", dash.no_update, dash.no_update, dash.no_update

    if not pending or len(pending) == 0:
        return dash.no_update, dash.no_update, dash.no_update, "At least one investment must be added to this scenario.", dash.no_update, dash.no_update, dash.no_update

    scenarios = list(scenarios or [])
    new_id = len(scenarios) + 1 if not scenarios else max(s.get('example_id', 0) for s in scenarios) + 1

    slug = tc_id.strip() if tc_id and tc_id.strip() else title.lower().replace(' ', '_').replace('-', '_')
    slug = "".join(c for c in slug if c.isalnum() or c == '_')

    total_cost = sum(float(inv.get('capital_cost', 0.0)) for inv in pending)
    scenario = {
        'example_id': new_id,
        'tech_change_id': slug,
        'change_type': 'tech_change',
        'title': title,
        'description': desc or "",
        'iterations': int(iters or 5),
        'investments_count': len(pending),
        'total_capital_cost': total_cost,
        'investments': list(pending)
    }
    scenarios.append(scenario)

    return scenarios, [], html.Span("No investments added to this scenario yet.", className="text-muted fst-italic"), html.Span("Scenario saved successfully.", style={'color': 'green'}), "", "", ""

@app.callback(
    Output('tech-changes-store', 'data', allow_duplicate=True),
    Input('remove-tc-scenario-btn', 'n_clicks'),
    State('tech-changes-grid', 'selectedRows'),
    State('tech-changes-store', 'data'),
    prevent_initial_call=True
)
def remove_tc_scenario(n, selected, scenarios):
    if not selected:
        return dash.no_update
    ids_to_remove = {s['example_id'] for s in selected}
    new_scenarios = [s for s in (scenarios or []) if s.get('example_id') not in ids_to_remove]
    for i, s in enumerate(new_scenarios):
        s['example_id'] = i + 1
    return new_scenarios

@app.callback(
    Output('tech-changes-grid', 'rowData'),
    Input('tech-changes-store', 'data')
)
def update_tech_changes_grid(scenarios):
    return scenarios or []

if __name__ == '__main__':
    print("Starting Sambaza-Sim Data Builder Utility on http://127.0.0.1:8051/")
    app.run(debug=True, port=8051)
