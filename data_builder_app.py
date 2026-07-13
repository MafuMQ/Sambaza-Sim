import os
import sys
import json
import pandas as pd
import dash
from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context
import dash_ag_grid as dag

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

app = Dash(__name__)

def serve_layout():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    import_options = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, f)) and not f.startswith('.'):
                if os.path.exists(os.path.join(data_dir, f, 'goods.csv')) and os.path.exists(os.path.join(data_dir, f, 'productions.csv')):
                    import_options.append({'label': f, 'value': f})

    return html.Div([
        dcc.Store(id='goods-store', data=[]),
        dcc.Store(id='productions-store', data=[]),
        
        dcc.ConfirmDialog(id='import-confirm-dialog', message='Importing will overwrite your current unsaved session data. Continue?'),
        dcc.ConfirmDialog(id='generate-confirm-dialog', message='This dataset folder already exists. Generating will overwrite it. Continue?'),
        
        html.H1("Sambaza-Sim Data Builder Utility"),
        html.P("Dynamically create and output input files for the simulator."),
        
        html.Div([
            html.H3("0. Import Existing Dataset"),
            html.Label("Select Dataset:"),
            dcc.Dropdown(id='import-dropdown', options=import_options, placeholder="Select an existing scenario to import...", style={'width': '300px', 'marginBottom': '10px'}),
            html.Button("Import Dataset", id='import-btn', n_clicks=0, style={'padding': '10px', 'backgroundColor': '#f39c12', 'color': 'white', 'cursor': 'pointer', 'border': 'none'}),
            html.Div(id='import-status', style={'marginTop': '10px'})
        ], style={'padding': '20px', 'border': '1px solid #ccc', 'marginBottom': '20px', 'backgroundColor': '#fdfbf7'}),
        
        html.Div([
            html.H3("1. Output Configuration"),
            html.Label("Target Folder Name (inside data/):"),
            dcc.Input(id='target-folder-input', type='text', placeholder='e.g., custom_scenario_1', style={'width': '300px'}),
            html.Div(id='folder-status', style={'color': 'red', 'marginTop': '5px'})
        ], style={'padding': '20px', 'border': '1px solid #ccc', 'marginBottom': '20px'}),
    
    # GOODS SECTION
    html.Div([
        html.H3("2. Add Goods"),
        
        dcc.Tabs([
            dcc.Tab(label='Direct Search', children=[
                html.Div([
                    html.Label("Search ISIC Sector:"),
                    dcc.Dropdown(id='isic-dropdown-direct', options=ISIC_OPTIONS, placeholder="Search ISIC sector..."),
                ], style={'padding': '15px'})
            ]),
            dcc.Tab(label='Hierarchical Selection', children=[
                html.Div([
                    html.Label("1. Section:"),
                    dcc.Dropdown(id='isic-section', options=ISIC_HIERARCHY['sections'], placeholder="Select Section..."),
                    html.Label("2. Division:"),
                    dcc.Dropdown(id='isic-division', placeholder="Select Division..."),
                    html.Label("3. Group:"),
                    dcc.Dropdown(id='isic-group', placeholder="Select Group..."),
                    html.Label("4. Class:"),
                    dcc.Dropdown(id='isic-class', placeholder="Select Class..."),
                ], style={'padding': '15px'})
            ])
        ]),
        
        # Hidden store for the ultimately selected ISIC
        dcc.Store(id='selected-isic-store', data=None),
        html.Div(id='selected-isic-display', style={'fontWeight': 'bold', 'color': '#2c3e50', 'margin': '10px 0'}),
        
        html.Div([
            html.Label("Custom ISIC Sub-Class 1 (defaults to 000):"),
            dcc.Input(id='isic-sub1-input', type='text', value='000', style={'marginRight': '15px'}),
            html.Label("Custom ISIC Sub-Class 2 (defaults to 000):"),
            dcc.Input(id='isic-sub2-input', type='text', value='000')
        ], style={'marginBottom': '10px'}),
        
        html.Label("Good Name:"),
        dcc.Input(id='good-name-input', type='text', placeholder="e.g., Steel"),
        
        html.Label("Description:"),
        dcc.Input(id='good-desc-input', type='text', placeholder="e.g., High-grade steel", style={'width': '100%'}),
        
        html.Button("Add Good", id='add-good-btn', n_clicks=0, style={'marginTop': '10px', 'marginRight': '10px'}),
        html.Button("Remove Selected Goods", id='remove-goods-btn', n_clicks=0, style={'marginTop': '10px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none', 'padding': '5px 10px', 'cursor': 'pointer'}),
        html.Div(id='add-good-status', style={'color': 'red', 'marginTop': '10px'}),
        
        html.H4("Current Goods (Edit Descriptive Name Directly)"),
        dag.AgGrid(
            id='goods-grid',
            columnDefs=[
                {'field': 'id', 'headerName': 'Internal ID', 'editable': False, 'checkboxSelection': True},
                {'field': 'name', 'editable': False},
                {'field': 'isic', 'headerName': 'Formatted ISIC', 'editable': False},
                {'field': 'descriptive_name', 'editable': True}
            ],
            rowData=[],
            dashGridOptions={'rowSelection': 'multiple'},
            style={'height': 200, 'width': '100%'}
        )
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'marginBottom': '20px'}),
    
    # PRODUCTIONS SECTION
    html.Div([
        html.H3("3. Add Production"),
        html.P("Note: Goods used as inputs must be defined in the Goods section above.", style={'fontStyle': 'italic'}),
        
        html.Label("Select Good to Produce:"),
        dcc.Dropdown(id='produce-dropdown', placeholder="Select good..."),
        
        html.Label("Producer ID (integer):"),
        dcc.Input(id='producer-id-input', type='number', value=1001),
        
        html.Label("Production Rate / Capacity:"),
        dcc.Input(id='production-rate-input', type='number', value=100),
        
        html.Label("Production Quantity:"),
        dcc.Input(id='production-qty-input', type='number', value=50),
        
        html.Div([
            dcc.Checklist(id='auto-price-checkbox', options=[{'label': ' Auto-calculate Price (Inputs + VA)', 'value': 'auto'}], value=['auto']),
            html.Label("Price:"),
            dcc.Input(id='price-input', type='number', value=0, disabled=True),
        ], style={'marginBottom': '15px'}),
        
        html.H4("Inputs (Requires Goods)"),
        dcc.Dropdown(id='input-good-dropdown', placeholder="Select input good..."),
        dcc.Input(id='input-qty', type='number', placeholder="Cost / Qty"),
        html.Button("Add Input", id='add-input-btn', n_clicks=0),
        html.Ul(id='current-inputs-list'),
        dcc.Store(id='current-inputs-store', data={}),
        
        html.H4("Value Added Components"),
        dcc.RadioItems(id='va-mode', options=[
            {'label': ' Absolute Values ($)', 'value': 'absolute'},
            {'label': ' Total VA + Percentages (%)', 'value': 'percentage'}
        ], value='absolute', style={'marginBottom': '10px'}),
        
        html.Div([
            html.Label("Total Value Added ($):"),
            dcc.Input(id='total-va-input', type='number', value=0)
        ], id='total-va-container', style={'display': 'none', 'marginBottom': '10px'}),
        
        html.Label("Wages ($):", id='va-wages-label'),
        dcc.Input(id='va-wages-input', type='number', value=10),
        html.Label("Surplus ($):", id='va-surplus-label'),
        dcc.Input(id='va-surplus-input', type='number', value=5),
        html.Label("Taxes ($):", id='va-taxes-label'),
        dcc.Input(id='va-taxes-input', type='number', value=2),
        
        html.Br(),html.Br(),
        html.Button("Add Production", id='add-prod-btn', n_clicks=0, style={'marginRight': '10px'}),
        html.Button("Remove Selected Productions", id='remove-productions-btn', n_clicks=0, style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none', 'padding': '5px 10px', 'cursor': 'pointer'}),
        html.Div(id='add-prod-status', style={'color': 'red', 'marginTop': '10px'}),
        
        html.H4("Current Productions"),
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
            style={'height': 200, 'width': '100%'}
        )
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'marginBottom': '20px'}),
    
    html.Div([
        html.Button("GENERATE FILES", id='generate-btn', n_clicks=0, style={'fontSize': '20px', 'padding': '15px', 'backgroundColor': '#27ae60', 'color': 'white', 'cursor': 'pointer'}),
        html.Div(id='generate-status', style={'fontSize': '18px', 'marginTop': '10px', 'fontWeight': 'bold'})
    ], style={'padding': '20px'})
])

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
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'import-btn':
        if not import_folder:
            return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, html.Span("Please select a dataset to import.", style={'color': 'red'})
        
        # If there is existing unsaved data, prompt
        if current_goods and len(current_goods) > 0:
            return True, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        # Otherwise, proceed to import naturally
        return perform_import(import_folder)
        
    if trigger_id == 'import-confirm-dialog':
        return perform_import(import_folder)
        
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

def perform_import(folder_name):
    base_dir = os.path.join(os.path.dirname(__file__), 'data', folder_name)
    try:
        goods_df = pd.read_csv(os.path.join(base_dir, 'goods.csv'))
        goods_df = goods_df.fillna('')
        goods_data = goods_df.to_dict('records')
        
        prods_df = pd.read_csv(os.path.join(base_dir, 'productions.csv'))
        prods_df = prods_df.fillna('')
        prods_data = prods_df.to_dict('records')
        
        return False, goods_data, prods_data, goods_data, prods_data, folder_name, html.Span(f"Imported '{folder_name}' successfully.", style={'color': 'green'})
    except Exception as e:
        return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, html.Span(f"Error importing dataset: {e}", style={'color': 'red'})

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
    Input('goods-store', 'data')
)
def update_good_dropdowns(goods):
    opts = [{'label': g['name'], 'value': g['isic']} for g in goods]
    return opts, opts

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
    Output('va-taxes-label', 'children'),
    Input('va-mode', 'value')
)
def toggle_va_mode(mode):
    if mode == 'percentage':
        return {'display': 'block', 'marginBottom': '10px'}, "Wages (%):", "Surplus (%):", "Taxes (%):"
    return {'display': 'none', 'marginBottom': '10px'}, "Wages ($):", "Surplus ($):", "Taxes ($):"

@app.callback(
    Output('price-input', 'value'),
    Output('price-input', 'disabled'),
    Input('auto-price-checkbox', 'value'),
    Input('current-inputs-store', 'data'),
    Input('va-mode', 'value'),
    Input('total-va-input', 'value'),
    Input('va-wages-input', 'value'),
    Input('va-surplus-input', 'value'),
    Input('va-taxes-input', 'value'),
    State('price-input', 'value')
)
def live_update_price(auto_price, current_inputs, va_mode, total_va, wages, surplus, taxes, current_price):
    is_auto = 'auto' in (auto_price or [])
    
    if not is_auto:
        return dash.no_update, False
        
    inputs_sum = sum(current_inputs.values()) if current_inputs else 0.0
    w = wages or 0.0
    s = surplus or 0.0
    t = taxes or 0.0
    
    if va_mode == 'absolute':
        va_sum = w + s + t
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
    State('va-taxes-input', 'value'),
    State('goods-store', 'data'),
    State('productions-store', 'data')
)
def manage_productions(add_n, remove_n, selected_rows, produce_isic, producer_id, p_rate, p_qty, price, inputs_dict, va_mode, total_va, wages, surplus, taxes, goods, productions):
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
    t = taxes or 0.0
    
    if va_mode == 'percentage':
        if abs(w + s + t - 100.0) > 0.001:
            return dash.no_update, dash.no_update, f"Error: Percentages sum to {w+s+t}% instead of 100%."
            
        tot = total_va or 0.0
        w = tot * (w / 100.0)
        s = tot * (s / 100.0)
        t = tot * (t / 100.0)
    
    va = {
        "wages": w,
        "surplus": s,
        "taxes": t
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
    State('productions-store', 'data')
)
def generate_files_check(n, folder_name, goods, productions):
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
        
    msg, style = perform_generation(folder_name, goods, productions)
    return False, msg, style

@app.callback(
    Output('generate-status', 'children', allow_duplicate=True),
    Output('generate-status', 'style', allow_duplicate=True),
    Input('generate-confirm-dialog', 'submit_n_clicks'),
    State('target-folder-input', 'value'),
    State('goods-store', 'data'),
    State('productions-store', 'data'),
    prevent_initial_call=True
)
def generate_files_confirmed(confirm_n, folder_name, goods, productions):
    if not confirm_n:
        return dash.no_update, dash.no_update
    return perform_generation(folder_name, goods, productions)

def perform_generation(folder_name, goods, productions):
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
    
    return f"Files generated successfully in data/{folder_name}!", {'color': 'green'}

if __name__ == '__main__':
    print("Starting Sambaza-Sim Data Builder Utility on http://127.0.0.1:8051/")
    app.run(debug=True, port=8051)
