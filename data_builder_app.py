import os
import sys
import json
import numpy as np
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

def compute_recipe_impact(prod, target_isic=None, action='replace_cost', action_val=0.0, sub_isic=None, sub_val=0.0,
                          ingredient_changes=None, substitute_goods=None):
    """
    Compute before-and-after recipe prices and Leontief technical coefficients
    from micro-level ingredient substitutions or adjustments.
    Supports either single-ingredient parameters or multi-ingredient dictionaries:
      ingredient_changes: dict of {isic: {'action': ..., 'val': ...}}
      substitute_goods: dict of {isic: val} or list of {isic: val}
    """
    if not prod:
        return None

    raw_inputs = prod.get('production_inputs', '{}')
    inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else dict(raw_inputs or {})

    raw_va = prod.get('production_added_values', '{}')
    va_dict = json.loads(raw_va) if isinstance(raw_va, str) else dict(raw_va or {})
    total_va = sum(float(v) for v in va_dict.values())

    p_old = sum(float(v) for v in inputs.values()) + total_va
    if p_old <= 0:
        p_old = float(prod.get('price', 1.0)) or 1.0

    new_inputs = {str(k): float(v) for k, v in inputs.items()}

    # Merge single-ingredient params if provided
    combined_ing_changes = dict(ingredient_changes or {})
    if target_isic and str(target_isic) not in combined_ing_changes:
        combined_ing_changes[str(target_isic)] = {
            'action': action or 'replace_cost',
            'val': action_val if action_val is not None else 0.0
        }

    # Merge substitute goods
    combined_subs = {}
    if isinstance(substitute_goods, dict):
        combined_subs.update(substitute_goods)
    elif isinstance(substitute_goods, list):
        for s in substitute_goods:
            if isinstance(s, dict) and 'isic' in s:
                combined_subs[str(s['isic'])] = s.get('val', 0.0)
    if sub_isic and str(sub_isic) not in combined_subs and (sub_val is not None and float(sub_val or 0.0) > 0):
        combined_subs[str(sub_isic)] = sub_val

    # Apply ingredient changes
    for ing_isic, chg in combined_ing_changes.items():
        k = str(ing_isic)
        c_old = new_inputs.get(k, 0.0)
        act = chg.get('action', 'replace_cost') if isinstance(chg, dict) else 'replace_cost'
        try:
            val = float(chg.get('val', 0.0) if isinstance(chg, dict) else chg)
        except (ValueError, TypeError):
            val = 0.0

        if act == 'multiply':
            c_new = max(0.0, c_old * val)
        elif act == 'add':
            c_new = max(0.0, c_old + val)
        else:  # replace_cost or set
            c_new = max(0.0, val)
        new_inputs[k] = c_new

    # Apply substitute goods
    for s_isic, s_val in combined_subs.items():
        sk = str(s_isic)
        try:
            val = float(s_val or 0.0)
        except (ValueError, TypeError):
            val = 0.0
        new_inputs[sk] = new_inputs.get(sk, 0.0) + max(0.0, val)

    p_new = sum(new_inputs.values()) + total_va
    if p_new <= 0:
        p_new = 1.0

    comparison = []
    all_keys = list(dict.fromkeys([str(k) for k in inputs.keys()] + [str(k) for k in new_inputs.keys()]))
    for k in all_keys:
        cost_old = float(inputs.get(k, 0.0))
        cost_new = float(new_inputs.get(k, 0.0))
        a_old = cost_old / p_old
        a_new = cost_new / p_new
        comparison.append({
            'input_isic': k,
            'cost_before': round(cost_old, 2),
            'cost_after': round(cost_new, 2),
            'a_before': round(a_old, 4),
            'a_after': round(a_new, 4),
            'delta_a': round(a_new - a_old, 4),
            'is_substitute': k in combined_subs and k not in inputs,
            'is_modified': k in combined_ing_changes
        })

    va_before = total_va / p_old
    va_after = total_va / p_new

    return {
        'price_before': round(p_old, 2),
        'price_after': round(p_new, 2),
        'delta_price': round(p_new - p_old, 2),
        'pct_price_change': round(((p_new - p_old) / p_old) * 100.0, 2) if p_old > 0 else 0.0,
        'total_va': round(total_va, 2),
        'va_before': round(va_before, 4),
        'va_after': round(va_after, 4),
        'delta_va': round(va_after - va_before, 4),
        'comparison': comparison,
        'new_inputs': new_inputs,
        'ingredient_changes': combined_ing_changes,
        'substitutes': combined_subs
    }

def compute_production_merit_ranks(productions, goods=None):
    """
    Annotate productions with merit order rank and active baseline status per good.
    Cheapest production for each good is Rank 1 (Active Baseline technology used in Leontief IO matrix A).
    """
    if not productions:
        return []

    groups = {}
    for p in productions:
        p_copy = dict(p)
        raw_inputs = p_copy.get('production_inputs', '{}')
        inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else dict(raw_inputs or {})
        raw_va = p_copy.get('production_added_values', '{}')
        va_dict = json.loads(raw_va) if isinstance(raw_va, str) else dict(raw_va or {})
        calc_price = sum(float(v) for v in inputs.values()) + sum(float(v) for v in va_dict.values())
        if calc_price <= 0:
            calc_price = float(p_copy.get('price', 0.0) or 0.0)
        p_copy['calc_price'] = round(calc_price, 2)
        p_copy['price'] = round(calc_price, 2)

        prod_key = str(p_copy.get('produce', ''))
        if not prod_key or prod_key == 'None' or prod_key == '':
            prod_key = str(p_copy.get('isic', 'unknown'))
        groups.setdefault(prod_key, []).append(p_copy)

    annotated = []
    for prod_key, prods in groups.items():
        sorted_prods = sorted(prods, key=lambda x: x['calc_price'])
        for rank, p in enumerate(sorted_prods, 1):
            p['tier_rank'] = rank
            p['is_baseline'] = (rank == 1)
            p['merit_status'] = f"Tier {rank}"
            annotated.append(p)

    annotated.sort(key=lambda x: int(x.get('id', 0)) if str(x.get('id', '')).isdigit() else 0)
    return annotated

def compute_curve_impact(goods, productions, target_isic, tier_idx,
                         cap_action='unchanged', cap_val=0.0,
                         price_action='unchanged', price_val=0.0,
                         tax_data=None, field=None, action=None, action_val=None):
    """
    Evaluate the economic impact of a supply curve tier modification (Middle / Curve Level)
    using the Leontief table and circular final demand.
    Supports modifying capacity, price, or both simultaneously under one scenario.
    """
    if not goods or not productions or not target_isic:
        return None

    # Backward compatibility with (field, action, action_val)
    if field is not None:
        if field == 'cap':
            cap_action = action or 'set'
            cap_val = action_val or 0.0
        elif field == 'price':
            price_action = action or 'set'
            price_val = action_val or 0.0

    sorted_goods = sorted(goods, key=lambda g: str(g.get('isic', '')))
    n = len(sorted_goods)
    if n == 0 or target_isic not in [g.get('isic') for g in sorted_goods]:
        return None

    isic_map = {g['isic']: i for i, g in enumerate(sorted_goods)}
    target_col = isic_map[target_isic]

    # Build A matrix and VA vector using cheapest baseline productions
    A_mon = np.zeros((n, n))
    VA_mon = np.zeros(n)

    annotated_prods = compute_production_merit_ranks(productions, sorted_goods)

    for g in sorted_goods:
        col_idx = isic_map[g['isic']]
        gid = str(g.get('id_number', ''))
        gisic = str(g.get('isic', ''))
        g_prods = [p for p in annotated_prods if (gid and str(p.get('produce')) == gid) or str(p.get('isic')) == gisic]
        if not g_prods:
            continue
        baseline_prod = next((p for p in g_prods if p.get('is_baseline')), g_prods[0])
        p_price = float(baseline_prod.get('price', 1.0)) or 1.0
        raw_inputs = baseline_prod.get('production_inputs', '{}')
        inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else dict(raw_inputs or {})
        for in_isic, in_cost in inputs.items():
            if in_isic in isic_map:
                A_mon[isic_map[in_isic], col_idx] = float(in_cost) / p_price

        raw_va = baseline_prod.get('production_added_values', '{}')
        va_dict = json.loads(raw_va) if isinstance(raw_va, str) else dict(raw_va or {})
        VA_mon[col_idx] = sum(float(v) for v in va_dict.values()) / p_price

    # Derive final demand Y
    Y = np.full(n, 100.0)
    if tax_data and len(tax_data) > 0:
        first_tax = tax_data[0]
        fd_raw = first_tax.get('final_demand')
        if fd_raw:
            try:
                if isinstance(fd_raw, str):
                    fd_vals = [float(v) for v in fd_raw.split(';') if v.strip()]
                elif isinstance(fd_raw, list):
                    fd_vals = [float(v) for v in fd_raw]
                else:
                    fd_vals = []
                for i, v in enumerate(fd_vals[:n]):
                    Y[i] = v
            except Exception:
                pass

    # Solve Leontief system: X = (I - A)^(-1) Y
    I_minus_A = np.eye(n) - A_mon
    try:
        X = np.linalg.solve(I_minus_A, Y)
        X = np.maximum(X, 0.0)
    except Exception:
        X = Y.copy()

    sector_gross_demand = float(X[target_col])

    # Get all tiers for target_isic
    target_gid = str(next((g.get('id_number', '') for g in sorted_goods if g.get('isic') == target_isic), ''))
    target_prods = [p for p in annotated_prods if (target_gid and str(p.get('produce')) == target_gid) or str(p.get('isic')) == target_isic]
    if not target_prods:
        return None

    sorted_tiers = sorted(target_prods, key=lambda p: float(p.get('price', 0.0)))
    tier_idx = int(tier_idx or 0)
    if tier_idx < 0 or tier_idx >= len(sorted_tiers):
        tier_idx = 0

    def run_dispatch(tiers_list):
        cum = 0.0
        dispatch_results = []
        total_cost = 0.0
        total_dispatched = 0.0
        for t in tiers_list:
            raw_qty = t.get('production_quantity', -1)
            cap = float(raw_qty) if raw_qty not in (-1, None, '') else float('inf')
            price = float(t.get('price', 0.0))

            rem = max(0.0, sector_gross_demand - cum)
            dispatched = min(rem, cap)
            util = (dispatched / cap * 100.0) if (cap > 0 and cap != float('inf')) else (100.0 if dispatched > 0 else 0.0)

            if dispatched >= cap and cap != float('inf'):
                status = "100% Saturated (Full)"
            elif dispatched > 0:
                status = "Marginal Active (Partial)"
            else:
                status = "Idle Reserve (0%)"

            dispatch_results.append({
                'prod_id': t.get('id', 0),
                'name': t.get('name', f"Tier {t.get('tier_rank', 1)}"),
                'cap': cap,
                'price': price,
                'dispatched': round(dispatched, 2),
                'utilization': round(util, 1),
                'status': status
            })
            total_cost += dispatched * price
            total_dispatched += dispatched
            cum += cap

        eff_price = (total_cost / total_dispatched) if total_dispatched > 0 else (tiers_list[0]['price'] if tiers_list else 0.0)
        return dispatch_results, round(eff_price, 2)

    before_tiers = [dict(t) for t in sorted_tiers]
    disp_before, p_eff_before = run_dispatch(before_tiers)

    after_tiers = [dict(t) for t in sorted_tiers]
    t_mod = after_tiers[tier_idx]

    # Apply capacity change if not unchanged
    if cap_action and cap_action != 'unchanged':
        old_cap = float(t_mod.get('production_quantity', 0.0) or 0.0)
        c_val = float(cap_val or 0.0)
        if cap_action == 'multiply':
            new_cap = max(0.0, old_cap * c_val)
        elif cap_action == 'add':
            new_cap = max(0.0, old_cap + c_val)
        else:  # set
            new_cap = max(0.0, c_val)
        t_mod['production_quantity'] = new_cap

    # Apply price change if not unchanged
    if price_action and price_action != 'unchanged':
        old_price = float(t_mod.get('price', 0.0))
        p_val = float(price_val or 0.0)
        if price_action == 'multiply':
            new_price = max(0.0, old_price * p_val)
        elif price_action == 'add':
            new_price = max(0.0, old_price + p_val)
        else:  # set
            new_price = max(0.0, p_val)
        t_mod['price'] = new_price

    after_tiers.sort(key=lambda p: float(p.get('price', 0.0)))
    disp_after, p_eff_after = run_dispatch(after_tiers)

    delta_peff = round(p_eff_after - p_eff_before, 2)
    pct_peff = round((delta_peff / p_eff_before) * 100.0, 2) if p_eff_before > 0 else 0.0

    return {
        'isic': target_isic,
        'sector_output_demand': round(sector_gross_demand, 2),
        'p_eff_before': p_eff_before,
        'p_eff_after': p_eff_after,
        'delta_peff': delta_peff,
        'pct_peff': pct_peff,
        'disp_before': disp_before,
        'disp_after': disp_after,
        'target_tier_index': tier_idx,
        'cap_action': cap_action,
        'cap_val': cap_val,
        'price_action': price_action,
        'price_val': price_val
    }

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
                    'input_isic': row.get('input_isic', ''),
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
        
        # Accordion wrapping all main sections
        dbc.Accordion([
            # Section 0A: Import Existing Dataset
            dbc.AccordionItem(
                title="0A. Import Existing Dataset",
                item_id="acc-import",
                children=[
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
                ]
            ),
            
            # Section 0B: Output Configuration
            dbc.AccordionItem(
                title="0B. Output Configuration",
                item_id="acc-output",
                children=[
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
                ]
            ),
            
            # Section 1: Add Goods
            dbc.AccordionItem(
                title="1. Add Goods",
                item_id="acc-goods",
                children=[
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
                ]
            ),
            
            # Section 2: Add Production
            dbc.AccordionItem(
                title="2. Add Production",
                item_id="acc-productions",
                children=[
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
                            {'field': 'produce_name', 'headerName': 'Produces', 'width': 130, 'checkboxSelection': True},
                            {'field': 'merit_status', 'headerName': 'Merit Order / Status', 'width': 130, 'cellStyle': {'fontWeight': 'bold'}},
                            {'field': 'price', 'headerName': 'Price ($)', 'width': 100, 'valueFormatter': {"function": "params.value != null ? ('$' + Number(params.value).toFixed(2)) : ''"}},
                            {'field': 'production_quantity', 'headerName': 'Capacity', 'width': 95},
                            {'field': 'producer', 'headerName': 'Producer ID', 'width': 110},
                            {'field': 'production_inputs', 'headerName': 'Inputs', 'flex': 1},
                            {'field': 'production_added_values', 'headerName': 'Value Added', 'flex': 1}
                        ],
                        rowData=[],
                        dashGridOptions={'rowSelection': 'multiple'},
                        style={'height': 300, 'width': '100%'}
                    )
                ]
            ),
            
            # Section 3: Tax Policies & Spending Profiles
            dbc.AccordionItem(
                title="3. Tax Policies & Spending Profiles",
                item_id="acc-tax",
                children=[
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
                ]
            ),

            # Section 4: Technological Changes & Capital Investments
            dbc.AccordionItem(
                title="4. Technological Changes & Capital Investments",
                item_id="acc-tech-changes",
                children=[
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
                        ], width=12, md=6),
                        dbc.Col([
                            html.Label("Scenario ID (slug):", className="mb-1 font-weight-bold"),
                            dbc.Input(id='tc-id-input', type='text', placeholder='e.g. energy_efficiency_capital', className="form-control"),
                        ], width=12, md=6),
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
                                    html.Label("Change Method / Scale:", className="mb-1 fw-bold"),
                                    dcc.Dropdown(
                                        id='tc-method-dropdown',
                                        options=[
                                            {'label': 'Production Recipe Substitution / Upgrade (Micro Level)', 'value': 'add_production_input_change'},
                                            {'label': 'Supply Curve Tier Modification (Middle / Curve Level)', 'value': 'add_curve_tier_change'},
                                            {'label': 'Single Coefficient A[i, j] (Macro Level)', 'value': 'add_coefficient_change'},
                                            {'label': 'Input Row Across All Sectors (i) (Macro Level)', 'value': 'add_input_change'},
                                            {'label': 'Sector Column Across All Inputs (j) (Macro Level)', 'value': 'add_sector_change'},
                                        ],
                                        value='add_production_input_change',
                                        clearable=False
                                    ),
                                ], width=12),
                            ], className="mb-3"),

                            # Macro level fields container
                            html.Div(id='tc-macro-inputs-container', style={'display': 'none'}, children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.Div(id='tc-sector-container', children=[
                                            html.Label("Sector (j - consumer/output):", className="mb-1"),
                                            dcc.Dropdown(id='tc-sector-dropdown', placeholder="Select producing sector..."),
                                        ]),
                                    ], width=12, md=6),
                                    dbc.Col([
                                        html.Div(id='tc-input-sector-container', children=[
                                            html.Label("Input Good (i - input/resource):", className="mb-1"),
                                            dcc.Dropdown(id='tc-input-sector-dropdown', placeholder="Select input good..."),
                                        ]),
                                    ], width=12, md=6),
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
                                    ], width=12, md=6),
                                    dbc.Col([
                                        html.Label("Value:", className="mb-1"),
                                        dbc.Input(id='tc-value-input', type='number', value=0.8, step='any', className="form-control"),
                                    ], width=12, md=6),
                                ], className="mb-3"),
                            ]),

                            # Micro recipe fields container
                            html.Div(id='tc-micro-recipe-container', children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Target Sector / Good:", className="mb-1 fw-bold text-primary"),
                                        dcc.Dropdown(id='tc-recipe-sector-dropdown', placeholder="Select sector / good to inspect productions..."),
                                    ], width=12, md=6),
                                    dbc.Col([
                                        html.Label("Search Productions:", className="mb-1 fw-bold text-muted"),
                                        dbc.Input(id='tc-recipe-search-input', type='text', placeholder="Search by name, ID, or producer...", className="form-control"),
                                    ], width=12, md=6),
                                ], className="mb-2"),

                                # Hidden sync dropdown for selected production ID
                                dcc.Dropdown(id='tc-recipe-prod-dropdown', style={'display': 'none'}),

                                # Table 1: Sector Production Tiers Table (Like the curves option)
                                dbc.Card([
                                    dbc.CardHeader([
                                        html.Span("Production Methods & Tiers for Selected Sector", className="fw-bold text-dark"),
                                        html.Small(" (Click 'Select' on a production to configure ingredient substitution)", className="text-muted ms-2 fst-italic")
                                    ], className="bg-light small"),
                                    dbc.CardBody(id='tc-recipe-tiers-table-container', className="p-2", children=[
                                        html.Div("Select a sector above to view its production methods and tiers.", className="text-muted small fst-italic p-2")
                                    ])
                                ], className="mb-3 border-secondary"),

                                # Step 2: The Other Table (Ingredient Substitution & Live Preview)
                                html.Div(id='tc-recipe-details-container', style={'display': 'none'}, children=[
                                    dcc.Store(id='tc-recipe-mods-store', data={'prod_id': None, 'ingredients': {}, 'substitutes': {}}),
                                    dbc.Card([
                                        dbc.CardHeader([
                                            dbc.Row([
                                                dbc.Col(html.Span(id='tc-recipe-selected-title', className="fw-bold text-primary small"), width=7),
                                                dbc.Col(html.Div(id='tc-recipe-status-msg', className="text-end"), width=2),
                                                dbc.Col(
                                                    dbc.Button("↺ Reset All Recipe Changes", id='tc-recipe-reset-all-btn', size="sm", color="outline-danger", className="py-0 px-2 float-end"),
                                                    width=3
                                                )
                                            ], align="center")
                                        ], className="bg-white border-bottom py-2"),
                                        dbc.CardBody([
                                            # Form A: Modify Existing Recipe Ingredient
                                            dbc.Card([
                                                dbc.CardHeader("Modify Existing Recipe Ingredient", className="py-1 px-3 small fw-bold bg-light text-dark"),
                                                dbc.CardBody([
                                                    dbc.Row([
                                                        dbc.Col([
                                                            html.Label("Ingredient to Modify:", className="mb-1 fw-bold text-dark small"),
                                                            dcc.Dropdown(id='tc-recipe-target-input-dropdown', placeholder="Select ingredient from current recipe..."),
                                                        ], width=12, md=4),
                                                        dbc.Col([
                                                            html.Label("Change Action:", className="mb-1 fw-bold text-dark small"),
                                                            dcc.Dropdown(
                                                                id='tc-recipe-action-dropdown',
                                                                options=[
                                                                    {'label': 'Set exact new cost ($)', 'value': 'replace_cost'},
                                                                    {'label': 'Multiply cost (e.g. 0.20 = -80%)', 'value': 'multiply'},
                                                                    {'label': 'Add / Subtract nominal cost ($)', 'value': 'add'},
                                                                ],
                                                                value='replace_cost',
                                                                clearable=False
                                                            ),
                                                        ], width=12, md=3),
                                                        dbc.Col([
                                                            html.Label("New Value / Cost ($):", className="mb-1 fw-bold text-dark small"),
                                                            dbc.Input(id='tc-recipe-val-input', type='number', value=0.0, step='any', className="form-control form-control-sm"),
                                                        ], width=12, md=3),
                                                        dbc.Col([
                                                            html.Label(" ", className="mb-1 d-block small"),
                                                            dbc.Button("✓ Apply Change", id='tc-recipe-apply-btn', color="primary", size="sm", className="w-100 fw-bold")
                                                        ], width=12, md=2),
                                                    ], align="end")
                                                ], className="p-2")
                                            ], className="mb-2 border-secondary"),

                                            # Form B: Introduce Substitute / New Good
                                            dbc.Card([
                                                dbc.CardHeader("Introduce Substitute / New Good (Optional)", className="py-1 px-3 small fw-bold bg-light text-dark"),
                                                dbc.CardBody([
                                                    dbc.Row([
                                                        dbc.Col([
                                                            html.Label("Substitute Good to Add:", className="mb-1 fw-bold text-dark small"),
                                                            dcc.Dropdown(id='tc-recipe-sub-good-dropdown', placeholder="Select replacement good..."),
                                                        ], width=12, md=6),
                                                        dbc.Col([
                                                            html.Label("Added Nominal Cost ($):", className="mb-1 fw-bold text-dark small"),
                                                            dbc.Input(id='tc-recipe-sub-val-input', type='number', min=0, value=0.0, step='any', className="form-control form-control-sm"),
                                                        ], width=12, md=3),
                                                        dbc.Col([
                                                            html.Label(" ", className="mb-1 d-block small"),
                                                            dbc.Button("+ Add Substitute", id='tc-recipe-add-sub-btn', color="info", size="sm", className="w-100 fw-bold")
                                                        ], width=12, md=3),
                                                    ], align="end")
                                                ], className="p-2")
                                            ], className="mb-3 border-secondary"),

                                            # Live Micro-to-Macro Calculation Preview (Table 2)
                                            dbc.Card([
                                                dbc.CardHeader([
                                                    dbc.Row([
                                                        dbc.Col(html.Span("Live Recipe Formulation & Macro Shift Preview", className="small fw-bold text-dark"), width=7),
                                                        dbc.Col(html.Span(id='tc-recipe-staged-summary', className="small float-end"), width=5)
                                                    ], align="center")
                                                ], className="py-1 px-3 bg-light border-bottom"),
                                                dbc.CardBody(id='tc-recipe-preview-display', className="p-3", children=[
                                                    html.Div("Select an ingredient above to preview live cost and coefficient shifts.", className="text-muted small fst-italic")
                                                ])
                                            ], className="border-secondary"),
                                        ])
                                    ], className="mb-3 border-primary shadow-sm")
                                ])
                            ]),

                            # Curve level fields container (Middle Level)
                            html.Div(id='tc-curve-inputs-container', style={'display': 'none'}, children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Target Good / Supply Curve:", className="mb-1 fw-bold text-primary"),
                                        dcc.Dropdown(id='tc-curve-good-dropdown', placeholder="Select good / supply curve..."),
                                    ], width=12, md=6),
                                    dbc.Col([
                                        html.Label("Target Curve Tier:", className="mb-1 fw-bold text-primary"),
                                        dcc.Dropdown(id='tc-curve-tier-dropdown', placeholder="Select tier to modify..."),
                                    ], width=12, md=6),
                                ], className="mb-2"),

                                # Row 1: Capacity (Cap) Modification
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Capacity Change Action:", className="mb-1 fw-bold text-dark"),
                                        dcc.Dropdown(
                                            id='tc-curve-cap-action-dropdown',
                                            options=[
                                                {'label': 'Unchanged (keep current capacity)', 'value': 'unchanged'},
                                                {'label': 'Set exact capacity', 'value': 'set'},
                                                {'label': 'Multiply capacity (e.g. 2.0 = double)', 'value': 'multiply'},
                                                {'label': 'Add / Subtract capacity (e.g. +100)', 'value': 'add'},
                                            ],
                                            value='set',
                                            clearable=False
                                        ),
                                    ], width=12, md=6),
                                    dbc.Col([
                                        html.Label("Capacity Value:", className="mb-1 fw-bold text-dark"),
                                        dbc.Input(id='tc-curve-cap-val-input', type='number', value=200.0, step='any', className="form-control"),
                                    ], width=12, md=6),
                                ], className="mb-2"),

                                # Row 2: Price Modification
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Unit Price Change Action:", className="mb-1 fw-bold text-dark"),
                                        dcc.Dropdown(
                                            id='tc-curve-price-action-dropdown',
                                            options=[
                                                {'label': 'Unchanged (keep current price)', 'value': 'unchanged'},
                                                {'label': 'Set exact price ($)', 'value': 'set'},
                                                {'label': 'Multiply price (e.g. 0.8 = -20%)', 'value': 'multiply'},
                                                {'label': 'Add / Subtract price ($) (e.g. -2.00)', 'value': 'add'},
                                            ],
                                            value='unchanged',
                                            clearable=False
                                        ),
                                    ], width=12, md=6),
                                    dbc.Col([
                                        html.Label("Price Value ($):", className="mb-1 fw-bold text-dark"),
                                        dbc.Input(id='tc-curve-price-val-input', type='number', value=0.0, step='any', className="form-control"),
                                    ], width=12, md=6),
                                ], className="mb-3"),

                                # Live Middle-Level Curve & Dispatch Preview
                                dbc.Card([
                                    dbc.CardHeader("Live Curve Merit-Order Dispatch & Market Price Preview", className="small fw-bold bg-light text-dark"),
                                    dbc.CardBody(id='tc-curve-preview-display', className="p-3", children=[
                                        html.Div("Select a good above to preview supply curve dispatch, gross output demand from the Leontief system, and effective market price.", className="text-muted small fst-italic")
                                    ])
                                ], className="mb-3 border-secondary"),
                            ]),

                            # Per-Investment Monetary Requirement
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Capital Cost ($):", className="mb-1 font-weight-bold text-success"),
                                    dbc.Input(id='tc-cost-input', type='number', min=0, value=0.0, step='any', placeholder="0 = free", className="form-control"),
                                ], width=12, md=6),
                                dbc.Col([
                                    html.Label("Build Duration (Iterations):", className="mb-1 font-weight-bold text-info"),
                                    dbc.Input(id='tc-duration-input', type='number', min=1, value=1, step=1, className="form-control"),
                                ], width=12, md=6),
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
                ]
            ),
        ], always_open=True, active_item=['acc-import', 'acc-output', 'acc-goods', 'acc-productions', 'acc-tax', 'acc-tech-changes'], id='main-accordion', className="shadow-sm mb-4")
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
        prods_data = compute_production_merit_ranks(prods_data, goods_data)

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
        annotated = compute_production_merit_ranks(updated_prods, goods)
        return annotated, annotated, f"Removed {len(ids_to_remove)} production(s)."

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
    annotated = compute_production_merit_ranks(productions, goods)
    
    return annotated, annotated, "Production added successfully."

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
    
    # Validation: Ensure domestic goods have at least 1 production (exempt Foreign Exchange / primary import currency)
    produced_isics = set(p.get('isic') for p in productions)
    missing_prods = [
        g.get('name', 'Unknown') for g in goods 
        if g.get('isic') not in produced_isics 
        and not (
            str(g.get('isic', '')).strip() == 'A9999_999_999' or
            str(g.get('id_number', '')).strip() == '9999' or
            'foreign exchange' in str(g.get('name', '')).strip().lower()
        )
    ]
    if missing_prods:
        return False, f"Validation Error: The following goods have no production defined: {', '.join(missing_prods)}", {'color': 'red'}
        
    # Validation: Ensure no production references a non-existent good (as input or produced)
    valid_isics = set(g.get('isic') for g in goods)
    for p in productions:
        if p.get('isic') not in valid_isics:
            return False, f"Validation Error: Production '{p.get('name', '')}' produces a good that no longer exists.", {'color': 'red'}
        raw_inputs = p.get('production_inputs', '{}')
        p_inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else (raw_inputs or {})
        for in_isic in p_inputs.keys():
            if in_isic not in valid_isics:
                return False, f"Validation Error: Production '{p.get('name', '')}' uses a non-existent input good ({in_isic}).", {'color': 'red'}
    
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
                if method == 'add_production_input_change':
                    row_base = {
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
                        'use_multi_level': True,
                        'sequence_number': seq,
                        'method': 'add_production_input_change',
                        'sector_idx': inv.get('sector_idx', ''),
                        'input_sector_idx': '',
                        'production_id': inv.get('production_id', ''),
                        'isic': '',
                        'tier_index': '',
                        'field': '',
                        'change_type_param': '',
                        'value': '',
                        'efficiency_type': '',
                        'va_component': '',
                        'input_isic': '',
                        'exclude_list': '',
                        'position': '',
                        'cap': '',
                        'price': '',
                        'va_components': '',
                        'requirements': '',
                        'investment_duration': ''
                    }

                    # Write row for each modified ingredient
                    ing_changes = inv.get('ingredient_changes')
                    if ing_changes:
                        for in_isic, chg in ing_changes.items():
                            row_ing = dict(row_base)
                            row_ing['sequence_number'] = seq
                            row_ing['input_isic'] = in_isic
                            row_ing['change_type_param'] = chg.get('action', 'replace_cost') if isinstance(chg, dict) else 'replace_cost'
                            row_ing['value'] = chg.get('val', 0.0) if isinstance(chg, dict) else chg
                            tc_rows.append(row_ing)
                            seq += 1
                    elif inv.get('input_isic'):
                        row_ing = dict(row_base)
                        row_ing['sequence_number'] = seq
                        row_ing['input_isic'] = inv['input_isic']
                        row_ing['change_type_param'] = inv.get('change_type', 'set')
                        row_ing['value'] = inv.get('value', '')
                        tc_rows.append(row_ing)
                        seq += 1

                    # Write row for each substitute good
                    subs = inv.get('substitutes')
                    if subs:
                        for s_isic, s_val in subs.items():
                            if float(s_val or 0.0) > 0:
                                sub_row = dict(row_base)
                                sub_row['sequence_number'] = seq
                                sub_row['input_isic'] = s_isic
                                sub_row['change_type_param'] = 'add'
                                sub_row['value'] = s_val
                                tc_rows.append(sub_row)
                                seq += 1
                    elif inv.get('substitute_isic') and float(inv.get('substitute_val', 0.0) or 0.0) > 0:
                        sub_row = dict(row_base)
                        sub_row['sequence_number'] = seq
                        sub_row['input_isic'] = inv['substitute_isic']
                        sub_row['change_type_param'] = 'add'
                        sub_row['value'] = inv['substitute_val']
                        tc_rows.append(sub_row)
                        seq += 1

                    # Macro coefficient shifts from recipe change
                    for item in inv.get('comparison', []):
                        if abs(item.get('delta_a', 0.0)) > 1e-4:
                            coeff_row = dict(row_base)
                            coeff_row['sequence_number'] = seq
                            coeff_row['method'] = 'add_coefficient_change'
                            coeff_row['input_sector_idx'] = item['input_isic']
                            coeff_row['production_id'] = ''
                            coeff_row['input_isic'] = ''
                            coeff_row['change_type_param'] = 'set'
                            coeff_row['value'] = item['a_after']
                            tc_rows.append(coeff_row)
                            seq += 1
                elif method == 'add_curve_tier_change':
                    cap_act = inv.get('cap_action', 'unchanged')
                    price_act = inv.get('price_action', 'unchanged')
                    row_base = {
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
                        'use_multi_level': True,
                        'sequence_number': seq,
                        'method': 'add_curve_tier_change',
                        'sector_idx': inv.get('sector_idx', ''),
                        'input_sector_idx': '',
                        'production_id': '',
                        'isic': inv.get('isic', ''),
                        'tier_index': inv.get('tier_index', ''),
                        'field': '',
                        'change_type_param': '',
                        'value': '',
                        'efficiency_type': '',
                        'va_component': '',
                        'input_isic': '',
                        'exclude_list': '',
                        'position': '',
                        'cap': '',
                        'price': '',
                        'va_components': '',
                        'requirements': '',
                        'investment_duration': ''
                    }

                    if cap_act != 'unchanged':
                        row_cap = dict(row_base)
                        row_cap['sequence_number'] = seq
                        row_cap['field'] = 'cap'
                        row_cap['change_type_param'] = cap_act
                        row_cap['value'] = inv.get('cap_val', '')
                        tc_rows.append(row_cap)
                        seq += 1

                    if price_act != 'unchanged':
                        row_price = dict(row_base)
                        row_price['sequence_number'] = seq
                        row_price['field'] = 'price'
                        row_price['change_type_param'] = price_act
                        row_price['value'] = inv.get('price_val', '')
                        tc_rows.append(row_price)
                        seq += 1

                    if cap_act == 'unchanged' and price_act == 'unchanged' and inv.get('field'):
                        row_leg = dict(row_base)
                        row_leg['sequence_number'] = seq
                        row_leg['field'] = inv.get('field', '')
                        row_leg['change_type_param'] = inv.get('change_type', 'set')
                        row_leg['value'] = inv.get('value', '')
                        tc_rows.append(row_leg)
                        seq += 1
                else:
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
                        'position': '',
                        'cap': '',
                        'price': '',
                        'va_components': '',
                        'requirements': '',
                        'investment_duration': ''
                    }
                    tc_rows.append(row)
                    seq += 1

                cost = float(inv.get('capital_cost', 0.0) or 0.0)
                dur = int(inv.get('investment_duration', 1) or 1)
                if cost > 0 or dur > 1:
                    req_row = dict(tc_rows[-1]) if tc_rows else {}
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
    Output('tc-recipe-prod-dropdown', 'options'),
    Output('tc-recipe-sub-good-dropdown', 'options'),
    Output('tc-curve-good-dropdown', 'options'),
    Output('tc-recipe-sector-dropdown', 'options'),
    Input('productions-store', 'data'),
    Input('goods-store', 'data')
)
def update_recipe_prods_dropdown(productions, goods):
    annotated = compute_production_merit_ranks(productions or [], goods or [])
    prod_opts = []
    for p in annotated:
        rank = p.get('tier_rank', 1)
        p_price = float(p.get('price', 0.0))
        tag = f"Tier {rank}"
        raw_qty = p.get('production_quantity', -1)
        cap_txt = f"Cap: {raw_qty}" if raw_qty not in (-1, None, '') else "Cap: ∞"
        prod_opts.append({
            'label': f"[{tag}] #{p['id']} {p['name']} - ${p_price:,.2f} ({cap_txt}) [Produces: {p.get('produce_name', p.get('isic', ''))}]",
            'value': p['id']
        })
    good_opts = [{'label': f"{g.get('name', 'Unknown')} ({g.get('isic', '')})", 'value': g.get('isic', '')} for g in (goods or [])]
    return prod_opts, good_opts, good_opts, good_opts

@app.callback(
    Output('tc-recipe-prod-dropdown', 'value'),
    Input({'type': 'recipe-tier-select-btn', 'index': ALL}, 'n_clicks'),
    Input('tc-recipe-sector-dropdown', 'value'),
    State('tc-recipe-prod-dropdown', 'value'),
    State('productions-store', 'data'),
    State('goods-store', 'data'),
    prevent_initial_call=True
)
def handle_recipe_prod_selection(btn_clicks, sector_isic, current_prod_id, productions, goods):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    trig = ctx.triggered[0]['prop_id']
    if 'recipe-tier-select-btn' in trig:
        btn_prop = json.loads(trig.split('.')[0])
        return int(btn_prop['index'])
    elif 'tc-recipe-sector-dropdown' in trig:
        if not sector_isic or not productions:
            return None
        target_gid = str(next((g.get('id_number', '') for g in (goods or []) if g.get('isic') == sector_isic), ''))
        sector_prods = [p for p in productions if (target_gid and str(p.get('produce')) == target_gid) or str(p.get('isic')) == sector_isic]
        if current_prod_id and any(p['id'] == current_prod_id for p in sector_prods):
            return current_prod_id
        sorted_p = sorted(sector_prods, key=lambda x: float(x.get('price', 0.0)))
        return sorted_p[0]['id'] if sorted_p else None
    return dash.no_update

@app.callback(
    Output('tc-recipe-details-container', 'style'),
    Output('tc-recipe-selected-title', 'children'),
    Output('tc-recipe-target-input-dropdown', 'options'),
    Output('tc-recipe-target-input-dropdown', 'value'),
    Input('tc-recipe-prod-dropdown', 'value'),
    State('productions-store', 'data'),
    State('goods-store', 'data')
)
def toggle_recipe_details(prod_id, productions, goods):
    if not prod_id or not productions:
        return {'display': 'none'}, "", [], None
    prod = next((p for p in (productions or []) if p['id'] == prod_id), None)
    if not prod:
        return {'display': 'none'}, "", [], None
    title = f"Selected Recipe: #{prod['id']} {prod.get('name', 'Production')} (Base Price: ${float(prod.get('price', 0.0)):,.2f})"

    raw_inputs = prod.get('production_inputs', '{}')
    inputs_dict = json.loads(raw_inputs) if isinstance(raw_inputs, str) else dict(raw_inputs or {})

    good_map = {g.get('isic'): g.get('name', g.get('isic')) for g in (goods or [])}
    id_map = {str(g.get('id_number')): g.get('name', str(g.get('id_number'))) for g in (goods or []) if g.get('id_number') is not None}
    id_num_map = {str(g.get('id')): g.get('name', str(g.get('id'))) for g in (goods or []) if g.get('id') is not None}

    opts = []
    for in_key, in_val in inputs_dict.items():
        str_key = str(in_key)
        name = good_map.get(str_key) or id_map.get(str_key) or id_num_map.get(str_key) or str_key
        opts.append({
            'label': f"{name} (${float(in_val):,.2f}) [{str_key}]",
            'value': str_key
        })
    default_val = opts[0]['value'] if opts else None
    return {'display': 'block'}, title, opts, default_val

# --- Micro-level Recipe Modifications Callbacks ---

@app.callback(
    Output('tc-recipe-mods-store', 'data'),
    Output('tc-recipe-sub-val-input', 'value'),
    Output('tc-recipe-status-msg', 'children'),
    Input('tc-recipe-prod-dropdown', 'value'),
    Input('tc-recipe-apply-btn', 'n_clicks'),
    Input('tc-recipe-add-sub-btn', 'n_clicks'),
    Input('tc-recipe-reset-all-btn', 'n_clicks'),
    Input({'type': 'recipe-reset-ing-btn', 'index': dash.ALL}, 'n_clicks'),
    Input({'type': 'recipe-remove-sub-btn', 'index': dash.ALL}, 'n_clicks'),
    State('tc-recipe-target-input-dropdown', 'value'),
    State('tc-recipe-action-dropdown', 'value'),
    State('tc-recipe-val-input', 'value'),
    State('tc-recipe-sub-good-dropdown', 'value'),
    State('tc-recipe-sub-val-input', 'value'),
    State('tc-recipe-mods-store', 'data'),
    State('productions-store', 'data'),
    State('goods-store', 'data'),
    prevent_initial_call=True
)
def manage_recipe_modifications(prod_id, apply_n, add_sub_n, reset_all_n,
                                reset_ing_clicks, remove_sub_clicks,
                                target_ing, action, val,
                                sub_good, sub_val,
                                current_mods, productions, goods):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    trig = ctx.triggered[0]['prop_id']
    mods = dict(current_mods or {'prod_id': prod_id, 'ingredients': {}, 'substitutes': {}})
    if mods.get('prod_id') != prod_id:
        mods = {'prod_id': prod_id, 'ingredients': {}, 'substitutes': {}}

    # If recipe selected changed
    if 'tc-recipe-prod-dropdown' in trig:
        return {'prod_id': prod_id, 'ingredients': {}, 'substitutes': {}}, 0.0, ""

    # Reset all
    if 'tc-recipe-reset-all-btn' in trig:
        mods['ingredients'] = {}
        mods['substitutes'] = {}
        return mods, 0.0, dbc.Badge("Reset all changes", color="secondary", className="p-1")

    # Apply ingredient change
    if 'tc-recipe-apply-btn' in trig:
        if not target_ing:
            return dash.no_update, dash.no_update, dbc.Badge("Select an ingredient", color="warning", className="p-1")
        try:
            v = float(val if val is not None else 0.0)
        except (ValueError, TypeError):
            v = 0.0
        good_name = next((g.get('name') for g in (goods or []) if g.get('isic') == target_ing), target_ing)
        mods.setdefault('ingredients', {})[str(target_ing)] = {
            'action': action or 'replace_cost',
            'val': v,
            'name': good_name
        }
        return mods, dash.no_update, dbc.Badge(f"✓ Applied {good_name}", color="success", className="p-1")

    # Add substitute good
    if 'tc-recipe-add-sub-btn' in trig:
        if not sub_good:
            return dash.no_update, dash.no_update, dbc.Badge("Select substitute good", color="warning", className="p-1")
        try:
            sv = float(sub_val if sub_val is not None else 0.0)
        except (ValueError, TypeError):
            sv = 0.0
        if sv <= 0:
            return dash.no_update, dash.no_update, dbc.Badge("Cost must be > 0", color="warning", className="p-1")
        good_name = next((g.get('name') for g in (goods or []) if g.get('isic') == sub_good), sub_good)
        mods.setdefault('substitutes', {})[str(sub_good)] = {
            'val': sv,
            'name': good_name
        }
        return mods, 0.0, dbc.Badge(f"✓ Added {good_name}", color="success", className="p-1")

    # Pattern matching reset single ingredient
    if 'recipe-reset-ing-btn' in trig:
        trig_json = json.loads(trig.split('.')[0])
        ing_key = str(trig_json['index'])
        if ing_key in mods.get('ingredients', {}):
            del mods['ingredients'][ing_key]
        return mods, dash.no_update, dbc.Badge("Reverted ingredient", color="info", className="p-1")

    # Pattern matching remove substitute
    if 'recipe-remove-sub-btn' in trig:
        trig_json = json.loads(trig.split('.')[0])
        sub_key = str(trig_json['index'])
        if sub_key in mods.get('substitutes', {}):
            del mods['substitutes'][sub_key]
        return mods, dash.no_update, dbc.Badge("Removed substitute", color="info", className="p-1")

    return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output('tc-recipe-target-input-dropdown', 'value', allow_duplicate=True),
    Output('tc-recipe-action-dropdown', 'value', allow_duplicate=True),
    Output('tc-recipe-val-input', 'value', allow_duplicate=True),
    Input({'type': 'recipe-edit-ing-btn', 'index': dash.ALL}, 'n_clicks'),
    State('tc-recipe-mods-store', 'data'),
    State('tc-recipe-prod-dropdown', 'value'),
    State('productions-store', 'data'),
    prevent_initial_call=True
)
def handle_edit_ing_click(n_clicks, mods, prod_id, productions):
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks):
        return dash.no_update, dash.no_update, dash.no_update
    trig = ctx.triggered[0]['prop_id']
    try:
        trig_json = json.loads(trig.split('.')[0])
        ing_isic = str(trig_json['index'])
    except Exception:
        return dash.no_update, dash.no_update, dash.no_update

    ing_info = (mods.get('ingredients') or {}).get(ing_isic)
    if ing_info:
        return ing_isic, ing_info.get('action', 'replace_cost'), ing_info.get('val', 0.0)

    # If not yet modified, load baseline cost
    prod = next((p for p in (productions or []) if p['id'] == prod_id), None)
    val = 0.0
    if prod:
        raw_inputs = prod.get('production_inputs', '{}')
        inputs_dict = json.loads(raw_inputs) if isinstance(raw_inputs, str) else dict(raw_inputs or {})
        val = float(inputs_dict.get(ing_isic, 0.0))
    return ing_isic, 'replace_cost', val

@app.callback(
    Output('tc-recipe-val-input', 'value', allow_duplicate=True),
    Input('tc-recipe-target-input-dropdown', 'value'),
    Input('tc-recipe-action-dropdown', 'value'),
    State('tc-recipe-prod-dropdown', 'value'),
    State('tc-recipe-mods-store', 'data'),
    State('productions-store', 'data'),
    prevent_initial_call=True
)
def update_recipe_val_default(target_isic, action, prod_id, mods, productions):
    if not prod_id or not productions or not target_isic:
        return dash.no_update
    ctx = callback_context
    trig = ctx.triggered[0]['prop_id'] if ctx.triggered else ''

    ing_mods = (mods or {}).get('ingredients', {})
    if 'tc-recipe-target-input-dropdown' in trig and str(target_isic) in ing_mods:
        return ing_mods[str(target_isic)].get('val', 0.0)

    prod = next((p for p in productions if p['id'] == prod_id), None)
    if not prod:
        return dash.no_update
    raw_inputs = prod.get('production_inputs', '{}')
    inputs_dict = json.loads(raw_inputs) if isinstance(raw_inputs, str) else dict(raw_inputs or {})
    current_cost = float(inputs_dict.get(target_isic, 0.0))
    if action == 'replace_cost':
        return current_cost
    elif action == 'multiply':
        return 0.8
    elif action == 'add':
        return 0.0
    return current_cost

@app.callback(
    Output('tc-recipe-preview-display', 'children'),
    Output('tc-recipe-staged-summary', 'children'),
    Input('tc-recipe-prod-dropdown', 'value'),
    Input('tc-recipe-mods-store', 'data'),
    State('productions-store', 'data'),
    State('goods-store', 'data')
)
def update_recipe_preview(prod_id, mods, productions, goods):
    if not prod_id or not productions:
        return html.Div("Select a production process above to preview recipe calculation.", className="text-muted small fst-italic"), ""
    prod = next((p for p in productions if p['id'] == prod_id), None)
    if not prod:
        return html.Div("Selected production not found.", className="text-danger small"), ""

    mods = mods or {'ingredients': {}, 'substitutes': {}}
    ing_mods = mods.get('ingredients', {})
    sub_mods = mods.get('substitutes', {})

    impact = compute_recipe_impact(
        prod=prod,
        ingredient_changes=ing_mods,
        substitute_goods=sub_mods
    )
    if not impact:
        return html.Div("Unable to calculate recipe shifts.", className="text-danger small"), ""

    good_map = {g.get('isic'): g.get('name', g.get('isic')) for g in (goods or [])}
    id_map = {str(g.get('id_number')): g.get('name', str(g.get('id_number'))) for g in (goods or []) if g.get('id_number') is not None}
    id_num_map = {str(g.get('id')): g.get('name', str(g.get('id'))) for g in (goods or []) if g.get('id') is not None}

    p_color = "success" if impact['delta_price'] < 0 else ("danger" if impact['delta_price'] > 0 else "secondary")

    rows = []
    for item in impact['comparison']:
        k = item['input_isic']
        in_name = good_map.get(k) or id_map.get(str(k)) or id_num_map.get(str(k)) or k
        c_b = item['cost_before']
        c_a = item['cost_after']
        a_b = item['a_before']
        a_a = item['a_after']
        da = item['delta_a']

        is_sub = item.get('is_substitute', False)
        is_mod = item.get('is_modified', False)

        c_disp = f"${c_b:,.2f} → ${c_a:,.2f}" if c_b != c_a else f"${c_b:,.2f}"
        a_disp = f"{a_b:.4f} → {a_a:.4f}" if a_b != a_a else f"{a_b:.4f}"
        da_style = {'color': '#16a34a', 'fontWeight': 'bold'} if da < 0 else ({'color': '#dc2626', 'fontWeight': 'bold'} if da > 0 else {'color': '#64748b'})
        da_str = f"{da:+.4f}" if da != 0 else "0.0000"

        if is_sub:
            badge_txt = dbc.Badge("Substitute", color="info", className="ms-2")
            action_desc = f"New Good (+${c_a:,.2f})"
            action_cell = dbc.Button("✕ Remove", id={'type': 'recipe-remove-sub-btn', 'index': k}, size="sm", color="outline-danger", className="py-0 px-2")
            row_class = "table-info"
        elif is_mod:
            chg = ing_mods[k]
            act = chg.get('action', 'replace_cost')
            val = chg.get('val', 0.0)
            if act == 'replace_cost':
                action_desc = f"Set: ${val:,.2f}"
            elif act == 'multiply':
                action_desc = f"Multiply: {val:.2f}x ({round((val-1)*100):+d}%)"
            else:
                action_desc = f"Add: {val:+,.2f}"
            badge_txt = dbc.Badge("Modified", color="primary", className="ms-2")
            action_cell = html.Div([
                dbc.Button("✎ Edit", id={'type': 'recipe-edit-ing-btn', 'index': k}, size="sm", color="outline-primary", className="py-0 px-1 me-1"),
                dbc.Button("↺ Reset", id={'type': 'recipe-reset-ing-btn', 'index': k}, size="sm", color="outline-secondary", className="py-0 px-1"),
            ])
            row_class = "table-primary"
        else:
            badge_txt = None
            action_desc = "Baseline (Unchanged)"
            action_cell = dbc.Button("✎ Modify", id={'type': 'recipe-edit-ing-btn', 'index': k}, size="sm", color="outline-secondary", className="py-0 px-1")
            row_class = ""

        rows.append(html.Tr([
            html.Td([html.Span(in_name, className="fw-bold" if (is_mod or is_sub) else ""), badge_txt] if badge_txt else in_name),
            html.Td(k, className="small text-muted"),
            html.Td(f"${c_b:,.2f}"),
            html.Td(html.Span(action_desc, className="small text-primary fw-bold" if is_mod else ("small text-info fw-bold" if is_sub else "small text-muted"))),
            html.Td(c_disp),
            html.Td(a_disp),
            html.Td(da_str, style=da_style),
            html.Td(action_cell, style={'textAlign': 'center'})
        ], className=row_class))

    # Row for Value Added
    va_b = impact['va_before']
    va_a = impact['va_after']
    va_disp = f"{va_b:.4f} → {va_a:.4f}" if va_b != va_a else f"{va_b:.4f}"
    rows.append(html.Tr([
        html.Td(html.Span("Value Added (Wages + Surplus)", className="fst-italic text-muted")),
        html.Td("—", className="small text-muted"),
        html.Td(f"${impact['total_va']:,.2f}"),
        html.Td("Fixed", className="small text-muted"),
        html.Td(f"${impact['total_va']:,.2f}"),
        html.Td(va_disp),
        html.Td(f"{va_a - va_b:+.4f}", className="text-muted"),
        html.Td("—", style={'textAlign': 'center', 'color': '#94a3b8'})
    ], className="table-light"))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Ingredient / Good"),
            html.Th("ISIC"),
            html.Th("Base Cost ($)"),
            html.Th("Configured Change"),
            html.Th("Effective Cost ($)"),
            html.Th("Direct Coeff A[i, j]"),
            html.Th("Shift (ΔA[i, j])"),
            html.Th("Action", style={'textAlign': 'center', 'width': '120px'})
        ])),
        html.Tbody(rows)
    ], bordered=True, hover=True, responsive=True, size="sm", className="mb-2")

    pct_txt = f"{impact['pct_price_change']:+.1f}%"
    summary_badges = html.Div([
        dbc.Badge(f"Recipe Unit Price: ${impact['price_before']:,.2f} → ${impact['price_after']:,.2f} ({impact['delta_price']:+,.2f}, {pct_txt})", color=p_color, className="me-2 mb-2 p-2", style={'fontSize': '0.85rem'}),
        dbc.Badge(f"Total Value Added: ${impact['total_va']:,.2f}", color="secondary", className="me-2 mb-2 p-2", style={'fontSize': '0.85rem'}),
    ])

    n_i = len(ing_mods)
    n_s = len(sub_mods)
    badge_label = f"{n_i} modified, {n_s} substitute(s)" if (n_i + n_s) > 0 else "0 changes staged"
    staged_badge = dbc.Badge(badge_label, color="primary" if (n_i + n_s) > 0 else "secondary", className="p-1")

    return html.Div([summary_badges, table]), staged_badge

@app.callback(
    Output('tc-recipe-tiers-table-container', 'children'),
    Input('tc-recipe-sector-dropdown', 'value'),
    Input('tc-recipe-search-input', 'value'),
    Input('tc-recipe-prod-dropdown', 'value'),
    Input('tc-recipe-mods-store', 'data'),
    State('productions-store', 'data'),
    State('goods-store', 'data')
)
def render_recipe_tiers_table(sector_isic, search_query, selected_prod_id,
                              recipe_mods, productions, goods):
    if not sector_isic:
        return html.Div("Select a sector above to view its production methods and tiers.", className="text-muted small fst-italic p-2")

    annotated = compute_production_merit_ranks(productions or [], goods or [])
    target_gid = str(next((g.get('id_number', '') for g in (goods or []) if g.get('isic') == sector_isic), ''))
    sector_prods = [p for p in annotated if (target_gid and str(p.get('produce')) == target_gid) or str(p.get('isic')) == sector_isic]

    if not sector_prods:
        return html.Div("No production processes defined for this sector.", className="text-warning small fst-italic p-2")

    sector_prods = sorted(sector_prods, key=lambda p: float(p.get('price', 0.0)))

    if search_query and search_query.strip():
        q = search_query.strip().lower()
        sector_prods = [p for p in sector_prods if q in str(p.get('name', '')).lower() or q in str(p.get('id', '')).lower() or q in str(p.get('producer', '')).lower()]

    if not sector_prods:
        return html.Div(f"No productions matching '{search_query}'.", className="text-muted small fst-italic p-2")

    has_mods = bool(recipe_mods and (recipe_mods.get('ingredients') or recipe_mods.get('substitutes')))

    rows = []
    for rank, p in enumerate(sector_prods, 1):
        is_selected = (selected_prod_id is not None and p['id'] == selected_prod_id)
        price_display = f"${p['price']:,.2f}"

        if is_selected and has_mods:
            impact = compute_recipe_impact(
                p,
                ingredient_changes=(recipe_mods or {}).get('ingredients', {}),
                substitute_goods=(recipe_mods or {}).get('substitutes', {})
            )
            if impact and impact.get('price_after') is not None:
                p_after = impact['price_after']
                if p_after != p['price']:
                    delta_color = '#16a34a' if impact['delta_price'] < 0 else '#d97706'
                    price_display = html.Span([
                        f"${p['price']:,.2f} → ",
                        html.B(f"${p_after:,.2f}", style={'color': delta_color})
                    ])

        raw_qty = p.get('production_quantity', -1)
        cap_str = f"{raw_qty:,.0f}" if raw_qty not in (-1, None, '') else "∞"
        btn_label = "Selected ✓" if is_selected else "Select"
        btn_color = "success" if is_selected else "outline-primary"
        row_class = "table-primary fw-bold" if is_selected else ""

        rows.append(html.Tr([
            html.Td(
                dbc.Button(
                    btn_label,
                    id={'type': 'recipe-tier-select-btn', 'index': p['id']},
                    size="sm",
                    color=btn_color,
                    className="py-0 px-2"
                ),
                style={'width': '110px', 'textAlign': 'center'}
            ),
            html.Td(f"Tier {rank}"),
            html.Td(f"#{p['id']} {p.get('name', 'Production')}"),
            html.Td(price_display),
            html.Td(cap_str),
            html.Td(str(p.get('producer', '1001'))),
        ], className=row_class))

    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Action", style={'width': '110px', 'textAlign': 'center'}),
            html.Th("Tier"),
            html.Th("Production Method"),
            html.Th("Unit Price ($)"),
            html.Th("Capacity"),
            html.Th("Producer ID"),
        ])),
        html.Tbody(rows)
    ], bordered=True, hover=True, responsive=True, size="sm", className="mb-0")

@app.callback(
    Output('tc-curve-tier-dropdown', 'options'),
    Output('tc-curve-tier-dropdown', 'value'),
    Input('tc-curve-good-dropdown', 'value'),
    State('productions-store', 'data'),
    State('goods-store', 'data')
)
def update_curve_tier_options(target_isic, productions, goods):
    if not target_isic or not productions:
        return [], None
    annotated = compute_production_merit_ranks(productions, goods)
    target_gid = str(next((g.get('id_number', '') for g in (goods or []) if g.get('isic') == target_isic), ''))
    target_prods = [p for p in annotated if (target_gid and str(p.get('produce')) == target_gid) or str(p.get('isic')) == target_isic]
    if not target_prods:
        return [], None
    sorted_tiers = sorted(target_prods, key=lambda p: float(p.get('price', 0.0)))
    tier_opts = []
    for idx, t in enumerate(sorted_tiers):
        raw_qty = t.get('production_quantity', -1)
        cap_str = f"Cap: {raw_qty}" if raw_qty not in (-1, None, '') else "Cap: ∞"
        price_str = f"${float(t.get('price', 0.0)):,.2f}"
        tier_opts.append({
            'label': f"Tier {idx+1} - {t.get('name', 'Production')} ({price_str}, {cap_str})",
            'value': idx
        })
    first_val = 0 if tier_opts else None
    return tier_opts, first_val

@app.callback(
    Output('tc-curve-cap-val-input', 'value'),
    Output('tc-curve-price-val-input', 'value'),
    Input('tc-curve-tier-dropdown', 'value'),
    State('tc-curve-good-dropdown', 'value'),
    State('productions-store', 'data'),
    State('goods-store', 'data')
)
def update_curve_inputs_defaults(tier_idx, target_isic, productions, goods):
    if tier_idx is None or not target_isic or not productions:
        return 200.0, 0.0
    annotated = compute_production_merit_ranks(productions, goods)
    target_gid = str(next((g.get('id_number', '') for g in (goods or []) if g.get('isic') == target_isic), ''))
    target_prods = [p for p in annotated if (target_gid and str(p.get('produce')) == target_gid) or str(p.get('isic')) == target_isic]
    sorted_tiers = sorted(target_prods, key=lambda p: float(p.get('price', 0.0)))
    if tier_idx < 0 or tier_idx >= len(sorted_tiers):
        return 200.0, 0.0
    selected_tier = sorted_tiers[tier_idx]
    raw_cap = selected_tier.get('production_quantity', 100)
    cap_val = float(raw_cap) if raw_cap not in (-1, None, '') else 200.0
    price_val = float(selected_tier.get('price', 0.0))
    return cap_val, price_val

@app.callback(
    Output('tc-curve-preview-display', 'children'),
    Input('tc-curve-good-dropdown', 'value'),
    Input('tc-curve-tier-dropdown', 'value'),
    Input('tc-curve-cap-action-dropdown', 'value'),
    Input('tc-curve-cap-val-input', 'value'),
    Input('tc-curve-price-action-dropdown', 'value'),
    Input('tc-curve-price-val-input', 'value'),
    State('goods-store', 'data'),
    State('productions-store', 'data'),
    State('tax-policies-store', 'data')
)
def update_curve_preview(target_isic, tier_idx, cap_action, cap_val, price_action, price_val, goods, productions, tax_data):
    if not target_isic or tier_idx is None:
        return html.Div("Select a good and tier above to preview supply curve dispatch, circular demand, and effective market price.", className="text-muted small fst-italic")

    impact = compute_curve_impact(
        goods=goods,
        productions=productions,
        target_isic=target_isic,
        tier_idx=tier_idx,
        cap_action=cap_action or 'unchanged',
        cap_val=cap_val if cap_val is not None else 0.0,
        price_action=price_action or 'unchanged',
        price_val=price_val if price_val is not None else 0.0,
        tax_data=tax_data
    )
    if not impact:
        return html.Div("Unable to calculate supply curve dispatch (check that good has defined productions).", className="text-danger small")

    p_color = "success" if impact['delta_peff'] < 0 else ("danger" if impact['delta_peff'] > 0 else "secondary")

    rows = []
    for idx, (b, a) in enumerate(zip(impact['disp_before'], impact['disp_after']), 1):
        cap_b = f"{b['cap']:,.0f}" if b['cap'] not in (-1, float('inf')) else "∞"
        cap_a = f"{a['cap']:,.0f}" if a['cap'] not in (-1, float('inf')) else "∞"
        price_display = f"${b['price']:,.2f} → ${a['price']:,.2f}" if b['price'] != a['price'] else f"${a['price']:,.2f}"
        delta_disp = a['dispatched'] - b['dispatched']
        disp_style = {'color': '#16a34a', 'fontWeight': 'bold'} if (delta_disp > 0 and b['price'] <= a['price']) or (delta_disp < 0 and b['price'] > a['price']) else {'color': '#334155'}

        status_badge_color = "success" if "100%" in a['status'] else ("warning" if "Marginal" in a['status'] else "secondary")

        rows.append(html.Tr([
            html.Td(f"Tier {idx}"),
            html.Td(a['name']),
            html.Td(price_display),
            html.Td(f"{cap_b} → {cap_a}"),
            html.Td(f"{b['dispatched']:,.1f} → {a['dispatched']:,.1f}", style=disp_style),
            html.Td(f"{a['utilization']:.1f}%"),
            html.Td(dbc.Badge(a['status'], color=status_badge_color, className="p-1"))
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Tier"),
            html.Th("Production Method"),
            html.Th("Price ($)"),
            html.Th("Capacity (Old → New)"),
            html.Th("Dispatched Output"),
            html.Th("Utilization (%)"),
            html.Th("Dispatch Status")
        ])),
        html.Tbody(rows)
    ], bordered=True, hover=True, responsive=True, size="sm", className="mb-2")

    good_name = next((g.get('name') for g in (goods or []) if g.get('isic') == target_isic), target_isic)
    summary_badges = html.Div([
        dbc.Badge(f"Sector Output Demand (X): {impact['sector_output_demand']:,.1f} units (from Leontief Matrix & Final Demand)", color="dark", className="me-2 mb-2 p-2", style={'fontSize': '0.85rem'}),
        dbc.Badge(f"Effective Market Price: ${impact['p_eff_before']:,.2f} → ${impact['p_eff_after']:,.2f} ({impact['delta_peff']:+,.2f}, {impact['pct_peff']:+.1f}%)", color=p_color, className="me-2 mb-2 p-2", style={'fontSize': '0.85rem'}),
    ])

    note = html.Small(
        f"✓ Total gross production of {impact['sector_output_demand']:,.1f} units is dispatched into {good_name}'s merit-order supply curve. Lower-price tiers saturate first; remaining demand spills into higher-cost reserve tiers.",
        className="text-muted d-block mt-1 fst-italic"
    )

    return html.Div([summary_badges, table, note])

@app.callback(
    Output('tc-macro-inputs-container', 'style'),
    Output('tc-micro-recipe-container', 'style'),
    Output('tc-curve-inputs-container', 'style'),
    Output('tc-sector-container', 'style'),
    Output('tc-input-sector-container', 'style'),
    Input('tc-method-dropdown', 'value')
)
def toggle_tc_method_fields(method):
    if method == 'add_production_input_change':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    elif method == 'add_curve_tier_change':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
    elif method == 'add_input_change':
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    elif method == 'add_sector_change':
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
    else:  # add_coefficient_change
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}

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
    State('tc-recipe-prod-dropdown', 'value'),
    State('tc-recipe-target-input-dropdown', 'value'),
    State('tc-recipe-action-dropdown', 'value'),
    State('tc-recipe-val-input', 'value'),
    State('tc-recipe-sub-good-dropdown', 'value'),
    State('tc-recipe-sub-val-input', 'value'),
    State('tc-recipe-mods-store', 'data'),
    State('tc-curve-good-dropdown', 'value'),
    State('tc-curve-tier-dropdown', 'value'),
    State('tc-curve-cap-action-dropdown', 'value'),
    State('tc-curve-cap-val-input', 'value'),
    State('tc-curve-price-action-dropdown', 'value'),
    State('tc-curve-price-val-input', 'value'),
    State('tc-cost-input', 'value'),
    State('tc-duration-input', 'value'),
    State('pending-investments-store', 'data'),
    State('productions-store', 'data'),
    State('goods-store', 'data'),
    prevent_initial_call=True
)
def manage_pending_investments(add_btn, clear_btn, method, sector, input_sector,
                               change_type, value,
                               rec_prod_id, rec_target, rec_action, rec_val, rec_sub_good, rec_sub_val,
                               recipe_mods,
                               curve_isic, curve_tier, curve_cap_action, curve_cap_val, curve_price_action, curve_price_val,
                               cost, duration, pending, productions, goods):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    pending = list(pending or [])

    if trigger_id == 'clear-investments-btn':
        return [], html.Span("No investments added to this scenario yet.", className="text-muted fst-italic"), ""

    if trigger_id == 'add-investment-btn':
        cost_val = float(cost or 0.0)
        dur_val = max(1, int(duration or 1))

        if method == 'add_production_input_change':
            if not rec_prod_id:
                return dash.no_update, dash.no_update, "Target production process is required."
            prod = next((p for p in (productions or []) if p['id'] == rec_prod_id), None)
            if not prod:
                return dash.no_update, dash.no_update, "Selected production process not found."

            mods = dict(recipe_mods or {'ingredients': {}, 'substitutes': {}})
            # If user filled in the input fields without pressing 'Apply', automatically include it
            if not mods.get('ingredients') and not mods.get('substitutes'):
                if rec_target:
                    mods['ingredients'] = {
                        str(rec_target): {
                            'action': rec_action or 'replace_cost',
                            'val': float(rec_val or 0.0)
                        }
                    }
                    if rec_sub_good and float(rec_sub_val or 0.0) > 0:
                        mods['substitutes'] = {str(rec_sub_good): float(rec_sub_val or 0.0)}

            if not mods.get('ingredients') and not mods.get('substitutes'):
                return dash.no_update, dash.no_update, "Please apply at least one ingredient modification or substitute good."

            impact = compute_recipe_impact(
                prod=prod,
                ingredient_changes=mods.get('ingredients', {}),
                substitute_goods=mods.get('substitutes', {})
            )
            if not impact:
                return dash.no_update, dash.no_update, "Error calculating recipe impact."

            # Fallback legacy single-item fields
            primary_ing = list(mods.get('ingredients', {}).keys())[0] if mods.get('ingredients') else ''
            primary_act = list(mods.get('ingredients', {}).values())[0].get('action', 'replace_cost') if mods.get('ingredients') else 'replace_cost'
            primary_val = list(mods.get('ingredients', {}).values())[0].get('val', 0.0) if mods.get('ingredients') else 0.0
            primary_sub = list(mods.get('substitutes', {}).keys())[0] if mods.get('substitutes') else ''
            primary_sub_val = list(mods.get('substitutes', {}).values())[0].get('val', 0.0) if mods.get('substitutes') else 0.0

            inv = {
                'method': 'add_production_input_change',
                'production_id': int(rec_prod_id),
                'sector_idx': prod.get('isic', ''),
                'prod_name': prod.get('name', ''),
                'produce_name': prod.get('produce_name', ''),
                'price_before': impact['price_before'],
                'price_after': impact['price_after'],
                'comparison': impact['comparison'],
                'capital_cost': cost_val,
                'investment_duration': dur_val,
                'ingredient_changes': mods.get('ingredients', {}),
                'substitutes': mods.get('substitutes', {}),
                'input_isic': primary_ing,
                'change_type': primary_act,
                'value': primary_val,
                'substitute_isic': primary_sub,
                'substitute_val': primary_sub_val
            }
            pending.append(inv)

        elif method == 'add_curve_tier_change':
            if not curve_isic:
                return dash.no_update, dash.no_update, "Target good is required."
            if curve_tier is None:
                return dash.no_update, dash.no_update, "Target tier is required."
            
            cap_act = curve_cap_action or 'unchanged'
            price_act = curve_price_action or 'unchanged'
            if cap_act == 'unchanged' and price_act == 'unchanged':
                return dash.no_update, dash.no_update, "Please set a Change Action for Capacity and/or Unit Price (both are currently unchanged)."
            
            if cap_act != 'unchanged' and curve_cap_val is None:
                return dash.no_update, dash.no_update, "Capacity value is required when capacity change action is selected."
            if price_act != 'unchanged' and curve_price_val is None:
                return dash.no_update, dash.no_update, "Price value is required when price change action is selected."

            sec_name = next((g.get('name') for g in (goods or []) if g.get('isic') == curve_isic), curve_isic)
            inv = {
                'method': 'add_curve_tier_change',
                'isic': curve_isic,
                'sector_idx': curve_isic,
                'tier_index': int(curve_tier),
                'cap_action': cap_act,
                'cap_val': float(curve_cap_val or 0.0),
                'price_action': price_act,
                'price_val': float(curve_price_val or 0.0),
                'capital_cost': cost_val,
                'investment_duration': dur_val,
                'sec_name': sec_name
            }
            pending.append(inv)

        else:
            if value is None:
                return dash.no_update, dash.no_update, "Value is required."
            if method in ('add_coefficient_change', 'add_sector_change') and not sector:
                return dash.no_update, dash.no_update, "Sector (j) is required for this method."
            if method in ('add_coefficient_change', 'add_input_change') and not input_sector:
                return dash.no_update, dash.no_update, "Input good (i) is required for this method."

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
        if m == 'add_production_input_change':
            n_i = len(inv.get('ingredient_changes', {}))
            n_s = len(inv.get('substitutes', {}))
            details = []
            if n_i > 0:
                details.append(f"{n_i} input(s) modified")
            elif inv.get('input_isic'):
                t_name = good_map.get(inv['input_isic'], inv['input_isic'])
                details.append(f"{t_name} {inv.get('change_type', 'set')} {inv.get('value', 0)}")
            if n_s > 0:
                details.append(f"{n_s} sub(s) added")
            elif inv.get('substitute_isic'):
                details.append(f"+sub {good_map.get(inv['substitute_isic'], inv['substitute_isic'])}")
            desc_str = ", ".join(details) if details else "Recipe modified"
            desc = f"Recipe #{inv['production_id']} ({inv.get('prod_name', '')}): {desc_str} [P: ${inv.get('price_before')}→${inv.get('price_after')}]"
            color = "info" if inv['capital_cost'] == 0 else "success"
        elif m == 'add_curve_tier_change':
            sec_name = inv.get('sec_name', good_map.get(inv.get('isic'), inv.get('isic')))
            tier_num = inv.get('tier_index', 0) + 1
            changes_desc = []
            if inv.get('cap_action') and inv['cap_action'] != 'unchanged':
                changes_desc.append(f"Cap: {inv['cap_action']} {inv.get('cap_val')}")
            if inv.get('price_action') and inv['price_action'] != 'unchanged':
                changes_desc.append(f"Price: {inv['price_action']} ${inv.get('price_val')}")
            if not changes_desc and inv.get('field'):
                changes_desc.append(f"{inv.get('field')}: {inv.get('change_type')} {inv.get('value')}")
            desc_str = " & ".join(changes_desc) if changes_desc else "No changes"
            desc = f"Supply Curve [{sec_name}] Tier {tier_num}: {desc_str}"
            color = "primary" if inv['capital_cost'] == 0 else "success"
        else:
            sec_name = good_map.get(inv['sector_idx'], inv['sector_idx']) if inv.get('sector_idx') else ''
            in_name = good_map.get(inv['input_sector_idx'], inv['input_sector_idx']) if inv.get('input_sector_idx') else ''
            if m == 'add_coefficient_change':
                desc = f"A[{in_name}, {sec_name}]: {inv['change_type']} {inv['value']}"
            elif m == 'add_input_change':
                desc = f"Input row [{in_name}] everywhere: {inv['change_type']} {inv['value']}"
            elif m == 'add_sector_change':
                desc = f"Sector [{sec_name}] all inputs: {inv['change_type']} {inv['value']}"
            else:
                desc = f"{m}: {inv['change_type']} {inv['value']}"
            color = "secondary" if inv['capital_cost'] == 0 else "success"

        cost_txt = f"${inv['capital_cost']:,.2f}" if inv['capital_cost'] > 0 else "Free ($0)"
        dur_txt = f"{inv['investment_duration']} iters" if inv['investment_duration'] > 1 else "1 iter"

        items.append(
            dbc.Badge(
                f"#{idx} {desc} | Req: {cost_txt}, {dur_txt}",
                color=color,
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
    State('pending-investments-store', 'data'),
    State('tech-changes-store', 'data'),
    prevent_initial_call=True
)
def add_tc_scenario(n, title, tc_id, desc, pending, scenarios):
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
        'iterations': 5,
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
