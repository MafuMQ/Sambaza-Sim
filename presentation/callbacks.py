import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, State, callback, html, no_update, ctx
import dash_ag_grid as dag

logger = logging.getLogger(__name__)

from presentation.layout import get_examples_config
from presentation.charts import (
    build_waterfall_chart,
    build_delta_bar_chart,
    build_ledger_trajectory_chart,
    build_savings_breakdown_chart,
)
from simulators.savings_ledger import SavingsLedger, InvestmentNotAffordableError


# ---------------------------------------------------------------------------
# Tech-change builder: show/hide fields based on change type
# ---------------------------------------------------------------------------

@callback(
    Output('tc-sector-container', 'style'),
    Output('tc-input-sector-container', 'style'),
    Output('tc-operation-container', 'style'),
    Output('tc-value-container', 'style'),
    Input('tc-change-type', 'value'),
)
def toggle_tc_fields(change_type):
    """Show/hide inputs depending on the selected change type."""
    show = {'display': 'block'}
    hide = {'display': 'none'}
    if change_type == 'add_sector_change':
        return show, hide, show, show
    elif change_type == 'add_input_change':
        return hide, show, show, show
    else:  # add_coefficient_change
        return show, show, show, show


# ---------------------------------------------------------------------------
# Tech-change builder: add / clear operations in the store
# ---------------------------------------------------------------------------

@callback(
    Output('store-tech-changes', 'data', allow_duplicate=True),
    Input('tc-add-button', 'n_clicks'),
    Input('tc-clear-button', 'n_clicks'),
    State('store-tech-changes', 'data'),
    State('tc-change-type', 'value'),
    State('tc-sector', 'value'),
    State('tc-input-sector', 'value'),
    State('tc-operation', 'value'),
    State('tc-value', 'value'),
    State('tc-cost', 'value'),
    State('tc-duration', 'value'),
    prevent_initial_call=True,
)
def manage_tech_changes(add_clicks, clear_clicks, current_changes, change_type,
                        sector, input_sector, operation, value,
                        tc_cost=0.0, tc_duration=1):
    triggered = ctx.triggered_id
    if triggered == 'tc-clear-button':
        return []

    changes = list(current_changes or [])

    # Validate inputs for the coefficient change action
    if value is None:
        return changes

    capital_cost = float(tc_cost) if tc_cost is not None and float(tc_cost) > 0 else 0.0
    duration = max(1, int(tc_duration)) if tc_duration is not None and int(tc_duration) > 0 else 1

    entry = {
        'method': change_type,
        'params': {
            'change_type': operation,
            'value': float(value),
        },
        'capital_cost': capital_cost,
        'duration': duration,
    }

    if change_type == 'add_sector_change':
        if not sector:
            return changes
        entry['params']['sector_idx'] = sector
    elif change_type == 'add_input_change':
        if not input_sector:
            return changes
        entry['params']['input_sector_idx'] = input_sector
    elif change_type == 'add_coefficient_change':
        if not sector or not input_sector:
            return changes
        entry['params']['sector_idx'] = sector
        entry['params']['input_sector_idx'] = input_sector

    changes.append(entry)
    return changes


# ---------------------------------------------------------------------------
# Tech-change builder: render the list of pending changes
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    'add_sector_change': 'All inputs of',
    'add_input_change': 'Usage of',
    'add_coefficient_change': 'Coefficient',
    'set_capital_requirements': 'Capital Investment Gate',
}

@callback(
    Output('tc-changes-display', 'children'),
    Input('store-tech-changes', 'data'),
)
def render_tc_changes(changes):
    if not changes:
        return html.P('No investments added yet.', style={'fontSize': '11px', 'color': '#64748b'})

    items = []
    for i, ch in enumerate(changes):
        method = ch.get('method', '')
        p = ch.get('params', {})
        cost = float(ch.get('capital_cost', 0.0))
        dur = max(1, int(ch.get('duration', 1)))

        if method == 'set_capital_requirements':
            reqs = p.get('requirements', {})
            dur = p.get('investment_duration', 1)
            total = sum(reqs.values()) if reqs else p.get('total_cost', 0.0)
            desc = f"Legacy Capital Gate: ${total:,.2f} ({dur} iter build)"
            border_c = '#f59e0b'
            badge = "💰 CAPITAL GATE"
        else:
            op = p.get('change_type', '')
            val = p.get('value', '')
            border_c = '#272b3d'
            badge = None
            if method == 'add_sector_change':
                desc = f"{METHOD_LABELS.get(method, method)} sector {p.get('sector_idx', '?')}: {op} {val}"
            elif method == 'add_input_change':
                desc = f"{METHOD_LABELS.get(method, method)} input {p.get('input_sector_idx', '?')} everywhere: {op} {val}"
            elif method == 'add_coefficient_change':
                desc = f"{METHOD_LABELS.get(method, method)} [{p.get('input_sector_idx', '?')} → {p.get('sector_idx', '?')}]: {op} {val}"
            elif method == 'add_production_input_change':
                in_isic = p.get('input_isic', '?')
                in_name = sector_name_map.get(in_isic, in_isic)
                desc = f"Recipe #{p.get('production_id', '?')} input {in_name}: {op} {val}"
            elif method == 'add_production_all_inputs_change':
                desc = f"Production #{p.get('production_id', '?')} all inputs: {op} {val}"
            elif method == 'add_production_efficiency_change':
                desc = f"Production #{p.get('production_id', '?')} {p.get('efficiency_type', '')} efficiency: {op} {val}"
            elif method == 'add_curve_new_tier':
                desc = f"Capacity tier for {p.get('isic', '?')}: cap={p.get('cap')}, price={p.get('price')}"
            elif method == 'add_curve_tier_change':
                desc = f"Curve tier change for {p.get('isic', '?')}: {p.get('field')} {op} {val}"
            elif method == 'add_curve_scale_all_tiers':
                desc = f"Scale curve tiers for {p.get('isic', '?')}: {p.get('field')} {op} {val}"
            else:
                desc = f"{method}: {p}"

        if cost > 0:
            req_badge = html.Span(
                f"💰 ${cost:,.2f} · {dur} iter{'s' if dur > 1 else ''}",
                style={
                    'backgroundColor': 'rgba(245, 158, 11, 0.15)',
                    'color': '#f59e0b',
                    'padding': '2px 7px',
                    'borderRadius': '4px',
                    'fontSize': '10px',
                    'fontWeight': '600',
                    'whiteSpace': 'nowrap',
                }
            )
        else:
            req_badge = html.Span(
                "⚡ Free",
                style={
                    'backgroundColor': 'rgba(100, 116, 139, 0.15)',
                    'color': '#94a3b8',
                    'padding': '2px 7px',
                    'borderRadius': '4px',
                    'fontSize': '10px',
                    'fontWeight': '500',
                    'whiteSpace': 'nowrap',
                }
            )

        items.append(
            html.Div(
                style={
                    'fontSize': '11px', 'padding': '6px 10px',
                    'backgroundColor': '#252836', 'borderRadius': '6px',
                    'marginBottom': '6px', 'border': f'1px solid {border_c}',
                    'color': '#e2e8f0', 'lineHeight': '1.4',
                    'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
                },
                children=[
                    html.Div([
                        html.Span(f"#{i+1} ", style={'fontWeight': '700', 'color': '#4f8ef7'}),
                        html.Span(f"[{badge}] " if badge else "", style={'color': '#f59e0b', 'fontWeight': '600'}),
                        html.Span(desc)
                    ], style={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'marginRight': '8px'}),
                    req_badge
                ]
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
    Output('store-tech-changes', 'data', allow_duplicate=True),
    Input('example-selector', 'value'),
    prevent_initial_call='initial_duplicate',
)
def update_controls(example_id):
    examples_config = get_examples_config()
    if example_id is not None:
        try:
            example_id = int(example_id)
        except (ValueError, TypeError):
            pass

    if not example_id or example_id not in examples_config:
        return "", None, True, "", 'supply_curves', 0.0, 0.0, 0.0, 0.0, []

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

    # Load tech change spec from example config into the pending-changes store
    loaded_changes = []
    tc_cfg = params.get('tech_change_config', {})
    spec = tc_cfg.get('spec', {}) if isinstance(tc_cfg, dict) else {}
    spec_changes = spec.get('changes', [])

    # Check if spec has a set_capital_requirements
    scen_cap_cost = 0.0
    scen_cap_dur = 1
    scen_reqs = None
    has_scen_cap = False
    for ch in spec_changes:
        if ch.get('method') == 'set_capital_requirements':
            p = ch.get('params', {})
            reqs = p.get('requirements', {})
            scen_reqs = reqs
            scen_cap_cost += sum(reqs.values()) if reqs else float(p.get('total_cost', 0.0))
            scen_cap_dur = max(scen_cap_dur, int(p.get('investment_duration', 1)))
            has_scen_cap = True

    for ch in spec_changes:
        method = ch.get('method')
        p = ch.get('params', {})
        if method != 'set_capital_requirements':
            entry = {
                'method': method,
                'params': p,
                'capital_cost': ch.get('capital_cost', scen_cap_cost if has_scen_cap else 0.0),
                'duration': ch.get('duration', scen_cap_dur if has_scen_cap else 1),
            }
            if scen_reqs:
                entry['requirements'] = scen_reqs
            loaded_changes.append(entry)

    return (
        desc, fd_total, fd_disabled, fd_hint, solver,
        inc_before, inc_after, corp_before, corp_after,
        loaded_changes,
    )

@callback(
    Output('summary-output', 'children'),
    Output('summary-output', 'style'),
    Output('summary-va', 'children'),
    Output('summary-va', 'style'),
    Output('summary-fd', 'children'),
    Output('summary-fd', 'style'),
    Output('summary-tax', 'children'),
    Output('summary-tax', 'style'),
    Output('summary-ledger', 'children'),
    Output('summary-ledger', 'style'),
    Output('financing-status-banner', 'children'),
    Output('financing-status-banner', 'style'),
    Output('fin-stat-initial', 'children'),
    Output('fin-stat-inflows', 'children'),
    Output('fin-stat-reserved', 'children'),
    Output('fin-stat-balance', 'children'),
    Output('fin-stat-status', 'children'),
    Output('fin-stat-status', 'style'),
    Output('graph-waterfall', 'figure'),
    Output('graph-delta-bar', 'figure'),
    Output('graph-ledger-trajectory', 'figure'),
    Output('graph-savings-breakdown', 'figure'),
    Output('table-results', 'rowData'),
    Output('table-demand-vector', 'rowData'),
    Output('table-va-coefficients', 'rowData'),
    Output('table-capital-requirements', 'rowData'),
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
    State('input-initial-savings', 'value'),
    State('input-savings-gate', 'value'),
)
def execute_simulation(n_clicks, example_id, iterations, solver,
                       inc_before, inc_after, corp_before, corp_after,
                       ui_tech_changes, total_fd_override,
                       wage_spend, surplus_spend, tax_spend, economy_type,
                       initial_savings_val=0.0, savings_gate=None,
                       **kwargs):
    if example_id is not None:
        try:
            example_id = int(example_id)
        except (ValueError, TypeError):
            pass
    iterations = int(iterations) if iterations else 5
    wage_spend = float(wage_spend) if wage_spend is not None else 1.0
    surplus_spend = float(surplus_spend) if surplus_spend is not None else 1.0
    tax_spend = float(tax_spend) if tax_spend is not None else 1.0
    initial_savings = float(initial_savings_val) if initial_savings_val is not None else 0.0
    enforce_gate = bool(savings_gate and 'enforce' in savings_gate)

    empty_matrix_store = {}
    examples_config = get_examples_config()
    empty_fig = go.Figure()

    if not example_id or example_id not in examples_config:
        return (
            "-", {}, "-", {}, "-", {}, "-", {}, "-", {},
            None, {'display': 'none'},
            "$0.00", "$0.00", "$0.00", "$0.00", "—", {'color': '#64748b'},
            empty_fig, empty_fig, empty_fig, empty_fig,
            [], [], [], [], empty_matrix_store
        )

    config = examples_config[example_id]
    params = config['params'].copy()

    # 1. Pipeline Gatekeeper: Get Pre-Calibrated Model
    from pipeline.setup_data import get_calibrated_model
    from core.io_matrix import IOModel
    from simulators.tech_change import TechnologicalChange
    from simulators.tech_change_loader import build_tech_change_from_spec
    from pipeline.db.repositories.good_repo import GoodsDatabase

    try:
        model, isic_map = get_calibrated_model(demoDB=False, loggingLevel=logging.WARNING)
    except Exception as e:
        print(f"Calibration error: {e}")
        err_fig = go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
        return (
            "Error", {'color': 'red'}, "-", {}, "-", {}, "-", {}, "-", {},
            html.Span(f"Error: {e}"), {'display': 'block', 'backgroundColor': 'rgba(239, 68, 68, 0.14)', 'padding': '10px'},
            "$0.00", "$0.00", "$0.00", "$0.00", "Error", {'color': '#ef4444'},
            err_fig, empty_fig, empty_fig, empty_fig,
            [], [], [], [], empty_matrix_store
        )

    # Sector name lookup for display
    sector_name_map = {}
    try:
        from presentation.layout import db_url
        gdb = GoodsDatabase(database_url=db_url)
        sector_name_map = {g.isic: g.name for g in gdb.get_all_goods()}
    except Exception:
        pass

    # 2. Setup Baseline Demand (Payload)
    if 'final_demand' in params and params['final_demand']:
        fd_raw_arr = np.array(params['final_demand'], dtype=float)
        if len(fd_raw_arr) == model.n:
            base_demand = fd_raw_arr
        elif len(fd_raw_arr) > model.n:
            base_demand = fd_raw_arr[:model.n]
        else:
            base_demand = np.pad(fd_raw_arr, (0, model.n - len(fd_raw_arr)), mode='constant', constant_values=100.0)
    else:
        # Fallback to uniform demand for testing
        base_demand = np.full(model.n, params.get('uniform_demand', 1000.0))

    # Parse Proportions Vectors
    def parse_proportions(key):
        if key in params and params[key]:
            val = params[key]
            try:
                if isinstance(val, list):
                    arr = np.array([float(x) for x in val])
                elif isinstance(val, str):
                    arr = np.array([float(x) for x in val.split(';')])
                else:
                    arr = None
                if arr is not None:
                    if len(arr) == model.n:
                        return arr
                    elif len(arr) > model.n:
                        return arr[:model.n]
                    else:
                        return np.pad(arr, (0, model.n - len(arr)), mode='constant', constant_values=0.0)
            except Exception:
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

    # 4. Simulate Baseline Tax Policy (Before)
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
        government_proportions=g_props,
        tech_change=None,
    )

    # 4b. Setup Tech Change & Savings Ledger
    ledger = SavingsLedger(initial_balance=initial_savings)

    # Collect investments
    raw_investments = []
    if ui_tech_changes:
        raw_investments = [dict(c) for c in ui_tech_changes]
    elif 'tech_change_config' in params and params['tech_change_config'].get('spec'):
        scen_spec = dict(params['tech_change_config']['spec'])
        scen_changes = list(scen_spec.get('changes', []))
        scen_cap_cost = 0.0
        scen_cap_dur = 1
        scen_reqs = None
        has_scen_cap = False
        for ch in scen_changes:
            if ch.get('method') == 'set_capital_requirements':
                p = ch.get('params', {})
                reqs = p.get('requirements', {})
                scen_reqs = reqs
                scen_cap_cost += sum(reqs.values()) if reqs else float(p.get('total_cost', 0.0))
                scen_cap_dur = max(scen_cap_dur, int(p.get('investment_duration', 1)))
                has_scen_cap = True
        for ch in scen_changes:
            if ch.get('method') != 'set_capital_requirements':
                entry = {
                    'method': ch.get('method'),
                    'params': ch.get('params', {}),
                    'capital_cost': ch.get('capital_cost', scen_cap_cost if has_scen_cap else 0.0),
                    'duration': ch.get('duration', scen_cap_dur if has_scen_cap else 1),
                }
                if scen_reqs:
                    entry['requirements'] = scen_reqs
                raw_investments.append(entry)

    # Backward compatibility with kwargs (e.g. legacy tests passing tc_has_cap_req / tc_cap_amount)
    legacy_cap = bool(kwargs.get('tc_has_cap_req') and 'has_req' in kwargs.get('tc_has_cap_req'))
    legacy_amount = kwargs.get('tc_cap_amount')
    legacy_duration = kwargs.get('tc_cap_duration')
    if legacy_cap and legacy_amount is not None and float(legacy_amount) > 0:
        if raw_investments:
            if raw_investments[0].get('capital_cost', 0.0) == 0.0:
                raw_investments[0]['capital_cost'] = float(legacy_amount)
                raw_investments[0]['duration'] = max(1, int(legacy_duration or 1))
        else:
            raw_investments.append({
                'method': 'noop',
                'params': {},
                'capital_cost': float(legacy_amount),
                'duration': max(1, int(legacy_duration or 1)),
            })

    # Sequential Reserve-on-Commit across each investment
    active_investments = []
    cap_req_rows = []
    total_reserved = 0.0
    num_financed = 0
    num_blocked = 0

    for idx, inv in enumerate(raw_investments):
        method = inv.get('method')
        p = inv.get('params', {})
        cost = float(inv.get('capital_cost', 0.0))
        dur = max(1, int(inv.get('duration', 1)))
        reqs = inv.get('requirements')

        # Build TechnologicalChange object for this investment
        t_obj = None
        if method and method != 'noop':
            try:
                sub_spec = {
                    "name": f"Investment #{idx+1}",
                    "description": f"Method {method}",
                    "changes": [{'method': method, 'params': p}],
                }
                t_obj = build_tech_change_from_spec(sub_spec, isic_map, ledger=None)
            except Exception as e:
                logger.error(f"Error building tech change for inv {idx}: {e}")

        # Human readable name
        inv_label = f"Investment #{idx+1}"
        if method == 'add_sector_change':
            sec = p.get('sector_idx', '?')
            s_name = sector_name_map.get(sec, sec)
            inv_label = f"Inputs of {s_name} ({p.get('change_type')} {p.get('value')})"
        elif method == 'add_input_change':
            sec = p.get('input_sector_idx', '?')
            s_name = sector_name_map.get(sec, sec)
            inv_label = f"Usage of {s_name} ({p.get('change_type')} {p.get('value')})"
        elif method == 'add_coefficient_change':
            inv_label = f"Coeff [{p.get('input_sector_idx', '?')} → {p.get('sector_idx', '?')}]"
        elif method == 'add_production_input_change':
            in_isic = p.get('input_isic', '?')
            in_name = sector_name_map.get(in_isic, in_isic)
            inv_label = f"Recipe #{p.get('production_id', '?')} {in_name} ({p.get('change_type')} {p.get('value')})"
        elif method == 'add_curve_tier_change':
            sec = p.get('isic', '?')
            s_name = sector_name_map.get(sec, sec)
            t_idx = p.get('tier_index', 0)
            inv_label = f"Curve {s_name} Tier {int(t_idx)+1} {p.get('field', '')} ({p.get('change_type', '')} {p.get('value', '')})"
        elif method == 'noop':
            inv_label = "Capital Project"

        # Check affordability
        if cost > 0:
            if enforce_gate:
                if ledger.withdraw(cost):
                    status = "Financed & Active"
                    total_reserved += cost
                    num_financed += 1
                    active_inv_entry = {
                        'name': inv_label,
                        'tech_change': t_obj,
                        'capital_cost': cost,
                        'duration': dur,
                        'status': status,
                    }
                    if reqs:
                        active_inv_entry['requirements'] = reqs
                    active_investments.append(active_inv_entry)
                else:
                    status = "Blocked (Insufficient Funds)"
                    num_blocked += 1
            else:
                status = "Financed (Gate Bypassed)"
                total_reserved += cost
                num_financed += 1
                active_inv_entry = {
                    'name': inv_label,
                    'tech_change': t_obj,
                    'capital_cost': cost,
                    'duration': dur,
                    'status': status,
                }
                if reqs:
                    active_inv_entry['requirements'] = reqs
                active_investments.append(active_inv_entry)

            if reqs and isinstance(reqs, dict):
                for sec, amt in reqs.items():
                    s_name = sector_name_map.get(sec, f"Sector {sec}")
                    cap_req_rows.append({
                        'Sector': f"{inv_label} → {s_name} ({sec})",
                        'Total_Amount': round(float(amt), 2),
                        'Per_Period_Demand': round(float(amt) / dur, 2),
                        'Duration': dur,
                        'Status': status,
                    })
            else:
                cap_req_rows.append({
                    'Sector': f"{inv_label} (Domestic Sectors)",
                    'Total_Amount': round(cost, 2),
                    'Per_Period_Demand': round(cost / dur, 2),
                    'Duration': dur,
                    'Status': status,
                })
        else:
            # Free investment ($0 cost)
            status = "Active (Free)"
            active_investments.append({
                'name': inv_label,
                'tech_change': t_obj,
                'capital_cost': 0.0,
                'duration': dur,
                'status': status,
            })

    # Compute overall financing status and banner
    if total_reserved > 0 or num_blocked > 0:
        if num_blocked == 0:
            investment_status = "Financed & Active"
            status_color = '#22c55e'
            gate_note = " (Savings Gate Enforced)" if enforce_gate else " (Gate Bypassed)"
            banner_children = [
                html.Span("✅ INVESTMENT FINANCED: ", style={'fontWeight': '700', 'color': '#22c55e'}),
                html.Span(f"All {num_financed} investment(s) approved and funded (${total_reserved:,.2f} reserved){gate_note}.")
            ]
            banner_style = {
                'display': 'block',
                'backgroundColor': 'rgba(34, 197, 94, 0.12)',
                'border': '1px solid #22c55e',
                'borderRadius': '8px',
                'padding': '11px 16px',
                'color': '#e2e8f0',
                'marginBottom': '14px',
            }
        elif num_financed > 0:
            investment_status = "Partially Financed"
            status_color = '#f59e0b'
            banner_children = [
                html.Span("⚠️ PARTIALLY FINANCED: ", style={'fontWeight': '700', 'color': '#f59e0b'}),
                html.Span(f"{num_financed} investment(s) funded (${total_reserved:,.2f}), but {num_blocked} investment(s) blocked due to insufficient savings.")
            ]
            banner_style = {
                'display': 'block',
                'backgroundColor': 'rgba(245, 158, 11, 0.14)',
                'border': '1px solid #f59e0b',
                'borderRadius': '8px',
                'padding': '11px 16px',
                'color': '#e2e8f0',
                'marginBottom': '14px',
            }
        else:
            investment_status = "Blocked (Insufficient Funds)"
            status_color = '#ef4444'
            banner_children = [
                html.Span("⚠️ ALL INVESTMENTS BLOCKED: ", style={'fontWeight': '700', 'color': '#ef4444'}),
                html.Span(f"Initial savings of ${initial_savings:,.2f} was insufficient for pending investment requirements. All technology changes were blocked; baseline technology was retained.")
            ]
            banner_style = {
                'display': 'block',
                'backgroundColor': 'rgba(239, 68, 68, 0.14)',
                'border': '1px solid #ef4444',
                'borderRadius': '8px',
                'padding': '11px 16px',
                'color': '#e2e8f0',
                'marginBottom': '14px',
            }
    else:
        investment_status = "No Capital Investment"
        status_color = '#64748b'
        banner_children = None
        banner_style = {'display': 'none'}

    reserved_amount = total_reserved

    # Determine post-simulation technology matrix for matrix explorer
    A_after_mat = model.A.copy()
    VA_after_mat = model.VA_coeffs.copy()
    for inv in active_investments:
        cost = float(inv.get('capital_cost', 0.0))
        dur = int(inv.get('duration', 0))
        if cost == 0.0 or iterations >= dur:
            t_obj = inv.get('tech_change')
            if t_obj is not None:
                A_after_mat, VA_after_mat = t_obj.apply(A_after_mat, VA_after_mat, isic_map)
    model_after = IOModel(A=A_after_mat, VA_coeffs=VA_after_mat)

    # 4c. Simulate Tax Policy (After) — with multi-investment technological changes and savings accumulation into ledger
    history_after = run_tax_simulation(
        model=model,
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
        government_proportions=g_props,
        ledger=ledger,
        investments=active_investments,
    )

    # 5. Extract Before and After states
    first_iter = history_before[-1]
    last_iter = history_after[-1]

    X_before = first_iter["X"]
    X_after = last_iter["X"]

    d_X = X_after.sum() - X_before.sum()
    d_FD = last_iter["total_demand"] - first_iter["total_demand"]

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

    # 6. Reconstruct Savings Ledger Trajectory
    commit_bal = initial_savings - reserved_amount
    iter_labels = ['Commit (t=0)']
    balances = [commit_bal]
    period_inflows = [0.0]

    curr_bal = commit_bal
    total_wage_savings = 0.0
    total_surplus_savings = 0.0

    for idx, h in enumerate(history_after):
        w_sav = h.get("total_wage_net", 0.0) * (1.0 - wage_spend)
        s_sav = h.get("total_surplus_net", 0.0) * (1.0 - surplus_spend)
        dep = w_sav + s_sav
        curr_bal += dep
        total_wage_savings += w_sav
        total_surplus_savings += s_sav
        iter_labels.append(f"Iter {idx + 1}")
        balances.append(round(curr_bal, 2))
        period_inflows.append(round(dep, 2))

    final_balance = curr_bal
    total_inflows = total_wage_savings + total_surplus_savings

    # Stat cards and KPI formatting
    stat_initial = f"${initial_savings:,.2f}"
    stat_inflows = f"+${total_inflows:,.2f}"
    stat_reserved = f"-${reserved_amount:,.2f}" if reserved_amount > 0 else "$0.00"
    stat_balance = f"${final_balance:,.2f}"
    stat_status_style = {'color': status_color, 'fontWeight': '700', 'fontSize': '14px', 'margin': 0}

    out_Ledger = f"${final_balance:,.2f}"
    style_Ledger = {
        'color': '#22c55e' if final_balance > 0 else '#64748b',
        'margin': 0,
        'fontSize': '24px',
        'fontWeight': '700',
        'lineHeight': '1',
    }

    # 7. Build Visualizations
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
    })
    df = df.sort_values(by='Output_Before', ascending=False)
    row_data = df.to_dict('records')

    fig_waterfall = build_waterfall_chart(X_before.sum(), X_after.sum())
    fig_delta = build_delta_bar_chart(df)
    fig_trajectory = build_ledger_trajectory_chart(
        iter_labels, balances, deposits=period_inflows, reserved_amount=reserved_amount
    )
    fig_breakdown = build_savings_breakdown_chart(
        initial_savings, total_wage_savings, total_surplus_savings, reserved_amount, final_balance
    )

    demand_vector_data = [{'Sector': s, 'Demand': round(float(d), 4)} for s, d in zip(sectors, first_iter["Y"])]

    va_coeff_data = []
    for i, s in enumerate(sectors):
        va_coeff_data.append({
            'Sector': s,
            'VA_Before': round(float(model.VA_coeffs[i]), 4),
            'VA_After': round(float(model_after.VA_coeffs[i]), 4),
            'VA_Delta': round(float(model_after.VA_coeffs[i] - model.VA_coeffs[i]), 4),
        })

    def _mat_to_list(m):
        return m.tolist() if m is not None else None

    Z_before_mat = model.A * X_before[np.newaxis, :]
    Z_after_mat = A_after_mat * X_after[np.newaxis, :]

    matrix_store = {
        'sectors': sectors,
        'A_before': _mat_to_list(model.A),
        'A_after': _mat_to_list(A_after_mat),
        'L_before': _mat_to_list(model.L),
        'L_after': _mat_to_list(model_after.L),
        'Z_before': _mat_to_list(Z_before_mat),
        'Z_after': _mat_to_list(Z_after_mat),
    }

    return (
        out_X, style_X, out_VA, style_VA, out_FD, style_FD, out_Tax, style_Tax,
        out_Ledger, style_Ledger,
        banner_children, banner_style,
        stat_initial, stat_inflows, stat_reserved, stat_balance, investment_status, stat_status_style,
        fig_waterfall, fig_delta, fig_trajectory, fig_breakdown,
        row_data, demand_vector_data, va_coeff_data, cap_req_rows, matrix_store
    )



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

