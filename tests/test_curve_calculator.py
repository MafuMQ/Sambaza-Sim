import pytest
import numpy as np
import json
import tempfile
import os
import pandas as pd
from unittest.mock import patch

from data_builder_app import (
    compute_production_merit_ranks,
    compute_curve_impact,
    parse_tech_changes_df
)
from simulators.tech_change_loader import load_tech_change_from_csv


def test_compute_production_merit_ranks():
    """Verify that multiple productions for a good are ranked correctly by unit cost."""
    goods = [
        {'id_number': 100, 'name': 'Electricity', 'isic': 'D3510_000_000'},
        {'id_number': 101, 'name': 'Wheat', 'isic': 'A0111_000_000'}
    ]

    productions = [
        # Electricity prods: Prod 1 is expensive coal ($25), Prod 2 is cheap solar ($10)
        {
            'id': 1,
            'name': 'Coal Power',
            'produce': 100,
            'isic': 'D3510_000_000',
            'production_quantity': 500,
            'production_inputs': json.dumps({'A0111_000_000': 15.0}),
            'production_added_values': json.dumps({'wages': 7.0, 'surplus': 3.0}) # Total = $25
        },
        {
            'id': 2,
            'name': 'Solar Farm',
            'produce': 100,
            'isic': 'D3510_000_000',
            'production_quantity': 200,
            'production_inputs': json.dumps({}),
            'production_added_values': json.dumps({'wages': 6.0, 'surplus': 4.0}) # Total = $10
        },
        # Wheat prod
        {
            'id': 3,
            'name': 'Wheat Farm',
            'produce': 101,
            'isic': 'A0111_000_000',
            'production_quantity': 1000,
            'production_inputs': json.dumps({}),
            'production_added_values': json.dumps({'wages': 10.0, 'surplus': 5.0}) # Total = $15
        }
    ]

    annotated = compute_production_merit_ranks(productions, goods)
    assert len(annotated) == 3

    # Check Solar Farm (id 2) became Rank 1 (Active Baseline) for Electricity
    solar = next(p for p in annotated if p['id'] == 2)
    assert solar['tier_rank'] == 1
    assert solar['is_baseline'] is True
    assert solar['merit_status'] == "Tier 1"
    assert np.isclose(solar['price'], 10.0)

    # Check Coal Power (id 1) became Rank 2 (Reserve)
    coal = next(p for p in annotated if p['id'] == 1)
    assert coal['tier_rank'] == 2
    assert coal['is_baseline'] is False
    assert coal['merit_status'] == "Tier 2"
    assert np.isclose(coal['price'], 25.0)

    # Wheat Farm (id 3) is alone, so it's Rank 1
    wheat = next(p for p in annotated if p['id'] == 3)
    assert wheat['tier_rank'] == 1
    assert wheat['is_baseline'] is True


def test_compute_curve_impact_dispatch_math():
    """
    Verify that expanding Tier 1 capacity displaces expensive Tier 2 output
    and lowers the effective market price based on Leontief circular demand.
    """
    goods = [
        {'id_number': 100, 'name': 'Electricity', 'isic': 'D3510'},
        {'id_number': 101, 'name': 'Manufacturing', 'isic': 'C1000'}
    ]

    # Electricity has two tiers:
    # Tier 1 (Solar): Price $10, Cap 100
    # Tier 2 (Peaker): Price $30, Cap 500
    # Manufacturing uses Electricity: $20 of Electricity per $100 of output (A[0,1] = 0.20)
    productions = [
        {
            'id': 1,
            'name': 'Peaker Gas',
            'produce': 100,
            'isic': 'D3510',
            'production_quantity': 500,
            'production_inputs': json.dumps({}),
            'production_added_values': json.dumps({'wages': 20.0, 'surplus': 10.0}), # $30
        },
        {
            'id': 2,
            'name': 'Solar Clean',
            'produce': 100,
            'isic': 'D3510',
            'production_quantity': 100,
            'production_inputs': json.dumps({}),
            'production_added_values': json.dumps({'wages': 6.0, 'surplus': 4.0}), # $10
        },
        {
            'id': 3,
            'name': 'Factory Mfg',
            'produce': 101,
            'isic': 'C1000',
            'production_quantity': 1000,
            'production_inputs': json.dumps({'D3510': 20.0}),
            'production_added_values': json.dumps({'wages': 50.0, 'surplus': 30.0}), # $100
        }
    ]

    # Final demand Y = [100.0, 100.0]
    # Sector 0 Gross Demand X0 = Y0 + A[0,1]*X1 = 100 + 0.20 * 100 = 120.0
    impact = compute_curve_impact(
        goods=goods,
        productions=productions,
        target_isic='D3510',
        tier_idx=0, # Tier 1 (Solar)
        field='cap',
        action='set',
        action_val=200.0, # Expand Solar capacity from 100 -> 200
        tax_data=[{'final_demand': '100.0;100.0'}]
    )

    assert impact is not None
    assert np.isclose(impact['sector_output_demand'], 120.0)

    # Before expansion:
    # Demand = 120.0
    # Solar (cap=100) dispatches 100.0 (100% util)
    # Peaker (cap=500) dispatches 20.0 (4% util)
    # P_eff_before = (100*10 + 20*30) / 120 = (1000 + 600) / 120 = $13.33
    disp_b = impact['disp_before']
    assert np.isclose(disp_b[0]['dispatched'], 100.0)
    assert np.isclose(disp_b[1]['dispatched'], 20.0)
    assert np.isclose(impact['p_eff_before'], 13.33, atol=0.02)

    # After expansion (Solar cap = 200):
    # Demand = 120.0
    # Solar (cap=200) dispatches all 120.0 (60% util)
    # Peaker dispatches 0.0 (0% util - completely displaced!)
    # P_eff_after = $10.00
    disp_a = impact['disp_after']
    assert np.isclose(disp_a[0]['dispatched'], 120.0)
    assert np.isclose(disp_a[1]['dispatched'], 0.0)
    assert np.isclose(impact['p_eff_after'], 10.00)
    assert impact['delta_peff'] < 0


def test_curve_generation_and_loader_roundtrip():
    """Verify that add_curve_tier_change generates correct CSV rows and loads via tech_change_loader."""
    scenario = {
        'example_id': 1,
        'tech_change_id': 'clean_energy_expansion',
        'title': 'Solar Capacity Expansion',
        'description': 'Expand solar capacity to displace peaker gas plants',
        'iterations': 5,
        'investments': [{
            'method': 'add_curve_tier_change',
            'isic': 'D3510',
            'sector_idx': 'D3510',
            'tier_index': 0,
            'field': 'cap',
            'change_type': 'set',
            'value': 250.0,
            'capital_cost': 150.0,
            'investment_duration': 2
        }]
    }

    # Verify CSV schema row building
    tc_rows = []
    seq = 1
    for inv in scenario['investments']:
        tc_rows.append({
            'example_id': scenario['example_id'],
            'tech_change_id': scenario['tech_change_id'],
            'change_type': 'tech_change',
            'title': scenario['title'],
            'description': scenario['description'],
            'final_demand': '',
            'income_tax_rate_before': '',
            'income_tax_rate_after': '',
            'corporate_tax_rate_before': '',
            'corporate_tax_rate_after': '',
            'income_tax_applies_to': '',
            'iterations': scenario['iterations'],
            'wage_proportions': '',
            'surplus_proportions': '',
            'government_proportions': '',
            'use_multi_level': True,
            'sequence_number': seq,
            'method': inv['method'],
            'sector_idx': inv['sector_idx'],
            'input_sector_idx': '',
            'production_id': '',
            'isic': inv['isic'],
            'tier_index': inv['tier_index'],
            'field': inv['field'],
            'change_type_param': inv['change_type'],
            'value': inv['value'],
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
        })
        seq += 1

        # Capital requirements row
        tc_rows.append({
            'example_id': scenario['example_id'],
            'tech_change_id': scenario['tech_change_id'],
            'change_type': 'tech_change',
            'title': scenario['title'],
            'description': scenario['description'],
            'final_demand': '',
            'income_tax_rate_before': '',
            'income_tax_rate_after': '',
            'corporate_tax_rate_before': '',
            'corporate_tax_rate_after': '',
            'income_tax_applies_to': '',
            'iterations': scenario['iterations'],
            'wage_proportions': '',
            'surplus_proportions': '',
            'government_proportions': '',
            'use_multi_level': True,
            'sequence_number': seq,
            'method': 'set_capital_requirements',
            'sector_idx': '',
            'input_sector_idx': '',
            'production_id': '',
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
            'requirements': json.dumps({"_total": inv['capital_cost']}),
            'investment_duration': inv['investment_duration']
        })

    df = pd.DataFrame(tc_rows)

    # Test round-trip with parse_tech_changes_df
    parsed = parse_tech_changes_df(df)
    assert len(parsed) == 1
    sc = parsed[0]
    assert sc['tech_change_id'] == 'clean_energy_expansion'
    assert len(sc['investments']) == 1
    p_inv = sc['investments'][0]
    assert p_inv['method'] == 'add_curve_tier_change'
    assert p_inv['isic'] == 'D3510'
    assert p_inv['tier_index'] == 0
    assert p_inv['field'] == 'cap'
    assert p_inv['capital_cost'] == 150.0
    assert p_inv['investment_duration'] == 2

    # Test database loader roundtrip with CSV and in-memory SQLite
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "tech_changes.csv")
        df.to_csv(csv_path, index=False)
        count = load_tech_change_from_csv(csv_path=csv_path, database_url="sqlite:///:memory:", clear_existing=True)
        assert count == 1


def test_compute_curve_impact_dual_simultaneous_modification():
    """Verify that both capacity and price can be modified simultaneously in one scenario."""
    goods = [
        {'id_number': 100, 'name': 'Electricity', 'isic': 'D3510'},
        {'id_number': 101, 'name': 'Manufacturing', 'isic': 'C1000'}
    ]

    productions = [
        {
            'id': 1,
            'name': 'Peaker Gas',
            'produce': 100,
            'isic': 'D3510',
            'production_quantity': 500,
            'production_inputs': json.dumps({}),
            'production_added_values': json.dumps({'wages': 20.0, 'surplus': 10.0}), # $30
        },
        {
            'id': 2,
            'name': 'Solar Clean',
            'produce': 100,
            'isic': 'D3510',
            'production_quantity': 100,
            'production_inputs': json.dumps({}),
            'production_added_values': json.dumps({'wages': 6.0, 'surplus': 4.0}), # $10
        },
        {
            'id': 3,
            'name': 'Factory Mfg',
            'produce': 101,
            'isic': 'C1000',
            'production_quantity': 1000,
            'production_inputs': json.dumps({'D3510': 20.0}),
            'production_added_values': json.dumps({'wages': 50.0, 'surplus': 30.0}), # $100
        }
    ]

    # Simultaneous change: Double Solar capacity (100 -> 200) AND reduce price ($10 -> $8)
    impact = compute_curve_impact(
        goods=goods,
        productions=productions,
        target_isic='D3510',
        tier_idx=0, # Solar
        cap_action='multiply',
        cap_val=2.0,
        price_action='add',
        price_val=-2.0,
        tax_data=[{'final_demand': '100.0;100.0'}]
    )

    assert impact is not None
    assert np.isclose(impact['sector_output_demand'], 120.0)

    # Before: Solar cap=100 price=$10, Peaker cap=500 price=$30. P_eff = $13.33
    assert np.isclose(impact['p_eff_before'], 13.33, atol=0.02)

    # After: Solar cap=200 price=$8, Solar supplies all 120.0 units. P_eff = $8.00
    disp_a = impact['disp_after']
    assert np.isclose(disp_a[0]['price'], 8.0)
    assert np.isclose(disp_a[0]['cap'], 200.0)
    assert np.isclose(disp_a[0]['dispatched'], 120.0)
    assert np.isclose(disp_a[1]['dispatched'], 0.0)
    assert np.isclose(impact['p_eff_after'], 8.00)
    assert np.isclose(impact['delta_peff'], -5.33, atol=0.02)

    # Unchanged test: If both actions are 'unchanged', delta_peff is 0
    impact_unchanged = compute_curve_impact(
        goods=goods,
        productions=productions,
        target_isic='D3510',
        tier_idx=0,
        cap_action='unchanged',
        price_action='unchanged',
        tax_data=[{'final_demand': '100.0;100.0'}]
    )
    assert np.isclose(impact_unchanged['delta_peff'], 0.0)
    assert np.isclose(impact_unchanged['p_eff_before'], impact_unchanged['p_eff_after'])


def test_curve_generation_dual_action_emission():
    """Verify that dual modifications emit both cap and price rows and load cleanly into SQLite."""
    scenario = {
        'example_id': 1,
        'tech_change_id': 'clean_energy_supercharge',
        'title': 'Solar Supercharge',
        'description': 'Expand solar capacity and lower tariff',
        'iterations': 5,
        'investments': [{
            'method': 'add_curve_tier_change',
            'isic': 'D3510',
            'sector_idx': 'D3510',
            'tier_index': 0,
            'cap_action': 'set',
            'cap_val': 300.0,
            'price_action': 'multiply',
            'price_val': 0.75,
            'capital_cost': 500.0,
            'investment_duration': 3
        }]
    }

    tc_rows = []
    seq = 1
    for inv in scenario['investments']:
        row_base = {
            'example_id': scenario['example_id'],
            'tech_change_id': scenario['tech_change_id'],
            'change_type': 'tech_change',
            'title': scenario['title'],
            'description': scenario['description'],
            'final_demand': '',
            'income_tax_rate_before': '',
            'income_tax_rate_after': '',
            'corporate_tax_rate_before': '',
            'corporate_tax_rate_after': '',
            'income_tax_applies_to': '',
            'iterations': scenario['iterations'],
            'wage_proportions': '',
            'surplus_proportions': '',
            'government_proportions': '',
            'use_multi_level': True,
            'sequence_number': seq,
            'method': 'add_curve_tier_change',
            'sector_idx': inv['sector_idx'],
            'input_sector_idx': '',
            'production_id': '',
            'isic': inv['isic'],
            'tier_index': inv['tier_index'],
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

        # Cap row
        if inv.get('cap_action') != 'unchanged':
            row_cap = dict(row_base)
            row_cap['sequence_number'] = seq
            row_cap['field'] = 'cap'
            row_cap['change_type_param'] = inv['cap_action']
            row_cap['value'] = inv['cap_val']
            tc_rows.append(row_cap)
            seq += 1

        # Price row
        if inv.get('price_action') != 'unchanged':
            row_price = dict(row_base)
            row_price['sequence_number'] = seq
            row_price['field'] = 'price'
            row_price['change_type_param'] = inv['price_action']
            row_price['value'] = inv['price_val']
            tc_rows.append(row_price)
            seq += 1

        # Capital requirements row
        tc_rows.append({
            'example_id': scenario['example_id'],
            'tech_change_id': scenario['tech_change_id'],
            'change_type': 'tech_change',
            'title': scenario['title'],
            'description': scenario['description'],
            'final_demand': '',
            'income_tax_rate_before': '',
            'income_tax_rate_after': '',
            'corporate_tax_rate_before': '',
            'corporate_tax_rate_after': '',
            'income_tax_applies_to': '',
            'iterations': scenario['iterations'],
            'wage_proportions': '',
            'surplus_proportions': '',
            'government_proportions': '',
            'use_multi_level': True,
            'sequence_number': seq,
            'method': 'set_capital_requirements',
            'sector_idx': '',
            'input_sector_idx': '',
            'production_id': '',
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
            'requirements': json.dumps({"_total": inv['capital_cost']}),
            'investment_duration': inv['investment_duration']
        })

    assert len(tc_rows) == 3 # 1 cap row + 1 price row + 1 capital req row
    df = pd.DataFrame(tc_rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "tech_changes.csv")
        df.to_csv(csv_path, index=False)
        count = load_tech_change_from_csv(csv_path=csv_path, database_url="sqlite:///:memory:", clear_existing=True)
        assert count == 1


def test_foreign_exchange_generation_validation():
    """Verify that datasets containing Foreign Exchange (like ex2) pass generation validation without requiring a domestic production."""
    from data_builder_app import generate_files_check

    goods = [
        {'id': 1, 'name': 'Foreign Exchange', 'id_number': 9999, 'isic': 'A9999_999_999'},
        {'id': 2, 'name': 'Rate Good', 'id_number': 2778, 'isic': 'A2327_978_13'}
    ]
    productions = [
        {
            'id': 1,
            'name': 'IMPORT',
            'id_number': 92778,
            'isic': 'A2327_978_13',
            'produce': 2778,
            'production_inputs': json.dumps({'A9999_999_999': 863}),
            'price': 912
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        folder = os.path.basename(tmpdir)
        displayed, msg, style = generate_files_check(
            n=1,
            folder_name=folder,
            goods=goods,
            productions=productions,
            tax_policies=[],
            tech_changes=[]
        )
        assert not (isinstance(msg, str) and "Validation Error" in msg)


