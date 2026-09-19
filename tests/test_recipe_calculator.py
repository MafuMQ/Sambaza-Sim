import os
import json
import shutil
import pytest
import numpy as np

from data_builder_app import compute_recipe_impact, perform_generation, perform_import
from simulators.tech_change_loader import load_tech_change_from_csv
from simulators.tax_simulator import run_tax_simulation
from simulators.tech_change import TechnologicalChange

def test_compute_recipe_impact_math():
    """Verify exact micro-to-macro price and Leontief coefficient formulas."""
    prod = {
        'id': 1,
        'name': 'Bakery Production',
        'isic': 'A1000_000_000',
        'produce_name': 'Food',
        'production_inputs': json.dumps({'A0111_olive': 15.0, 'A2000_flour': 10.0}),
        'production_added_values': json.dumps({'wages': 3.0, 'surplus': 2.0}),
        'price': 30.0
    }

    # Substitute Olive Oil with Sunflower Oil ($8.00) and drop Olive Oil to $2.00
    res = compute_recipe_impact(
        prod=prod,
        target_isic='A0111_olive',
        action='replace_cost',
        action_val=2.0,
        sub_isic='A0112_sunflower',
        sub_val=8.0
    )

    assert res['price_before'] == 30.0
    # New price = 2.0 (olive) + 10.0 (flour) + 8.0 (sunflower) + 5.0 (VA) = 25.0
    assert res['price_after'] == 25.0
    assert res['delta_price'] == -5.0
    assert res['pct_price_change'] == pytest.approx(-16.67, 0.01)

    comp_dict = {item['input_isic']: item for item in res['comparison']}
    
    # Olive Oil: 15/30 = 0.50 -> 2/25 = 0.08
    assert comp_dict['A0111_olive']['a_before'] == 0.50
    assert comp_dict['A0111_olive']['a_after'] == 0.08
    assert comp_dict['A0111_olive']['delta_a'] == -0.42

    # Flour: 10/30 = 0.3333 -> 10/25 = 0.40 (price deflation effect!)
    assert comp_dict['A2000_flour']['a_before'] == pytest.approx(0.3333, 0.001)
    assert comp_dict['A2000_flour']['a_after'] == 0.40
    assert comp_dict['A2000_flour']['delta_a'] == pytest.approx(0.0667, 0.001)

    # Sunflower Oil: 0/30 = 0.0 -> 8/25 = 0.32
    assert comp_dict['A0112_sunflower']['a_before'] == 0.0
    assert comp_dict['A0112_sunflower']['a_after'] == 0.32
    assert comp_dict['A0112_sunflower']['delta_a'] == 0.32

    # Verify all coefficients + VA sum to 1.0 exactly
    total_coeff_after = sum(item['a_after'] for item in res['comparison']) + res['va_after']
    assert total_coeff_after == pytest.approx(1.0, 0.0001)


def test_recipe_generation_and_loader_roundtrip():
    """Verify that data_builder_app generates tech_changes.csv with recipe changes and loader parses them."""
    tmp_folder = "test_recipe_gen_tmp"
    tmp_path = os.path.join("data", tmp_folder)
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    # Load ex2 baseline as base data
    _, goods, prods, taxes, _, _, _, _, _ = perform_import('ex2')

    # Create a recipe substitution scenario
    target_prod = prods[0] # first production
    raw_inputs = json.loads(target_prod['production_inputs'])
    target_ingredient = list(raw_inputs.keys())[0]

    impact = compute_recipe_impact(
        prod=target_prod,
        target_isic=target_ingredient,
        action='multiply',
        action_val=0.5,
        sub_isic=None,
        sub_val=0.0
    )

    recipe_scenario = {
        'example_id': 1,
        'tech_change_id': 'recipe_efficiency_upgrade',
        'change_type': 'tech_change',
        'title': 'Bakery Recipe Modernization',
        'description': 'Reduced ingredient input by 50%',
        'iterations': 5,
        'investments': [{
            'method': 'add_production_input_change',
            'production_id': target_prod['id'],
            'input_isic': target_ingredient,
            'change_type': 'multiply',
            'value': 0.5,
            'substitute_isic': '',
            'substitute_val': 0.0,
            'sector_idx': target_prod['isic'],
            'comparison': impact['comparison'],
            'capital_cost': 250.0,
            'investment_duration': 2
        }]
    }

    msg, style = perform_generation(tmp_folder, goods, prods, taxes, [recipe_scenario])
    assert "successfully" in msg

    tc_csv = os.path.join(tmp_path, "tech_changes.csv")
    assert os.path.exists(tc_csv)

    # Load via tech_change_loader into in-memory SQLite
    count = load_tech_change_from_csv(csv_path=tc_csv, database_url="sqlite:///:memory:")
    assert count == 1, f"Expected 1 tech change scenario loaded, got {count}"

    # Cleanup
    shutil.rmtree(tmp_path)


def test_recipe_multi_iteration_simulation():
    """Verify run_tax_simulation handles multi-iteration phased recipe investments."""
    from core.io_matrix import IOModel
    from unittest.mock import patch

    isic_map = {'A1': 0, 'A2': 1}
    A_before = np.array([[0.1, 0.4],
                         [0.2, 0.1]])
    model = IOModel(A=A_before, VA_coeffs=np.array([0.7, 0.5]))
    
    # Say the recipe change shifts column 0 (A1) to: A[0,0]=0.05, A[1,0]=0.10
    tc = TechnologicalChange("Recipe Switch")
    tc.add_coefficient_change(sector_idx='A1', input_sector_idx='A1', change_type='set', value=0.05)
    tc.add_coefficient_change(sector_idx='A1', input_sector_idx='A2', change_type='set', value=0.10)

    investments = [{
        'name': 'Flour Milling Recipe Upgrade',
        'tech_change': tc,
        'capital_cost': 200.0,
        'duration': 2,
        'status': 'Financed & Active'
    }]

    fake_va_coeffs = {
        "minWages": np.array([0.0, 0.0]),
        "bonusWages": np.array([0.0, 0.0]),
        "wages": np.array([0.4, 0.3]),
        "surplus": np.array([0.3, 0.2]),
    }

    with patch("simulators.tax_simulator.build_va_component_matrix", return_value=fake_va_coeffs):
        history = run_tax_simulation(
            model=model,
            isic_map=isic_map,
            base_demand=np.array([100.0, 100.0]),
            iterations=5,
            investments=investments
        )

    assert len(history) == 5

    # During iterations 1 & 2 (index 0 & 1), build phase runs on baseline technology
    assert history[0]['phase'] == 'investment'
    assert np.isclose(history[0]['capital_demand_total'], 100.0)
    assert np.allclose(history[0]['capital_injected'], [50.0, 50.0])

    # At iteration 3 (index 2), capital injection finishes and new technology matrix is active
    assert history[2]['phase'] == 'operational'
    assert np.isclose(history[2]['capital_demand_total'], 0.0)
    assert np.isclose(history[2]['model'].A[0, 0], 0.05)
    assert np.isclose(history[2]['model'].A[1, 0], 0.10)


def test_compute_recipe_impact_multi_ingredients():
    """Verify simultaneous multi-ingredient modifications and substitutes on the micro level."""
    prod = {
        'id': 1,
        'name': 'Commercial Brewery',
        'isic': 'A1100_beverage',
        'produce_name': 'Beer',
        'production_inputs': json.dumps({
            'A0111_barley': 30.0,
            'A0112_hops': 10.0,
            'A3600_water': 10.0
        }),
        'production_added_values': json.dumps({'wages': 10.0, 'surplus': 10.0}),
        'price': 70.0
    }

    # Modify barley: multiply by 0.5 (30 -> 15)
    # Modify hops: add +5 (10 -> 15)
    # Introduce yeast substitute: +$5
    # Total inputs = 15 (barley) + 15 (hops) + 10 (water) + 5 (yeast) = 45
    # Total VA = 20
    # New price = 65
    res = compute_recipe_impact(
        prod=prod,
        ingredient_changes={
            'A0111_barley': {'action': 'multiply', 'val': 0.5},
            'A0112_hops': {'action': 'add', 'val': 5.0}
        },
        substitute_goods={
            'A0113_yeast': 5.0
        }
    )

    assert res['price_before'] == 70.0
    assert res['price_after'] == 65.0
    assert res['delta_price'] == -5.0

    comp_dict = {item['input_isic']: item for item in res['comparison']}
    assert comp_dict['A0111_barley']['cost_before'] == 30.0
    assert comp_dict['A0111_barley']['cost_after'] == 15.0
    assert comp_dict['A0111_barley']['is_modified'] is True

    assert comp_dict['A0112_hops']['cost_before'] == 10.0
    assert comp_dict['A0112_hops']['cost_after'] == 15.0
    assert comp_dict['A0112_hops']['is_modified'] is True

    assert comp_dict['A3600_water']['cost_before'] == 10.0
    assert comp_dict['A3600_water']['cost_after'] == 10.0
    assert comp_dict['A3600_water']['is_modified'] is False

    assert comp_dict['A0113_yeast']['cost_before'] == 0.0
    assert comp_dict['A0113_yeast']['cost_after'] == 5.0
    assert comp_dict['A0113_yeast']['is_substitute'] is True

    # Coefficients + VA must sum to exactly 1.0
    total_coeffs = sum(item['a_after'] for item in res['comparison']) + res['va_after']
    assert total_coeffs == pytest.approx(1.0, 0.0001)


def test_multi_ingredient_generation_and_loader():
    """Verify data_builder_app generates tech_changes.csv with multiple ingredient changes and tech_change_loader loads them."""
    tmp_folder = "test_multi_recipe_gen_tmp"
    tmp_path = os.path.join("data", tmp_folder)
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    _, goods, prods, taxes, _, _, _, _, _ = perform_import('ex2')
    target_prod = prods[0]
    raw_inputs = json.loads(target_prod['production_inputs'])
    ing_keys = list(raw_inputs.keys())

    # Modify first two ingredients if available
    ing_changes = {}
    if len(ing_keys) >= 2:
        ing_changes[ing_keys[0]] = {'action': 'multiply', 'val': 0.8}
        ing_changes[ing_keys[1]] = {'action': 'add', 'val': 2.0}
    else:
        ing_changes[ing_keys[0]] = {'action': 'multiply', 'val': 0.8}

    sub_goods = {goods[-1]['isic']: 5.0}

    impact = compute_recipe_impact(
        prod=target_prod,
        ingredient_changes=ing_changes,
        substitute_goods=sub_goods
    )

    multi_scenario = {
        'example_id': 1,
        'tech_change_id': 'multi_ingredient_upgrade',
        'change_type': 'tech_change',
        'title': 'Advanced Multi-Ingredient Overhaul',
        'description': 'Upgraded multiple inputs and added substitute good',
        'iterations': 5,
        'investments': [{
            'method': 'add_production_input_change',
            'production_id': target_prod['id'],
            'sector_idx': target_prod['isic'],
            'ingredient_changes': ing_changes,
            'substitutes': sub_goods,
            'comparison': impact['comparison'],
            'capital_cost': 500.0,
            'investment_duration': 3
        }]
    }

    msg, style = perform_generation(tmp_folder, goods, prods, taxes, [multi_scenario])
    assert "successfully" in msg

    tc_csv = os.path.join(tmp_path, "tech_changes.csv")
    assert os.path.exists(tc_csv)

    # Loader verification
    count = load_tech_change_from_csv(csv_path=tc_csv, database_url="sqlite:///:memory:")
    assert count == 1

    shutil.rmtree(tmp_path)
