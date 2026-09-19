import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from data_builder_app import perform_import
from simulators.tech_change_loader import load_tech_changes_from_csv, rebuild_examples_dict_from_db
from pipeline.db.repositories.tech_change_repo import TechChangeDatabase

def test_ex3_and_ex4_import_and_structure():
    """Verify that both ex3 and ex4 import cleanly into data_builder_app and satisfy schema."""
    for folder in ['ex3', 'ex4']:
        success, goods, prods, taxes, tech, _, _, imported_folder, status_msg = perform_import(folder)
        assert success is False  # Modal closed
        assert imported_folder == folder
        assert len(goods) == 6
        assert len(prods) == 25
        # Tax policies have zero tax hikes/cuts (neutral baseline or ignored)
        assert len(taxes) == 0

        # Verify goods contain real sectors and Foreign Exchange
        isics = [g['isic'] for g in goods]
        assert 'A9999_999_999' in isics
        assert 'A0111_100_01' in isics
        assert 'A3510_200_02' in isics
        assert 'A2810_300_03' in isics
        assert 'A4923_400_04' in isics
        assert 'A6910_500_05' in isics

        # Verify all productions satisfy price = inputs + value added
        for p in prods:
            inputs = json.loads(p['production_inputs']) if isinstance(p['production_inputs'], str) else p['production_inputs']
            vas = json.loads(p['production_added_values']) if isinstance(p['production_added_values'], str) else p['production_added_values']
            tot = sum(inputs.values()) + sum(vas.values())
            assert abs(tot - float(p['price'])) < 1e-4

def test_ex3_national_and_firm_tech_changes():
    """Verify that ex3 contains 10 technological changes (5 National Scale + 5 Firm Scale) without tax changes."""
    temp_db = "sqlite:///data_test_verify_ex3.db"
    count = load_tech_changes_from_csv(
        tax_policy_csv='data/ex3/tax_policies.csv',
        tech_change_csv='data/ex3/tech_changes.csv',
        database_url=temp_db,
        clear_existing=True
    )
    assert count == 10
    examples = rebuild_examples_dict_from_db(temp_db)
    assert len(examples) == 10

    # Verify Examples 1-5 are National Scale
    assert "National Clean Power Grid Efficiency" in examples[1]['title']
    assert "National Intermodal Freight" in examples[2]['title']
    assert "Industrial Automation & IT Substitution" in examples[3]['title']
    assert "Strategic Domestic Machinery Capacity" in examples[4]['title']
    assert "National Green Industry & Energy Transition" in examples[5]['title']

    # Verify Examples 6-10 are Firm Scale
    assert "Farm-Level Precision Agritech Recipe" in examples[6]['title']
    assert "Firm Recipe: Domestic Bio-Composite" in examples[7]['title']
    assert "Plant Automation: Assembly Line Lean" in examples[8]['title']
    assert "Firm Scale: Multimodal Freight Terminal" in examples[9]['title']
    assert "Firm Scale: Gas Turbine Heat-Recovery" in examples[10]['title']

    # Dispose engine to release file lock on Windows before cleanup
    tcdb = TechChangeDatabase(database_url=temp_db)
    tcdb.engine.dispose()
    try:
        Path("data_test_verify_ex3.db").unlink(missing_ok=True)
    except PermissionError:
        pass

def test_ex4_dedicated_firm_scale():
    """Verify that ex4 contains the dedicated 5 Firm Scale technological change scenarios."""
    temp_db = "sqlite:///data_test_verify_ex4.db"
    count = load_tech_changes_from_csv(
        tax_policy_csv='data/ex4/tax_policies.csv',
        tech_change_csv='data/ex4/tech_changes.csv',
        database_url=temp_db,
        clear_existing=True
    )
    assert count == 5
    examples = rebuild_examples_dict_from_db(temp_db)
    assert len(examples) == 5

    # Dispose engine to release file lock on Windows before cleanup
    tcdb = TechChangeDatabase(database_url=temp_db)
    tcdb.engine.dispose()
    try:
        Path("data_test_verify_ex4.db").unlink(missing_ok=True)
    except PermissionError:
        pass
