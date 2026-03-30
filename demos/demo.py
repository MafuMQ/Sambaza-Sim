import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
from pipeline.setup_data import setup_data
from core.simulation import run_simulation
from core.tech_change import TechnologicalChange
from core.io_matrix import build_io_matrix

loggingLevel = logging.WARNING
logging.basicConfig(level=loggingLevel)
logger = logging.getLogger(__name__)

"""
Unified Demo Script for Input-Output Economic Modeling
=======================================================

This module provides a comprehensive demonstration of Input-Output economic modeling,
including:

1. Tax Policy Analysis (Examples 1-6):
   - Circular flow dynamics
   - Income and corporate tax impacts
   - Demand shocks and spillover effects
   - Differentiated C/I/G distributions

2. Technological Change Analysis (Examples 7-15):
   - Energy efficiency improvements
   - Productivity gains
   - Automation and capital-labor substitution
   - Material efficiency and green transitions
   - Multi-level production changes
   - Combined technology + tax policy scenarios

Key Features:
- Examples loaded from database (CSV → SQLite → Runtime)
- Value Added (VA) = Final Demand (FD) identity maintained
- Circular flow: VA from period t becomes C+I+G in period t+1
- Multiple demand specification methods
- Support for both Leontief inverse and supply curve solvers
- Comprehensive output displays with iteration tracking
"""

def load_examples_from_database(database_url: str = "sqlite:///data.db"):
    """Load examples from database instead of hardcoded Python dicts."""
    from pipeline.load_tech_changes import rebuild_examples_dict_from_db
    return rebuild_examples_dict_from_db(database_url=database_url)


def run_demo(example_id, examples_config=None, database_url: str = "sqlite:///data.db"):
    """
    Run a specific demo example.
    
    Parameters:
    -----------
    example_id : int or str
        The example number (1-15) to run:
        - 1-6: Tax policy examples
        - 7-12: Tech change examples (matrix level)
        - 13-15: Tech change examples (multi-level and combined)
    examples_config : dict, optional
        Dictionary containing all example configurations.
        If None, will load from database.
    database_url : str
        SQLite database URL for loading examples
    """
    # Load examples from database if not provided
    if examples_config is None:
        examples_config = load_examples_from_database(database_url)
    # Convert to int if string number provided
    if isinstance(example_id, str) and example_id.isdigit():
        example_id = int(example_id)
    
    # Get example config
    if example_id not in examples_config:
        available = ', '.join(map(str, examples_config.keys()))
        raise ValueError(f"Example {example_id} not found. Available examples: {available}")
    
    config = examples_config[example_id]
    
    # Print header
    print("\n" + "="*100)
    print(f"EXAMPLE {example_id}: {config['title']}")
    print("="*100)
    for line in config['description']:
        print(line)
    print()
    
    # Check if this is a technological change comparison
    if config['params'].get('is_tech_comparison', False):
        run_technological_change_comparison(config)
    else:
        # Run tax policy simulation with configured parameters
        params = config['params'].copy()
        params.setdefault('solver_type', 'supply_curves')
        params.setdefault('before_name', 'Before Tax Change')
        params.setdefault('after_name', 'After Tax Change')
        # Map consumption_distribution to demand_distribution if present
        if 'consumption_distribution' in params:
            params.setdefault('demand_distribution', params.pop('consumption_distribution'))
        run_simulation(**params)


def run_technological_change_comparison(config):
    """
    Run a technological change comparison using the same final demand.
    
    This demonstrates how to compare baseline vs. changed technology
    while holding final demand constant to isolate the pure effect
    of technological innovation.
    
    Supports both:
    - Matrix-level changes (direct A matrix modification)
    - Multi-level changes (production/curve level, requires rebuild)
    """
    params = config['params']
    final_demand = np.array(params['final_demand'], dtype=float)
    tech_config = params['tech_change_config']
    use_multi_level = params.get('use_multi_level', False)
    solver_type = params.get('solver_type', 'leontief')
    
    print("="*100)
    print("TECHNOLOGICAL CHANGE COMPARISON MODE")
    print("="*100)
    print(f"\nThis demo compares two scenarios with THE SAME final demand:")
    print(f"  1. Baseline (current technology)")
    print(f"  2. {tech_config['name']}")
    print(f"\nFinal Demand (constant): {final_demand}")
    print(f"Total FD: {final_demand.sum():.2f}")
    
    # Create the technological change object
    tech_change = tech_config['tech_change_builder']({})  # Pass empty isic_map initially
    
    if use_multi_level:
        # For multi-level changes, let run_simulation handle everything
        print(f"\n[Multi-Level Mode] Changes will be applied at: {', '.join(tech_change.get_change_levels())}")
        print(tech_change.get_summary())
        print()
        
        iterations = params.get('iterations', 1)
        # Support separate before/after tax rates for combined tech+tax examples
        income_tax_before = params.get('income_tax_rate_before', 0.0)
        income_tax_after = params.get('income_tax_rate_after', income_tax_before)
        corp_tax_before = params.get('corporate_tax_rate_before', 0.0)
        corp_tax_after = params.get('corporate_tax_rate_after', corp_tax_before)
        
        # Use tech_change parameter - simulation will rebuild as needed
        comparison = run_simulation(
            final_demand=final_demand,
            tech_change=tech_change,  # Pass the TechnologicalChange object
            before_name="Baseline Technology",
            after_name=tech_config['name'],
            iterations=iterations,
            demand_distribution=params.get('demand_distribution', 'proportional'),
            income_tax_rate_before=income_tax_before,
            income_tax_rate_after=income_tax_after,
            corporate_tax_rate_before=corp_tax_before,
            corporate_tax_rate_after=corp_tax_after,
            income_tax_applies_to=params.get('income_tax_applies_to', 'bonusWages'),
            consumption_proportions=params.get('consumption_proportions', None),
            investment_proportions=params.get('investment_proportions', None),
            government_proportions=params.get('government_proportions', None),
            consumption_rate=params.get('consumption_rate', 1.0),
            solver_type=solver_type,
        )
    else:
        # Matrix-level only: manual approach with full display
        print()
        
        # Build baseline IO matrix
        print("Building baseline Input-Output matrix...")
        A_baseline, VA_baseline, isic_map = build_io_matrix(demoDB=False, loggingLevel=loggingLevel)
        n_sectors = len(isic_map)
        print(f"[OK] Matrix built with {n_sectors} sectors")
        
        # Rebuild tech_change with actual isic_map
        tech_change = tech_config['tech_change_builder'](isic_map)
        
        # Display baseline matrix structure
        print("\nBaseline Technical Coefficient Matrix (A):")
        print(f"{'':>5}", end='')
        for j in range(n_sectors):
            print(f"{j:>8}", end='')
        print()
        for i in range(n_sectors):
            print(f"{i:>3} |", end='')
            for j in range(n_sectors):
                print(f"{A_baseline[i,j]:>8.4f}", end='')
            print()
        
        print("\nBaseline Value Added Coefficients (VA):")
        for j in range(n_sectors):
            print(f"  Sector {j}: {VA_baseline[j]:.4f}")
        print()
        
        # Create and apply technological change using the builder function
        print("="*100)
        print("Defining Technological Change")
        print("="*100)
        print(tech_change.get_summary())
        print()
        
        # Apply technological change
        print("="*100)
        print("Applying Technological Change")
        print("="*100)
        A_changed, VA_changed = tech_change.apply(A_baseline, VA_baseline, isic_map)
        print("[OK] Technological change applied")
        
        # Show what changed in the A matrix
        print("\nChanges in Technical Coefficients (ΔA):")
        delta_A = A_changed - A_baseline
        max_change = np.abs(delta_A).max()
        if max_change > 0.0001:
            print(f"{'':>5}", end='')
            for j in range(n_sectors):
                print(f"{j:>8}", end='')
            print()
            for i in range(n_sectors):
                print(f"{i:>3} |", end='')
                for j in range(n_sectors):
                    val = delta_A[i,j]
                    if abs(val) > 0.0001:
                        print(f"{val:>+8.4f}", end='')
                    else:
                        print(f"{'':>8}", end='')
                print()
        else:
            print("  (No significant changes in A matrix)")
        
        print("\nChanges in Value Added Coefficients (ΔVA):")
        delta_VA = VA_changed - VA_baseline
        for j in range(n_sectors):
            if abs(delta_VA[j]) > 0.0001:
                print(f"  Sector {j}: {delta_VA[j]:+.4f} (from {VA_baseline[j]:.4f} to {VA_changed[j]:.4f})")
        
        print("\n" + "="*100)
        print("Final Demand (SAME for both scenarios)")
        print("="*100)
        print(f"\nFinal Demand Vector: {final_demand}")
        print(f"Total Final Demand: {final_demand.sum():.2f}")
        
        # Compare scenarios
        print("\n" + "="*100)
        print("SCENARIO COMPARISON")
        print("="*100)
        iterations = params.get('iterations', 1)
        if iterations > 1:
            print(f"\nComparing outcomes with THE SAME initial final demand over {iterations} iterations...")
            print("This isolates the pure effect of technological change and shows how it compounds over time.")
        else:
            print("\nComparing outcomes with THE SAME final demand...")
            print("This isolates the pure effect of technological change.")
        
        # Run unified simulation
        # Support separate before/after tax rates for combined tech+tax examples
        income_tax_before = params.get('income_tax_rate_before', 0.0)
        income_tax_after = params.get('income_tax_rate_after', income_tax_before)
        corp_tax_before = params.get('corporate_tax_rate_before', 0.0)
        corp_tax_after = params.get('corporate_tax_rate_after', corp_tax_before)
        
        comparison = run_simulation(
            final_demand=final_demand,
            A_before=A_baseline,
            A_after=A_changed,
            VA_before=VA_baseline,
            VA_after=VA_changed,
            isic_map=isic_map,
            before_name="Baseline Technology",
            after_name=tech_config['name'],
            iterations=iterations,
            demand_distribution=params.get('demand_distribution', 'proportional'),
            income_tax_rate_before=income_tax_before,
            income_tax_rate_after=income_tax_after,
            corporate_tax_rate_before=corp_tax_before,
            corporate_tax_rate_after=corp_tax_after,
            income_tax_applies_to=params.get('income_tax_applies_to', 'bonusWages'),
            consumption_proportions=params.get('consumption_proportions', None),
            investment_proportions=params.get('investment_proportions', None),
            government_proportions=params.get('government_proportions', None),
            consumption_rate=params.get('consumption_rate', 1.0),
            solver_type=solver_type,
        )
    
    # Additional interpretation for tech change examples
    if comparison and iterations == 1:
        print("\n" + "="*100)
        print("INTERPRETATION")
        print("="*100)
        
        delta_X_total = comparison['deltas']['X'].sum()
        delta_VA = comparison['deltas']['VA']
        
        if delta_X_total < 0:
            print(f"\n✓ Technology reduces total gross output by {abs(delta_X_total):.2f}")
            print(f"  This means LESS production is needed to satisfy the SAME final demand.")
            print(f"  The economy becomes more EFFICIENT.")
        elif delta_X_total > 0:
            print(f"\n✗ Technology increases total gross output by {delta_X_total:.2f}")
            print(f"  This means MORE production is needed for the SAME final demand.")
            print(f"  This could indicate substitution toward more input-intensive methods.")
        else:
            print(f"\n- No change in total gross output")
        
        if delta_VA > 0:
            print(f"\n✓ Value added increases by {delta_VA:.2f}")
            print(f"  More value is retained in the economy (less spent on intermediate inputs).")
        elif delta_VA < 0:
            print(f"\n✗ Value added decreases by {abs(delta_VA):.2f}")
            print(f"  Less value is retained (more spent on intermediate inputs).")
        
        # Resource efficiency
        baseline_efficiency = comparison['before']['VA'] / comparison['before']['X'].sum()
        changed_efficiency = comparison['after']['VA'] / comparison['after']['X'].sum()
        
        print(f"\nResource Efficiency (VA/Gross Output):")
        print(f"  Baseline:  {baseline_efficiency:.4f}")
        print(f"  Changed:   {changed_efficiency:.4f}")
        print(f"  Change:    {(changed_efficiency - baseline_efficiency):+.4f}")
    elif comparison and iterations > 1:
        print("\n" + "="*100)
        print("INTERPRETATION")
        print("="*100)
        print(f"\nWith {iterations} iterations, we see how the technological change affects the economy")
        print(f"over multiple periods. Each iteration's value added becomes the next period's final demand,")
        print(f"showing whether the technology leads to sustained growth or efficiency improvements.")


def list_examples(database_url: str = "sqlite:///data.db"):
    """List all available examples with their titles from database."""
    from db.repositories.tech_change_repo import TechChangeDatabase
    
    db = TechChangeDatabase(database_url=database_url)
    tech_changes = db.get_all_tech_changes()
    
    if not tech_changes:
        print("\n" + "="*100)
        print("NO EXAMPLES FOUND IN DATABASE")
        print("="*100)
        print("\nPlease run setup_data() to load examples from CSV files.")
        return
    
    print("\n" + "="*100)
    print("AVAILABLE EXAMPLES FROM DATABASE")
    print("="*100)
    
    # Group by type
    tax_policy = [tc for tc in tech_changes if tc.change_type == 'tax_policy']
    tech_change = [tc for tc in tech_changes if tc.change_type == 'tech_change']
    
    if tax_policy:
        print("\nTax Policy Examples:")
        print("-" * 100)
        for tc in tax_policy:
            print(f"  {tc.example_id}. {tc.title}")
    
    if tech_change:
        print("\nTechnological Change Examples:")
        print("-" * 100)
        for tc in tech_change:
            print(f"  {tc.example_id}. {tc.title}")
            if tc.use_multi_level:
                print(f"      (Multi-Level)")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    from pathlib import Path
    
    print("="*100)
    print("INPUT-OUTPUT ECONOMIC MODELING - UNIFIED DEMO")
    print("="*100)
    
    # Check if database exists, if not setup data
    db_file = Path("data.db")
    if not db_file.exists():
        print("\nDatabase not found. Setting up data...")
        setup_data(source="data/ex2", overwrite_existing_data=True, logging_level=loggingLevel)
        print("\nData setup complete.\n")
    else:
        print("\nUsing existing database.\n")
    
    # List all available examples
    list_examples()
    
    # Run specific example(s)
    if len(sys.argv) > 1:
        try:
            example_number = int(sys.argv[1])
            print(f"\nRunning example {example_number}...\n")
            run_demo(example_number)
            print("\n" + "="*100)
            print(f'Example {example_number} completed!')
            print("="*100)
        except ValueError:
            print(f"Invalid example number: {sys.argv[1]}")
    else:
        # Default: run example 7
        example_number = 7
        print(f"\nRunning example {example_number}...\n")
        run_demo(example_number)
        print("\n" + "="*100)
        print(f'Example {example_number} completed!')
        print("="*100)
        print("\nUsage: python demo.py [example_number]")
        print("  e.g., python demo.py 11")

