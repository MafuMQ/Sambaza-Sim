"""
Unified Input-Output Simulation Engine
========================================

This module provides a single unified simulation function for Input-Output economic
modeling that supports:

- **Technological change analysis**: Compare different production technologies (A matrices)
- **Tax policy analysis**: Compare different tax regimes (before/after rates)
- **Combined analysis**: Both technological change AND tax policy changes simultaneously
- **Circular flow dynamics**: Multi-iteration GDP -> C+I+G -> FD feedback loops
- **Flexible demand distribution**: Separate proportion vectors for C, I, G
- **Multiple solver modes**: Leontief inverse or DynamicEquilibriumSolver (supply curves)

Key Identity: VA = FD = C + I + G
  where C = Wages net of income tax
        I = Surplus net of corporate tax
        G = Total tax revenue

Usage:
------
    from Input_Output_Model.demos.util.simulation import run_simulation

    # Pure tax policy comparison (A matrix built from database)
    run_simulation(
        total_demand=1000.0, proportions=[0.25, 0.20, 0.30, 0.15, 0.10, 0.0],
        income_tax_rate_before=0.10, income_tax_rate_after=0.20,
        corporate_tax_rate_before=0.25, corporate_tax_rate_after=0.25,
        iterations=5, solver_type="supply_curves"
    )

    # Pure technological change comparison
    run_simulation(
        final_demand=final_demand,
        A_before=A_baseline, A_after=A_changed,
        VA_before=VA_baseline, VA_after=VA_changed,
        isic_map=isic_map,
        before_name="Baseline", after_name="Energy Efficiency",
        iterations=3
    )

    # Combined: tech change + tax policy change
    run_simulation(
        final_demand=final_demand,
        A_before=A_baseline, A_after=A_changed,
        VA_before=VA_baseline, VA_after=VA_changed,
        isic_map=isic_map,
        income_tax_rate_before=0.10, income_tax_rate_after=0.15,
        iterations=3, solver_type="leontief"
    )
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import logging
import random
from Input_Output_Model.models.entities.SupplyCurve import SupplyCurveDatabase
from Input_Output_Model.models.entities.Production import ProductionsDatabase
from Input_Output_Model.models.table.solver import DynamicEquilibriumSolver
from Input_Output_Model.util.Evaluators import build_io_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# Helper Functions
# ==============================================================================

def build_va_component_matrix(isic_map):
    """
    Build matrices for each VA component from production_added_values in the database.
    
    Returns dictionaries mapping sector index to VA coefficient for each component:
    - minWages, bonusWages, wages (total), surplus
    
    These are monetary coefficients: $ of component per $ of output.
    """
    ptdb = ProductionsDatabase()
    
    # Initialize component vectors (per dollar of output)
    n = len(isic_map)
    minWages_coeff = np.zeros(n)
    bonusWages_coeff = np.zeros(n)
    wages_coeff = np.zeros(n)
    surplus_coeff = np.zeros(n)
    
    # For each sector, get the cheapest production and extract VA components
    for isic, idx in isic_map.items():
        from Input_Output_Model.models.entities.Good import GoodsDatabase
        gdb = GoodsDatabase()
        goods = gdb.get_all_goods()
        
        # Find the good with this ISIC
        good = next((g for g in goods if g.isic == isic), None)
        if not good:
            continue
            
        prods = ptdb.get_all_productions_by_good(int(good.id_number))
        if not prods:
            continue
        
        # Use cheapest production (same as in build_io_matrix)
        production = min(prods, key=lambda p: float(p.price) if p.price else float('inf'))
        
        output_price = float(production.price) if production.price else 1.0
        if output_price <= 0:
            output_price = 1.0
        
        # Extract VA components from production_added_values JSON
        va_components = production.production_added_values
        
        if va_components:
            # Normalize to per-dollar-of-output coefficients
            minWages_coeff[idx] = float(va_components.get('minWages', 0)) / output_price
            bonusWages_coeff[idx] = float(va_components.get('bonusWages', 0)) / output_price
            wages_coeff[idx] = float(va_components.get('wages', 0)) / output_price
            surplus_coeff[idx] = float(va_components.get('surplus', 0)) / output_price
    
    return {
        'minWages': minWages_coeff,
        'bonusWages': bonusWages_coeff,
        'wages': wages_coeff,
        'surplus': surplus_coeff
    }


def apply_taxes(income, income_tax_rate, corporate_tax_rate, income_tax_applies_to):
    """
    Apply taxes to income components.
    
    Income tax reduces worker take-home pay.
    Corporate tax reduces business profits.
    
    Parameters:
    -----------
    income : dict
        Dictionary with 'minWages', 'bonusWages', 'wages', 'surplus' arrays
    income_tax_rate : float
        Tax rate on wages (0.0 to 1.0)
    corporate_tax_rate : float
        Tax rate on surplus (0.0 to 1.0)
    income_tax_applies_to : str
        Which wage component to tax: "bonusWages", "wages", or "both"
    
    Returns:
    --------
    dict with gross, net, and tax arrays for each component
    """
    minWages = income['minWages'].copy()
    bonusWages = income['bonusWages'].copy()
    wages = income['wages'].copy()
    surplus = income['surplus'].copy()
    
    # Calculate income tax
    if income_tax_applies_to == "bonusWages":
        income_tax = bonusWages * income_tax_rate
        bonusWages_net = bonusWages * (1 - income_tax_rate)
        minWages_net = minWages  # Not taxed
        wages_net = minWages_net + bonusWages_net
    elif income_tax_applies_to == "wages":
        income_tax = wages * income_tax_rate
        wages_net = wages * (1 - income_tax_rate)
        minWages_net = minWages * (1 - income_tax_rate)
        bonusWages_net = bonusWages * (1 - income_tax_rate)
    elif income_tax_applies_to == "both":
        income_tax = wages * income_tax_rate
        wages_net = wages * (1 - income_tax_rate)
        minWages_net = minWages * (1 - income_tax_rate)
        bonusWages_net = bonusWages * (1 - income_tax_rate)
    else:
        raise ValueError(f"Invalid income_tax_applies_to: {income_tax_applies_to}")
    
    # Calculate corporate tax
    corporate_tax = surplus * corporate_tax_rate
    surplus_net = surplus * (1 - corporate_tax_rate)
    
    total_tax = income_tax + corporate_tax
    
    return {
        'minWages_gross': minWages,
        'bonusWages_gross': bonusWages,
        'wages_gross': wages,
        'surplus_gross': surplus,
        'minWages_net': minWages_net,
        'bonusWages_net': bonusWages_net,
        'wages_net': wages_net,
        'surplus_net': surplus_net,
        'income_tax': income_tax,
        'corporate_tax': corporate_tax,
        'total_tax': total_tax
    }


def _distribute_demand(amount, explicit_proportions, demand_distribution,
                       initial_demand_proportions, domestic_indices, n):
    """
    Distribute an aggregate demand amount across sectors using the specified strategy.
    
    Order of precedence:
    1. explicit_proportions (if provided) - must sum to 1.0
    2. demand_distribution == "proportional" with initial_demand_proportions
    3. demand_distribution == "uniform" across domestic sectors
    4. Fallback: initial_demand_proportions or uniform
    """
    if amount == 0:
        return np.zeros(n)
    if explicit_proportions is not None:
        props = np.array(explicit_proportions, dtype=float)
        if len(props) != n:
            raise ValueError(f"Proportions length {len(props)} does not match sectors {n}")
        if not np.isclose(props.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Proportions must sum to 1.0, got {props.sum()}")
        return props * amount
    elif demand_distribution == "proportional" and initial_demand_proportions is not None:
        return initial_demand_proportions * amount
    elif demand_distribution == "uniform":
        result = np.zeros(n)
        per_sector = amount / len(domestic_indices) if domestic_indices else 0
        for i in domestic_indices:
            result[i] = per_sector
        return result
    else:
        if initial_demand_proportions is not None:
            return initial_demand_proportions * amount
        else:
            return np.full(n, amount / n)


def _load_supply_curves(isic_map):
    """Load supply curves from the database for use with DynamicEquilibriumSolver."""
    scdb = SupplyCurveDatabase()
    all_curves = scdb.get_all_supply_curves()
    
    supply_data = {}
    for good in all_curves:
        if good.isic in isic_map:
            tiers = good.price
            if good.id_number == 9999:  # Foreign Exchange
                continue
            if not tiers:
                logger.warning(f"Sector {good.isic} has no price tiers! Using defaults.")
                tiers = [{'cap': -1, 'price': 1.0}]
            supply_data[good.isic] = tiers
    
    return supply_data


def _build_demand_vector(n_sectors, isic_map, final_demand=None, demand_vector=None,
                         target_isic=None, demand_shock=None, uniform_demand=None,
                         total_demand=None, proportions=None):
    """
    Build a demand vector from one of several specification modes.
    
    Modes (in order of precedence):
    1. final_demand - direct vector
    2. demand_vector - direct vector (alias)
    3. target_isic + demand_shock - sector-specific
    4. demand_shock alone - random sector
    5. uniform_demand - same for all sectors
    6. total_demand + proportions - proportional distribution
    """
    if final_demand is not None:
        demand = np.array(final_demand, dtype=float)
        if len(demand) != n_sectors:
            raise ValueError(f"final_demand length {len(demand)} != {n_sectors} sectors")
        print(f"\nDemand Mode: Direct final demand vector")
        return demand
    
    if demand_vector is not None:
        demand = np.array(demand_vector, dtype=float)
        if len(demand) != n_sectors:
            raise ValueError(f"demand_vector length {len(demand)} != {n_sectors} sectors")
        print(f"\nDemand Mode: Direct demand vector")
        return demand
    
    if target_isic is not None and demand_shock is not None:
        if target_isic not in isic_map:
            raise ValueError(f"Target ISIC '{target_isic}' not found in sector map")
        demand = np.zeros(n_sectors)
        idx = isic_map[target_isic]
        demand[idx] = demand_shock
        print(f"\nDemand Mode: Sector-specific")
        print(f"  Target: [{idx}] {target_isic}")
        print(f"  Demand: ${demand_shock:,.2f}")
        return demand
    
    if demand_shock is not None:
        domestic_isics = [isic for isic in isic_map.keys() if isic != "A9999_999_999"]
        if not domestic_isics:
            raise ValueError("No domestic sectors found")
        target_isic = random.choice(domestic_isics)
        demand = np.zeros(n_sectors)
        idx = isic_map[target_isic]
        demand[idx] = demand_shock
        print(f"\nDemand Mode: Random sector")
        print(f"  Selected: [{idx}] {target_isic}")
        print(f"  Demand: ${demand_shock:,.2f}")
        return demand
    
    if uniform_demand is not None:
        demand = np.full(n_sectors, uniform_demand, dtype=float)
        print(f"\nDemand Mode: Uniform (${uniform_demand:,.2f} per sector)")
        return demand
    
    if total_demand is not None and proportions is not None:
        props = np.array(proportions, dtype=float)
        if len(props) != n_sectors:
            raise ValueError(f"Proportions length {len(props)} != {n_sectors} sectors")
        if not np.isclose(props.sum(), 1.0):
            raise ValueError(f"Proportions must sum to 1.0, got {props.sum()}")
        demand = props * total_demand
        print(f"\nDemand Mode: Proportional distribution")
        print(f"  Total demand: ${total_demand:,.2f}")
        return demand
    
    raise ValueError(
        "Must provide one of: final_demand, demand_vector, (target_isic + demand_shock), "
        "demand_shock, uniform_demand, or (total_demand + proportions)"
    )


# ==============================================================================
# Scenario Execution (shared iteration loop)
# ==============================================================================

def _run_scenario(solver_fn, demand, A_matrix, VA_coeffs, va_components, scale_factors,
                  income_tax_rate, corporate_tax_rate, income_tax_applies_to,
                  iterations, consumption_proportions, investment_proportions,
                  government_proportions, demand_distribution,
                  initial_demand_proportions, domestic_indices, n,
                  scenario_name, has_taxes, verbose=True):
    """
    Run one scenario for the given number of iterations.
    
    Returns a list of per-iteration result dictionaries.
    """
    history = []
    current_demand = demand.copy()
    current_C = None
    current_I = None
    current_G = None
    
    for iteration in range(iterations):
        if iterations > 1 and verbose:
            print(f"\n>>> Iteration {iteration + 1}/{iterations}")
            print(f"    Final Demand: ${current_demand.sum():>12,.2f}")
        
        # Solve for output
        output, prices = solver_fn(current_demand)
        
        # Calculate value added by sector
        VA_by_sector = VA_coeffs * output
        VA_total = VA_by_sector.sum()
        
        # Calculate intermediate inputs by sector
        intermediate_by_sector = A_matrix.sum(axis=0) * output
        intermediate_total = intermediate_by_sector.sum()
        
        # Compute VA sub-components (income distribution)
        income = {
            'minWages': output * va_components['minWages'] * scale_factors,
            'bonusWages': output * va_components['bonusWages'] * scale_factors,
            'wages': output * va_components['wages'] * scale_factors,
            'surplus': output * va_components['surplus'] * scale_factors
        }
        
        # Apply taxes
        after_tax = apply_taxes(income, income_tax_rate, corporate_tax_rate, income_tax_applies_to)
        
        # GDP expenditure components
        C = after_tax['wages_net'].sum()
        I_val = after_tax['surplus_net'].sum()
        G = after_tax['total_tax'].sum()
        
        # Store iteration results
        history.append({
            'iteration': iteration + 1,
            'demand': current_demand.copy(),
            'X': output.copy(),
            'VA_by_sector': VA_by_sector.copy(),
            'VA_total': VA_total,
            'intermediate_by_sector': intermediate_by_sector.copy(),
            'intermediate_total': intermediate_total,
            'income': income,
            'after_tax': after_tax,
            'C': C,
            'I': I_val,
            'G': G,
            'demand_C': current_C.copy() if current_C is not None else None,
            'demand_I': current_I.copy() if current_I is not None else None,
            'demand_G': current_G.copy() if current_G is not None else None,
        })
        
        if iterations > 1 and verbose:
            print(f"    Gross Output:   ${output.sum():>12,.2f}")
            print(f"    Value Added:    ${VA_total:>12,.2f}")
            print(f"      C (Consumption): ${C:>12,.2f}")
            print(f"      I (Investment):  ${I_val:>12,.2f}")
            if has_taxes:
                print(f"      G (Government):  ${G:>12,.2f}")
                print(f"      Tax Revenue:     ${after_tax['total_tax'].sum():>12,.2f}")
        
        # Calculate next iteration's demand from this iteration's VA
        if iteration < iterations - 1:
            current_C = _distribute_demand(
                C, consumption_proportions, demand_distribution,
                initial_demand_proportions, domestic_indices, n)
            current_I = _distribute_demand(
                I_val, investment_proportions, demand_distribution,
                initial_demand_proportions, domestic_indices, n)
            current_G = _distribute_demand(
                G, government_proportions, demand_distribution,
                initial_demand_proportions, domestic_indices, n)
            current_demand = current_C + current_I + current_G
    
    return history


# ==============================================================================
# Display Functions (shared)
# ==============================================================================

def _display_scenario(name, history, isic_map, has_taxes, iterations):
    """Display full results for a single scenario."""
    last = history[-1]
    X = last['X']
    VA_by_sector = last['VA_by_sector']
    VA_total = last['VA_total']
    intermediate_by_sector = last['intermediate_by_sector']
    intermediate_total = last['intermediate_total']
    income = last['income']
    at = last['after_tax']
    
    if iterations > 1:
        print(f"\n--- Summary After {iterations} Iterations ---")
    
    # Aggregate Results
    print(f"\nAggregate Results:")
    print(f"  Gross Output (X):           {X.sum():>12,.2f}")
    print(f"  Final Demand (FD):          {last['demand'].sum():>12,.2f}")
    print(f"  Value Added (VA):           {VA_total:>12,.2f}")
    print(f"  Total Intermediate Inputs:  {intermediate_total:>12,.2f}")
    
    # VA sub-components (income distribution)
    print(f"\n  Value Added Components (Income Distribution):")
    print(f"    Min Wages:     ${income['minWages'].sum():>12,.2f}")
    print(f"    Bonus Wages:   ${income['bonusWages'].sum():>12,.2f}")
    print(f"    Total Wages:   ${income['wages'].sum():>12,.2f}")
    print(f"    Surplus:       ${income['surplus'].sum():>12,.2f}")
    print(f"    TOTAL VA:      ${(income['wages'].sum() + income['surplus'].sum()):>12,.2f}")
    
    if has_taxes:
        print(f"\n  After-Tax Income:")
        print(f"    Wages (gross):   ${at['wages_gross'].sum():>12,.2f}  ->  Wages (net):   ${at['wages_net'].sum():>12,.2f}  (Income Tax: ${at['income_tax'].sum():>12,.2f})")
        print(f"    Surplus (gross): ${at['surplus_gross'].sum():>12,.2f}  ->  Surplus (net): ${at['surplus_net'].sum():>12,.2f}  (Corp Tax:   ${at['corporate_tax'].sum():>12,.2f})")
        print(f"    Total Tax Revenue: ${at['total_tax'].sum():>12,.2f}")
    
    # GDP expenditure components
    print(f"\n  GDP Expenditure Components:")
    if has_taxes:
        print(f"    C (Consumption):  ${last['C']:>12,.2f}  (= Wages after income tax)")
        print(f"    I (Investment):   ${last['I']:>12,.2f}  (= Surplus after corporate tax)")
        print(f"    G (Government):   ${last['G']:>12,.2f}  (= Tax revenue)")
    else:
        print(f"    C (Consumption = Wages):   ${last['C']:>12,.2f}")
        print(f"    I (Investment = Surplus):  ${last['I']:>12,.2f}")
        print(f"    G (Government = Taxes):    ${last['G']:>12,.2f}")
    print(f"    FD (C + I + G):            ${(last['C'] + last['I'] + last['G']):>12,.2f}")
    
    # Final Demand Composition
    print(f"\nFinal Demand Composition:")
    print(f"{'Sector':<20} {'Amount':<15} {'% of Total FD':<15}")
    print("-" * 50)
    fd_total = last['demand'].sum()
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        if last['demand'][idx] > 0.01:
            fd_pct = (last['demand'][idx] / fd_total * 100) if fd_total > 0 else 0
            print(f"[{idx}] {isic[:17]:<17} "
                  f"{last['demand'][idx]:>12.2f}   "
                  f"{fd_pct:>12.2f}%")
    print("-" * 50)
    print(f"{'TOTAL':<20} {fd_total:>12.2f}   {'100.00%':>15}")
    
    # Value Added Composition
    print(f"\nValue Added Composition:")
    print(f"{'Sector':<20} {'Amount':<15} {'% of Total VA':<15}")
    print("-" * 50)
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        if VA_by_sector[idx] > 0.01:
            va_pct = (VA_by_sector[idx] / VA_total * 100) if VA_total > 0 else 0
            print(f"[{idx}] {isic[:17]:<17} "
                  f"{VA_by_sector[idx]:>12.2f}   "
                  f"{va_pct:>12.2f}%")
    print("-" * 50)
    print(f"{'TOTAL':<20} {VA_total:>12.2f}   {'100.00%':>15}")
    
    # Sector-by-Sector Breakdown
    print(f"\nSector-by-Sector Breakdown:")
    print(f"{'Sector':<20} {'Output (X)':<15} {'Value Added':<15} {'Intermediate':<15}")
    print("-" * 65)
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        print(f"[{idx}] {isic[:17]:<17} "
              f"{X[idx]:>12.2f}   "
              f"{VA_by_sector[idx]:>12.2f}   "
              f"{intermediate_by_sector[idx]:>12.2f}")
    print("-" * 65)
    print(f"{'TOTAL':<20} "
          f"{X.sum():>12.2f}   "
          f"{VA_total:>12.2f}   "
          f"{intermediate_total:>12.2f}")
    
    # VA Components by Sector
    print(f"\nValue Added Components by Sector:")
    print(f"{'Sector':<20} {'minWages':<14} {'bonusWages':<14} {'Total Wages':<14} {'Surplus':<14} {'Total VA':<14}")
    print("-" * 95)
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        if VA_by_sector[idx] > 0.01:
            print(f"[{idx}] {isic[:17]:<17} "
                  f"{income['minWages'][idx]:>12.2f}  "
                  f"{income['bonusWages'][idx]:>12.2f}  "
                  f"{income['wages'][idx]:>12.2f}  "
                  f"{income['surplus'][idx]:>12.2f}  "
                  f"{VA_by_sector[idx]:>12.2f}")
    print("-" * 95)
    print(f"{'TOTAL':<20} "
          f"{income['minWages'].sum():>12.2f}  "
          f"{income['bonusWages'].sum():>12.2f}  "
          f"{income['wages'].sum():>12.2f}  "
          f"{income['surplus'].sum():>12.2f}  "
          f"{VA_total:>12.2f}")
    
    # FD Components by Sector
    print(f"\nFinal Demand Components by Sector:")
    if last['demand_C'] is not None:
        print(f"{'Sector':<20} {'C (Wages)':<14} {'I (Surplus)':<14} {'G (Taxes)':<14} {'Total FD':<14}")
        print("-" * 80)
        for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
            if last['demand'][idx] > 0.01:
                print(f"[{idx}] {isic[:17]:<17} "
                      f"{last['demand_C'][idx]:>12.2f}  "
                      f"{last['demand_I'][idx]:>12.2f}  "
                      f"{last['demand_G'][idx]:>12.2f}  "
                      f"{last['demand'][idx]:>12.2f}")
        print("-" * 80)
        print(f"{'TOTAL':<20} "
              f"{last['demand_C'].sum():>12.2f}  "
              f"{last['demand_I'].sum():>12.2f}  "
              f"{last['demand_G'].sum():>12.2f}  "
              f"{last['demand'].sum():>12.2f}")
    else:
        print(f"  (Exogenous initial demand - no C/I/G decomposition)")
        print(f"  Total FD: ${last['demand'].sum():>12,.2f}")
    
    # Per-scenario iteration evolution (if multi-iteration)
    if iterations > 1:
        print(f"\n--- GDP Expenditure Evolution ({name}) ---")
        print(f"{'Iter':<6} {'C (Wages)':<18} {'I (Surplus)':<18} {'G (Taxes)':<18} {'FD (C+I+G)':<18}")
        print("-" * 80)
        for hist in history:
            print(f"{hist['iteration']:<6} "
                  f"${hist['C']:>15,.2f}  "
                  f"${hist['I']:>15,.2f}  "
                  f"${hist['G']:>15,.2f}  "
                  f"${(hist['C'] + hist['I'] + hist['G']):>15,.2f}")
        
        print(f"\n--- Income Component Evolution ({name}) ---")
        print(f"{'Iter':<6} {'minWages':<15} {'bonusWages':<15} {'Wages':<15} {'Surplus':<15} {'Total VA':<15}")
        print("-" * 85)
        for hist in history:
            inc = hist['income']
            print(f"{hist['iteration']:<6} "
                  f"${inc['minWages'].sum():>12,.2f}  "
                  f"${inc['bonusWages'].sum():>12,.2f}  "
                  f"${inc['wages'].sum():>12,.2f}  "
                  f"${inc['surplus'].sum():>12,.2f}  "
                  f"${hist['VA_total']:>12,.2f}")


def _display_comparison(before_history, after_history, before_name, after_name,
                        isic_map, has_taxes, iterations, initial_demand_sum,
                        has_tech_change, has_tax_change):
    """Display comparison between two scenarios. Returns deltas dict."""
    last_b = before_history[-1]
    last_a = after_history[-1]
    
    X_b, X_a = last_b['X'], last_a['X']
    VA_b, VA_a = last_b['VA_by_sector'], last_a['VA_by_sector']
    VA_total_b, VA_total_a = last_b['VA_total'], last_a['VA_total']
    int_b, int_a = last_b['intermediate_by_sector'], last_a['intermediate_by_sector']
    int_total_b, int_total_a = last_b['intermediate_total'], last_a['intermediate_total']
    inc_b, inc_a = last_b['income'], last_a['income']
    at_b, at_a = last_b['after_tax'], last_a['after_tax']
    
    delta_X = X_a - X_b
    delta_VA_by_sector = VA_a - VA_b
    delta_VA = VA_total_a - VA_total_b
    delta_int_by_sector = int_a - int_b
    delta_int = int_total_a - int_total_b
    
    # Determine comparison title
    if has_tech_change and has_tax_change:
        title = "Impact of Combined Technology & Tax Policy Change"
    elif has_tech_change:
        title = "Impact of Technological Change"
    else:
        title = "Impact of Tax Policy Change"
    
    print(f"\n{'='*100}")
    print(f"COMPARISON: {title}")
    print(f"{'='*100}")
    
    # Aggregate changes
    print(f"\nAggregate Changes (initial FD = ${initial_demand_sum:,.2f}):")
    print(f"  Change in Gross Output:     {delta_X.sum():>+12,.2f}  ({(delta_X.sum()/X_b.sum()*100):>+7.2f}%)")
    print(f"  Change in Value Added:      {delta_VA:>+12,.2f}  ({(delta_VA/VA_total_b*100):>+7.2f}%)")
    print(f"  Change in Interm. Inputs:   {delta_int:>+12,.2f}  ({(delta_int/int_total_b*100 if int_total_b > 0 else 0):>+7.2f}%)")
    
    # VA component changes
    d_minw = inc_a['minWages'].sum() - inc_b['minWages'].sum()
    d_bonusw = inc_a['bonusWages'].sum() - inc_b['bonusWages'].sum()
    d_wages = inc_a['wages'].sum() - inc_b['wages'].sum()
    d_surplus = inc_a['surplus'].sum() - inc_b['surplus'].sum()
    
    print(f"\n  Value Added Component Changes:")
    print(f"    \u0394Min Wages:    {d_minw:>+12,.2f}  ({(d_minw/inc_b['minWages'].sum()*100 if inc_b['minWages'].sum() > 0 else 0):>+7.2f}%)")
    print(f"    \u0394Bonus Wages:  {d_bonusw:>+12,.2f}  ({(d_bonusw/inc_b['bonusWages'].sum()*100 if inc_b['bonusWages'].sum() > 0 else 0):>+7.2f}%)")
    print(f"    \u0394Total Wages:  {d_wages:>+12,.2f}  ({(d_wages/inc_b['wages'].sum()*100 if inc_b['wages'].sum() > 0 else 0):>+7.2f}%)")
    print(f"    \u0394Surplus:      {d_surplus:>+12,.2f}  ({(d_surplus/inc_b['surplus'].sum()*100 if inc_b['surplus'].sum() > 0 else 0):>+7.2f}%)")
    
    # GDP expenditure changes
    d_C = last_a['C'] - last_b['C']
    d_I = last_a['I'] - last_b['I']
    d_G = last_a['G'] - last_b['G']
    
    print(f"\n  GDP Expenditure Component Changes:")
    print(f"    \u0394C (Consumption): {d_C:>+12,.2f}  ({(d_C/last_b['C']*100 if last_b['C'] > 0 else 0):>+7.2f}%)")
    print(f"    \u0394I (Investment):  {d_I:>+12,.2f}  ({(d_I/last_b['I']*100 if last_b['I'] > 0 else 0):>+7.2f}%)")
    print(f"    \u0394G (Government):  {d_G:>+12,.2f}")
    
    # Tax revenue changes
    if has_taxes:
        d_income_tax = at_a['income_tax'].sum() - at_b['income_tax'].sum()
        d_corp_tax = at_a['corporate_tax'].sum() - at_b['corporate_tax'].sum()
        d_total_tax = at_a['total_tax'].sum() - at_b['total_tax'].sum()
        d_wages_net = at_a['wages_net'].sum() - at_b['wages_net'].sum()
        d_surplus_net = at_a['surplus_net'].sum() - at_b['surplus_net'].sum()
        
        print(f"\n  Tax Revenue Changes:")
        print(f"    \u0394Income Tax:     {d_income_tax:>+12,.2f}  ({(d_income_tax/at_b['income_tax'].sum()*100 if at_b['income_tax'].sum() > 0 else 0):>+7.2f}%)")
        print(f"    \u0394Corporate Tax:  {d_corp_tax:>+12,.2f}  ({(d_corp_tax/at_b['corporate_tax'].sum()*100 if at_b['corporate_tax'].sum() > 0 else 0):>+7.2f}%)")
        print(f"    \u0394Total Tax:      {d_total_tax:>+12,.2f}  ({(d_total_tax/at_b['total_tax'].sum()*100 if at_b['total_tax'].sum() > 0 else 0):>+7.2f}%)")
        print(f"\n  After-Tax Income Changes:")
        print(f"    \u0394Wages (net):    {d_wages_net:>+12,.2f}  ({(d_wages_net/at_b['wages_net'].sum()*100 if at_b['wages_net'].sum() > 0 else 0):>+7.2f}%)")
        print(f"    \u0394Surplus (net):  {d_surplus_net:>+12,.2f}  ({(d_surplus_net/at_b['surplus_net'].sum()*100 if at_b['surplus_net'].sum() > 0 else 0):>+7.2f}%)")
        
        # Winners and losers
        print(f"\n  Distribution Impact:")
        if d_wages_net > 0:
            print(f"    Workers gain ${d_wages_net:,.2f}")
        elif d_wages_net < 0:
            print(f"    Workers lose ${-d_wages_net:,.2f}")
        if d_surplus_net > 0:
            print(f"    Businesses gain ${d_surplus_net:,.2f}")
        elif d_surplus_net < 0:
            print(f"    Businesses lose ${-d_surplus_net:,.2f}")
        if d_total_tax > 0:
            print(f"    Government gains ${d_total_tax:,.2f}")
        elif d_total_tax < 0:
            print(f"    Government loses ${-d_total_tax:,.2f}")
    
    # FD component changes by sector
    if last_b['demand_C'] is not None and last_a['demand_C'] is not None:
        print(f"\n{'='*100}")
        print(f"FINAL DEMAND COMPONENT CHANGES BY SECTOR")
        print(f"{'='*100}")
        hdr_dc = "\u0394C (Wages)"; hdr_di = "\u0394I (Surplus)"; hdr_dg = "\u0394G (Taxes)"; hdr_df = "\u0394Total FD"
        print(f"\n{'Sector':<20} {hdr_dc:<14} {hdr_di:<14} {hdr_dg:<14} {hdr_df:<14}")
        print("-" * 80)
        delta_FD = last_a['demand'] - last_b['demand']
        for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
            dc = last_a['demand_C'][idx] - last_b['demand_C'][idx]
            di = last_a['demand_I'][idx] - last_b['demand_I'][idx]
            dg = last_a['demand_G'][idx] - last_b['demand_G'][idx]
            if abs(dc) > 0.01 or abs(di) > 0.01 or abs(dg) > 0.01 or abs(delta_FD[idx]) > 0.01:
                print(f"[{idx}] {isic[:17]:<17} "
                      f"{dc:>+12.2f}  "
                      f"{di:>+12.2f}  "
                      f"{dg:>+12.2f}  "
                      f"{delta_FD[idx]:>+12.2f}")
        print("-" * 80)
        print(f"{'TOTAL':<20} "
              f"{(last_a['demand_C'].sum() - last_b['demand_C'].sum()):>+12.2f}  "
              f"{(last_a['demand_I'].sum() - last_b['demand_I'].sum()):>+12.2f}  "
              f"{(last_a['demand_G'].sum() - last_b['demand_G'].sum()):>+12.2f}  "
              f"{delta_FD.sum():>+12.2f}")
    
    # VA component changes by sector
    print(f"\n{'='*100}")
    print(f"VALUE ADDED COMPONENT CHANGES BY SECTOR")
    print(f"{'='*100}")
    h_mw = "\u0394minWages"; h_bw = "\u0394bonusWages"; h_w = "\u0394Wages"; h_s = "\u0394Surplus"; h_tv = "\u0394Total VA"
    print(f"\n{'Sector':<20} {h_mw:<14} {h_bw:<14} {h_w:<14} {h_s:<14} {h_tv:<14}")
    print("-" * 95)
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        if abs(delta_VA_by_sector[idx]) > 0.01:
            dm = inc_a['minWages'][idx] - inc_b['minWages'][idx]
            db = inc_a['bonusWages'][idx] - inc_b['bonusWages'][idx]
            dw = inc_a['wages'][idx] - inc_b['wages'][idx]
            ds = inc_a['surplus'][idx] - inc_b['surplus'][idx]
            print(f"[{idx}] {isic[:17]:<17} "
                  f"{dm:>+12.2f}  "
                  f"{db:>+12.2f}  "
                  f"{dw:>+12.2f}  "
                  f"{ds:>+12.2f}  "
                  f"{delta_VA_by_sector[idx]:>+12.2f}")
    print("-" * 95)
    print(f"{'TOTAL':<20} "
          f"{d_minw:>+12.2f}  "
          f"{d_bonusw:>+12.2f}  "
          f"{d_wages:>+12.2f}  "
          f"{d_surplus:>+12.2f}  "
          f"{delta_VA:>+12.2f}")
    
    # Detailed sector-by-sector changes
    print(f"\nDetailed Sector-by-Sector Changes:")
    h_out = "\u0394Output"; h_va = "\u0394Value Added"; h_int = "\u0394Intermediate"
    print(f"{'Sector':<20} {h_out:<15} {'% Chg':<10} {h_va:<15} {h_int:<15}")
    print("-" * 85)
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        pct = ((X_a[idx] - X_b[idx]) / X_b[idx] * 100) if X_b[idx] != 0 else 0
        print(f"[{idx}] {isic[:17]:<17} "
              f"{delta_X[idx]:>+12.2f}   "
              f"{pct:>+7.2f}%   "
              f"{delta_VA_by_sector[idx]:>+12.2f}   "
              f"{delta_int_by_sector[idx]:>+12.2f}")
    print("-" * 85)
    print(f"{'TOTAL':<20} "
          f"{delta_X.sum():>+12.2f}   "
          f"{(delta_X.sum()/X_b.sum()*100):>+7.2f}%   "
          f"{delta_VA:>+12.2f}   "
          f"{delta_int:>+12.2f}")
    
    # After-tax sector breakdown (when taxes present)
    if has_taxes:
        print(f"\n{'='*100}")
        print(f"SECTOR-BY-SECTOR AFTER-TAX BREAKDOWN")
        print(f"{'='*100}")
        print(f"\n{before_name}:")
        print(f"{'Sector':<20} {'Output':<12} {'MinWage':<12} {'Bonus':<12} {'Profit':<12} {'IncomeTax':<12} {'CorpTax':<12}")
        print("-" * 110)
        for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
            print(f"[{idx}] {isic[:17]:<17} "
                  f"${X_b[idx]:>10.2f}  "
                  f"${at_b['minWages_net'][idx]:>10.2f}  "
                  f"${at_b['bonusWages_net'][idx]:>10.2f}  "
                  f"${at_b['surplus_net'][idx]:>10.2f}  "
                  f"${at_b['income_tax'][idx]:>10.2f}  "
                  f"${at_b['corporate_tax'][idx]:>10.2f}")
        print("-" * 110)
        print(f"{'TOTAL':<20} "
              f"${X_b.sum():>10.2f}  "
              f"${at_b['minWages_net'].sum():>10.2f}  "
              f"${at_b['bonusWages_net'].sum():>10.2f}  "
              f"${at_b['surplus_net'].sum():>10.2f}  "
              f"${at_b['income_tax'].sum():>10.2f}  "
              f"${at_b['corporate_tax'].sum():>10.2f}")
        
        print(f"\n{after_name}:")
        print(f"{'Sector':<20} {'Output':<12} {'MinWage':<12} {'Bonus':<12} {'Profit':<12} {'IncomeTax':<12} {'CorpTax':<12}")
        print("-" * 110)
        for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
            print(f"[{idx}] {isic[:17]:<17} "
                  f"${X_a[idx]:>10.2f}  "
                  f"${at_a['minWages_net'][idx]:>10.2f}  "
                  f"${at_a['bonusWages_net'][idx]:>10.2f}  "
                  f"${at_a['surplus_net'][idx]:>10.2f}  "
                  f"${at_a['income_tax'][idx]:>10.2f}  "
                  f"${at_a['corporate_tax'][idx]:>10.2f}")
        print("-" * 110)
        print(f"{'TOTAL':<20} "
              f"${X_a.sum():>10.2f}  "
              f"${at_a['minWages_net'].sum():>10.2f}  "
              f"${at_a['bonusWages_net'].sum():>10.2f}  "
              f"${at_a['surplus_net'].sum():>10.2f}  "
              f"${at_a['income_tax'].sum():>10.2f}  "
              f"${at_a['corporate_tax'].sum():>10.2f}")
        
        # Delta table
        print(f"\nChanges (\u0394 = {after_name} - {before_name}):")
        d_output = X_a - X_b
        d_mw_net = at_a['minWages_net'] - at_b['minWages_net']
        d_bw_net = at_a['bonusWages_net'] - at_b['bonusWages_net']
        d_s_net = at_a['surplus_net'] - at_b['surplus_net']
        d_it = at_a['income_tax'] - at_b['income_tax']
        d_ct = at_a['corporate_tax'] - at_b['corporate_tax']
        
        h_o="\u0394Output"; h_mw2="\u0394MinWage"; h_b="\u0394Bonus"; h_p="\u0394Profit"; h_it="\u0394IncomeTax"; h_ct="\u0394CorpTax"
        print(f"{'Sector':<20} {h_o:<12} {h_mw2:<12} {h_b:<12} {h_p:<12} {h_it:<12} {h_ct:<12}")
        print("-" * 110)
        for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
            print(f"[{idx}] {isic[:17]:<17} "
                  f"{d_output[idx]:>+10.2f}  "
                  f"${d_mw_net[idx]:>+10.2f}  "
                  f"${d_bw_net[idx]:>+10.2f}  "
                  f"${d_s_net[idx]:>+10.2f}  "
                  f"${d_it[idx]:>+10.2f}  "
                  f"${d_ct[idx]:>+10.2f}")
        print("-" * 110)
        print(f"{'TOTAL':<20} "
              f"{d_output.sum():>+10.2f}  "
              f"${d_mw_net.sum():>+10.2f}  "
              f"${d_bw_net.sum():>+10.2f}  "
              f"${d_s_net.sum():>+10.2f}  "
              f"${d_it.sum():>+10.2f}  "
              f"${d_ct.sum():>+10.2f}")
    
    # VA Composition Changes
    print(f"\n{'='*100}")
    print(f"VALUE ADDED COMPOSITION CHANGES")
    print(f"{'='*100}")
    h_dva = "\u0394VA"
    print(f"\n{'Sector':<20} {'Before VA':<15} {'% of VA':<12} {'After VA':<15} {'% of VA':<12} {h_dva:<15}")
    print("-" * 95)
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        if VA_b[idx] > 0.01 or VA_a[idx] > 0.01:
            pct_b = (VA_b[idx] / VA_total_b * 100) if VA_total_b > 0 else 0
            pct_a = (VA_a[idx] / VA_total_a * 100) if VA_total_a > 0 else 0
            print(f"[{idx}] {isic[:17]:<17} "
                  f"{VA_b[idx]:>12.2f}   "
                  f"{pct_b:>9.2f}%   "
                  f"{VA_a[idx]:>12.2f}   "
                  f"{pct_a:>9.2f}%   "
                  f"{delta_VA_by_sector[idx]:>+12.2f}")
    print("-" * 95)
    print(f"{'TOTAL':<20} "
          f"{VA_total_b:>12.2f}   "
          f"{'100.00%':>12}   "
          f"{VA_total_a:>12.2f}   "
          f"{'100.00%':>12}   "
          f"{delta_VA:>+12.2f}")
    
    # FD Composition Changes (if demand differs between scenarios)
    delta_FD_by_sector = last_a['demand'] - last_b['demand']
    if np.abs(delta_FD_by_sector).sum() > 0.01:
        fd_b_total = last_b['demand'].sum()
        fd_a_total = last_a['demand'].sum()
        
        print(f"\n{'='*100}")
        print(f"FINAL DEMAND COMPOSITION CHANGES")
        print(f"{'='*100}")
        h_dfd = "\u0394FD"
        print(f"\n{'Sector':<20} {'Before FD':<15} {'% of FD':<12} {'After FD':<15} {'% of FD':<12} {h_dfd:<15}")
        print("-" * 95)
        for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
            if last_b['demand'][idx] > 0.01 or last_a['demand'][idx] > 0.01:
                pct_b = (last_b['demand'][idx] / fd_b_total * 100) if fd_b_total > 0 else 0
                pct_a = (last_a['demand'][idx] / fd_a_total * 100) if fd_a_total > 0 else 0
                print(f"[{idx}] {isic[:17]:<17} "
                      f"{last_b['demand'][idx]:>12.2f}   "
                      f"{pct_b:>9.2f}%   "
                      f"{last_a['demand'][idx]:>12.2f}   "
                      f"{pct_a:>9.2f}%   "
                      f"{delta_FD_by_sector[idx]:>+12.2f}")
        print("-" * 95)
        print(f"{'TOTAL':<20} "
              f"{fd_b_total:>12.2f}   "
              f"{'100.00%':>12}   "
              f"{fd_a_total:>12.2f}   "
              f"{'100.00%':>12}   "
              f"{delta_FD_by_sector.sum():>+12.2f}")
    
    # Iteration-by-iteration comparison
    if iterations > 1:
        print(f"\n{'='*100}")
        print(f"ITERATION-BY-ITERATION COMPARISON")
        print(f"{'='*100}")
        
        print(f"\n{'Iteration':<12} {'Before Output':<20} {'After Output':<20} {'Delta Output':<20} {'% Change'}")
        print("-" * 85)
        for i in range(iterations):
            b_iter = before_history[i]
            a_iter = after_history[i]
            d_iter = a_iter['X'].sum() - b_iter['X'].sum()
            pct = (d_iter / b_iter['X'].sum() * 100) if b_iter['X'].sum() != 0 else 0
            print(f"{i+1:<12} "
                  f"{b_iter['X'].sum():>18,.2f}  "
                  f"{a_iter['X'].sum():>18,.2f}  "
                  f"{d_iter:>+18,.2f}  "
                  f"{pct:>+7.2f}%")
        
        print(f"\n{'Iteration':<12} {'Before VA':<20} {'After VA':<20} {'Delta VA':<20} {'% Change'}")
        print("-" * 85)
        for i in range(iterations):
            b_iter = before_history[i]
            a_iter = after_history[i]
            d_va = a_iter['VA_total'] - b_iter['VA_total']
            pct = (d_va / b_iter['VA_total'] * 100) if b_iter['VA_total'] != 0 else 0
            print(f"{i+1:<12} "
                  f"{b_iter['VA_total']:>18,.2f}  "
                  f"{a_iter['VA_total']:>18,.2f}  "
                  f"{d_va:>+18,.2f}  "
                  f"{pct:>+7.2f}%")
    
    # Return deltas
    return {
        'X': delta_X,
        'VA': delta_VA,
        'VA_by_sector': delta_VA_by_sector,
        'intermediate_inputs': delta_int,
        'intermediate_by_sector': delta_int_by_sector,
        'minWages': d_minw,
        'bonusWages': d_bonusw,
        'wages': d_wages,
        'surplus': d_surplus,
        'C': d_C,
        'I': d_I,
        'G': d_G,
    }


# ==============================================================================
# Main Unified Simulation Function
# ==============================================================================

def run_simulation(
    # ---- Demand Specification (multiple modes) ----
    final_demand=None,
    demand_vector=None,
    target_isic=None,
    demand_shock=None,
    uniform_demand=None,
    total_demand=None,
    proportions=None,
    
    # ---- Technology Configuration ----
    A_before=None,
    VA_before=None,
    A_after=None,
    VA_after=None,
    isic_map=None,
    
    # ---- Multi-Level Technological Change ----
    tech_change=None,  # TechnologicalChange object for multi-level changes
    
    # ---- Tax Policy (before/after) ----
    income_tax_rate_before=0.0,
    income_tax_rate_after=0.0,
    corporate_tax_rate_before=0.0,
    corporate_tax_rate_after=0.0,
    income_tax_applies_to="bonusWages",
    
    # ---- Proportions for C/I/G Distribution ----
    consumption_proportions=None,
    investment_proportions=None,
    government_proportions=None,
    
    # ---- Circular Flow ----
    iterations=1,
    demand_distribution="proportional",
    consumption_rate=1.0,
    
    # ---- Solver ----
    solver_type="leontief",
    
    # ---- Display Names ----
    before_name="Before",
    after_name="After",
    
    # ---- Logging ----
    loggingLevel=logging.WARNING,
):
    """
    Unified simulation function for Input-Output economic modeling.
    
    Supports technological change analysis, tax policy analysis, or both simultaneously.
    Runs two scenarios (before/after) and displays a comprehensive comparison.
    
    Parameters:
    -----------
    DEMAND SPECIFICATION (provide ONE):
        final_demand : array-like
            Direct final demand vector
        demand_vector : array-like
            Direct demand vector (alias for final_demand)
        target_isic + demand_shock : str, float
            Demand for a specific sector
        demand_shock : float (alone)
            Demand for a random sector
        uniform_demand : float
            Same demand for all sectors
        total_demand + proportions : float, list
            Distribute total_demand across sectors by proportions
    
    TECHNOLOGY:
        A_before : np.ndarray
            Before/baseline technical coefficient matrix (None = build from DB)
        VA_before : np.ndarray
            Before/baseline VA coefficients
        A_after : np.ndarray
            After/changed technical coefficient matrix (None = same as before)
        VA_after : np.ndarray
            After/changed VA coefficients (None = same as before)
        isic_map : dict
            Mapping of ISIC codes to matrix indices (None = build from DB)
        tech_change : TechnologicalChange
            Multi-level TechnologicalChange object. If provided, changes are applied:
            1. Production-level changes -> rebuild A matrix
            2. Curve-level changes -> rebuild supply data
            3. Matrix-level changes -> direct A matrix modifications
    
    TAX POLICY:
        income_tax_rate_before/after : float
            Income tax rate for each scenario (0.0 to 1.0)
        corporate_tax_rate_before/after : float
            Corporate tax rate for each scenario (0.0 to 1.0)
        income_tax_applies_to : str
            Which wages to tax: "bonusWages", "wages", or "both"
    
    CIRCULAR FLOW:
        consumption_proportions : list
            How to distribute C across sectors (sum to 1.0)
        investment_proportions : list
            How to distribute I across sectors (sum to 1.0)
        government_proportions : list
            How to distribute G across sectors (sum to 1.0)
        iterations : int
            Number of circular flow iterations (default: 1)
        demand_distribution : str
            Fallback distribution: "proportional" or "uniform"
        consumption_rate : float
            Fraction of income spent (default: 1.0, for future use)
    
    SOLVER:
        solver_type : str
            "leontief" (Leontief inverse) or "supply_curves" (DynamicEquilibriumSolver)
    
    DISPLAY:
        before_name : str
            Label for scenario 1
        after_name : str
            Label for scenario 2
    
    Returns:
    --------
    dict with 'before', 'after', 'deltas', 'config', 'before_history', 'after_history', 'isic_map'
    """
    
    # ==================================================================
    # 1. Build IO matrices (if not provided)
    # ==================================================================
    built_from_db = (A_before is None)
    
    if A_before is None:
        print("=" * 100)
        print("BUILDING COEFFICIENT MATRICES FROM DATABASE")
        print("=" * 100)
        A_before, VA_before, isic_map = build_io_matrix(demoDB=False, loggingLevel=loggingLevel)
        print(f"\n[OK] IO Matrix Built. Sectors: {len(isic_map)}")
    
    # ==================================================================
    # 1b. Apply multi-level technological change (if provided)
    # ==================================================================
    supply_data_after = None  # For curve-level changes
    
    if tech_change is not None:
        from Input_Output_Model.demos.util.technological_change import LEVEL_MATRIX, LEVEL_PRODUCTION, LEVEL_CURVE
        
        change_levels = tech_change.get_change_levels()
        print("\n" + "=" * 100)
        print(f"APPLYING MULTI-LEVEL TECHNOLOGICAL CHANGE: {tech_change.name}")
        print("=" * 100)
        print(f"Change levels: {', '.join(change_levels)}")
        
        # Start with baseline as A_after
        if A_after is None:
            A_after = A_before.copy()
        if VA_after is None:
            VA_after = VA_before.copy()
        
        # Level 3 (deepest): Production-level changes (cascade: productions → curves → matrix)
        if tech_change.has_production_changes():
            print(f"\n[Level 3: Productions] Applying {len(tech_change.production_changes)} changes...")
            ptdb = ProductionsDatabase()
            scdb = SupplyCurveDatabase()
            result = tech_change.apply_to_productions(
                ptdb, scdb, rebuild_curves=True, rebuild_matrix=True, loggingLevel=loggingLevel
            )
            
            if result['A_matrix'] is not None:
                A_after = result['A_matrix']
                VA_after = result['VA_vector']
                isic_map = result['isic_map']
                print(f"  [OK] Cascade rebuild: {len(result['productions'])} productions → curves → coefficient matrix")
            if result.get('supply_data'):
                supply_data_after = result['supply_data']
        
        # Level 2: Curve-level changes (cascade: curves → matrix)
        if tech_change.has_curve_changes():
            print(f"\n[Level 2: Curves] Applying {len(tech_change.curve_changes)} changes...")
            ptdb = ProductionsDatabase()
            scdb = SupplyCurveDatabase()
            result = tech_change.apply_to_curves(
                scdb, ptdb=ptdb, isic_map=isic_map, rebuild_matrix=True, loggingLevel=loggingLevel
            )
            supply_data_after = result['supply_data']
            if result['A_matrix'] is not None:
                A_after = result['A_matrix']
                VA_after = result['VA_vector']
                isic_map = result['isic_map']
            print(f"  [OK] Cascade rebuild: {len(result['modified_curves'])} curves → coefficient matrix")
        
        # Level 1: Matrix-level changes (direct A matrix modification)
        if tech_change.has_matrix_changes():
            print(f"\n[Matrix Level] Applying {len(tech_change.matrix_changes)} changes...")
            A_after, VA_after = tech_change.apply(A_after, VA_after, isic_map)
            print(f"  [OK] A matrix and VA coefficients updated")
        
        print(tech_change.get_summary())
    
    elif A_after is None:
        A_after = A_before.copy()
    if VA_after is None:
        VA_after = VA_before.copy()
    
    n = A_before.shape[0]
    
    # ==================================================================
    # 2. Determine what's changing
    # ==================================================================
    has_tech_change = not (np.allclose(A_before, A_after) and np.allclose(VA_before, VA_after))
    has_tax_change = (income_tax_rate_before != income_tax_rate_after or
                      corporate_tax_rate_before != corporate_tax_rate_after)
    has_taxes_before = (income_tax_rate_before > 0 or corporate_tax_rate_before > 0)
    has_taxes_after = (income_tax_rate_after > 0 or corporate_tax_rate_after > 0)
    has_any_taxes = has_taxes_before or has_taxes_after
    
    # ==================================================================
    # 3. Print configuration header
    # ==================================================================
    sim_type = []
    if has_tech_change:
        sim_type.append("Technological Change")
    if has_tax_change:
        sim_type.append("Tax Policy Change")
    if not sim_type:
        sim_type.append("Scenario Comparison")
    
    print("\n" + "=" * 100)
    print(f"INPUT-OUTPUT SIMULATION: {' + '.join(sim_type)}")
    if iterations > 1:
        print(f"(Circular Flow Model - {iterations} iterations)")
    print("=" * 100)
    
    if has_any_taxes or has_tax_change:
        print(f"\nTax Policy Configuration:")
        print(f"  Income Tax ({before_name}):    {income_tax_rate_before*100:.1f}% on {income_tax_applies_to}")
        print(f"  Income Tax ({after_name}):     {income_tax_rate_after*100:.1f}% on {income_tax_applies_to}")
        print(f"  Corporate Tax ({before_name}): {corporate_tax_rate_before*100:.1f}% on surplus")
        print(f"  Corporate Tax ({after_name}):  {corporate_tax_rate_after*100:.1f}% on surplus")
    
    if iterations > 1:
        print(f"\nCircular Flow Parameters:")
        print(f"  Iterations: {iterations}")
        print(f"  Distribution: {demand_distribution}")
        if consumption_proportions is not None:
            print(f"  Consumption Proportions: {consumption_proportions}")
        if investment_proportions is not None:
            print(f"  Investment Proportions:   {investment_proportions}")
        if government_proportions is not None:
            print(f"  Government Proportions:   {government_proportions}")
    
    # ==================================================================
    # 4. Build demand vector
    # ==================================================================
    print("\n" + "=" * 100)
    print("DEMAND SPECIFICATION")
    print("=" * 100)
    
    demand = _build_demand_vector(
        n, isic_map, final_demand, demand_vector,
        target_isic, demand_shock, uniform_demand,
        total_demand, proportions
    )
    
    # Display demand
    print(f"\nDemand Vector:")
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        if isic != "A9999_999_999":
            print(f"  [{idx}] {isic[:20]:<20}: ${demand[idx]:>10.2f}")
    print(f"\nTotal Final Demand: ${demand.sum():,.2f}")
    
    # ==================================================================
    # 5. Compute shared setup
    # ==================================================================
    # VA component matrices from database
    va_components = build_va_component_matrix(isic_map)
    db_va_total = va_components['wages'] + va_components['surplus']
    scale_factors_before = np.divide(VA_before, db_va_total,
                                     where=db_va_total!=0, out=np.ones_like(VA_before))
    scale_factors_after = np.divide(VA_after, db_va_total,
                                    where=db_va_total!=0, out=np.ones_like(VA_after))
    
    # Display VA component coefficients
    print(f"\nValue Added Component Coefficients ($ per $ of output):")
    print(f"{'Sector':<20} {'minWages':<12} {'bonusWages':<12} {'wages':<12} {'surplus':<12} {'Total VA':<12}")
    print("-" * 95)
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        print(f"[{idx}] {isic[:17]:<17} "
              f"{va_components['minWages'][idx]:>10.4f}  "
              f"{va_components['bonusWages'][idx]:>10.4f}  "
              f"{va_components['wages'][idx]:>10.4f}  "
              f"{va_components['surplus'][idx]:>10.4f}  "
              f"{VA_before[idx]:>10.4f}")
    
    # Domestic indices and initial demand proportions
    domestic_indices = [idx for isic, idx in isic_map.items() if isic != "A9999_999_999"]
    total_domestic = sum(demand[i] for i in domestic_indices)
    initial_demand_proportions = None
    if total_domestic > 0:
        initial_demand_proportions = np.zeros(n)
        for i in domestic_indices:
            initial_demand_proportions[i] = demand[i] / total_domestic
    
    # ==================================================================
    # 6. Create solver functions
    # ==================================================================
    if solver_type == "leontief":
        try:
            L_before = np.linalg.inv(np.eye(n) - A_before)
        except np.linalg.LinAlgError:
            print("ERROR: Could not invert (I-A) for before scenario!")
            return None
        try:
            L_after = np.linalg.inv(np.eye(n) - A_after)
        except np.linalg.LinAlgError:
            print("ERROR: Could not invert (I-A) for after scenario!")
            return None
        
        solver_before = lambda d: (L_before @ d, np.ones(n))
        solver_after = lambda d: (L_after @ d, np.ones(n))
        
    elif solver_type == "supply_curves":
        print("\n" + "=" * 100)
        print("LOADING SUPPLY CURVES")
        print("=" * 100)
        supply_data = _load_supply_curves(isic_map)
        print(f"[OK] Loaded supply curves for {len(supply_data)} sectors.")
        
        solver_before_obj = DynamicEquilibriumSolver(A_before, isic_map, supply_data)
        
        # Use modified supply data for "after" scenario if curve-level changes were applied
        if supply_data_after is not None:
            print(f"[OK] Using modified supply curves for '{after_name}' scenario.")
            solver_after_obj = DynamicEquilibriumSolver(A_after, isic_map, supply_data_after)
        elif has_tech_change:
            solver_after_obj = DynamicEquilibriumSolver(A_after, isic_map, supply_data)
        else:
            solver_after_obj = solver_before_obj
        
        def _make_solver(solver_obj):
            def solve(d):
                result = solver_obj.solve(d, verbose=False)
                if not result or result['status'] != 'converged':
                    raise RuntimeError("Solver failed to converge")
                return result['output'], result['prices']
            return solve
        
        solver_before = _make_solver(solver_before_obj)
        solver_after = _make_solver(solver_after_obj)
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}. Use 'leontief' or 'supply_curves'.")
    
    # ==================================================================
    # 7. Run BEFORE scenario
    # ==================================================================
    print(f"\n{'='*100}")
    if iterations > 1:
        print(f"SCENARIO 1: {before_name} (Circular Flow - {iterations} iterations)")
    else:
        print(f"SCENARIO 1: {before_name}")
    print(f"{'='*100}")
    
    before_history = _run_scenario(
        solver_fn=solver_before,
        demand=demand,
        A_matrix=A_before,
        VA_coeffs=VA_before,
        va_components=va_components,
        scale_factors=scale_factors_before,
        income_tax_rate=income_tax_rate_before,
        corporate_tax_rate=corporate_tax_rate_before,
        income_tax_applies_to=income_tax_applies_to,
        iterations=iterations,
        consumption_proportions=consumption_proportions,
        investment_proportions=investment_proportions,
        government_proportions=government_proportions,
        demand_distribution=demand_distribution,
        initial_demand_proportions=initial_demand_proportions,
        domestic_indices=domestic_indices,
        n=n,
        scenario_name=before_name,
        has_taxes=has_taxes_before,
    )
    
    _display_scenario(before_name, before_history, isic_map, has_taxes_before, iterations)
    
    # ==================================================================
    # 8. Run AFTER scenario
    # ==================================================================
    print(f"\n{'='*100}")
    if iterations > 1:
        print(f"SCENARIO 2: {after_name} (Circular Flow - {iterations} iterations)")
    else:
        print(f"SCENARIO 2: {after_name}")
    print(f"{'='*100}")
    
    after_history = _run_scenario(
        solver_fn=solver_after,
        demand=demand,
        A_matrix=A_after,
        VA_coeffs=VA_after,
        va_components=va_components,
        scale_factors=scale_factors_after,
        income_tax_rate=income_tax_rate_after,
        corporate_tax_rate=corporate_tax_rate_after,
        income_tax_applies_to=income_tax_applies_to,
        iterations=iterations,
        consumption_proportions=consumption_proportions,
        investment_proportions=investment_proportions,
        government_proportions=government_proportions,
        demand_distribution=demand_distribution,
        initial_demand_proportions=initial_demand_proportions,
        domestic_indices=domestic_indices,
        n=n,
        scenario_name=after_name,
        has_taxes=has_taxes_after,
    )
    
    _display_scenario(after_name, after_history, isic_map, has_taxes_after, iterations)
    
    # ==================================================================
    # 9. Display comparison
    # ==================================================================
    deltas = _display_comparison(
        before_history, after_history,
        before_name, after_name,
        isic_map, has_any_taxes, iterations,
        demand.sum(),
        has_tech_change, has_tax_change
    )
    
    # ==================================================================
    # 10. Summary
    # ==================================================================
    print("\n" + "=" * 100)
    print("SIMULATION COMPLETE")
    print("=" * 100)
    
    # Resource efficiency comparison (if tech change)
    if has_tech_change:
        last_b = before_history[-1]
        last_a = after_history[-1]
        eff_b = last_b['VA_total'] / last_b['X'].sum() if last_b['X'].sum() > 0 else 0
        eff_a = last_a['VA_total'] / last_a['X'].sum() if last_a['X'].sum() > 0 else 0
        
        print(f"\nResource Efficiency (VA / Gross Output):")
        print(f"  {before_name}:  {eff_b:.4f}")
        print(f"  {after_name}:   {eff_a:.4f}")
        print(f"  Change:         {(eff_a - eff_b):+.4f}")
        
        delta_X_total = deltas['X'].sum()
        if delta_X_total < 0:
            print(f"\nTechnology reduces gross output by {abs(delta_X_total):.2f} for the SAME demand.")
            print(f"  The economy becomes more EFFICIENT.")
        elif delta_X_total > 0:
            print(f"\nTechnology increases gross output by {delta_X_total:.2f} for the SAME demand.")
            print(f"  This could indicate substitution toward more input-intensive methods.")
    
    # ==================================================================
    # 11. Return structured result
    # ==================================================================
    last_b = before_history[-1]
    last_a = after_history[-1]
    
    return {
        'before': {
            'name': before_name,
            'X': last_b['X'],
            'VA': last_b['VA_total'],
            'VA_by_sector': last_b['VA_by_sector'],
            'FD': demand.sum(),
            'intermediate_inputs': last_b['intermediate_total'],
            'intermediate_by_sector': last_b['intermediate_by_sector'],
            'income': last_b['income'],
            'after_tax': last_b['after_tax'],
            'C': last_b['C'],
            'I': last_b['I'],
            'G': last_b['G']
        },
        'after': {
            'name': after_name,
            'X': last_a['X'],
            'VA': last_a['VA_total'],
            'VA_by_sector': last_a['VA_by_sector'],
            'FD': demand.sum(),
            'intermediate_inputs': last_a['intermediate_total'],
            'intermediate_by_sector': last_a['intermediate_by_sector'],
            'income': last_a['income'],
            'after_tax': last_a['after_tax'],
            'C': last_a['C'],
            'I': last_a['I'],
            'G': last_a['G']
        },
        'deltas': deltas,
        'config': {
            'income_tax_rate_before': income_tax_rate_before,
            'income_tax_rate_after': income_tax_rate_after,
            'corporate_tax_rate_before': corporate_tax_rate_before,
            'corporate_tax_rate_after': corporate_tax_rate_after,
            'income_tax_applies_to': income_tax_applies_to,
            'has_tech_change': has_tech_change,
            'has_tax_change': has_tax_change,
            'has_any_taxes': has_any_taxes,
            'iterations': iterations,
            'solver_type': solver_type,
        },
        'before_history': before_history,
        'after_history': after_history,
        'isic_map': isic_map
    }


# ==============================================================================
# Backward Compatibility Alias
# ==============================================================================

def run_tax_policy_simulation(**kwargs):
    """
    Backward-compatible wrapper for run_simulation.
    
    Maps the old parameter names to the new unified interface.
    Use run_simulation() directly for new code.
    """
    # The old function defaults to supply_curves solver
    kwargs.setdefault('solver_type', 'supply_curves')
    kwargs.setdefault('before_name', 'Before Tax Change')
    kwargs.setdefault('after_name', 'After Tax Change')
    
    # Map consumption_distribution to demand_distribution
    if 'consumption_distribution' in kwargs:
        kwargs.setdefault('demand_distribution', kwargs.pop('consumption_distribution'))
    
    return run_simulation(**kwargs)
