import numpy as np
from typing import Dict, List, Any

from core.io_matrix import IOModel
from core.income import build_va_component_matrix, apply_taxes
from core.demand import distribute_demand

import logging

logger = logging.getLogger(__name__)

def run_tax_simulation(
    model: IOModel,
    isic_map: Dict[str, int],
    base_demand: np.ndarray,
    iterations: int = 1,
    income_tax_rate: float = 0.0,
    corporate_tax_rate: float = 0.0,
    income_tax_applies_to: str = "wages",
    wage_spend_rate: float = 1.0,
    surplus_spend_rate: float = 1.0,
    tax_spend_rate: float = 1.0,
    economy_type: str = "open",
    wage_proportions: np.ndarray = None,
    surplus_proportions: np.ndarray = None,
    government_proportions: np.ndarray = None,
) -> List[Dict[str, Any]]:
    """
    Run an iterative circular-flow tax simulation.
    
    Returns a history list of iteration results.
    Each result contains X, Y, tax splits, and summary totals.
    """
    n = model.n
    history = []
    
    # Calculate VA components (coefficients) for this model
    va_coeffs = build_va_component_matrix(isic_map)
    
    # Initial demand distribution proportions for "proportional" mapping (fallback)
    base_demand_sum = base_demand.sum()
    if base_demand_sum > 0:
        initial_proportions = base_demand / base_demand_sum
    else:
        initial_proportions = np.full(n, 1.0 / n)
        
    domestic_indices = [i for i, isic in enumerate(isic_map) if isic != "A9999_999_999"]
    import_idx = isic_map.get("A9999_999_999", None)

    current_demand = base_demand.copy()

    for i in range(iterations):
        # 1. Output Calculation (Leontief)
        X = model.simulate(current_demand)
        
        # 2. Income Component Calculation (Gross)
        gross_income = {
            "minWages": va_coeffs["minWages"] * X,
            "bonusWages": va_coeffs["bonusWages"] * X,
            "wages": va_coeffs["wages"] * X,
            "surplus": va_coeffs["surplus"] * X,
        }
        
        # 3. Apply Taxes
        tax_results = apply_taxes(
            income=gross_income,
            income_tax_rate=income_tax_rate,
            corporate_tax_rate=corporate_tax_rate,
            income_tax_applies_to=income_tax_applies_to
        )
        
        total_wage_net = tax_results["wages_net"].sum()
        total_surplus_net = tax_results["surplus_net"].sum()
        total_tax_rev = tax_results["total_tax"].sum()
        total_va = (model.VA_coeffs * X).sum()
        unallocated_va = total_va - (gross_income["wages"].sum() + gross_income["surplus"].sum())
        
        # Import Leakage Handling
        import_leakage = 0.0
        if import_idx is not None:
            import_leakage = X[import_idx]
            
        recycled_leakage = 0.0
        if economy_type == "closed":
            recycled_leakage = import_leakage
        
        logger.info(f"--- Iteration {i+1} ---")
        logger.info(f"Output (X): {X.sum():.2f}, Total Demand (Y): {current_demand.sum():.2f}")
        logger.info(f"Net Wages: {total_wage_net:.2f}, Net Surplus: {total_surplus_net:.2f}, Taxes: {total_tax_rev:.2f}")
        logger.info(f"Unallocated VA: {unallocated_va:.2f}, Import Leakage: {import_leakage:.2f} (Recycled: {recycled_leakage:.2f})")
        
        # Save snapshot
        snapshot = {
            "iteration": i + 1,
            "Y": current_demand.copy(),
            "X": X.copy(),
            "gross_income": gross_income,
            "tax_results": tax_results,
            "total_demand": current_demand.sum(),
            "total_output": X.sum()
        }
        history.append(snapshot)
        
        # 4. Prepare next iteration's demand if not on last iteration
        if i < iterations - 1:
            # Bucket 1: Wage Spending (Wages + Unallocated)
            wage_spending = total_wage_net * wage_spend_rate + unallocated_va * 1.0
            wage_demand = distribute_demand(
                amount=wage_spending,
                explicit_proportions=wage_proportions,
                demand_distribution="proportional" if wage_proportions is None else "custom",
                initial_demand_proportions=initial_proportions,
                domestic_indices=domestic_indices,
                n=n
            )
            
            # Bucket 2: Surplus/Surplus Spending (Surplus + Recycled Imports)
            surplus_spending = total_surplus_net * surplus_spend_rate + recycled_leakage * 1.0
            surplus_demand = distribute_demand(
                amount=surplus_spending,
                explicit_proportions=surplus_proportions,
                demand_distribution="proportional" if surplus_proportions is None else "custom",
                initial_demand_proportions=initial_proportions,
                domestic_indices=domestic_indices,
                n=n
            )
            
            # Bucket 3: Government Spending
            gov_spending = total_tax_rev * tax_spend_rate
            gov_demand = distribute_demand(
                amount=gov_spending,
                explicit_proportions=government_proportions,
                demand_distribution="proportional" if government_proportions is None else "custom",
                initial_demand_proportions=initial_proportions,
                domestic_indices=domestic_indices,
                n=n
            )
            
            total_spending = wage_spending + surplus_spending + gov_spending
            logger.info(f"Next-Iteration Spending: {total_spending:.2f} (Wage: {wage_spending:.2f}, Surplus: {surplus_spending:.2f}, Gov: {gov_spending:.2f})")
            
            current_demand = wage_demand + surplus_demand + gov_demand
            
    return history
