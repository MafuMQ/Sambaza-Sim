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
    gov_target_sector: str = None,
    demand_distribution: str = "proportional",
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
    
    # Initial demand distribution proportions for "proportional" mapping
    base_demand_sum = base_demand.sum()
    if base_demand_sum > 0:
        initial_proportions = base_demand / base_demand_sum
    else:
        initial_proportions = np.full(n, 1.0 / n)
        
    domestic_indices = [i for i, isic in enumerate(isic_map) if isic != "A9999_999_999"]

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
        
        logger.info(f"--- Iteration {i+1} ---")
        logger.info(f"Output (X): {X.sum():.2f}, Total Demand (Y): {current_demand.sum():.2f}")
        logger.info(f"Net Wages: {total_wage_net:.2f}, Net Surplus: {total_surplus_net:.2f}, Taxes: {total_tax_rev:.2f}")
        logger.info(f"Unallocated VA: {unallocated_va:.2f}")
        
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
            # Distribute Consumer + Unallocated Spending proportionally
            consumer_spending = (
                total_wage_net * wage_spend_rate +
                total_surplus_net * surplus_spend_rate +
                unallocated_va * 1.0  # Prevent demand leakage
            )
            
            consumer_demand = distribute_demand(
                amount=consumer_spending,
                explicit_proportions=None,
                demand_distribution=demand_distribution,
                initial_demand_proportions=initial_proportions,
                domestic_indices=domestic_indices,
                n=n
            )
            
            # Distribute Government Spending
            gov_spending = total_tax_rev * tax_spend_rate
            gov_demand = np.zeros(n)
            
            if gov_target_sector and gov_target_sector in isic_map:
                gov_demand[isic_map[gov_target_sector]] = gov_spending
            else:
                gov_demand = distribute_demand(
                    amount=gov_spending,
                    explicit_proportions=None,
                    demand_distribution=demand_distribution,
                    initial_demand_proportions=initial_proportions,
                    domestic_indices=domestic_indices,
                    n=n
                )
            
            total_spending = consumer_spending + gov_spending
            logger.info(f"Next-Iteration Spending: {total_spending:.2f} (Consumer: {consumer_spending:.2f}, Gov: {gov_spending:.2f})")
            
            current_demand = consumer_demand + gov_demand
            
    return history
