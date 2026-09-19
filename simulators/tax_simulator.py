import numpy as np
from typing import TYPE_CHECKING, Dict, List, Any, Optional

from core.io_matrix import IOModel
from core.income import build_va_component_matrix, apply_taxes
from core.demand import distribute_demand

import logging

if TYPE_CHECKING:
    from simulators.savings_ledger import SavingsLedger
    from simulators.tech_change import TechnologicalChange

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
    ledger: Optional['SavingsLedger'] = None,
    tech_change: Optional['TechnologicalChange'] = None,
    investments: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Run an iterative circular-flow tax simulation with optional multi-investment technological change.

    Each investment in ``investments`` has its own monetary requirement (capital_cost)
    and investment phase duration (iterations):
      - While iteration i < duration, the investment is in construction: its per-period
        capital demand is injected into domestic sectors, and its new technology is NOT active.
      - Once iteration i >= duration (or if capital_cost == 0), the investment phase is
        complete and its technological change activates.

    For backward compatibility, ``tech_change`` is also accepted as a single investment.
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

    # Normalize single tech_change into investments list if provided
    if investments is None and tech_change is not None:
        has_cap = tech_change.has_capital_requirements()
        investments = [{
            'name': tech_change.name,
            'tech_change': tech_change,
            'capital_cost': tech_change.get_total_capital_cost() if has_cap else 0.0,
            'duration': tech_change.investment_duration if has_cap else 0,
            'requirements': tech_change.capital_requirements if has_cap else None,
            'status': 'Active',
        }]

    current_demand = base_demand.copy()

    for i in range(iterations):
        # 1. Compute capital demand injections for iteration i
        injected_capital = np.zeros(n)
        if investments:
            for inv in investments:
                cost = float(inv.get('capital_cost', 0.0))
                dur = max(1, int(inv.get('duration', 1)))
                if cost > 0 and i < dur:
                    per_period = cost / dur
                    reqs = inv.get('requirements')
                    if reqs and isinstance(reqs, dict):
                        for sec_id, sec_amt in reqs.items():
                            s_idx = isic_map.get(sec_id) if isinstance(sec_id, str) else sec_id
                            if s_idx is not None and 0 <= s_idx < n:
                                injected_capital[s_idx] += (float(sec_amt) / dur)
                    elif domestic_indices:
                        per_sector = per_period / len(domestic_indices)
                        for d_idx in domestic_indices:
                            injected_capital[d_idx] += per_sector

        in_investment_phase = bool(injected_capital.sum() > 0)
        effective_demand = current_demand + injected_capital

        # 2. Determine active technology for iteration i
        active_A = model.A.copy()
        active_VA = model.VA_coeffs.copy()
        if investments:
            for inv in investments:
                cost = float(inv.get('capital_cost', 0.0))
                dur = int(inv.get('duration', 0))
                if cost == 0.0 or i >= dur:
                    t_obj = inv.get('tech_change')
                    if t_obj is not None:
                        active_A, active_VA = t_obj.apply(active_A, active_VA, isic_map)

        active_model = IOModel(A=active_A, VA_coeffs=active_VA)

        # 3. Output Calculation (Leontief)
        X = active_model.simulate(effective_demand)

        
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
        total_va = (active_model.VA_coeffs * X).sum()
        unallocated_va = total_va - (gross_income["wages"].sum() + gross_income["surplus"].sum())
        
        # Import Leakage Handling
        import_leakage = 0.0
        if import_idx is not None:
            import_leakage = X[import_idx]
            
        recycled_leakage = 0.0
        if economy_type == "closed":
            recycled_leakage = import_leakage
        
        logger.info(f"--- Iteration {i+1} [{'Investment Phase' if in_investment_phase else 'Operational Phase'}] ---")
        logger.info(f"Output (X): {X.sum():.2f}, Total Demand (Y): {effective_demand.sum():.2f}")
        logger.info(f"Net Wages: {total_wage_net:.2f}, Net Surplus: {total_surplus_net:.2f}, Taxes: {total_tax_rev:.2f}")
        logger.info(f"Unallocated VA: {unallocated_va:.2f}, Import Leakage: {import_leakage:.2f} (Recycled: {recycled_leakage:.2f})")

        # Deposit unspent VA income into the savings ledger (if attached).
        # Unspent = the fraction of net income not recirculated as demand.
        # This represents household / corporate savings that can later
        # finance capital investment via set_capital_requirements.
        if ledger is not None:
            savings_this_period = (
                total_wage_net * (1.0 - wage_spend_rate)
                + total_surplus_net * (1.0 - surplus_spend_rate)
            )
            if savings_this_period > 0:
                ledger.deposit(savings_this_period, source_label="va")
                logger.info(
                    f"Iteration {i+1}: deposited {savings_this_period:.2f} into savings ledger "
                    f"(balance now: {ledger.balance:.2f})"
                )

        # Save snapshot
        snapshot = {
            "iteration": i + 1,
            "phase": "investment" if in_investment_phase else "operational",
            "capital_injected": injected_capital.copy(),
            "capital_demand_total": float(injected_capital.sum()),
            "Y": effective_demand.copy(),
            "base_circular_demand": current_demand.copy(),
            "X": X.copy(),
            "gross_income": gross_income,
            "tax_results": tax_results,
            "total_demand": float(effective_demand.sum()),
            "total_output": float(X.sum()),
            "model": active_model,
            "A_matrix": active_A.copy()
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
