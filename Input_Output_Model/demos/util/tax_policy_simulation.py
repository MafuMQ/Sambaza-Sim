import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import logging
from Input_Output_Model.models.entities.SupplyCurve import SupplyCurveDatabase
from Input_Output_Model.models.entities.Production import ProductionsDatabase
from Input_Output_Model.models.table.solver import DynamicEquilibriumSolver 
from Input_Output_Model.util.Evaluators import build_io_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_va_component_matrix(isic_map):
    """
    Build matrices for each VA component from production_added_values in the database.
    
    Returns dictionaries mapping sector index to VA coefficient for each component:
    - minWages, bonusWages, wages (total), surplus
    
    These are monetary coefficients: $ of component per $ of output
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
        # Get all productions for this ISIC
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


def run_tax_policy_simulation(
    demand_vector=None,
    target_isic: str = None, 
    demand_shock: float = None,
    uniform_demand: float = None,
    total_demand: float = None,
    proportions: list = None,
    consumption_proportions: list = None,
    investment_proportions: list = None,
    government_proportions: list = None,
    income_tax_rate_before: float = 0.0,
    income_tax_rate_after: float = 0.0,
    corporate_tax_rate_before: float = 0.0,
    corporate_tax_rate_after: float = 0.0,
    income_tax_applies_to: str = "bonusWages",
    loggingLevel=logging.WARNING,
    iterations: int = 1,
    consumption_rate: float = 1.0,
    consumption_distribution: str = "proportional"
):
    """
    Simulate how changes in tax policy and demand affect income distribution.
    
    Uses ACTUAL VA components from database (minWages, bonusWages, wages, surplus)
    and applies specified tax rates to show:
    - How much government collects
    - How much workers take home
    - How much businesses keep
    
    DEMAND SPECIFICATION (provide ONE of the following):
    ---------------------------------------------------
    demand_vector : np.ndarray or list
        Direct specification of final demand for each sector (in dollars)
    
    target_isic + demand_shock : str, float
        Demand for a specific sector: e.g., target_isic="A2327_978_13", demand_shock=100
        All other sectors get zero demand
    
    demand_shock : float (alone)
        Demand for a randomly selected sector (in dollars)
        All other sectors get zero demand
    
    uniform_demand : float
        Apply the same demand value to ALL sectors (in dollars)
    
    total_demand + proportions : float, list
        Distribute total_demand across sectors according to proportions
        e.g., total_demand=1000, proportions=[0.3, 0.2, 0.2, 0.2, 0.1, 0.0]
        NOTE: This is only used for initial demand. For circular flow iterations,
        use consumption_proportions, investment_proportions, and government_proportions.
    
    consumption_proportions : list
        How to distribute consumption (C) across sectors during circular flow iterations.
        Must sum to 1.0. If not provided, falls back to consumption_distribution setting.
    
    investment_proportions : list
        How to distribute investment (I) across sectors during circular flow iterations.
        Must sum to 1.0. If not provided, falls back to consumption_distribution setting.
    
    government_proportions : list
        How to distribute government spending (G) across sectors during circular flow iterations.
        Must sum to 1.0. If not provided, falls back to consumption_distribution setting.
    
    TAX PARAMETERS:
    ---------------
    income_tax_rate_before : float
        Income tax rate BEFORE policy change (0.0 to 1.0)
    income_tax_rate_after : float
        Income tax rate AFTER policy change (0.0 to 1.0)
    corporate_tax_rate_before : float
        Corporate tax rate on surplus BEFORE policy change (0.0 to 1.0)
    corporate_tax_rate_after : float
        Corporate tax rate on surplus AFTER policy change (0.0 to 1.0)
    income_tax_applies_to : str
        Which wages to tax: "bonusWages" (default), "wages", or "both"
    iterations : int
        Number of iterations to run the circular flow model (default: 1)
        If > 1, after the first iteration, final demand is calculated from VA
    consumption_rate : float
        Fraction of after-tax income that is consumed/spent (0.0 to 1.0, default: 1.0)
        Applied to both wages and surplus when calculating new final demand
    consumption_distribution : str
        How to distribute consumption across sectors: "proportional" (default), "uniform", or "manual"
        - "proportional": distribute based on initial demand proportions
        - "uniform": distribute equally across all domestic sectors
        - "manual": use the proportions parameter for distribution
    """
    
    print("=" * 100)
    print("TAX POLICY IMPACT SIMULATION")
    if iterations > 1:
        print(f"(Circular Flow Model - {iterations} iterations)")
    print("=" * 100)
    print(f"\nTax Policy Configuration:")
    print(f"  Income Tax (BEFORE): {income_tax_rate_before*100:.1f}% on {income_tax_applies_to}")
    print(f"  Income Tax (AFTER):  {income_tax_rate_after*100:.1f}% on {income_tax_applies_to}")
    print(f"  Corporate Tax (BEFORE): {corporate_tax_rate_before*100:.1f}% on surplus")
    print(f"  Corporate Tax (AFTER):  {corporate_tax_rate_after*100:.1f}% on surplus")
    if iterations > 1:
        print(f"\nCircular Flow Parameters:")
        print(f"  Iterations: {iterations}")
        print(f"  Consumption Rate: {consumption_rate*100:.1f}%")
        print(f"  Distribution: {consumption_distribution}")
    print()
    
    # 1. Build the monetary coefficient matrix
    print("=" * 100)
    print("--- 1. BUILDING COEFFICIENT MATRICES ---")
    print("=" * 100)
    A_mon, VA_mon, isic_map = build_io_matrix(demoDB=False, loggingLevel=loggingLevel)
    
    print(f"\n[OK] IO Matrix Built. Sectors: {len(isic_map)}")
    
    # Build VA component matrices from actual database values
    print("\nExtracting Value Added components from database...")
    va_components = build_va_component_matrix(isic_map)
    
    # Scale VA components to match total VA from IO matrix
    # The database only has wages+surplus, but total VA = 1 - sum(inputs)
    # We need to scale the components proportionally to match VA_mon
    db_va_total = va_components['wages'] + va_components['surplus']
    scale_factors = np.divide(VA_mon, db_va_total, where=db_va_total!=0, out=np.ones_like(VA_mon))
    
    print("\nValue Added Component Coefficients ($ per $ of output):")
    print(f"{'Sector':<20} {'minWages':<12} {'bonusWages':<12} {'wages':<12} {'surplus':<12} {'Total VA':<12}")
    print("-" * 95)
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        print(f"[{idx}] {isic[:17]:<17} "
              f"{va_components['minWages'][idx]:>10.4f}  "
              f"{va_components['bonusWages'][idx]:>10.4f}  "
              f"{va_components['wages'][idx]:>10.4f}  "
              f"{va_components['surplus'][idx]:>10.4f}  "
              f"{VA_mon[idx]:>10.4f}")
    
    # 2. Load supply curves
    print("\n" + "=" * 100)
    print("--- 2. LOADING SUPPLY CURVES ---")
    print("=" * 100)
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
    
    print(f"[OK] Loaded supply curves for {len(supply_data)} sectors.")
    
    # 2b. Create demand vector based on user input
    print("\n" + "=" * 100)
    print("--- 3. CREATING DEMAND VECTOR ---")
    print("=" * 100)
    
    n_sectors = len(isic_map)
    
    # Store initial demand proportions for circular flow
    initial_demand_proportions = None
    
    # Determine demand based on input mode
    if demand_vector is not None:
        # Mode 1: Direct vector specification
        demand = np.array(demand_vector, dtype=float)
        if len(demand) != n_sectors:
            raise ValueError(f"Demand vector length {len(demand)} does not match number of sectors {n_sectors}")
        print(f"\nMode: Direct demand vector provided")
        print(f"Demand vector: {demand}")
        
    elif target_isic is not None and demand_shock is not None:
        # Mode 2: Specific sector demand
        if target_isic not in isic_map:
            raise ValueError(f"Target ISIC '{target_isic}' not found in sector map")
        demand = np.zeros(n_sectors)
        idx = isic_map[target_isic]
        demand[idx] = demand_shock
        print(f"\nMode: Sector-specific demand")
        print(f"Target: [{idx}] {target_isic}")
        print(f"Demand: ${demand_shock:,.2f}")
        
    elif demand_shock is not None:
        # Mode 3: Single random sector gets all demand
        import random
        domestic_isics = [isic for isic in isic_map.keys() if isic != "A9999_999_999"]
        if not domestic_isics:
            raise ValueError("No domestic sectors found")
        target_isic = random.choice(domestic_isics)
        demand = np.zeros(n_sectors)
        idx = isic_map[target_isic]
        demand[idx] = demand_shock
        print(f"\nMode: Random sector demand")
        print(f"Randomly selected: [{idx}] {target_isic}")
        print(f"Demand: ${demand_shock:,.2f}")
        
    elif uniform_demand is not None:
        # Mode 4: Uniform demand across all sectors
        demand = np.full(n_sectors, uniform_demand, dtype=float)
        print(f"\nMode: Uniform demand")
        print(f"Demand: ${uniform_demand:,.2f} for each sector")
        
    elif total_demand is not None and proportions is not None:
        # Mode 5: Proportional distribution
        proportions_array = np.array(proportions, dtype=float)
        if len(proportions_array) != n_sectors:
            raise ValueError(f"Proportions length {len(proportions_array)} does not match number of sectors {n_sectors}")
        if not np.isclose(proportions_array.sum(), 1.0):
            raise ValueError(f"Proportions must sum to 1.0, got {proportions_array.sum()}")
        demand = proportions_array * total_demand
        print(f"\nMode: Proportional distribution")
        print(f"Total demand: ${total_demand:,.2f}")
        print(f"Proportions: {proportions}")
        print(f"Resulting demand vector: {demand}")
        
    else:
        raise ValueError(
            "Must provide one of: demand_vector, (target_isic + demand_shock), "
            "demand_shock, uniform_demand, or (total_demand + proportions)"
        )
    
    # Display the demand vector
    print(f"\nDemand Vector:")
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        if isic != "A9999_999_999":
            print(f"  [{idx}] {isic[:20]:<20}: ${demand[idx]:>10.2f}")
    
    # Calculate initial demand proportions for circular flow
    domestic_indices = [idx for isic, idx in isic_map.items() if isic != "A9999_999_999"]
    total_domestic_demand = sum(demand[i] for i in domestic_indices)
    if total_domestic_demand > 0:
        initial_demand_proportions = np.zeros(n_sectors)
        for i in domestic_indices:
            initial_demand_proportions[i] = demand[i] / total_domestic_demand
    
    # 3. Run BEFORE scenario (with old tax rates)
    print("\n" + "=" * 100)
    if iterations > 1:
        print(f"--- 4. SCENARIO: BEFORE TAX CHANGE (Circular Flow - {iterations} iterations) ---")
    else:
        print("--- 4. SCENARIO: BEFORE TAX CHANGE ---")
    print("=" * 100)
    
    solver = DynamicEquilibriumSolver(A_mon, isic_map, supply_data)
    
    # Initialize for circular flow iterations
    current_demand = demand.copy()
    iteration_history = []
    
    # GDP Components tracking
    # For first iteration, initial demand is treated as mixed exogenous demand (I + G + X)
    # For subsequent iterations, we separate C, I, G, NX
    consumption_demand = np.zeros(n_sectors)
    investment_demand = demand.copy()  # Initial demand treated as investment/exogenous
    government_demand = np.zeros(n_sectors)
    exports = np.zeros(n_sectors)
    imports = np.zeros(n_sectors)
    
    for iteration in range(iterations):
        if iterations > 1:
            print(f"\n>>> Iteration {iteration + 1}/{iterations}")
            print(f"    Final Demand: ${current_demand.sum():>12,.2f}")
        else:
            print(f"\nRunning economy with old tax rates...")
        
        before_result = solver.solve(current_demand, verbose=False)
        
        if not before_result or before_result['status'] != 'converged':
            print(f"[ERROR] Iteration {iteration + 1} failed to converge!")
            break
        
        before_output = before_result['output']
        before_prices = before_result['prices']
        
        before_income = calculate_income_components(
            before_output, before_prices, va_components, VA_mon, scale_factors, isic_map
        )
        
        before_after_tax = apply_taxes(
            before_income,
            income_tax_rate_before,
            corporate_tax_rate_before,
            income_tax_applies_to
        )
        
        # Store iteration results
        iteration_history.append({
            'iteration': iteration + 1,
            'demand': current_demand.copy(),
            'consumption': consumption_demand.copy(),
            'investment': investment_demand.copy(),
            'government': government_demand.copy(),
            'exports': exports.copy(),
            'imports': imports.copy(),
            'output': before_output.copy(),
            'income': before_income.copy(),
            'after_tax': before_after_tax.copy()
        })
        
        if iterations > 1:
            total_income = (before_after_tax['wages_net'].sum() + 
                          before_after_tax['surplus_net'].sum())
            print(f"    Total After-Tax Income: ${total_income:,.2f}")
            print(f"    Total Tax Collected: ${before_after_tax['total_tax'].sum():,.2f}")
        
        # Calculate next iteration's demand from THIS iteration's VA (which becomes next period's C+I+G)
        if iteration < iterations - 1:  # Not the last iteration
            # Use GROSS (pre-tax) income from CURRENT iteration, because the full VA becomes next period's expenditure
            total_wages_gross = before_income['wages'].sum()
            total_surplus_gross = before_income['surplus'].sum()
            # This total should equal current_demand.sum() by the IO identity
            total_va = total_wages_gross + total_surplus_gross
            
            # Calculate how much tax will be collected when this VA is split
            # Apply tax rates to get the split
            if income_tax_applies_to == "bonusWages":
                income_tax = before_income['bonusWages'].sum() * income_tax_rate_before
            elif income_tax_applies_to == "wages":
                income_tax = before_income['wages'].sum() * income_tax_rate_before
            elif income_tax_applies_to == "both":
                income_tax = (before_income['minWages'].sum() + before_income['bonusWages'].sum()) * income_tax_rate_before
            else:
                income_tax = 0.0
            corporate_tax = before_income['surplus'].sum() * corporate_tax_rate_before
            total_tax = income_tax + corporate_tax
            
            # GDP IDENTITY: Y = C + I + G + (X - M)
            # Split the GROSS VA into C, I, G based on who gets it
            # CONSUMPTION (C): Wages AFTER income tax
            total_consumption = total_wages_gross - income_tax
            
            # INVESTMENT (I): Surplus AFTER corporate tax
            total_investment = total_surplus_gross - corporate_tax
            
            # GOVERNMENT SPENDING (G): ALL taxes collected
            total_government = total_tax
            
            # Verify: C + I + G should equal total VA from this period
            total_cig = total_consumption + total_investment + total_government
            assert abs(total_cig - total_va) < 0.01, f"C+I+G ({total_cig}) should equal VA ({total_va})"
            
            # NET EXPORTS (X - M): Zero for closed economy
            total_exports = 0.0
            total_imports = 0.0
            
            # Distribute consumption across sectors
            if consumption_proportions is not None:
                # Use explicit consumption proportions
                c_props = np.array(consumption_proportions, dtype=float)
                if len(c_props) != n_sectors:
                    raise ValueError(f"consumption_proportions length {len(c_props)} does not match sectors {n_sectors}")
                if not np.isclose(c_props.sum(), 1.0, atol=1e-6):
                    raise ValueError(f"consumption_proportions must sum to 1.0, got {c_props.sum()}")
                consumption_demand = c_props * total_consumption
            elif consumption_distribution == "proportional" and initial_demand_proportions is not None:
                # Use initial demand proportions
                consumption_demand = initial_demand_proportions * total_consumption
            elif consumption_distribution == "uniform":
                # Distribute equally across domestic sectors
                per_sector_c = total_consumption / len(domestic_indices)
                consumption_demand = np.zeros(n_sectors)
                for i in domestic_indices:
                    consumption_demand[i] = per_sector_c
            elif consumption_distribution == "manual" and proportions is not None:
                # Use provided proportions
                consumption_demand = np.array(proportions) * total_consumption
            else:
                # Default: proportional
                if initial_demand_proportions is not None:
                    consumption_demand = initial_demand_proportions * total_consumption
                else:
                    # Fallback to uniform
                    per_sector_c = total_consumption / len(domestic_indices)
                    consumption_demand = np.zeros(n_sectors)
                    for i in domestic_indices:
                        consumption_demand[i] = per_sector_c
            
            # Distribute investment across sectors
            if investment_proportions is not None:
                # Use explicit investment proportions
                i_props = np.array(investment_proportions, dtype=float)
                if len(i_props) != n_sectors:
                    raise ValueError(f"investment_proportions length {len(i_props)} does not match sectors {n_sectors}")
                if not np.isclose(i_props.sum(), 1.0, atol=1e-6):
                    raise ValueError(f"investment_proportions must sum to 1.0, got {i_props.sum()}")
                investment_demand = i_props * total_investment
            elif consumption_distribution == "proportional" and initial_demand_proportions is not None:
                investment_demand = initial_demand_proportions * total_investment
            elif consumption_distribution == "uniform":
                per_sector_i = total_investment / len(domestic_indices)
                investment_demand = np.zeros(n_sectors)
                for i in domestic_indices:
                    investment_demand[i] = per_sector_i
            elif consumption_distribution == "manual" and proportions is not None:
                investment_demand = np.array(proportions) * total_investment
            else:
                if initial_demand_proportions is not None:
                    investment_demand = initial_demand_proportions * total_investment
                else:
                    per_sector_i = total_investment / len(domestic_indices)
                    investment_demand = np.zeros(n_sectors)
                    for i in domestic_indices:
                        investment_demand[i] = per_sector_i
            
            # Distribute government spending across sectors
            if government_proportions is not None:
                # Use explicit government proportions
                g_props = np.array(government_proportions, dtype=float)
                if len(g_props) != n_sectors:
                    raise ValueError(f"government_proportions length {len(g_props)} does not match sectors {n_sectors}")
                if not np.isclose(g_props.sum(), 1.0, atol=1e-6):
                    raise ValueError(f"government_proportions must sum to 1.0, got {g_props.sum()}")
                government_demand = g_props * total_government
            elif consumption_distribution == "proportional" and initial_demand_proportions is not None:
                government_demand = initial_demand_proportions * total_government
            elif consumption_distribution == "uniform":
                per_sector_g = total_government / len(domestic_indices)
                government_demand = np.zeros(n_sectors)
                for i in domestic_indices:
                    government_demand[i] = per_sector_g
            elif consumption_distribution == "manual" and proportions is not None:
                government_demand = np.array(proportions) * total_government
            else:
                if initial_demand_proportions is not None:
                    government_demand = initial_demand_proportions * total_government
                else:
                    per_sector_g = total_government / len(domestic_indices)
                    government_demand = np.zeros(n_sectors)
                    for i in domestic_indices:
                        government_demand[i] = per_sector_g
            
            # Exports and Imports (closed economy - no external trade)
            exports = np.zeros(n_sectors)
            imports = np.zeros(n_sectors)
            
            # Total Final Demand = C + I + G + (X - M)
            # Since X = M = 0, this simplifies to: FD = C + I + G
            current_demand = consumption_demand + investment_demand + government_demand
    
    # Use the last iteration's results
    before_income = iteration_history[-1]['income']
    before_after_tax = iteration_history[-1]['after_tax']
    before_output = iteration_history[-1]['output']
    
    if iterations > 1:
        print(f"\n--- Summary After {iterations} Iterations ---")
    print_income_summary("Economy with Old Tax Rates", before_income, before_after_tax)
    
    if iterations > 1:
        # Show GDP breakdown evolution
        print("\n--- GDP Expenditure Evolution Across Iterations ---")
        print(f"{'Iter':<6} {'C':<15} {'I':<15} {'G':<15} {'X':<15} {'M':<15} {'FD (C+I+G+NX)':<15}")
        print("-" * 100)
        for hist in iteration_history:
            c = hist['consumption'].sum()
            i = hist['investment'].sum()
            g = hist['government'].sum()
            x = hist['exports'].sum()
            m = hist['imports'].sum()
            fd = hist['demand'].sum()
            print(f"{hist['iteration']:<6} "
                  f"${c:>13,.2f}  "
                  f"${i:>13,.2f}  "
                  f"${g:>13,.2f}  "
                  f"${x:>13,.2f}  "
                  f"${m:>13,.2f}  "
                  f"${fd:>13,.2f}")
        
        # Show traditional summary
        print("\n--- Income & Output Evolution Across Iterations ---")
        print(f"{'Iter':<6} {'Wages (net)':<15} {'Surplus (net)':<15} {'Total Tax':<15} {'Total FD':<15} {'Total Output':<15} {'Total VA (=FD)':<15}")
        print("-" * 105)
        for hist in iteration_history:
            # VA should equal FD (GDP identity)
            total_va = hist['demand'].sum()  # By definition in IO model
            print(f"{hist['iteration']:<6} "
                  f"${hist['after_tax']['wages_net'].sum():>13,.2f}  "
                  f"${hist['after_tax']['surplus_net'].sum():>13,.2f}  "
                  f"${hist['after_tax']['total_tax'].sum():>13,.2f}  "
                  f"${hist['demand'].sum():>13,.2f}  "
                  f"${hist['output'].sum():>13,.2f}  "
                  f"${total_va:>13,.2f}")
    
    # 4. Run AFTER scenario with new tax rates
    print("\n" + "=" * 100)
    if iterations > 1:
        print(f"--- 5. SCENARIO: AFTER TAX CHANGE (Circular Flow - {iterations} iterations) ---")
    else:
        print("--- 5. SCENARIO: AFTER TAX CHANGE ---")
    print("=" * 100)
    
    if iterations == 1:
        print(f"\nApplying NEW tax rates to same demand...")
        
        after_income = before_income.copy()  # Same pre-tax income
        
        after_after_tax = apply_taxes(
            after_income,
            income_tax_rate_after,
            corporate_tax_rate_after,
            income_tax_applies_to
        )
    else:
        # Run circular flow with new tax rates
        current_demand = demand.copy()
        iteration_history_new = []
        
        # GDP Components tracking for new tax scenario
        consumption_demand_new = np.zeros(n_sectors)
        investment_demand_new = demand.copy()
        government_demand_new = np.zeros(n_sectors)
        exports_new = np.zeros(n_sectors)
        imports_new = np.zeros(n_sectors)
        
        for iteration in range(iterations):
            print(f"\n>>> Iteration {iteration + 1}/{iterations}")
            print(f"    Final Demand: ${current_demand.sum():>12,.2f}")
            
            after_result = solver.solve(current_demand, verbose=False)
            
            if not after_result or after_result['status'] != 'converged':
                print(f"[ERROR] Iteration {iteration + 1} failed to converge!")
                break
            
            after_output = after_result['output']
            after_prices = after_result['prices']
            
            after_income = calculate_income_components(
                after_output, after_prices, va_components, VA_mon, scale_factors, isic_map
            )
            
            after_after_tax = apply_taxes(
                after_income,
                income_tax_rate_after,
                corporate_tax_rate_after,
                income_tax_applies_to
            )
            
            # Store iteration results
            iteration_history_new.append({
                'iteration': iteration + 1,
                'demand': current_demand.copy(),
                'consumption': consumption_demand_new.copy(),
                'investment': investment_demand_new.copy(),
                'government': government_demand_new.copy(),
                'exports': exports_new.copy(),
                'imports': imports_new.copy(),
                'output': after_output.copy(),
                'income': after_income.copy(),
                'after_tax': after_after_tax.copy()
            })
            
            total_income = (after_after_tax['wages_net'].sum() + 
                          after_after_tax['surplus_net'].sum())
            print(f"    Total After-Tax Income: ${total_income:,.2f}")
            print(f"    Total Tax Collected: ${after_after_tax['total_tax'].sum():,.2f}")
            
            # Calculate next iteration's demand
            if iteration < iterations - 1:
                # Use GROSS (pre-tax) income from CURRENT iteration
                total_wages_gross = after_income['wages'].sum()
                total_surplus_gross = after_income['surplus'].sum()
                total_va = total_wages_gross + total_surplus_gross
                
                # Calculate taxes
                if income_tax_applies_to == "bonusWages":
                    income_tax = after_income['bonusWages'].sum() * income_tax_rate_after
                elif income_tax_applies_to == "wages":
                    income_tax = after_income['wages'].sum() * income_tax_rate_after
                elif income_tax_applies_to == "both":
                    income_tax = (after_income['minWages'].sum() + after_income['bonusWages'].sum()) * income_tax_rate_after
                else:
                    income_tax = 0.0
                corporate_tax = after_income['surplus'].sum() * corporate_tax_rate_after
                total_tax = income_tax + corporate_tax
                
                # GDP IDENTITY: Y = C + I + G + (X - M)
                # CONSUMPTION (C): Wages after income tax
                total_consumption = total_wages_gross - income_tax
                
                # INVESTMENT (I): Surplus after corporate tax
                total_investment = total_surplus_gross - corporate_tax
                
                # GOVERNMENT SPENDING (G): ALL taxes collected
                total_government = total_tax
                
                # Verify
                total_cig = total_consumption + total_investment + total_government
                assert abs(total_cig - total_va) < 0.01, f"C+I+G ({total_cig}) should equal VA ({total_va})"
                
                # NET EXPORTS: Zero for closed economy
                total_exports = 0.0
                total_imports = 0.0
                
                if consumption_proportions is not None:
                    # Use explicit consumption proportions
                    c_props = np.array(consumption_proportions, dtype=float)
                    if len(c_props) != n_sectors:
                        raise ValueError(f"consumption_proportions length {len(c_props)} does not match sectors {n_sectors}")
                    if not np.isclose(c_props.sum(), 1.0, atol=1e-6):
                        raise ValueError(f"consumption_proportions must sum to 1.0, got {c_props.sum()}")
                    consumption_demand_new = c_props * total_consumption
                elif consumption_distribution == "proportional" and initial_demand_proportions is not None:
                    consumption_demand_new = initial_demand_proportions * total_consumption
                elif consumption_distribution == "uniform":
                    per_sector_c = total_consumption / len(domestic_indices)
                    consumption_demand_new = np.zeros(n_sectors)
                    for i in domestic_indices:
                        consumption_demand_new[i] = per_sector_c
                elif consumption_distribution == "manual" and proportions is not None:
                    consumption_demand_new = np.array(proportions) * total_consumption
                else:
                    if initial_demand_proportions is not None:
                        consumption_demand_new = initial_demand_proportions * total_consumption
                    else:
                        per_sector_c = total_consumption / len(domestic_indices)
                        consumption_demand_new = np.zeros(n_sectors)
                        for i in domestic_indices:
                            consumption_demand_new[i] = per_sector_c
                
                # Distribute investment across sectors
                if investment_proportions is not None:
                    # Use explicit investment proportions
                    i_props = np.array(investment_proportions, dtype=float)
                    if len(i_props) != n_sectors:
                        raise ValueError(f"investment_proportions length {len(i_props)} does not match sectors {n_sectors}")
                    if not np.isclose(i_props.sum(), 1.0, atol=1e-6):
                        raise ValueError(f"investment_proportions must sum to 1.0, got {i_props.sum()}")
                    investment_demand_new = i_props * total_investment
                elif consumption_distribution == "proportional" and initial_demand_proportions is not None:
                    investment_demand_new = initial_demand_proportions * total_investment
                elif consumption_distribution == "uniform":
                    per_sector_i = total_investment / len(domestic_indices)
                    investment_demand_new = np.zeros(n_sectors)
                    for i in domestic_indices:
                        investment_demand_new[i] = per_sector_i
                elif consumption_distribution == "manual" and proportions is not None:
                    investment_demand_new = np.array(proportions) * total_investment
                else:
                    if initial_demand_proportions is not None:
                        investment_demand_new = initial_demand_proportions * total_investment
                    else:
                        per_sector_i = total_investment / len(domestic_indices)
                        investment_demand_new = np.zeros(n_sectors)
                        for i in domestic_indices:
                            investment_demand_new[i] = per_sector_i
                
                # Distribute government spending across sectors
                if government_proportions is not None:
                    # Use explicit government proportions
                    g_props = np.array(government_proportions, dtype=float)
                    if len(g_props) != n_sectors:
                        raise ValueError(f"government_proportions length {len(g_props)} does not match sectors {n_sectors}")
                    if not np.isclose(g_props.sum(), 1.0, atol=1e-6):
                        raise ValueError(f"government_proportions must sum to 1.0, got {g_props.sum()}")
                    government_demand_new = g_props * total_government
                elif consumption_distribution == "proportional" and initial_demand_proportions is not None:
                    government_demand_new = initial_demand_proportions * total_government
                elif consumption_distribution == "uniform":
                    per_sector_g = total_government / len(domestic_indices)
                    government_demand_new = np.zeros(n_sectors)
                    for i in domestic_indices:
                        government_demand_new[i] = per_sector_g
                elif consumption_distribution == "manual" and proportions is not None:
                    government_demand_new = np.array(proportions) * total_government
                else:
                    if initial_demand_proportions is not None:
                        government_demand_new = initial_demand_proportions * total_government
                    else:
                        per_sector_g = total_government / len(domestic_indices)
                        government_demand_new = np.zeros(n_sectors)
                        for i in domestic_indices:
                            government_demand_new[i] = per_sector_g
                
                exports_new = np.zeros(n_sectors)
                imports_new = np.zeros(n_sectors)
                
                # Total Final Demand = C + I + G (closed economy, no external trade)
                current_demand = consumption_demand_new + investment_demand_new + government_demand_new
        
        # Use the last iteration's results
        after_income = iteration_history_new[-1]['income']
        after_after_tax = iteration_history_new[-1]['after_tax']
        after_output = iteration_history_new[-1]['output']
        
        print(f"\n--- Summary After {iterations} Iterations ---")
        print_income_summary("Economy with New Tax Rates", after_income, after_after_tax)
        
        # Show GDP breakdown evolution
        print("\n--- GDP Expenditure Evolution Across Iterations ---")
        print(f"{'Iter':<6} {'C':<15} {'I':<15} {'G':<15} {'X':<15} {'M':<15} {'FD (C+I+G+NX)':<15}")
        print("-" * 100)
        for hist in iteration_history_new:
            c = hist['consumption'].sum()
            i = hist['investment'].sum()
            g = hist['government'].sum()
            x = hist['exports'].sum()
            m = hist['imports'].sum()
            fd = hist['demand'].sum()
            print(f"{hist['iteration']:<6} "
                  f"${c:>13,.2f}  "
                  f"${i:>13,.2f}  "
                  f"${g:>13,.2f}  "
                  f"${x:>13,.2f}  "
                  f"${m:>13,.2f}  "
                  f"${fd:>13,.2f}")
        
        # Show iteration evolution
        print("\n--- Income & Output Evolution Across Iterations ---")
        print(f"{'Iter':<6} {'Wages (net)':<15} {'Surplus (net)':<15} {'Total Tax':<15} {'Total FD':<15} {'Total Output':<15} {'Total VA (=FD)':<15}")
        print("-" * 105)
        for hist in iteration_history_new:
            # VA should equal FD (GDP identity)
            total_va = hist['demand'].sum()  # By definition in IO model
            print(f"{hist['iteration']:<6} "
                  f"${hist['after_tax']['wages_net'].sum():>13,.2f}  "
                  f"${hist['after_tax']['surplus_net'].sum():>13,.2f}  "
                  f"${hist['after_tax']['total_tax'].sum():>13,.2f}  "
                  f"${hist['demand'].sum():>13,.2f}  "
                  f"${hist['output'].sum():>13,.2f}  "
                  f"${total_va:>13,.2f}")
    
    if iterations == 1:
        print_income_summary("Economy with New Tax Rates", after_income, after_after_tax)
    
    # 5. Compare all scenarios
    print("\n" + "=" * 100)
    print("--- 6. COMPARATIVE ANALYSIS ---")
    print("=" * 100)
    
    print("\n" + "="*100)
    print("IMPACT OF TAX POLICY CHANGE")
    print("="*100)
    
    compare_scenarios(
        "Before (Old Taxes)",
        before_after_tax,
        "After (New Taxes)",
        after_after_tax
    )
    
    # 6. Sector breakdown
    print("\n" + "=" * 100)
    print("--- 7. SECTOR-BY-SECTOR BREAKDOWN (Tax Policy Impact) ---")
    print("=" * 100)
    
    # First show absolute values
    print("\nAbsolute Values (Before and After):")
    print(f"\n{'Sector':<20} {'Output':<12} {'MinWage':<12} {'Bonus':<12} {'Profit':<12} {'IncomeTax':<12} {'CorpTax':<12}")
    print("-" * 110)
    print("BEFORE TAX CHANGE:")
    for isic, i in sorted(isic_map.items(), key=lambda x: x[1]):
        print(f"[{i}] {isic[:17]:<17} "
              f"${before_output[i]:>10.2f}  "
              f"${before_after_tax['minWages_net'][i]:>10.2f}  "
              f"${before_after_tax['bonusWages_net'][i]:>10.2f}  "
              f"${before_after_tax['surplus_net'][i]:>10.2f}  "
              f"${before_after_tax['income_tax'][i]:>10.2f}  "
              f"${before_after_tax['corporate_tax'][i]:>10.2f}")
    print("-" * 110)
    print(f"{'TOTAL':<20} "
          f"${before_output.sum():>10.2f}  "
          f"${before_after_tax['minWages_net'].sum():>10.2f}  "
          f"${before_after_tax['bonusWages_net'].sum():>10.2f}  "
          f"${before_after_tax['surplus_net'].sum():>10.2f}  "
          f"${before_after_tax['income_tax'].sum():>10.2f}  "
          f"${before_after_tax['corporate_tax'].sum():>10.2f}")
    
    print("\nAFTER TAX CHANGE:")
    for isic, i in sorted(isic_map.items(), key=lambda x: x[1]):
        print(f"[{i}] {isic[:17]:<17} "
              f"${after_output[i]:>10.2f}  "
              f"${after_after_tax['minWages_net'][i]:>10.2f}  "
              f"${after_after_tax['bonusWages_net'][i]:>10.2f}  "
              f"${after_after_tax['surplus_net'][i]:>10.2f}  "
              f"${after_after_tax['income_tax'][i]:>10.2f}  "
              f"${after_after_tax['corporate_tax'][i]:>10.2f}")
    print("-" * 110)
    print(f"{'TOTAL':<20} "
          f"${after_output.sum():>10.2f}  "
          f"${after_after_tax['minWages_net'].sum():>10.2f}  "
          f"${after_after_tax['bonusWages_net'].sum():>10.2f}  "
          f"${after_after_tax['surplus_net'].sum():>10.2f}  "
          f"${after_after_tax['income_tax'].sum():>10.2f}  "
          f"${after_after_tax['corporate_tax'].sum():>10.2f}")
    
    # Then show deltas
    print("\nChanges (Δ = After - Before):")
    
    delta_output = after_output - before_output
    delta_minwages = after_after_tax['minWages_net'] - before_after_tax['minWages_net']
    delta_bonus = after_after_tax['bonusWages_net'] - before_after_tax['bonusWages_net']
    delta_surplus_net = after_after_tax['surplus_net'] - before_after_tax['surplus_net']
    delta_income_tax = after_after_tax['income_tax'] - before_after_tax['income_tax']
    delta_corp_tax = after_after_tax['corporate_tax'] - before_after_tax['corporate_tax']
    
    print(f"\n{'Sector':<20} {'ΔOutput':<12} {'ΔMinWage':<12} {'ΔBonus':<12} {'ΔProfit':<12} {'ΔIncomeTax':<12} {'ΔCorpTax':<12}")
    print("-" * 110)
    
    for isic, i in sorted(isic_map.items(), key=lambda x: x[1]):
        print(f"[{i}] {isic[:17]:<17} "
              f"{delta_output[i]:>+10.2f}  "
              f"${delta_minwages[i]:>+10.2f}  "
              f"${delta_bonus[i]:>+10.2f}  "
              f"${delta_surplus_net[i]:>+10.2f}  "
              f"${delta_income_tax[i]:>+10.2f}  "
              f"${delta_corp_tax[i]:>+10.2f}")
    
    print("-" * 110)
    print(f"{'TOTAL':<20} "
          f"{delta_output.sum():>+10.2f}  "
          f"${delta_minwages.sum():>+10.2f}  "
          f"${delta_bonus.sum():>+10.2f}  "
          f"${delta_surplus_net.sum():>+10.2f}  "
          f"${delta_income_tax.sum():>+10.2f}  "
          f"${delta_corp_tax.sum():>+10.2f}")
    
    print("\n" + "=" * 100)
    print("SIMULATION COMPLETE")
    print("=" * 100)
    
    return {
        'before': before_after_tax,
        'after': after_after_tax,
        'isic_map': isic_map
    }


def calculate_income_components(output, prices, va_components, VA_mon, scale_factors, isic_map):
    """
    Calculate monetary income for each VA component by sector.
    
    IMPORTANT: The solver returns 'output' as MONETARY VALUE (dollars) because 
    we use a monetary coefficient matrix A_mon. Therefore, output IS already 
    in currency units. We do NOT multiply by prices again (that would be "dollars squared").
    
    Uses VA_mon (correct total VA from IO matrix) and scales the database components
    proportionally to match. This ensures VA = FD identity is preserved.
    """
    # output is already in dollars from the monetary IO solver
    output_value = output
    
    # Calculate total VA using correct coefficients from IO matrix
    # Then split into components based on database proportions (scaled)
    return {
        'minWages': output_value * va_components['minWages'] * scale_factors,
        'bonusWages': output_value * va_components['bonusWages'] * scale_factors,
        'wages': output_value * va_components['wages'] * scale_factors,
        'surplus': output_value * va_components['surplus'] * scale_factors
    }


def apply_taxes(income, income_tax_rate, corporate_tax_rate, income_tax_applies_to):
    """
    Apply taxes to income components.
    
    Income tax reduces worker take-home pay.
    Corporate tax reduces business profits.
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
        # Proportionally reduce both components
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


def print_income_summary(title, pre_tax, after_tax):
    """Print a formatted summary of income distribution."""
    print(f"\n{title}:")
    print(f"  PRE-TAX Income:")
    print(f"    Min Wages:     ${pre_tax['minWages'].sum():>12,.2f}")
    print(f"    Bonus Wages:   ${pre_tax['bonusWages'].sum():>12,.2f}")
    print(f"    Total Wages:   ${pre_tax['wages'].sum():>12,.2f}")
    print(f"    Surplus:       ${pre_tax['surplus'].sum():>12,.2f}")
    print(f"    TOTAL VA:      ${(pre_tax['wages'].sum() + pre_tax['surplus'].sum()):>12,.2f}")
    
    print(f"\n  AFTER-TAX Income:")
    print(f"    Min Wages:     ${after_tax['minWages_net'].sum():>12,.2f}")
    print(f"    Bonus Wages:   ${after_tax['bonusWages_net'].sum():>12,.2f}")
    print(f"    Total Wages:   ${after_tax['wages_net'].sum():>12,.2f}")
    print(f"    Surplus (net): ${after_tax['surplus_net'].sum():>12,.2f}")
    
    print(f"\n  TAX REVENUE:")
    print(f"    Income Tax:    ${after_tax['income_tax'].sum():>12,.2f}")
    print(f"    Corporate Tax: ${after_tax['corporate_tax'].sum():>12,.2f}")
    print(f"    TOTAL TAX:     ${after_tax['total_tax'].sum():>12,.2f}")


def compare_scenarios(name1, data1, name2, data2):
    """Compare two scenarios and show changes."""
    print(f"\nChanges from '{name1}' to '{name2}':")
    
    delta_minwages = data2['minWages_net'].sum() - data1['minWages_net'].sum()
    delta_bonus = data2['bonusWages_net'].sum() - data1['bonusWages_net'].sum()
    delta_wages = data2['wages_net'].sum() - data1['wages_net'].sum()
    delta_surplus = data2['surplus_net'].sum() - data1['surplus_net'].sum()
    delta_income_tax = data2['income_tax'].sum() - data1['income_tax'].sum()
    delta_corp_tax = data2['corporate_tax'].sum() - data1['corporate_tax'].sum()
    delta_total_tax = data2['total_tax'].sum() - data1['total_tax'].sum()
    
    print(f"  Min Wages (net):      ${delta_minwages:>+12,.2f}")
    print(f"  Bonus Wages (net):    ${delta_bonus:>+12,.2f}")
    print(f"  Total Wages (net):    ${delta_wages:>+12,.2f}")
    print(f"  Surplus (net):        ${delta_surplus:>+12,.2f}")
    print(f"  Income Tax Revenue:   ${delta_income_tax:>+12,.2f}")
    print(f"  Corporate Tax Revenue:${delta_corp_tax:>+12,.2f}")
    print(f"  TOTAL Tax Revenue:    ${delta_total_tax:>+12,.2f}")
    
    # Winners and losers
    print(f"\n  Distribution:")
    if delta_wages > 0:
        print(f"    ✓ Workers gain ${delta_wages:,.2f}")
    elif delta_wages < 0:
        print(f"    ✗ Workers lose ${-delta_wages:,.2f}")
    
    if delta_surplus > 0:
        print(f"    ✓ Businesses gain ${delta_surplus:,.2f}")
    elif delta_surplus < 0:
        print(f"    ✗ Businesses lose ${-delta_surplus:,.2f}")
    
    if delta_total_tax > 0:
        print(f"    ✓ Government gains ${delta_total_tax:,.2f}")
    elif delta_total_tax < 0:
        print(f"    ✗ Government loses ${-delta_total_tax:,.2f}")
