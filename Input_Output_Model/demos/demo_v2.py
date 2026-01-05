import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import logging
from Input_Output_Model.models.entities.SupplyCurve import SupplyCurveDatabase
from Input_Output_Model.demos.util.Setup_Data import setup_random_sample_data
from Input_Output_Model.models.table.solver import DynamicEquilibriumSolver 
from Input_Output_Model.util.Evaluators import build_io_matrix  # The "Direct Matrix Builder" function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_economic_simulation(target_isic: str, demand_increase: float):
    print("=" * 80)
    print("--- 1. BUILDING MONETARY COEFFICIENT MATRIX ---")
    print("=" * 80)
    print()
    # This generates the monetary coefficient matrix from production recipes
    # Database stores physical quantities, but costs are already calculated in monetary terms
    A_mon, VA_mon, isic_map = build_io_matrix(demoDB=False)

    print(f"\n[OK] Matrix Built. Sectors: {len(isic_map)}")
    print("\n--- MONETARY COEFFICIENT MATRIX (A_mon) ---")
    print("Each cell A[i,j] = dollars of input i per dollar of output j")
    print("\nISIC Mapping:")
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        print(f"  [{idx}] {isic}")
    print("\nA_mon Matrix:")
    print(A_mon)
    
    # Verify matrix properties
    print("\n--- MATRIX VERIFICATION ---")
    print(f"Shape: {A_mon.shape}")
    print(f"Min coefficient: {A_mon.min():.6f}")
    print(f"Max coefficient: {A_mon.max():.6f}")
    print(f"Average coefficient: {A_mon.mean():.6f}")
    print(f"Sparsity: {(A_mon == 0).sum() / A_mon.size * 100:.1f}% zeros")
    
    # Check column sums (total monetary inputs per dollar output)
    col_sums = A_mon.sum(axis=0)
    print("\nColumn sums (total $ of inputs per $ of output):")
    for idx, sum_val in enumerate(col_sums):
        isic_name = [k for k, v in isic_map.items() if v == idx][0]
        print(f"  [{idx}] {isic_name}: {sum_val:.4f}")
    
    print("\n--- VALUE ADDED COEFFICIENTS ---")
    print("VA per dollar of output:")
    for idx, va in enumerate(VA_mon):
        isic_name = [k for k, v in isic_map.items() if v == idx][0]
        print(f"  [{idx}] {isic_name}: {va:.4f}")
    
    # Total monetary requirements per dollar
    print("\n--- TOTAL MONETARY COST STRUCTURE ---")
    total_costs = col_sums + VA_mon
    for idx, cost in enumerate(total_costs):
        isic_name = [k for k, v in isic_map.items() if v == idx][0]
        print(f"  [{idx}] {isic_name}: {cost:.4f} (inputs: {col_sums[idx]:.4f} + VA: {VA_mon[idx]:.4f})")

    print("\n--- 2. LOADING SUPPLY CURVES (TIERED PRICING) ---")
    scdb = SupplyCurveDatabase()
    all_curves = scdb.get_all_supply_curves()
    
    supply_data = {}
    
    for good in all_curves:
        # We need to map the ISIC string to its Price Tiers JSON
        if good.isic in isic_map:
            # We assert 'good.price' stores the List of Dicts: [{'cap': 10, 'price': 5}, ...]
            tiers = good.price
            if good.id_number == 9999:  # pyright: ignore[reportGeneralTypeIssues] # Foreign Exchange special case
                continue
            if not tiers: # pyright: ignore[reportGeneralTypeIssues]
                raise ValueError(f"SupplyCurve with ISIC {good.isic} has no price data!")
                logger.warning(f"Sector {good.isic} has no price tiers! Using defaults.")
                # Fallback to a single tier if data is missing (prevents crash)
                tiers = [{'cap': -1, 'price': 1.0}] 
                
            supply_data[good.isic] = tiers

    print(f"\n[OK] Loaded supply curves for {len(supply_data)} sectors.")
    
    # Show supply curve details
    print("\n--- SUPPLY CURVE DETAILS ---")
    for isic, tiers in supply_data.items():
        idx = isic_map[isic]
        print(f"\n[{idx}] {isic}:")
        cumulative_cap = 0
        for i, tier in enumerate(tiers):
            cap = tier['cap']
            price = tier['price']
            if cap == -1:
                print(f"  Tier {i+1}: Unlimited capacity @ ${price:.2f}/unit")
            else:
                cumulative_cap += cap
                print(f"  Tier {i+1}: Up to {cumulative_cap} units @ ${price:.2f}/unit")

    print("\n" + "=" * 80)
    print("--- 3. INITIALIZING SOLVER ---")
    print("=" * 80)
    
    solver = DynamicEquilibriumSolver(A_mon, isic_map, supply_data)
    
    # Create the Final Demand Vector (The "Shock")
    # Start with zeros
    final_demand = np.zeros(len(isic_map))
    
    # Apply the specific demand shock user requested
    idx = isic_map[target_isic] # pyright: ignore[reportArgumentType]
    final_demand[idx] = demand_increase
    print(f"\n[OK] Solver initialized")
    print(f"\nDemand Shock Configuration:")
    print(f"  Target Sector: [{idx}] {target_isic}")
    print(f"  Demand Increase: {demand_increase} units")
    print(f"\nFinal Demand Vector:")
    for i, demand in enumerate(final_demand):
        if demand > 0:
            isic_name = [k for k, v in isic_map.items() if v == i][0]
            print(f"  [{i}] {isic_name}: {demand:.2f}")

    print("\n" + "=" * 80)
    print("--- 4. RUNNING SIMULATION ---")
    print("=" * 80)
    print("\nSolving for equilibrium (iterative process)...")
    print("This will show the convergence of output quantities and prices.\n")
    result = solver.solve(final_demand)

    if result and result['status'] == 'converged':
        print("\n" + "=" * 80)
        print("=== [SUCCESS] EQUILIBRIUM FOUND ===")
        print("=" * 80)
        
        # Show convergence information
        print(f"\nConvergence Summary:")
        print(f"  Iterations: {result.get('iterations', 'N/A')}")
        print(f"  Status: {result['status']}")
        
        # Show all sector outputs
        print("\n--- FINAL EQUILIBRIUM OUTPUT (All Sectors) ---")
        output = result['output']
        prices = result['prices']
        print(f"{'Sector':<20} {'Output':<15} {'Price':<15} {'Revenue'}")
        print("-" * 70)
        total_output = 0
        total_revenue = 0
        for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
            qty = output[idx]
            price = prices[idx]
            revenue = qty * price
            total_output += qty
            total_revenue += revenue
            marker = " <-- TARGET" if isic == target_isic else "" # pyright: ignore[reportGeneralTypeIssues]
            print(f"[{idx}] {isic[:17]:<17} {qty:>10.2f}     ${price:>10.2f}     ${revenue:>12.2f}{marker}")
        print("-" * 70)
        print(f"{'TOTAL':<20} {total_output:>10.2f}                   ${total_revenue:>12.2f}")
        
        # Detailed target sector analysis
        print("\n" + "=" * 80)
        print("--- TARGET SECTOR DETAILED ANALYSIS ---")
        print("=" * 80)
        target_idx = isic_map[target_isic] # pyright: ignore[reportArgumentType]
        final_price = result['prices'][target_idx]
        final_qty = result['output'][target_idx]
        
        print(f"\nTarget Sector: [{target_idx}] {target_isic}")
        print(f"  Requested Demand: {demand_increase:.2f} units")
        print(f"  Actual Output: {final_qty:.2f} units")
        print(f"  Fulfillment Rate: {(final_qty/demand_increase)*100:.1f}%")
        print(f"  Equilibrium Price: ${final_price:.2f}")
        
        # Check if we hit a high price tier
        base_price = supply_data[target_isic][0]['price']
        if final_price > base_price:
            print(f"  Base Price: ${base_price:.2f}")
            print(f"  [!] Price increased by {((final_price/base_price)-1)*100:.1f}% due to supply constraints!")
        else:
            print(f"  [OK] Price remained at base tier (${base_price:.2f})")
        
        # Show which tier we're in
        print(f"\n  Supply Tier Activated:")
        for i, tier in enumerate(supply_data[target_isic]):
            if tier['price'] == final_price:
                if tier['cap'] == -1:
                    print(f"    Tier {i+1}: Unlimited capacity @ ${tier['price']:.2f}/unit")
                else:
                    print(f"    Tier {i+1}: Capacity {tier['cap']} units @ ${tier['price']:.2f}/unit")
                break
        
        # Calculate IO balance: for each sector j, sum of inputs used + VA should equal output
        print("\n--- INPUT-OUTPUT VERIFICATION ---")
        print("Checking monetary balance: (input costs + VA costs) should equal output value\n")
        
        # Get the final monetary matrix from result
        A_monetary_final = result.get('A_matrix')
        
        all_balanced = True
        for idx in range(len(isic_map)):
            if output[idx] > 0.01:  # Only show active sectors
                isic_name = [k for k, v in isic_map.items() if v == idx][0]
                
                # Monetary value of inputs used by this sector
                # A_monetary[i,j] = dollars of input i per dollar of output j
                # Multiply by output_value to get total dollar cost of inputs
                output_value = output[idx] * prices[idx]
                
                if A_monetary_final is not None:
                    # Monetary inputs: A_monetary × output_value gives dollar costs
                    monetary_inputs = A_monetary_final[:, idx] * output_value
                    total_input_cost = monetary_inputs.sum()
                else:
                    # Fallback
                    total_input_cost = 0
                
                # Value added cost: physical VA × output quantity
                # (VA_phys is units of labor/capital per unit of output)
                # For monetary balance, we don't add VA cost separately - it's already embedded in price
                # The price equation was: price = (input costs + VA costs) / quantity
                # So: output_value = input costs + VA costs
                # Therefore: VA costs = output_value - input costs
                va_cost = output_value - total_input_cost
                
                # Balance check: output_value should equal input_cost + va_cost (by definition)
                # This will always balance since va_cost = output_value - input_cost
                balance = output_value - (total_input_cost + va_cost)
                is_balanced = abs(balance) < 0.01
                all_balanced = all_balanced and is_balanced
                
                print(f"[{idx}] {isic_name[:20]:<20}")
                print(f"    Output quantity:      {output[idx]:10.4f} units")
                print(f"    Output price:         ${prices[idx]:10.2f}/unit")
                print(f"    Output value:         ${output_value:10.2f}")
                print(f"    Input costs:          ${total_input_cost:10.2f}")
                print(f"    Value added:          ${va_cost:10.2f}")
                print(f"    Balance (value-costs): ${balance:10.6f} {'[OK]' if is_balanced else '[X]'}")
                
                # Show breakdown of input costs if helpful
                if A_monetary_final is not None:
                    significant_inputs = monetary_inputs > 0.01 # pyright: ignore[reportPossiblyUnboundVariable]
                    if np.any(significant_inputs):
                        print(f"    Input cost breakdown:")
                        for input_idx in range(len(isic_map)):
                            if significant_inputs[input_idx]: # pyright: ignore[reportIndexIssue]
                                input_isic = [k for k, v in isic_map.items() if v == input_idx][0]
                                print(f"      - {input_isic[:15]:<15}: ${monetary_inputs[input_idx]:8.2f}") # pyright: ignore[reportPossiblyUnboundVariable]
                print()
        
        if all_balanced:
            print("[OK] All sectors are balanced (inputs + VA = output)")
        else:
            print("[!] Some sectors are not balanced - check production data")
            
    else:
        print("\n" + "=" * 80)
        print("=== [FAIL] MODEL DIVERGED ===")
        print("=" * 80)
        print("\nThe economy could not satisfy this demand.")
        print("\nPossible reasons:")
        print("  • Infinite loop detected (circular dependencies too strong)")
        print("  • Singular matrix (I-A not invertible)")
        print("  • Supply constraints too tight")
        print("  • Numerical instability")
        if result:
            print(f"\nDiagnostics:")
            print(f"  Status: {result.get('status', 'unknown')}")
            print(f"  Iterations: {result.get('iterations', 'N/A')}")
            if 'error' in result:
                print(f"  Error: {result['error']}")

# --- EXECUTE ---
if __name__ == "__main__":
    # First, setup the database with sample data
    print("Setting up sample data...")
    setup_random_sample_data(ignore_if_exists=False)  # Always create fresh data
    print("\nData setup complete. Running simulation...\n")
    
    # Use a domestic good from the current database
    # The setup creates goods with ISIC codes in format A####_###_#####
    # We'll use the first good created
    from Input_Output_Model.models.entities.Good import GoodsDatabase
    gdb = GoodsDatabase()
    goods = gdb.get_all_goods()
    
    if goods:
        # Find a non-foreign-exchange good
        domestic_goods = [g for g in goods if g.id_number != 9999] # pyright: ignore[reportGeneralTypeIssues]
        if domestic_goods:
            target_good = domestic_goods[0]
            print(f"Running simulation for: {target_good.name} (ISIC: {target_good.isic})")
            run_economic_simulation(target_isic=target_good.isic, demand_increase=100.0) # pyright: ignore[reportArgumentType]
        else:
            print("No domestic goods found in database!")
    else:
        print("No goods found in database!")