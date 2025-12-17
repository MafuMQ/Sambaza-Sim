import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import logging
from Input_Output_Model.models.entities.Good_Indice import GoodsIndiceDatabase

# Import the components we built in the previous turns
from Input_Output_Model.models.table.solver import DynamicEquilibriumSolver 
from Input_Output_Model.util.Evaluators import evaluate_indicies_production_inputs_to_matrix # The "Direct Matrix Builder" function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_economic_simulation(target_isic: str, demand_increase: float):
    print("=" * 80)
    print("--- 1. BUILDING PHYSICAL SKELETON ---")
    print("=" * 80)
    print()
    # This generates the static A_phys matrix directly from Production recipes
    A_phys, VA_phys, isic_map = evaluate_indicies_production_inputs_to_matrix(demoDB=False)

    print(f"\n[OK] Skeleton Built. Sectors: {len(isic_map)}")
    print("\n--- TECHNICAL COEFFICIENTS MATRIX (A_phys) ---")
    print("Each cell A[i,j] = input i needed per unit of output j")
    print("\nISIC Mapping:")
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        print(f"  [{idx}] {isic}")
    print("\nA_phys Matrix:")
    print(A_phys)
    
    # Verify matrix properties
    print("\n--- MATRIX VERIFICATION ---")
    print(f"Shape: {A_phys.shape}")
    print(f"Min coefficient: {A_phys.min():.6f}")
    print(f"Max coefficient: {A_phys.max():.6f}")
    print(f"Average coefficient: {A_phys.mean():.6f}")
    print(f"Sparsity: {(A_phys == 0).sum() / A_phys.size * 100:.1f}% zeros")
    
    # Check column sums (total inputs per unit output)
    col_sums = A_phys.sum(axis=0)
    print("\nColumn sums (total intermediate inputs per unit output):")
    for idx, sum_val in enumerate(col_sums):
        isic_name = [k for k, v in isic_map.items() if v == idx][0]
        print(f"  [{idx}] {isic_name}: {sum_val:.4f}")
    
    print("\n--- VALUE ADDED VECTOR (VA_phys) ---")
    print("VA per unit of output (labor, capital, etc.):")
    for idx, va in enumerate(VA_phys):
        isic_name = [k for k, v in isic_map.items() if v == idx][0]
        print(f"  [{idx}] {isic_name}: {va:.4f}")
    
    # Total cost per unit = intermediate inputs + value added
    print("\n--- TOTAL COST PER UNIT ---")
    total_costs = col_sums + VA_phys
    for idx, cost in enumerate(total_costs):
        isic_name = [k for k, v in isic_map.items() if v == idx][0]
        print(f"  [{idx}] {isic_name}: {cost:.4f} (inputs: {col_sums[idx]:.4f} + VA: {VA_phys[idx]:.4f})")

    print("\n--- 2. LOADING MUSCLES (SUPPLY TIERS) ---")
    gidb = GoodsIndiceDatabase()
    all_indices = gidb.get_all_good_indices()
    
    supply_data = {}
    
    for good in all_indices:
        # We need to map the ISIC string to its Price Tiers JSON
        if good.isic in isic_map:
            # We assert 'good.price' stores the List of Dicts: [{'cap': 10, 'price': 5}, ...]
            tiers = good.price
            if good.id_number == 9999:  # Foreign Exchange special case
                continue
            if not tiers:
                raise ValueError(f"GoodIndice with ISIC {good.isic} has no price data!")
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
    
    solver = DynamicEquilibriumSolver(A_phys, isic_map, supply_data)
    
    # Create the Final Demand Vector (The "Shock")
    # Start with zeros
    final_demand = np.zeros(len(isic_map))
    
    # Apply the specific demand shock user requested
    idx = isic_map[target_isic]
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
            marker = " <-- TARGET" if isic == target_isic else ""
            print(f"[{idx}] {isic[:17]:<17} {qty:>10.2f}     ${price:>10.2f}     ${revenue:>12.2f}{marker}")
        print("-" * 70)
        print(f"{'TOTAL':<20} {total_output:>10.2f}                   ${total_revenue:>12.2f}")
        
        # Detailed target sector analysis
        print("\n" + "=" * 80)
        print("--- TARGET SECTOR DETAILED ANALYSIS ---")
        print("=" * 80)
        target_idx = isic_map[target_isic]
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
        print("For each sector: sum(inputs used) + value_added = output\n")
        
        all_balanced = True
        for idx in range(len(isic_map)):
            if output[idx] > 0.01:  # Only show active sectors
                isic_name = [k for k, v in isic_map.items() if v == idx][0]
                
                # Inputs used by this sector (column j of A_phys, scaled by output j)
                inputs_used = A_phys[:, idx] * output[idx]
                total_inputs = inputs_used.sum()
                
                # Value added by this sector
                va_generated = VA_phys[idx] * output[idx]
                
                # Total cost of production
                total_cost = total_inputs + va_generated
                
                # Output value
                actual_output = output[idx]
                
                # Balance (should be close to zero)
                balance = actual_output - total_cost
                is_balanced = abs(balance) < 0.01
                all_balanced = all_balanced and is_balanced
                
                print(f"[{idx}] {isic_name[:20]:<20}")
                print(f"    Output produced:      {actual_output:10.4f} units")
                print(f"    Intermediate inputs:  {total_inputs:10.4f} units")
                print(f"    Value added:          {va_generated:10.4f} units")
                print(f"    Total cost:           {total_cost:10.4f} units")
                print(f"    Balance (out-cost):   {balance:10.6f} {'[OK]' if is_balanced else '[X]'}")
                
                # Show breakdown of inputs if helpful
                if np.any(inputs_used > 0.01):
                    print(f"    Input breakdown:")
                    for input_idx in range(len(isic_map)):
                        if inputs_used[input_idx] > 0.01:
                            input_isic = [k for k, v in isic_map.items() if v == input_idx][0]
                            print(f"      - {input_isic[:15]:<15}: {inputs_used[input_idx]:8.4f} units")
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
    # Use a domestic good (not an import) for demand shock
    # A0778_505_72 is "She Good" - has 4 domestic productions
    run_economic_simulation(target_isic="A0778_505_72", demand_increase=100.0)