import os
import sys
import numpy as np
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.setup_data import get_calibrated_model
from core.io_matrix import IOModel

logging.basicConfig(level=logging.WARNING)

def run_first_principles_demo():
    print("=" * 80)
    print("Sambaza-Sim: First Principles IOModel Demo")
    print("=" * 80)
    
    # 1. Initialize Baseline Model from Pipeline Gatekeeper
    print("\n1. Initializing Model from Database...")
    try:
        model, isic_map = get_calibrated_model(demoDB=False)
        n = model.n
        sectors = [k for k, v in sorted(isic_map.items(), key=lambda item: item[1])]
    except Exception as e:
        print(f"Failed to build model: {e}")
        return

    print(f"   Successfully built stateless IOModel for {n} sectors.")
    
    # 2. Establish Baseline Scenario (Uniform Demand)
    print("\n2. Simulating Baseline Demand")
    base_demand = np.full(n, 1000.0)
    X_base = model.simulate(base_demand)
    print(f"   Total Gross Output (Baseline): {X_base.sum():.2f}")

    # 3. Simulate a Final Demand Shock
    print("\n3. Scenario A: Final Demand Shock (+100 to Sector 0)")
    shock_demand = base_demand.copy()
    shock_demand[0] += 100.0
    
    X_shock = model.simulate(shock_demand)
    print(f"   Total Gross Output (Post-Shock): {X_shock.sum():.2f}")
    print(f"   Delta Gross Output: {(X_shock - X_base).sum():.2f}")
    
    # 4. Simulate a Technological Change (A Matrix modification)
    print("\n4. Scenario B: Technological Change (10% efficiency gain in Sector 1 inputs)")
    A_new = model.A.copy()
    
    # Reduce all inputs required by Sector 1 by 10%
    A_new[:, 1] = A_new[:, 1] * 0.90
    
    # Create a new statless model calibrated to the new tech
    tech_model = IOModel(A=A_new, VA_coeffs=model.VA_coeffs)
    
    X_tech = tech_model.simulate(base_demand)
    print(f"   Total Gross Output (Post-Tech Change): {X_tech.sum():.2f}")
    print(f"   Delta Gross Output: {(X_tech - X_base).sum():.2f}")

    print("\n" + "=" * 80)
    print("Demo Complete")
    print("=" * 80)

if __name__ == "__main__":
    run_first_principles_demo()


