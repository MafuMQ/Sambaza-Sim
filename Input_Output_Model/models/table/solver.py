import numpy as np
import logging

logger = logging.getLogger(__name__)

class DynamicEquilibriumSolver:
    def __init__(self, A_phys, isic_map, supply_data):
        """
        :param A_phys: The Physical Matrix (Inputs per Unit Output)
        :param isic_map: Dict mapping {ISIC_String: Matrix_Index_Int}
        :param supply_data: Dict mapping {ISIC_String: Tier_List} 
                            (The JSON structure you stored)
        """
        self.A_phys = A_phys
        self.isic_map = isic_map
        self.n = len(isic_map)
        
        # Pre-process supply data into fast NumPy arrays
        self.supply_tiers = self._vectorize_supply_data(supply_data)

    def _vectorize_supply_data(self, supply_data):
        """
        Converts JSON list of tiers into NumPy arrays for fast lookup.
        Returns a dict: {row_idx: (bounds_array, caps_array, prices_array)}
        """
        vectorized = {}
        
        for isic, tiers in supply_data.items():
            if isic not in self.isic_map:
                continue
            
            idx = self.isic_map[isic]
            
            # Extract 'cap' and 'price' from your tier dictionaries
            # Assuming tier structure: [{'cap': 100, 'price': 50}, ...]
            n_tiers = len(tiers)
            caps = np.zeros(n_tiers)
            prices = np.zeros(n_tiers)
            
            for i, t in enumerate(tiers):
                c = t.get('cap')
                # Handle Infinite/None caps
                if c is None or c == -1: 
                    caps[i] = np.inf
                else:
                    caps[i] = float(c)
                
                prices[i] = float(t.get('price', 0.0))
            
            # Create cumulative bounds (0, 10, 30...)
            bounds = np.concatenate(([0], np.cumsum(caps)[:-1]))
            
            vectorized[idx] = (bounds, caps, prices)
            
        return vectorized

    def get_market_prices(self, current_output):
        """
        Step 1: Calculate Prices based on current Demand (Output).
        """
        current_prices = np.zeros(self.n)
        
        for i in range(self.n):
            if i not in self.supply_tiers:
                # Fallback: if no supply data, assume price = 1.0 or 0
                current_prices[i] = 1.0 
                continue

            bounds, caps, prices = self.supply_tiers[i]
            demand = current_output[i]
            
            if demand <= 0:
                current_prices[i] = prices[0] # Base price
                continue

            # Vectorized Weighted Average Calculation
            amounts = np.clip(demand - bounds, 0, caps)
            total_cost = np.dot(amounts, prices)
            current_prices[i] = total_cost / demand
            
        return current_prices

    def update_value_matrix(self, current_prices):
        """
        Step 2: Convert Physical Matrix to Monetary Matrix.
        A_monetary[i,j] = (A_phys[i,j] * P_i) / P_j
        = dollars of input i needed per dollar of output j
        
        Where:
        - A_phys[i,j] = physical units of input i per physical unit of output j  
        - P_i = price of input i ($/unit)
        - P_j = price of output j ($/unit)
        """
        # Avoid division by zero
        P_out = np.where(current_prices == 0, 1e-9, current_prices)
        
        # Numerator: value of inputs (A_phys * input prices)
        value_inputs = self.A_phys * current_prices[:, None]
        
        # Denominator: output prices (broadcast across rows)
        A_monetary = value_inputs / P_out[None, :]
        
        return A_monetary

    def solve(self, final_demand, max_iter=50, tol=1e-3, verbose=True):
        """
        The Main Loop.
        :param final_demand: Array of final demand quantities (Household consumption)
        :param verbose: If True, print iteration details
        """
        # Initial Guess: Total Output = Final Demand (assuming 0 intermediate use initially)
        current_output = np.array(final_demand, dtype=float)
        current_prices = np.zeros(self.n)
        
        print(f"Starting Solver for {self.n} sectors...")
        if verbose:
            print(f"\nInitial State:")
            print(f"  Final Demand: {final_demand}")
            print(f"  Initial Output Guess: {current_output}\n")

        for iteration in range(max_iter):
            prev_output = current_output.copy()
            
            # 1. Update Prices based on current Quantity
            current_prices = self.get_market_prices(current_output)
            
            if verbose and iteration < 10:  # Only show first 10 iterations
                print(f"Iteration {iteration + 1}:")
                print(f"  Prices: {np.round(current_prices, 2)}")
            
            # 2. Update Matrix based on new Prices
            A_monetary = self.update_value_matrix(current_prices)
            
            if verbose and iteration < 3:  # Show matrix for first few iterations
                print(f"  A_monetary (first 3x3):")
                print(f"  {np.round(A_monetary[:3, :3], 4)}")
            
            # 3. Solve Leontief: X = (I - A)^-1 * F
            I = np.eye(self.n)
            I_minus_A = I - A_monetary
            
            # Debug: Check if matrix is productive (Hawkins-Simon condition)
            if verbose and iteration == 0:
                col_sums = A_monetary.sum(axis=0)
                print(f"  A_monetary column sums: {np.round(col_sums, 4)}")
                bad_cols = np.where(col_sums >= 1.0)[0]
                if len(bad_cols) > 0:
                    print(f"  WARNING: Columns {bad_cols} have sum >= 1.0 (non-productive!)")
            
            try:
                L_inv = np.linalg.inv(I_minus_A)
            except np.linalg.LinAlgError:
                logger.error("Matrix became singular (unsolvable)!")
                print(f"  [X] SINGULAR MATRIX at iteration {iteration + 1}")
                print(f"  (I-A) column sums: {np.round((I - A_monetary).sum(axis=0), 4)}")
                return None

            current_output = L_inv @ final_demand
            
            if verbose and iteration < 10:
                print(f"  New Output: {np.round(current_output, 2)}")
                print(f"  Change: {np.linalg.norm(current_output - prev_output):.6f}\n")
            
            # 4. Check Convergence
            # We check if the Output vector has stopped changing
            diff = np.linalg.norm(current_output - prev_output)
            if diff < tol:
                print(f"[OK] Converged in {iteration+1} iterations (change={diff:.6f} < tolerance={tol})")
                return {
                    "status": "converged",
                    "output": current_output,
                    "prices": current_prices,
                    "A_matrix": A_monetary,
                    "iterations": iteration + 1
                }
                
        print(f"[X] Max iterations ({max_iter}) reached without convergence.")
        return {
            "status": "failed",
            "output": current_output,
            "prices": current_prices,
            "iterations": max_iter
        }