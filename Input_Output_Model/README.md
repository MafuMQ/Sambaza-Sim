## What the Input_Output_Model App Does

This is an **economic Input-Output (I-O) model** that simulates how demand shocks propagate through an economy using **Leontief's matrix framework** with **tiered supply curves** and **dynamic pricing**.

### Core Purpose
The app calculates how an increase in demand for one good ripples through the economy, determining:
- How much each sector must produce to satisfy direct and indirect requirements
- What prices emerge when supply is constrained
- Whether the economic system can reach equilibrium

---

## How It Works (Step-by-Step)

### **1. Database Structure (SQLAlchemy ORM)**

The app stores economic data in SQLite with three main entities:

**Production** (individual production methods):
- `production_inputs`: JSON dict storing **monetary costs** of inputs, e.g., `{"A1234_567_8": 45, "A9876_543_2": 20}` means this production method requires $45 worth of good A1234_567_8 and $20 worth of A9876_543_2
- `production_added_values`: JSON dict of value-added components (wages, profits), e.g., `{"wages": 30, "profit": 15}`
- `total_inputs_cost`: Sum of all input costs ($65 in example above)
- `total_value_added`: Sum of all VA components ($45 in example above)
- `price`: Total cost = `total_inputs_cost + total_value_added` ($110 in example)
- `production_quantity`: Maximum capacity this production method can achieve

**GoodIndice/SupplyCurve** (aggregated supply curves per good):
- `price`: JSON list of tiers `[{'cap': 100, 'price': 50}, {'cap': 200, 'price': 75}, ...]`
  - This represents a **merit-order supply curve**: cheapest 100 units cost $50/unit, next 200 units cost $75/unit, etc.
- Built by sorting all Production methods for a good by price (cheapest first)

**Good** (ISIC classification):
- Stores industry codes and descriptive names

---

### **2. Data Setup (Setup_Data.py)**

The `test_good()` function generates sample economic data:

1. Creates random goods with ISIC codes
2. For each good, creates multiple domestic production methods with:
   - Random input requirements (stored as **monetary costs**)
   - Random value-added components
   - Random production capacities
3. Creates an IMPORT production method using foreign exchange good (ISIC A9999_999_999)
4. Calls `evaluate_productions_price()` to calculate production prices
5. Calls `evaluate_indicies_with_functions()` to build tiered supply curves

**Key insight**: The database stores **monetary values**, not physical quantities. When it says `production_inputs = {isic: 45}`, that means "$45 worth of that input," not "45 units."

---

### **3. Matrix Building (Evaluators.py)**

`evaluate_indicies_production_inputs_to_matrix()` constructs the **monetary coefficient matrix**:

**Input**: Production recipes from database  
**Output**: 
- `A_mon[i,j]` = dollars of input i per dollar of output j
- `VA_mon[j]` = dollars of value added per dollar of output j
- `isic_map` = mapping from ISIC code → matrix index

**Algorithm**:
```
For each output good j:
    1. Find cheapest production method (including imports)
    2. Get output_price for that method
    3. For each input i with cost C_i:
         A_mon[i,j] = C_i / output_price  # Normalize to per-dollar basis
    4. VA_mon[j] = total_value_added / output_price
```

**Properties of A_mon**:
- Each column sums to ≤ 1.0 (representing cost breakdown per dollar of output)
- Column sum + VA coefficient = 1.0 (since price = inputs + VA)
- This is the **monetary technical coefficient matrix** used in Leontief analysis

---

### **4. Supply Curves Loading**

From `GoodIndice.price` JSON, the app loads tiered supply curves for each good:

```python
supply_data = {
    "A7455_757_2": [
        {'cap': 50, 'price': 100},   # First 50 units @ $100/unit
        {'cap': 30, 'price': 150},   # Next 30 units @ $150/unit
        {'cap': -1, 'price': 200}    # Unlimited @ $200/unit
    ],
    ...
}
```

This represents a **merit-order dispatch** model: the economy uses cheapest capacity first, then moves up the cost curve as demand increases.

---

### **5. Equilibrium Solver (solver.py)**

The `DynamicEquilibriumSolver` finds equilibrium through iteration:

**Initialization**:
- Pre-processes supply tiers into NumPy arrays for fast lookup
- Stores the monetary coefficient matrix A_mon

**Core Loop** (in `solve()` method):
```
Input: final_demand vector (exogenous demand, e.g., [0, 0, 100, 0, ...])
Initialize: current_output = final_demand

Repeat until convergence:
    1. Calculate prices based on current_output:
         - For each sector, find which supply tier is activated
         - Use weighted average if output spans multiple tiers
    
    2. Use monetary matrix directly (already normalized):
         A_monetary = A_mon  # No conversion needed
    
    3. Solve Leontief equation:
         X = (I - A_monetary)^-1 × final_demand
         
         This calculates total output needed in each sector
         to satisfy final demand AND all intermediate requirements
    
    4. Check if output changed:
         If ||X_new - X_old|| < tolerance → converged
         Else → update current_output and repeat
```

**get_market_prices()** method:
- Given output quantity Q for a good
- Find which tiers are activated (e.g., if Q=70, uses tier 1 fully + 20 from tier 2)
- Calculate weighted average price: (50×$100 + 20×$150) / 70 = $114.29/unit

**Convergence**: 
- In simple cases with no tight constraints, converges in 1-2 iterations
- The Leontief equation already gives the equilibrium; iteration handles price updates if needed

---

### **6. Demo Execution (demo_v2.py)**

`run_economic_simulation()` orchestrates the full workflow:

**Phase 1: Matrix Building**
```python
A_mon, VA_mon, isic_map = evaluate_indicies_production_inputs_to_matrix()
```
- Reads Production table
- Constructs monetary coefficient matrix
- Prints matrix properties (shape, sparsity, column sums)
- **Verification**: Checks that column_sum + VA = 1.0 for all sectors

**Phase 2: Supply Curve Loading**
```python
supply_data = {isic: good.price for good in all_indices}
```
- Reads GoodIndice table
- Loads tiered supply curves
- Displays tier details (capacity and price at each level)

**Phase 3: Solver Initialization**
```python
solver = DynamicEquilibriumSolver(A_mon, isic_map, supply_data)
final_demand = [0, 0, ..., 100, ..., 0]  # Shock target sector
```

**Phase 4: Simulation**
```python
result = solver.solve(final_demand)
```
- Iterates to find equilibrium
- Prints convergence details

**Phase 5: Results Analysis**
- Shows output quantities and prices for all sectors
- Calculates target sector fulfillment rate
- Identifies which supply tier was activated
- **I-O Balance Verification**: For each sector j:
  ```
  output_value_j = Σ(input_costs_i→j) + value_added_j
  ```
  This must hold (by construction, since prices were set that way)

---

## Example Flow

**Scenario**: Demand shock of 100 units for good "A7455_757_2"

1. **Matrix shows interdependencies**:
   - A7455_757_2 requires $0.40 of A1234_567_8 per dollar of output
   - A1234_567_8 requires $0.25 of A9876_543_2 per dollar
   - Creates chain reaction

2. **Leontief multiplier effect**:
   ```
   Direct: 100 units of A7455_757_2
   Indirect: 
     - 40 units of A1234_567_8 (to supply inputs for A7455_757_2)
     - 10 units of A9876_543_2 (to supply inputs for A1234_567_8)
     - ...feedback loops continue...
   ```

3. **Supply constraints activate higher tiers**:
   - If 100 units exceeds tier 1 capacity (50), price jumps to tier 2
   - Solver balances quantity and price until stable

4. **Equilibrium output**:
   - All sectors produce exact amounts needed
   - Prices reflect scarcity (higher tiers activated for constrained goods)
   - I-O accounts balance: for every sector, revenue = input costs + value added

---

## Key Design Decisions

1. **Monetary vs Physical**: Database stores dollar values, not quantities. This simplifies calculations since you don't multiply quantities by prices—the costs are already in dollars.

2. **Normalized coefficients**: A[i,j] is "per dollar of output," making the matrix dimensionally consistent and interpretable.

3. **Tiered supply**: Unlike basic Leontief models (fixed prices), this allows price discovery based on scarcity.

4. **Import handling**: Foreign exchange good (A9999_999_999) represents the "resource" consumed to purchase imports, treating imports as a production process.

5. **Merit order**: Productions sorted by price ensure economy uses cheapest capacity first (economic efficiency principle).

---

This app is essentially a **computable general equilibrium (CGE) simulator** with:
- Leontief production technology (fixed coefficients)
- Tiered supply curves (increasing marginal costs)
- Iterative solution method (dynamic equilibrium finding)

The output tells you: *"If final demand for X increases by Y units, how much does each sector need to produce, and what prices emerge?"*