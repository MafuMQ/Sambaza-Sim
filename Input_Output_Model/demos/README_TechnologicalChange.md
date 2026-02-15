# Technological Change in Input-Output Models

## Overview

This module provides a framework for modeling and analyzing **technological change** in Input-Output (IO) economic models. Technological change represents improvements or modifications to production processes that alter the technical coefficients (production recipes) in the IO table.

## Key Concept

The fundamental principle when analyzing technological change is:

**Hold Final Demand Constant → Compare outcomes with different technologies**

By using the **same final demand (FD)** for both baseline and changed scenarios, we isolate the **pure effect** of technological change on:
- **Gross output required** (total production needed)
- **Value added generated** (income retained in the economy)  
- **Resource efficiency** (how much input is needed per unit of final demand)
- **Sectoral impacts** (which industries benefit or are affected)

## Why This Matters

When technology improves (e.g., energy efficiency, automation, process innovation):
- **Less gross output** may be needed to satisfy the same final demand
- **Value added** may increase (less spent on intermediate inputs)
- **Employment patterns** may change (automation reduces labor)
- **Environmental impacts** may decrease (better resource efficiency)

## Core Components

### 1. TechnologicalChange Class

The `TechnologicalChange` class describes modifications to the technical coefficient matrix (A).

```python
from Input_Output_Model.demos.util.technological_change import TechnologicalChange

# Create a technological change
tech = TechnologicalChange(
    name="Energy Efficiency Improvement",
    description="30% reduction in energy consumption across all sectors"
)

# Add changes to technical coefficients
tech.add_input_change(
    input_sector_idx=1,  # Energy sector
    change_type="multiply",
    value=0.70  # Reduce by 30% (multiply by 0.70)
)

# Apply to IO matrix
A_new, VA_new = tech.apply(A_baseline, VA_baseline)
```

### 2. Change Types

Three types of modifications are supported:

| Change Type | Description | Example |
|------------|-------------|---------|
| `multiply` | Scale coefficient by a factor | `value=0.80` → 20% reduction |
| `add` | Add/subtract a fixed amount | `value=-0.05` → reduce by 0.05 units |
| `set` | Set coefficient to specific value | `value=0.15` → set to exactly 0.15 |

### 3. Scope of Changes

Changes can be applied at different levels:

#### a) Specific Coefficient (single cell in A matrix)
```python
tech.add_coefficient_change(
    sector_idx=3,           # Producing sector (column)
    input_sector_idx=2,     # Input sector (row)
    change_type="multiply",
    value=0.85              # 15% reduction
)
```

#### b) Entire Input Across All Sectors (row in A matrix)
```python
tech.add_input_change(
    input_sector_idx=1,     # Energy input
    change_type="multiply",
    value=0.75,             # 25% reduction
    exclude_sectors=[5]     # Optional: skip certain sectors
)
```

#### c) All Inputs for One Sector (column in A matrix)
```python
tech.add_sector_change(
    sector_idx=2,           # Manufacturing sector
    change_type="multiply",
    value=0.90,             # 10% productivity improvement
    exclude_inputs=[0]      # Optional: skip certain inputs
)
```

### 4. Comparison Function

```python
from Input_Output_Model.demos.util.simulation import run_simulation

result = run_simulation(
    final_demand=final_demand,  # SAME for both scenarios!
    A_before=A_baseline,
    A_after=A_changed,
    VA_before=VA_baseline,
    VA_after=VA_changed,
    isic_map=isic_map,
    before_name="Current Technology",
    after_name="With Energy Efficiency",
    solver_type="leontief"
)
```

This produces a detailed comparison showing:
- Change in gross output
- Change in value added
- Change in intermediate inputs
- Sector-by-sector impacts
- Efficiency metrics

## Usage Examples

### Example 1: Energy Efficiency Improvement

```python
from Input_Output_Model.demos.util.technological_change import create_energy_efficiency_change
from Input_Output_Model.util.Evaluators import build_io_matrix
import numpy as np

# Build baseline IO matrix
A_baseline, VA_baseline, isic_map = build_io_matrix()

# Define technological change: 30% energy efficiency gain
tech_change = create_energy_efficiency_change(
    energy_sector_idx=1,
    efficiency_gain=0.30
)

# Apply change
A_changed, VA_changed = tech_change.apply(A_baseline, VA_baseline)

# Compare with SAME final demand
final_demand = np.array([250, 200, 300, 150, 100, 0])

run_simulation(
    final_demand=final_demand,
    A_before=A_baseline, A_after=A_changed,
    VA_before=VA_baseline, VA_after=VA_changed,
    isic_map=isic_map,
    solver_type="leontief"
)
```

**Expected Result:** With the same final demand, gross output decreases because production requires less energy input per unit of output.

### Example 2: Automation (Labor-Capital Substitution)

```python
# Create automation change
tech = TechnologicalChange(
    name="Manufacturing Automation",
    description="Replace labor with machinery"
)

# Reduce labor input (sector 0) by 40%
tech.add_coefficient_change(
    sector_idx=3,           # Manufacturing sector
    input_sector_idx=0,     # Labor services
    change_type="multiply",
    value=0.60              # 40% reduction
)

# Increase machinery input (sector 2) by 15%
tech.add_coefficient_change(
    sector_idx=3,
    input_sector_idx=2,     # Machinery
    change_type="multiply",
    value=1.15              # 15% increase
)

# Apply and compare...
```

**Expected Result:** Manufacturing requires less labor but more capital equipment. Total value added may change depending on the relative prices.

### Example 3: Economy-Wide Green Transition

```python
tech = TechnologicalChange(
    name="Green Technology Transition",
    description="Multiple efficiency improvements"
)

# 1. Energy efficiency (all sectors)
tech.add_input_change(
    input_sector_idx=1,     # Energy
    change_type="multiply",
    value=0.75              # 25% reduction
)

# 2. Material efficiency (manufacturing sectors)
for sector in [2, 3]:
    tech.add_sector_change(
        sector_idx=sector,
        change_type="multiply",
        value=0.85          # 15% reduction in all inputs
    )

# 3. Services productivity
tech.add_sector_change(
    sector_idx=4,
    change_type="multiply",
    value=0.90              # 10% reduction
)
```

**Expected Result:** Comprehensive efficiency gains across the economy, reducing both gross output and intermediate input requirements.

### Example 4: Running Pre-Built Demos

```python
# Run comprehensive demos from unified demo script
from Input_Output_Model.demos.demo import run_demo

# Example 7: Energy efficiency
run_demo(7)

# Example 8: Productivity improvement
run_demo(8)

# Example 9: Automation
run_demo(9)

# ... Examples 10-12 available (see demo.py)
```

Or run from the main demo file:

```python
from Input_Output_Model.demos.demo import run_demo, setup_data

# Setup data first
setup_data(source="data/ex2", overwrite_existing_data=True)

# Run Example 7 (Technological Change Comparison)
run_demo(7)
```

## Technical Details

### Value Added Adjustment

When technical coefficients are modified, the value added (VA) vector is automatically adjusted to maintain the identity:

**A + VA = 1** (for each sector)

This ensures that:
- Total inputs + Value added = Total output (always equals 1 in coefficient form)
- When inputs decrease, value added increases proportionally
- The model remains internally consistent

### Matrix Relationships

The Input-Output model uses the Leontief inverse:

**X = (I - A)^(-1) × F**

Where:
- **X** = Gross output vector (total production)
- **A** = Technical coefficient matrix (input requirements)
- **F** = Final demand vector (consumption, investment, government, exports)
- **I** = Identity matrix

When A changes (technological improvement), X changes for the **same** F, which shows us the pure efficiency effect.

### Comparison Methodology

1. **Baseline Scenario:**
   - Use original A matrix (A_baseline)
   - Calculate X_baseline = (I - A_baseline)^(-1) × F
   - Calculate VA_baseline = VA_baseline^T × X_baseline

2. **Changed Scenario:**
   - Use modified A matrix (A_changed)
   - Calculate X_changed = (I - A_changed)^(-1) × F  (SAME F!)
   - Calculate VA_changed = VA_changed^T × X_changed

3. **Compare:**
   - ΔX = X_changed - X_baseline (change in gross output)
   - ΔVA = VA_changed - VA_baseline (change in value added)
   - Efficiency = VA / X (value added per unit of gross output)

## Practical Applications

### Policy Analysis
- **Green energy transition:** Model switch from fossil fuels to renewables
- **Circular economy:** Analyze recycling and waste reduction impacts
- **Industrial policy:** Assess R&D investments in productivity

### Business Strategy
- **Technology adoption:** Compare costs/benefits of new production methods
- **Supply chain optimization:** Evaluate process improvements
- **Sustainability planning:** Quantify resource efficiency gains

### Research
- **Historical analysis:** Study technological progress over time
- **Forecasting:** Project future technological scenarios
- **Structural change:** Understand how economies evolve

## Files Structure

```
Input_Output_Model/
├── demos/
│   ├── demo.py                          # Unified demo file (Examples 1-12)
│   ├── configs/                         # Configuration files
│   │   ├── tax_policy_examples.py       # Tax policy configs (1-6)
│   │   └── tech_change_examples.py      # Tech change configs (7-12)
│   └── util/
│       ├── technological_change.py      # Core TechnologicalChange class
│       ├── Setup_Data.py                # Data loading utilities
│       └── simulation.py                # Unified simulation runner
├── models/
│   ├── entities/                        # Database models (Good, Production)
│   └── table/
│       └── solver.py                    # DynamicEquilibriumSolver
└── util/
    └── Evaluators.py                    # build_io_matrix() function
```

## Best Practices

### 1. Always Use the Same Final Demand
```python
# ✓ CORRECT
final_demand = np.array([200, 180, 220, 150, 100, 0])
run_simulation(
    final_demand=final_demand,
    A_before=A_baseline, A_after=A_changed,
    VA_before=VA_baseline, VA_after=VA_changed,
    isic_map=isic_map
)  # Same FD for both

# \u2717 INCORRECT - using different final demands defeats the purpose
```

### 2. Document Your Changes
```python
tech = TechnologicalChange(
    name="Clear descriptive name",
    description="Explain WHAT changes and WHY"
)
print(tech.get_summary())  # Always review before applying
```

### 3. Validate Technical Feasibility
```python
# Check that changes make sense
A_new, VA_new = tech.apply(A, VA)

# Verify: each column should sum to ≤ 1.0
column_sums = A_new.sum(axis=0) + VA_new
assert all(column_sums <= 1.001), "Invalid coefficients!"
```

### 4. Interpret Results Carefully
- **Decreased X:** More efficient (good for resources, may reduce employment)
- **Increased VA:** More value retained domestically (generally positive)
- **Sectoral shifts:** Winners and losers from technological change

## Further Reading

### Academic Background
- **Leontief Input-Output Analysis:** Wassily Leontief (Nobel Prize 1973)
- **Structural Change:** Study of how economies evolve over time
- **Productivity Analysis:** Measurement of technical efficiency

### Related Concepts
- **Total Factor Productivity (TFP):** Overall efficiency improvement
- **Technical Change vs. Efficiency Change:** Different sources of productivity growth
- **Embodied vs. Disembodied Technical Change:** In capital vs. general knowledge

## Common Questions

**Q: Why not just change prices instead of coefficients?**  
A: Technical coefficients represent *physical* relationships (kg of steel per car). Technological change alters these physical requirements, not just prices.

**Q: Can I model learning-by-doing or gradual improvements?**  
A: Yes! Apply progressively larger changes over time (e.g., multiply by 0.95, 0.90, 0.85...) in sequence.

**Q: What if my change makes the system infeasible (column sums > 1)?**  
A: This means your technology requires more inputs than outputs, which is impossible. Revise the change parameters.

**Q: How do I model innovation in a new sector?**  
A: Add a new row/column to your matrices, define its technical coefficients, and compare scenarios with/without it.

## Contact and Contributions

This module is part of the Sambaza-Sim Input-Output modeling framework. 

For questions, suggestions, or contributions, please refer to the main project documentation.

---

**Last Updated:** February 2026  
**Version:** 1.0
