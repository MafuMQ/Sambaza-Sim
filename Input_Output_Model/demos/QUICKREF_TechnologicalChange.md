# Technological Change Quick Reference

## Quick Start

### 1. Basic Setup
```python
from Input_Output_Model.demos.util.technological_change import TechnologicalChange
from Input_Output_Model.demos.util.simulation import run_simulation
from Input_Output_Model.util.Evaluators import build_io_matrix
import numpy as np

# Build your IO matrix
A, VA, isic_map = build_io_matrix()
```

### 2. Create a Technological Change
```python
tech = TechnologicalChange(
    name="My Innovation",
    description="What it does and why"
)
```

### 3. Add Changes
```python
# Option A: Single coefficient (one cell)
tech.add_coefficient_change(sector_idx=2, input_sector_idx=1, 
                            change_type="multiply", value=0.80)

# Option B: One input across all sectors (row)
tech.add_input_change(input_sector_idx=1, 
                     change_type="multiply", value=0.75)

# Option C: All inputs for one sector (column)
tech.add_sector_change(sector_idx=2, 
                      change_type="multiply", value=0.90)
```

### 4. Apply and Compare
```python
# Apply change
A_new, VA_new = tech.apply(A, VA)

# Compare with SAME final demand using unified simulation
final_demand = np.array([200, 180, 220, 150, 100, 0])
run_simulation(
    final_demand=final_demand,
    A_before=A, A_after=A_new,
    VA_before=VA, VA_after=VA_new,
    isic_map=isic_map,
    solver_type="leontief"
)
```

## Change Types

| Type | Effect | Example |
|------|--------|---------|
| `multiply` | Scale by factor | `value=0.80` = 20% reduction<br>`value=1.15` = 15% increase |
| `add` | Add constant | `value=-0.05` = reduce by 0.05<br>`value=0.03` = increase by 0.03 |
| `set` | Set to value | `value=0.15` = set exactly to 0.15 |

## Common Scenarios

### Energy Efficiency (30% reduction)
```python
tech = TechnologicalChange("Energy Efficiency", "Reduce energy use")
tech.add_input_change(input_sector_idx=1, change_type="multiply", value=0.70)
A_new, VA_new = tech.apply(A, VA)
```

### Automation (Labor ↓ Capital ↑)
```python
tech = TechnologicalChange("Automation", "Replace workers with machines")
tech.add_coefficient_change(3, 0, "multiply", 0.60)  # Labor -40%
tech.add_coefficient_change(3, 2, "multiply", 1.20)  # Capital +20%
A_new, VA_new = tech.apply(A, VA)
```

### Productivity Improvement (All inputs)
```python
tech = TechnologicalChange("Productivity", "General efficiency gain")
tech.add_sector_change(sector_idx=2, change_type="multiply", value=0.85)
A_new, VA_new = tech.apply(A, VA)
```

### Material Efficiency (Specific input)
```python
tech = TechnologicalChange("Material Efficiency", "Less waste")
tech.add_coefficient_change(4, 1, "multiply", 0.75)  # -25% materials
A_new, VA_new = tech.apply(A, VA)
```

## Pre-Built Templates

```python
from Input_Output_Model.demos.util.technological_change import (
    create_energy_efficiency_change,
    create_productivity_improvement
)

# Energy efficiency
tech = create_energy_efficiency_change(energy_sector_idx=1, efficiency_gain=0.30)

# Productivity
tech = create_productivity_improvement(sector_idx=2, productivity_gain=0.15)

# Apply as usual
A_new, VA_new = tech.apply(A, VA)
```

## Run Pre-Built Demos

### Technological Change Examples (7-12)
```python
from Input_Output_Model.demos.demo import run_demo
from Input_Output_Model.demos.util.Setup_Data import setup_data

# Setup data
setup_data(source="data/ex2", overwrite_existing_data=True)

# Run demos
run_demo(7)   # Energy efficiency
run_demo(8)   # Productivity improvement
run_demo(9)   # Automation
run_demo(10)  # Material efficiency
run_demo(11)  # Green transition (multiple changes)
run_demo(12)  # Custom changes
```

### Tax Policy Examples (1-6)
```python
from Input_Output_Model.demos.demo import run_demo

run_demo(1)  # Circular Flow Model
run_demo(2)  # Income Tax Increase
```

## Interpreting Results

### Gross Output (X)
- **Decrease:** More efficient, need less production for same demand ✓
- **Increase:** Less efficient, need more production ✗

### Value Added (VA)
- **Increase:** More value retained in economy ✓
- **Decrease:** More spent on intermediate inputs ✗

### Efficiency Ratio (VA/X)
- **Higher:** Better resource efficiency ✓
- **Lower:** Worse resource efficiency ✗

## Common Patterns

### Technology reduces inputs → X decreases, VA increases
```
✓ Efficiency improvement
  Less gross output needed
  More value added generated
  Example: Energy efficiency, waste reduction
```

### Technology substitutes inputs → X varies, VA varies
```
⚖️ Input substitution
  Capital for labor: may increase X if capital-intensive
  May change VA depending on relative costs
  Example: Automation, mechanization
```

### Technology increases inputs → X increases, VA decreases
```
✗ Inefficiency
  Usually indicates error in specification
  Check your change parameters!
```

## Checklist

- [ ] Load baseline data: `A, VA, isic_map = build_io_matrix()`
- [ ] Create tech change: `tech = TechnologicalChange(name, description)`
- [ ] Add modifications: `tech.add_coefficient_change(...)` or similar
- [ ] Review changes: `print(tech.get_summary())`
- [ ] Apply changes: `A_new, VA_new = tech.apply(A, VA)`
- [ ] Define **same** final demand for both scenarios
- [ ] Compare: `run_simulation(final_demand=FD, A_before=A, A_after=A_new, VA_before=VA, VA_after=VA_new, isic_map=isic_map)`
- [ ] Interpret results (check X, VA, efficiency)

## Key Principle

**🔑 Always use the SAME final demand for baseline and changed scenarios!**

This isolates the pure effect of technological change from demand-side effects.

```python
# ✓ CORRECT
final_demand = np.array([200, 180, 220, 150, 100, 0])
run_simulation(
    final_demand=final_demand,
    A_before=A_baseline, A_after=A_changed,
    VA_before=VA_baseline, VA_after=VA_changed,
    isic_map=isic_map
)  # Same FD!

# ✗ WRONG - using different final demands defeats the purpose
```

## Troubleshooting

**Problem:** Column sums > 1 after change  
**Solution:** Your coefficients are too large. Reduce the change magnitude or use `multiply` instead of `add`.

**Problem:** Leontief inverse fails (singular matrix)  
**Solution:** Your technology makes (I-A) non-invertible. Some sector may have inputs ≥ 1.

**Problem:** Results don't make sense  
**Solution:** Review tech.get_summary(), check sector indices, verify change_type and values.

**Problem:** Want to apply multiple changes  
**Solution:** Call add_coefficient_change() multiple times before apply(), or use add_sector_change() for bulk modifications.

## File Locations

- **Core module:** `Input_Output_Model/demos/util/technological_change.py`
- **Simulation engine:** `Input_Output_Model/demos/util/simulation.py`
- **Demo file:** `Input_Output_Model/demos/demo.py` (Examples 7-12 for tech change)
- **Configuration:** `Input_Output_Model/demos/configs/tech_change_examples.py`
- **Documentation:** `Input_Output_Model/demos/README_TechnologicalChange.md`

## Next Steps

1. Read full documentation: `README_TechnologicalChange.md`
2. Run example demos to see it in action
3. Create your own technological change scenarios
4. Integrate with tax policy simulations for comprehensive analysis

---
*Quick Reference v1.0 - February 2026*
