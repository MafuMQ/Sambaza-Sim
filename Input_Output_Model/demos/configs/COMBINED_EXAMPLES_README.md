# Combined Technological Change + Tax Policy Examples

## Overview

The simulation framework supports **simultaneous** technological change and tax policy modifications. This allows you to model comprehensive policy packages like:

- Green transition with carbon taxation
- Automation with universal basic income (funded by corporate tax)
- Trade liberalization with revenue-neutral tax reform

## How It Works

The `run_simulation()` function accepts both:
- `tech_change` parameter (TechnologicalChange object)
- Tax policy parameters (income/corporate tax rates before/after)

When both are present, the simulation:
1. Applies technological changes (matrix rebuild if needed)
2. Applies tax policy to the modified economy
3. Shows the combined impact in the comparison output

## Example: Green Transition + Carbon Tax

**Example 15** demonstrates this combination:

```python
# From tech_change_examples.py, example 9:
{
    "tech_change_builder": lambda isic_map: _create_green_tech_with_carbon_tax(),
    "final_demand": [250.0, 200.0, 280.0, 180.0, 170.0, 0.0],
    "use_multi_level": True,
    
    # Tax policy parameters
    "income_tax_rate_before": 0.15,
    "income_tax_rate_after": 0.20,      # +5% increase
    "corporate_tax_rate_before": 0.25,
    "corporate_tax_rate_after": 0.35,    # +10% increase (carbon tax)
    "income_tax_applies_to": "bonusWages",
    
    # Circular flow
    "iterations": 3,
    "consumption_proportions": [0.30, 0.25, 0.20, 0.15, 0.10, 0.0],
    "investment_proportions": [0.10, 0.15, 0.40, 0.25, 0.10, 0.0],
    "government_proportions": [0.20, 0.25, 0.25, 0.20, 0.10, 0.0]
}
```

## Running Combined Examples

```python
from Input_Output_Model.demos.demo import run_demo, list_examples
from Input_Output_Model.demos.util.Setup_Data import setup_data

# Setup data
setup_data(source='data/ex2')

# List all examples (combined examples are at the end)
list_examples()

# Run example 15 (Green Transition + Carbon Tax)
run_demo(15)
```

## Creating Your Own Combined Examples

### Method 1: Add to tech_change_examples.py

```python
def _create_automation_with_ubi():
    """Automation technology + UBI funded by corporate tax."""
    tech = TechnologicalChange(
        name="Automation + Universal Basic Income",
        description="Labor-saving automation with tax-funded income support"
    )
    
    # Reduce labor inputs (sector 0)
    tech.add_input_change(
        input_sector_idx=0,
        change_type="multiply",
        value=0.60  # 40% labor reduction
    )
    
    # Increase capital inputs (sector 2)
    tech.add_input_change(
        input_sector_idx=2,
        change_type="multiply",
        value=1.20  # 20% capital increase
    )
    
    return tech

# In TECH_CHANGE_EXAMPLES dict:
10: {
    "title": "COMBINED: Automation + UBI",
    "tech_change_builder": lambda isic_map: _create_automation_with_ubi(),
    "final_demand": [200.0, 200.0, 250.0, 180.0, 170.0, 0.0],
    "use_multi_level": True,
    
    # Tax change to fund UBI
    "income_tax_rate_before": 0.15,
    "income_tax_rate_after": 0.15,       # No change
    "corporate_tax_rate_before": 0.25,
    "corporate_tax_rate_after": 0.40,    # +15% to fund UBI
    "income_tax_applies_to": "bonusWages",
    
    "iterations": 3,
    "consumption_proportions": [0.35, 0.30, 0.20, 0.10, 0.05, 0.0],  # Higher consumption
    "investment_proportions": [0.10, 0.15, 0.45, 0.20, 0.10, 0.0],   # More automation investment
    "government_proportions": [0.25, 0.25, 0.20, 0.20, 0.10, 0.0]    # UBI distribution
}
```

### Method 2: Call run_simulation() Directly

```python
from Input_Output_Model.demos.util.simulation import run_simulation
from Input_Output_Model.demos.util.technological_change import TechnologicalChange

# Create tech change
tech = TechnologicalChange(name="My Tech Change")
tech.add_input_change(input_sector_idx=1, change_type="multiply", value=0.70)

# Run with both tech and tax changes
run_simulation(
    final_demand=[200, 200, 200, 200, 200, 0],
    tech_change=tech,  # Tech change object
    
    # Tax policy (before/after different)
    income_tax_rate_before=0.15,
    income_tax_rate_after=0.20,
    corporate_tax_rate_before=0.25,
    corporate_tax_rate_after=0.35,
    
    iterations=3,
    consumption_proportions=[0.30, 0.25, 0.20, 0.15, 0.10, 0.0],
    investment_proportions=[0.10, 0.15, 0.40, 0.25, 0.10, 0.0],
    government_proportions=[0.20, 0.25, 0.25, 0.20, 0.10, 0.0]
)
```

## Use Cases

### 1. **Climate Policy Package**
- Tech: Energy efficiency improvements, renewable energy adoption
- Tax: Carbon tax (corporate tax increase), revenue recycling via green investment

### 2. **Automation + Labor Adjustment**
- Tech: Robotics, AI reducing labor inputs
- Tax: Robot tax (corporate tax) funding worker retraining and UBI

### 3. **Trade Liberalization + Tax Reform**
- Tech: Improved logistics, reduced trade barriers (lower input costs)
- Tax: Revenue-neutral tax shift (lower corporate, higher consumption)

### 4. **Circular Economy Transition**
- Tech: Material efficiency, recycling (reduced virgin material inputs)
- Tax: Virgin material tax, recycled material subsidies (via government spending)

## Output Interpretation

When running combined examples, the simulation shows:

1. **Technological Effect**: Changes in A matrix, resource efficiency
2. **Tax Effect**: Changes in VA distribution (C/I/G split)
3. **Combined Effect**: Net impact on GDP, sectoral output, income distribution

The "Technological Change + Tax Policy Change" label appears in the simulation header when both are detected.

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tech_change` | TechnologicalChange object | None |
| `income_tax_rate_before` | Income tax rate in baseline | 0.0 |
| `income_tax_rate_after` | Income tax rate after changes | Before rate |
| `corporate_tax_rate_before` | Corporate tax rate in baseline | 0.0 |
| `corporate_tax_rate_after` | Corporate tax rate after changes | Before rate |
| `iterations` | Number of circular flow iterations | 1 |
| `consumption_proportions` | How C is distributed | Proportional |
| `investment_proportions` | How I is distributed | Proportional |
| `government_proportions` | How G is distributed | Proportional |

## Notes

- Both tech change and tax policy modifications are relative to the **same baseline**
- If only tech changes exist, tax rates default to 0.0 (no taxes)
- If only tax changes exist, A matrix remains unchanged (baseline technology)
- Combined examples show the **interaction effects** between technology and policy
- Use `iterations > 1` to see how effects compound over multiple periods
