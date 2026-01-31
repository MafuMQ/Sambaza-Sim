# Input-Output Economic Model

A comprehensive Input-Output (IO) economic modeling system with support for circular flow dynamics, tax policy analysis, and dynamic equilibrium solving. This subproject implements Leontief Input-Output tables with multi-tiered supply curves and value-added decomposition.

## Features

### Core Capabilities
- **Dynamic Equilibrium Solving**: Multi-tiered supply curve modeling with price discovery
- **Input-Output Matrix Construction**: Automatic generation of monetary coefficient matrices ($ input per $ output)
- **Value Added Decomposition**: Tracks minimum wages, bonus wages, total wages, and surplus
- **Tax Policy Simulation**: Compare economic impacts of income and corporate tax changes
- **Circular Flow Dynamics**: Model how value added becomes consumption and investment in subsequent periods
- **ISIC Classification**: Full support for International Standard Industrial Classification codes

### Economic Modeling
- **VA = FD Identity**: Maintains Value Added = Final Demand equilibrium through proper scaling
- **Multi-Period Simulation**: Circular flow iterations where VA(t) → C+I+G(t+1)
- **Flexible Demand Specification**: Direct vectors, uniform, proportional, or sector-specific demand shocks
- **Foreign Exchange Handling**: Special good for representing import costs

## Project Structure

```
Input_Output_Model/
├── demos/
│   ├── demo.py                         # Example simulations and usage demonstrations
│   └── util/
│       ├── Setup_Data.py               # Data initialization and foreign exchange setup
│       └── tax_policy_simulation.py    # Tax policy simulation engine
├── models/
│   ├── entities/
│   │   ├── Good.py                     # Good entity with ISIC classification
│   │   ├── Production.py               # Production method entity with inputs/VA
│   │   └── SupplyCurve.py              # Multi-tiered supply curve entity
│   └── table/
│       ├── solver.py                   # Dynamic equilibrium solver
│       └── issues.md                   # Known issues and technical notes
└── util/
    └── Evaluators.py                   # IO matrix builders and evaluators
```

## Installation

### Prerequisites
- Python 3.8+
- Required packages (from root `requirements.txt`):

```bash
pip install sqlalchemy sympy numpy pandas faker matplotlib ipywidgets seaborn
```

### Setup
1. Ensure you're in the project root directory
2. Data files should be in `data/` directory with structure:
   ```
   data/
   ├── ex1/
   │   ├── goods.csv
   │   └── productions.csv
   └── ex2/
       ├── goods.csv
       └── productions.csv
   ```

## Quick Start

### Basic Example

```python
from Input_Output_Model.demos.util.Setup_Data import setup_data
from Input_Output_Model.demos.util.tax_policy_simulation import run_tax_policy_simulation

# Setup sample data
setup_data(source="data/ex2", overwrite_existing_data=True)

# Run a tax policy simulation
run_tax_policy_simulation(
    total_demand=1000.0,
    proportions=[0.25, 0.20, 0.30, 0.15, 0.10, 0.0],
    income_tax_rate_before=0.10,
    income_tax_rate_after=0.15,
    corporate_tax_rate_before=0.25,
    corporate_tax_rate_after=0.30,
    income_tax_applies_to="bonusWages",
    iterations=5
)
```

### Running Demo Examples

```bash
python Input_Output_Model/demos/demo.py
```

The demo file includes 6 comprehensive examples:
1. **Circular Flow Model**: Demonstrates stable GDP in closed economy
2. **Income Tax Increase**: Impact of raising income tax from 10% to 20%
3. **Corporate Tax Increase**: Effects of corporate tax changes on investment
4. **Sector-Specific Demand Shock**: Targeted demand increase in specific sector
5. **Uniform Demand Distribution**: Equal demand across all sectors
6. **Direct Demand Vector**: Custom demand specification

## Usage Guide

### Demand Specification

Choose **ONE** of the following methods:

#### 1. Direct Vector
```python
demand_vector = [150.0, 120.0, 180.0, 90.0, 60.0, 0.0]
run_tax_policy_simulation(demand_vector=demand_vector, ...)
```

#### 2. Total Demand with Proportions
```python
run_tax_policy_simulation(
    total_demand=1000.0,
    proportions=[0.25, 0.20, 0.30, 0.15, 0.10, 0.0],
    ...
)
```

#### 3. Uniform Demand
```python
run_tax_policy_simulation(uniform_demand=75.0, ...)
```

#### 4. Sector-Specific Shock
```python
run_tax_policy_simulation(
    target_isic="A2327_978_13",
    demand_shock=200.0,
    ...
)
```

### Tax Parameters

#### Income Tax
- `income_tax_rate_before`: Initial income tax rate (0.0 to 1.0)
- `income_tax_rate_after`: New income tax rate for comparison
- `income_tax_applies_to`: What to tax
  - `"minWages"`: Tax only minimum wages
  - `"bonusWages"`: Tax only bonus wages
  - `"wages"`: Tax total wages
  - `"both"`: Tax both min and bonus separately

#### Corporate Tax
- `corporate_tax_rate_before`: Initial corporate tax on surplus
- `corporate_tax_rate_after`: New corporate tax rate

### Circular Flow Parameters

- `iterations`: Number of circular flow iterations (default=1)
- `consumption_rate`: Share of after-tax income consumed (0.0 to 1.0, default=1.0)
- `consumption_distribution`: How to distribute consumption
  - `"proportional"`: Based on sector output shares (default)
  - `"uniform"`: Equal across all sectors
  - `"manual"`: Use the proportions parameter

#### Separate FD Component Proportions (New Feature)

For multi-iteration simulations, you can now specify different distribution patterns for each Final Demand component:

- `consumption_proportions`: Distribution vector for Consumption (C)
- `investment_proportions`: Distribution vector for Investment (I)
- `government_proportions`: Distribution vector for Government spending (G)

Each vector must sum to 1.0. This allows realistic modeling where:
- Consumption favors consumer goods sectors
- Investment favors capital goods sectors  
- Government spending follows policy priorities

**Example:**
```python
# Consumption favors consumer sectors
consumption_props = [0.35, 0.30, 0.20, 0.10, 0.05, 0.0]
# Investment favors capital goods
investment_props = [0.10, 0.15, 0.40, 0.25, 0.10, 0.0]
# Government more balanced
government_props = [0.20, 0.20, 0.20, 0.20, 0.20, 0.0]

run_tax_policy_simulation(
    total_demand=500.0,
    proportions=[0.30, 0.25, 0.20, 0.15, 0.10, 0.0],  # Initial demand only
    consumption_proportions=consumption_props,
    investment_proportions=investment_props,
    government_proportions=government_props,
    iterations=5
)
```

If these are not provided, the system falls back to the `consumption_distribution` setting.

## Key Concepts

### Input-Output Matrix
The system builds a monetary coefficient matrix **A** where:
- `A[i,j]` = dollars of good i needed per dollar of output of good j

### Value Added Components
Each production method decomposes value added into:
- **minWages**: Minimum/base wages
- **bonusWages**: Performance/incentive wages
- **wages**: Total wages (minWages + bonusWages)
- **surplus**: Gross operating surplus (profits)

### Circular Flow
1. Initial demand creates production: **X** = (I - A)⁻¹ · **d**
2. Production generates value added: **VA** = wages + surplus
3. After-tax VA becomes next period's demand (consumption + investment)
4. System maintains VA = FD equilibrium

### Dynamic Equilibrium
The solver handles multi-tiered supply curves:
- Each sector has tiers with different capacities and prices
- Solver finds equilibrium quantities and prices
- Supports scarcity pricing when demand exceeds capacity

## Database Schema

### Goods Table
- ISIC classification (section, division, group, class, sub-classes)
- Descriptive names and ID numbers
- Foreign exchange special good (ISIC: A9999_999_999)

### Productions Table
- Production inputs (JSON: {isic: quantity})
- Value added components (JSON: {minWages, bonusWages, wages, surplus})
- Efficiency metrics (material, labour, energy)
- Producer information

### Supply Curves Table
- Multi-tier pricing: [{cap, price}, ...]
- Capacity constraints per tier
- ISIC linkage to goods

## Output Metrics

Each simulation provides:
- **Economic Aggregates**: GDP, Total VA, Final Demand
- **Income Distribution**: Before/after tax wages and surplus
- **Tax Revenue**: Income and corporate tax collections
- **Sector Details**: Production, prices, VA by sector
- **Comparison Tables**: Before vs. After policy changes

## Examples from Demo

### Example 1: Stable Circular Flow
Demonstrates VA = FD identity maintenance across 5 iterations with no savings.

### Example 2: Income Tax Impact
Shows how doubling income tax (10% → 20%) affects:
- Worker take-home pay
- Government revenue
- GDP trajectory
- Consumption patterns

### Example 3: Corporate Tax Impact
Analyzes corporate tax increase (25% → 35%) impact on:
- Business surplus retention
- Investment capacity
- Economic growth

## Technical Notes

### Foreign Exchange
A special good represents foreign currency for imports:
- ISIC: `A9999_999_999`
- Fixed exchange rate (default 1.0)
- Automatically created during data setup

### Logging
Adjust logging level for different verbosity:
```python
import logging
setup_data(source="data/ex2", logging_level=logging.DEBUG)
```

### Database Backend
Uses SQLAlchemy ORM with SQLite:
- Goods: `sqlite:///goods.db`
- Productions: `sqlite:///productions.db`
- Supply Curves: `sqlite:///supply_curves.db`

## Known Issues

See [models/table/issues.md](models/table/issues.md) for detailed technical issues and workarounds.

## Contributing

When adding new features:
1. Maintain VA = FD identity
2. Update relevant database schemas
3. Add example to demo.py
4. Document parameters in docstrings

## Related Subprojects

- **Growth/**: Harrod-Domar and Leontief growth models
- **MarxReproductionModel/**: Marx's reproduction schemes
- **Optimization/**: Economic optimization algorithms

## References

- Leontief, W. (1986). *Input-Output Economics*
- Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions*
- United Nations. (2018). *International Standard Industrial Classification of All Economic Activities (ISIC), Rev.4*

## License

Part of the Sambaza-Sim project.
