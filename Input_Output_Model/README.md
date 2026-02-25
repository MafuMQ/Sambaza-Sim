# Input-Output Economic Model

A computational Input-Output (IO) economic modeling system built on Leontief's framework, extended with multi-tiered supply curves, value-added decomposition, tax policy simulation, and multi-level technological change analysis.

The system models an economy as a network of interdependent sectors. Each sector requires inputs from other sectors to produce its output, captured in a monetary coefficient matrix **A**. From this structure, the model derives equilibrium production, income distribution, and the effects of policy or technology shocks.

## Core Ideas

### The Leontief System

Given a final demand vector **d**, gross output **X** is:

$$\mathbf{X} = (\mathbf{I} - \mathbf{A})^{-1} \cdot \mathbf{d}$$

where **A** is the monetary technical coefficient matrix: `A[i,j]` = dollars of input *i* per dollar of output *j*.

### Data Flow: Three Levels of Abstraction

The model maintains three levels of economic data, each building on the one below:

```
┌─────────────────────────────────────────────────────────┐
│  Level 1: Coefficient Matrix (A matrix)                 │
│  ← Direct coefficients for Leontief solving             │
│  ← Rebuilt from ↓                                       │
├─────────────────────────────────────────────────────────┤
│  Level 2: Supply Curves (Flow Matrix)                   │
│  ← Tiered price/capacity structures per good            │
│  ← Merit-order sorted (cheapest first)                  │
│  ← Rebuilt from ↓                                       │
├─────────────────────────────────────────────────────────┤
│  Level 3: Productions (Firm-Level)                      │
│  ← Individual production methods with input recipes     │
│  ← Value-added decomposition (wages, surplus)           │
│  ← The ground truth of the economy                      │
└─────────────────────────────────────────────────────────┘
```

**Productions** (firms with specific input recipes) are aggregated into **Supply Curves** (merit-order tiered market structures), which are then used to build the **Coefficient Matrix** for Leontief analysis.

### Circular Flow

The model implements the national income identity **VA = FD = C + I + G**:

1. Final demand **d** drives production: **X** = (I − A)⁻¹ · **d**
2. Production generates value added: **VA** = wages + surplus
3. After-tax VA is decomposed into Consumption, Investment, and Government spending
4. These become the next period's final demand → iterate

### Value Added Decomposition

Each production method breaks down its value added into:
- **minWages** — Base/minimum wages
- **bonusWages** — Performance/incentive wages
- **surplus** — Gross operating surplus (profits)

This decomposition enables tax policy analysis where income tax applies to wages and corporate tax applies to surplus.

## Project Structure

```
Input_Output_Model/
├── README.md
├── notes.txt                           # Design notes, limitations, and roadmap
│
├── models/
│   ├── entities/
│   │   ├── Good.py                     # Good entity — ISIC classified products
│   │   ├── Production.py              # Production method — input recipes & VA
│   │   ├── SupplyCurve.py             # Tiered supply curve — price/capacity steps
│   │   └── TechChange.py              # TechChange entity — DB-backed example store
│   └── table/
│       └── solver.py                   # DynamicEquilibriumSolver — supply curve solver
│
├── util/
│   └── Evaluators.py                   # Matrix builders, curve builders, price evaluation
│
└── demos/
    ├── demo.py                         # Unified demo runner (examples 7–16)
    ├── test_capital_investment.py      # Tests: 2-phase capital investment (3 tests)
    ├── configs/
    │   ├── __init__.py
    │   └── load_tech_changes.py        # CSV → DB loader + rebuild_examples_dict_from_db()
    └── util/
        ├── Setup_Data.py               # Data loading (CSV or random generation)
        ├── simulation.py               # Unified simulation engine + 2-phase capital
        └── technological_change.py     # Multi-level tech change + capital requirements
```

All tech change examples are driven by **data files** (`data/ex2/tech_changes.csv` and `data/ex2/tax_policies.csv`) rather than hardcoded Python. `setup_data()` loads them into `data.db` automatically at step [4/4].

## Data Model

All entities are persisted in SQLite via SQLAlchemy ORM (`data.db`).

### Good

A product or service classified by ISIC code.

| Field | Description |
|-------|-------------|
| `name` | Short name (e.g., "Steel") |
| `id_number` | Unique numeric identifier |
| `isic` | ISIC code (e.g., `A2410_100_1`) |
| `isic_section`, `isic_division`, ... | Parsed ISIC components |

### Production

A specific method of producing a Good—the fundamental unit of the model. Each Good can have multiple competing Production methods.

| Field | Description |
|-------|-------------|
| `produce` | ID of the Good this method produces |
| `production_inputs` | JSON: `{isic: monetary_cost}` — dollars of each input needed |
| `production_added_values` | JSON: `{minWages, bonusWages, wages, surplus}` |
| `production_quantity` | Capacity (units this method can produce) |
| `price` | Total cost = Σ inputs + Σ value added |
| `production_material_efficiency` | Efficiency metrics |

A special `IMPORT` production exists for each Good, using the Foreign Exchange good (`A9999_999_999`) as its sole input.

### Supply Curve

Aggregates all Production methods for a Good into a tiered merit-order supply curve.

| Field | Description |
|-------|-------------|
| `price` | JSON: `[{cap, price}, ...]` — tiers sorted cheapest first |
| `total_inputs_cost` | JSON: `[{cap, price}, ...]` — input cost curve |
| `total_value_added` | JSON: `[{cap, price}, ...]` — VA curve |

Each tier represents a production method's capacity and cost. The solver fills cheapest tiers first (merit order).

## Key Components

### Evaluators (`util/Evaluators.py`)

The mathematical core:

- **`evaluate_productions_price()`** — Computes production prices: price = Σ(input costs) + Σ(value added)
- **`build_tiered_supply_curve()`** — Sorts productions by merit order, builds three parallel tier arrays (price, input cost, VA)
- **`build_io_matrix()`** — Constructs the monetary coefficient matrix **A** and VA vector from the cheapest production of each Good
- **`create_leontief_inverse()`** — Computes (I − A)⁻¹ from a transaction matrix

### DynamicEquilibriumSolver (`models/table/solver.py`)

An iterative solver that uses supply curve tiers instead of fixed coefficients:

1. Estimate output quantities
2. Calculate prices from supply curve tiers (weighted average across active tiers)
3. Solve the Leontief system with monetary coefficients
4. Iterate until convergence

### Technological Change (`demos/util/technological_change.py`)

A multi-level system for modeling how technology shocks propagate through the economy. Changes at any level cascade upward:

| Level | Target | Rebuild Chain | Use Case |
|-------|--------|---------------|----------|
| **1 — Matrix** | A matrix coefficients directly | None | Quick what-if analysis |
| **2 — Curves** | Supply curve tiers | Curves → Matrix | Capacity expansion, cost changes |
| **3 — Productions** | Firm-level input recipes | Productions → Curves → Matrix | Process improvements |

See [Usage Reference](#usage-reference) for comprehensive API documentation and examples.

### Simulation Engine (`demos/util/simulation.py`)

Unified function supporting all analysis modes:

- **Tax policy comparison** — Before/after tax rate changes with circular flow
- **Technology comparison** — Baseline vs. modified technology, same final demand
- **Combined** — Simultaneous tech change + tax policy shift
- **Multi-period** — Circular flow iterations where VA(t) → FD(t+1)
- **2-Phase capital investment** — When `tech_change.has_capital_requirements()` is true, simulation automatically runs an Investment phase before the technology phase (see [Capital Investment](#capital-investment))

Supports two solver modes:
- `leontief` — Standard Leontief inverse
- `supply_curves` — DynamicEquilibriumSolver with tiered pricing

### Data Setup (`demos/util/Setup_Data.py`)

Two data sources:
- **CSV files** — Load from `data/ex1/` or `data/ex2/` (`goods.csv` + `productions.csv` + optional `tech_changes.csv` + `tax_policies.csv`)
- **Random generation** — Creates a synthetic economy with 5 goods, 4 domestic productions each, plus import options

When a `tech_changes.csv` is present in the source directory it is automatically loaded into the database at step [4/4] of `setup_data()`.

## Getting Started

### Prerequisites

```bash
pip install sqlalchemy sympy numpy pandas faker matplotlib
```

### Quick Start Examples

#### Setup Data

All examples start with loading data:

```python
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from Input_Output_Model.demos.util.Setup_Data import setup_data

# Load from CSV or generate random data
setup_data(source="data/ex2", overwrite_existing_data=True)
```

#### Tax Policy Analysis

```python
from Input_Output_Model.demos.util.simulation import run_simulation

# Compare before/after tax rate changes with circular flow
run_simulation(
    total_demand=1000.0,
    proportions=[0.25, 0.20, 0.30, 0.15, 0.10, 0.0],
    income_tax_rate_before=0.10,
    income_tax_rate_after=0.20,
    corporate_tax_rate_before=0.25,
    corporate_tax_rate_after=0.25,
    income_tax_applies_to="bonusWages",
    iterations=5
)
```

#### Technological Change — Level 1 (Matrix)

Direct modifications to the coefficient matrix for quick what-if analysis:

```python
from Input_Output_Model.demos.util.technological_change import TechnologicalChange
from Input_Output_Model.util.Evaluators import build_io_matrix

# Build baseline matrix
result = build_io_matrix()
A_baseline = result['A_matrix']

# Create tech change: 30% energy efficiency improvement
tc = TechnologicalChange(name="Energy Efficiency")
tc.add_coefficient_change(
    from_isic="A3510_100_1",  # Energy sector
    to_isic="all",             # All consuming sectors
    change_type="multiply",
    value=0.70                 # 30% reduction
)

result = tc.apply()
A_modified = result['A_matrix']
```

#### Technological Change — Level 2 (Supply Curves)

Modify tiered supply curves, then rebuild the coefficient matrix:

```python
from Input_Output_Model.models.entities.SupplyCurve import SupplyCurveDatabase

scdb = SupplyCurveDatabase()

# Add new low-cost production capacity
tc = TechnologicalChange(name="Capacity Expansion")
tc.add_curve_new_tier(
    isic="A01",
    tier_price=80.0,
    tier_cap=150.0,
    tier_inputs_cost=50.0,
    tier_va=30.0,
    position=0  # Insert as cheapest tier
)

result = tc.apply_to_curves(scdb, rebuild_matrix=True)
A_modified = result['A_matrix']
```

#### Technological Change — Level 3 (Productions)

Modify firm-level production methods, cascade through curves to matrix:

```python
from Input_Output_Model.models.entities.Production import ProductionsDatabase
from Input_Output_Model.models.entities.Good import GoodDatabase

ptdb = ProductionsDatabase()
scdb = SupplyCurveDatabase()
gdb = GoodDatabase()

# Improve a specific production's efficiency
tc = TechnologicalChange(name="Factory Modernization")

# Reduce a specific input by 15%
tc.add_production_input_change(
    production_id=5,
    input_isic="A2410_100_1",
    change_type="multiply",
    value=0.85
)

# Reduce all inputs by 10%
tc.add_production_all_inputs_change(
    production_id=5,
    change_type="multiply",
    value=0.90
)

# Full cascade: productions → curves → coefficient matrix
result = tc.apply_to_productions(ptdb, scdb, gdb, 
                                  rebuild_curves=True, 
                                  rebuild_matrix=True)
A_modified = result['A_matrix']
supply_curves = result['supply_data']
```

#### Combined: Tech Change + Tax Policy

```python
# Create technological change
tc = TechnologicalChange(name="Green Transition")
tc.add_coefficient_change(
    from_isic="A3510_100_1",
    to_isic="all",
    change_type="multiply",
    value=0.70
)

# Run simulation with tech change AND tax policy shift
run_simulation(
    total_demand=1000.0,
    proportions=[0.25, 0.20, 0.30, 0.15, 0.10, 0.0],
    income_tax_rate_before=0.10,
    income_tax_rate_after=0.15,
    corporate_tax_rate_before=0.25,
    corporate_tax_rate_after=0.25,
    income_tax_applies_to="bonusWages",
    iterations=5,
    tech_change=tc  # Apply tech change in both scenarios
)
```

### Running Demos

```bash
# From project root:
python Input_Output_Model/demos/demo.py          # default: example 7
python Input_Output_Model/demos/demo.py 16       # run specific example
```

Pass the example number as a command-line argument (7–16). The demo auto-lists all available examples on startup.

### Running Tests

```bash
# From project root:
python Input_Output_Model/demos/test_capital_investment.py
```

Runs 3 tests and exits 0 if all pass:
1. **Capital investment** — 2-phase run: asserts correct phase labels, FD conservation, and FD == baseline throughout
2. **No capital** — control run: asserts all phases are `None`, FD conserved
3. **Via CSV** — end-to-end: loads `data/ex2/tech_changes.csv` → DB → builder → `run_simulation`, asserts all of the above

## Demo Examples

### Tax Policy (Examples 1–6)

| # | Title | Description |
|---|-------|-------------|
| 1 | Circular Flow — Stable GDP | VA = FD identity over 5 iterations, no savings |
| 2 | Income Tax Increase | 10% → 20%, impact on consumption and growth |
| 3 | Corporate Tax Increase | 25% → 35%, impact on investment capacity |
| 4 | Demand Shock | Targeted sector-specific demand increase |
| 5 | Uniform Demand | Equal demand across all sectors |
| 6 | Differentiated C/I/G | Separate distribution vectors for C, I, G |

### Technological Change — Matrix Level (Examples 7–12)

| # | Title | Description |
|---|-------|-------------|
| 7 | Energy Efficiency | 30% energy input reduction economy-wide |
| 8 | Productivity Improvement | 20% all-input reduction in one sector |
| 9 | Automation Substitution | Reduce labor 40%, increase capital 15% |
| 10 | Material Efficiency | 25% material waste reduction |
| 11 | Green Transition | Economy-wide multi-sector efficiency gains |
| 12 | Custom Coefficients | Fine-grained coefficient targeting |

### Multi-Level (Examples 13–15)

| # | Title | Description |
|---|-------|-------------|
| 13 | Production-Level Modernization | Level 3 + Level 1 changes, matrix rebuilt from modified productions |
| 14 | Capacity Expansion | Level 2 curve tier changes, new capacity at lower cost |
| 15 | Green Transition + Carbon Tax | Simultaneous tech change (efficiency) + tax policy (carbon pricing) |

### Capital Investment (Example 16)

| # | Title | Description |
|---|-------|-------------|
| 16 | Energy Efficiency with Capital Investment | 20% energy reduction requiring $300 upfront capital — 2 investment iterations then 3 new-technology iterations |

## Usage Reference

### Demand Specification

Choose **one** method to specify final demand:

| Method | Parameters | Example |
|--------|-----------|---------|
| Direct vector | `final_demand=[250, 200, ...]` | Full control over each sector |
| Total + proportions | `total_demand=1000, proportions=[0.25, ...]` | Distribute a total amount |
| Uniform | `uniform_demand=75.0` | Equal demand, all sectors |
| Sector shock | `target_isic="A01", demand_shock=200` | Single sector increase |

### Tax Parameters

| Parameter | Description |
|-----------|-------------|
| `income_tax_rate_before/after` | Income tax rate (0.0–1.0) |
| `corporate_tax_rate_before/after` | Corporate tax rate on surplus |
| `income_tax_applies_to` | `"minWages"`, `"bonusWages"`, `"wages"`, or `"both"` |
| `consumption_proportions` | How C is distributed across sectors |
| `investment_proportions` | How I is distributed across sectors |
| `government_proportions` | How G is distributed across sectors |
| `iterations` | Number of circular flow periods |
| `consumption_rate` | Share of after-tax income consumed (default 1.0) |

### Technological Change API

The `TechnologicalChange` class provides methods for each level. Chain multiple changes in one object:

#### Level 1 — Matrix Methods

```python
# Modify specific coefficient
tc.add_coefficient_change(from_isic="A01", to_isic="A02", 
                          change_type="multiply", value=0.85)

# Modify all inputs to a sector
tc.add_coefficient_change(from_isic="all", to_isic="A02",
                          change_type="multiply", value=0.90)

# Modify one input across all sectors
tc.add_coefficient_change(from_isic="A3510_100_1", to_isic="all",
                          change_type="multiply", value=0.70)

# Modify entire sector's row and column
tc.add_sector_change(sector_isic="A02", change_type="multiply", value=0.80)

# Apply all Level 1 changes
result = tc.apply()
```

#### Level 2 — Supply Curve Methods

```python
# Modify specific tier's price
tc.add_curve_tier_change(isic="A01", tier_index=0, field="price",
                         change_type="multiply", value=0.90)

# Add new capacity tier
tc.add_curve_new_tier(isic="A01", tier_price=75.0, tier_cap=200.0,
                      tier_inputs_cost=45.0, tier_va=30.0, position=0)

# Remove a tier
tc.add_curve_remove_tier(isic="A01", tier_index=2)

# Scale all tiers in a curve
tc.add_curve_scale_all_tiers(isic="A01", field="price",
                              change_type="multiply", value=0.95)

# Apply and rebuild matrix
result = tc.apply_to_curves(scdb, rebuild_matrix=True)
```

#### Level 3 — Production Methods

```python
# Modify specific input
tc.add_production_input_change(production_id=5, input_isic="A01",
                               change_type="multiply", value=0.85)

# Modify all inputs
tc.add_production_all_inputs_change(production_id=5,
                                    change_type="multiply", value=0.90)

# Modify value added component
tc.add_production_va_change(production_id=5, va_component="surplus",
                            change_type="multiply", value=1.10)

# Modify efficiency
tc.add_production_efficiency_change(production_id=5, efficiency_field="material",
                                    change_type="set", value=0.92)

# Apply and cascade through all levels
result = tc.apply_to_productions(ptdb, scdb, gdb,
                                  rebuild_curves=True,
                                  rebuild_matrix=True)
```
#### Capital Investment

Attach capital spending requirements to any `TechnologicalChange`. Before the new technology takes effect the simulation runs `investment_duration` iterations where final demand is **reallocated** (not increased) toward the specified capital sectors.

```python
tc = TechnologicalChange(name="Energy Efficiency Upgrade")

# Structural change: 20% reduction in energy input across all sectors
tc.add_input_change(
    input_sector_idx="A6178_413_70647",
    change_type="multiply",
    value=0.80,
)

# Capital requirement: $300 must be spent before the change takes effect
tc.set_capital_requirements(
    requirements={
        "A2227_974_71103": 200.0,   # $200 machinery
        "A3790_132_63":   100.0,   # $100 installation
    },
    investment_duration=2,          # 2 investment iterations, then new tech
)

print(tc.has_capital_requirements())   # True
print(tc.get_total_capital_cost())     # 300.0

# Pass to run_simulation — phases are handled automatically
result = run_simulation(
    uniform_demand=100.0,
    tech_change=tc,
    iterations=5,           # iters 1–2 = Investment, iters 3–5 = New Technology
    solver_type="leontief",
)
```

**2-Phase mechanics:**

| Phase | Iterations | A matrix | Final Demand |
|-------|-----------|----------|--------------|
| Investment | 1 … `investment_duration` | baseline (old technology) | reallocated toward capital sectors; total FD unchanged |
| New Technology | remainder | updated (new technology) | restored to original distribution |

Capital spending is a **reallocation** — it reduces demand from other sectors proportionally, so total FD = VA is conserved every iteration. No new money is created.

Each iteration's phase label is stored in `result['after_history'][i]['phase']` (`"Investment"` or `"New Technology"`).

#### Capital Requirements in CSV

To specify capital requirements in `tech_changes.csv`, add a `set_capital_requirements` entry in the `tech_change_params` JSON array:

```csv
example_id,... ,use_multi_level,tech_change_params
16,tech_change,...,True,"[{""method"": ""add_input_change"", ""params"": {...}}, {""method"": ""set_capital_requirements"", ""params"": {""requirements"": {""A2227_974_71103"": 200.0, ""A3790_132_63"": 100.0}, ""investment_duration"": 2}}]"
```

`set_capital_requirements` is dispatched via `getattr` like every other method, so no code changes are needed in the loader.
#### Change Types

| Type | Description | Example |
|------|-------------|---------|
| `multiply` | Scale by factor | `value=0.85` → 15% reduction |
| `add` | Add fixed amount | `value=5.0` → increase by 5 |
| `set` | Set to specific value | `value=100.0` → set to 100 |

#### Return Values

All `apply*` methods return a dictionary:

```python
{
    'A_matrix': np.ndarray,          # Modified coefficient matrix (if rebuilt)
    'VA_vector': np.ndarray,         # Modified VA vector (if rebuilt)
    'isic_map': dict,                # ISIC -> matrix index mapping
    'supply_data': dict,             # Rebuilt supply curves (Level 2/3)
    'productions': dict              # Modified productions (Level 3)
}
```

## Architecture

```
                         ┌──────────────┐
                         │   demo.py    │  Entry point: picks example, calls simulation
                         └──────┬───────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
             ┌──────┴──────┐        ┌───────┴────────┐
             │ simulation  │        │ technological  │
             │   .py       │◄───────│  _change.py    │  Applies tech changes,
             │             │        │                │  rebuilds matrices
             └──────┬──────┘        └───────┬────────┘
                    │                       │
          ┌─────────┴─────────┐    ┌────────┴────────┐
          │                   │    │                  │
    ┌─────┴─────┐     ┌──────┴────┴──┐      ┌───────┴──────┐
    │ solver.py │     │ Evaluators.py │      │ Setup_Data.py│
    │           │     │               │      │              │
    │ Dynamic   │     │ build_io_     │      │ CSV / random │
    │ Equil.    │     │ matrix()      │      │ data loader  │
    │ Solver    │     │ build_curves()│      │              │
    └─────┬─────┘     └──────┬────────┘      └───────┬──────┘
          │                  │                        │
          └──────────────────┴────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │    data.db      │
                    │  (SQLite/ORM)   │
                    │                 │
                    │ Good            │
                    │ Production      │
                    │ SupplyCurve     │
                    └─────────────────┘
```

## Known Limitations

1. **No price-driven substitution** — Technology changes affect VA proportionally; reduced input cost increases surplus/wages rather than reducing prices. Growth only occurs when excess VA feeds back as higher FD.
2. **Static prices** — Good prices don't change dynamically; elasticity and substitution are not modeled.
3. **No savings/leakage** — The circular flow assumes full spending (consumption rate = 1.0 by default).
4. **Hardcoded labor** — Bonus wages are embedded in production recipes, not adjustable as labor market variables.
5. **No diminishing returns** — Investment returns are linear; no ICOR dynamics yet.
6. **Cannot mix Level 2 and Level 3 changes in one TechnologicalChange object** — If you add both production changes (Level 3) and curve changes (Level 2) to a single `TechnologicalChange` object, calling `apply_to_productions()` will rebuild the supply curves entirely from production records, which overwrites any Level 2 curve modifications. Use separate objects or apply Level 2 changes after the Level 3 cascade completes.
7. **Capital investment is always a reallocation** — `set_capital_requirements` re-routes existing demand; it does not model external financing, borrowing, or savings drawdown. The total FD is strictly conserved across all investment iterations.

See [notes.txt](notes.txt) for the full design notes and roadmap.

## Roadmap

- [x] Government-driven investment applying technological change (2-phase capital + tech)
- [ ] External financing / savings drawdown for capital investment
- [ ] Savings, debt tracking, and leakage modeling
- [ ] Diminishing returns on investment
- [ ] Investment ranking and multi-option optimization (Harrod-Domar/ICOR)
- [ ] International trade modeling
- [ ] Price elasticity and substitution effects
- [ ] Time-rate dynamics (production delays)

## References

- Leontief, W. (1986). *Input-Output Economics*
- Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions*
- United Nations. (2018). *ISIC Rev.4*

## Related Subprojects

- **Growth/** — Harrod-Domar and Leontief growth models
- **MarxReproductionModel/** — Marx's reproduction schemes
- **Optimization/** — Economic optimization algorithms
