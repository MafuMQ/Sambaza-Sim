# Sambaza-Sim: Input-Output Economic Model

A computational Input-Output (IO) economic modeling system built on Leontief's framework, recently completely refactored into a scalable, object-oriented architecture. The system models an economy as a network of interdependent sectors, deriving equilibrium production, income distribution, and the cascading effects of tax policy or technology shocks.

## Core Ideas

### The Leontief System

Given a final demand vector **d**, gross output **X** is:

$$\mathbf{X} = (\mathbf{I} - \mathbf{A})^{-1} \cdot \mathbf{d}$$

where **A** is the monetary technical coefficient matrix: `A[i,j]` = dollars of input *i* per dollar of output *j*.

### Data Flow: Three Levels of Abstraction

The model maintains three levels of economic data, each building on the one below:

```text
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

## Project Structure

The project has been refactored for clarity and separation of concerns:

```text
Sambaza-Sim/
├── README.md
├── requirements.txt
├── app.py                          # Dashboard application entry point
├── data.db                         # SQLite local database (generated)
│
├── core/                           # Core economic engine algorithms
│   ├── io_matrix.py                # Matrix evaluation
│   ├── simulation.py               # Orchestrator & Scenario logic
│   ├── tech_change.py              # Multi-tiered technological shock model
│   ├── demand.py                   # Demand calculations
│   ├── income.py                   # Tax application
│   └── solvers/                    # Dynamic equilibrium & Leontief solvers
│
├── db/                             # SQLAlchemy persistence layer
│   ├── base.py                     # Database engine configuration mixin
│   └── repositories/               # ORM controllers (Goods, Productions, SupplyCurves)
│
├── pipeline/                       # Data seeding layer
│   ├── setup_data.py               # Instantiates or resets the database from scratch or CSV
│   └── load_tech_changes.py        # Maps tax/tech policies from CSV
│
├── ui/                             # Dash/Plotly Dashboard Layer
│   ├── layout.py                   # HTML layout nodes
│   ├── charts.py                   # Plotly figure generation
│   └── callbacks.py                # State transitions and simulation binds
│
├── demos/                          # CLI Sandbox and automated runners
│   ├── demo.py                     # Primary examples entrypoint
│   └── demo_db.py                  # Database-backed demo queries
│
└── tests/                          # Automated validation suite
    └── test_smoke.py
```

## Getting Started

### Prerequisites

All Python dependencies are listed in `requirements.txt`. Install them using pip:

```bash
pip install -r requirements.txt
# Alternatively natively:
pip install dash dash-ag-grid plotly pandas numpy sqlalchemy sympy faker
```

### 1. Database Initialization

All examples and interactive dashboards rely on the underlying SQLite database (`data.db`). **To set this up, run the data pipeline:**

```bash
# This will ingest `data/ex2/` CSVs or generate synthetic tables natively
python pipeline/setup_data.py
```

*Note: If `data.db` exists remotely it will prompt to clear or run cleanly. You can also run it programmatically via `setup_data(overwrite_existing_data=True)`.*

### 2. Running the Interactive Dashboard

Launch the local Dash server to explore the scenarios visually:

```bash
python app.py
```

Open your browser to `http://127.0.0.1:8050/`. The dashboard allows you to step through all defined tax policy and technology examples interactively, showing graphical outputs and sector-by-sector comparisons dynamically.

### 3. Running CLI Demos

To execute a scenario via the unified CLI runner:

```bash
python demos/demo.py      # Lists options and runs a default
python demos/demo.py 16   # Executes specific example directly
```

## Features Deep Dive

### Simulation Engine (`core/simulation.py`)

A unified function supporting multiple analysis iterations:
- **Tax policy comparison** — Before/after tax rate adjustments with circular flow.
- **Technology comparison** — Isolation of technology changes holding final demand constant.
- **Combined Analysis** — Simultaneous technology shock AND taxation changes in the same tick.

Supports two native solvers within `core/solvers/`:
- `leontief` — Standard direct Leontief Inverse computation.
- `supply_curves` — Dynamic Equilibrium Tiered Solver based on capacity allocation.

### Technological Change (`core/tech_change.py`)

A multi-level system cascading technology shocks recursively:
- **Level 1 (Matrix)**: Modifies `A` matrix directly for quick ad-hoc analysis.
- **Level 2 (Curves)**: Modifies supply curve limits enabling capacity augmentation.
- **Level 3 (Productions)**: Modifies root firm-level input equations directly and propagates downstream to Curves and Matrices automatically.

*Capital Investment features exist natively: You can attach capital limits causing simulations to dynamically redirect resources to specified 'investment required' sectors before enabling the technology matrix changes!*

## Development & Verification

### Running the Suite

Smoke tests have been provided to validate mathematical consistency and matrix building:

```bash
python -m pytest tests/test_smoke.py -v
```

### Known Limitations

1. **No price-driven substitution** — Technology changes affect VA proportionally; reduced input cost increases surplus/wages rather than reducing prices dynamically yet.
2. **Static prices** — Sector prices don't adjust elastically against substitution yet.
3. **Strict Iteration Reallocation** — Capital requirements strictly 'reallocate' demand instead of simulating exterior capital financing injections. No new money is artificially injected outside of tracking identity laws.

## References

- Leontief, W. (1986). *Input-Output Economics*
- Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions*
- United Nations. (2018). *ISIC Rev.4*
