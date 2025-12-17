# Input_Output Subproject

## Overview

The `Input_Output` subproject implements an Input-Output (IO) economic model, including data structures, database management, and computational routines for evaluating production, goods, and technological changes. It is designed to:

- Model economic sectors, goods, and production methods using ISIC codes
- Store and manage data using SQLite and SQLAlchemy ORM
- Compute IO tables, Leontief inverse matrices, and simulate technological changes
- Provide demonstration scripts for running and evaluating the model

## Key Features

- **Goods, Producers, and Productions**: Each is represented as a database entity with ISIC classification and relevant attributes.
- **Good Indices**: Track sectoral indices, prices, and value added.
- **Production Methods**: Store input requirements, value added, and efficiency metrics.
- **Evaluators**: Functions to compute IO matrices, value added, and Leontief inverses.
- **Demonstrators**: Scripts to showcase model capabilities, including demand shocks and technological improvements.

## Directory Structure

- `models/entities/` — ORM models for Goods, Good Indices, Producers, Productions, etc.
- `util/` — Evaluation and utility functions for IO analysis
- `demos/` — Demo scripts and setup utilities
- `tests/` — Test scripts for model integrity and acceptance

## Usage

1. **Setup**: Ensure Python 3.8+ and install dependencies from `requirements.txt`.
2. **Run Demo**: Execute the main demo script:
   ```powershell
   & C:/Python313/python.exe Input_Output/demos/demo.py
   ```
   This will:
   - Initialize the database
   - Add sample producers, goods, and productions
   - Compute IO matrices and Leontief inverses
   - Simulate technological changes and output results

3. **Database**: Data is stored in `data.db` (or `dataDEMO.db` for demo runs).

## Main Components

- **GoodsDatabase, GoodIndiceDatabase, ProductionsDatabase**: Classes for managing entities in SQLite.
- **Evaluators**: Functions for:
  - ISIC code parsing and assignment
  - Price and value added calculation
  - IO matrix and Leontief inverse computation
  - Technological change simulation
- **Demo Script**: `demos/demo.py` demonstrates the full workflow, including:
  - Data setup
  - Matrix construction
  - Leontief inverse calculation
  - Demand and technology shock simulation

## Output

The demo prints:
- Database initialization and entity creation logs
- IO matrices and Leontief inverses (before/after technological change)
- Value added by sector and changes due to technology
- Verification of IO accounting

## Extending

- Add new goods, producers, or production methods by extending the ORM models
- Implement new evaluators or demonstrators for custom scenarios
- Integrate with other economic models or data sources as needed

## References
- [Leontief Input-Output Model](https://en.wikipedia.org/wiki/Input–output_model)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)

---

For details, see the code in `util/Evaluators.py`, `models/entities/`, and the demo script in `demos/demo.py`.
