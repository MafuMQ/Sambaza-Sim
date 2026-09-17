# Sambaza-Sim Input Data Structure

The application is designed to ingest macroeconomic data via a set of structured CSV files. This documentation details the expected format and purpose of these files, commonly located in environments like `data/ex2/`.

> [!NOTE]
> **No Pre-Calculated Coefficient Tables:** Currently, the system requires the base goods and production data to dynamically calculate supply curves and construct the Input-Output (IO) matrix. Supplying pre-calculated IO coefficient tables is not supported because it bypasses the system's ability to calculate dynamic supply curves.

---

## Core Data Files

These files are mandatory for initializing the economy's base state.

### 1. `goods.csv`
Defines all the commodities and services available within the simulated economy.

**Key Columns:**
- `id`: Unique internal integer identifier.
- `name`: Short, human-readable name of the good.
- `descriptive_name`: A detailed description of the good.
- `id_number`: An alternative numeric identifier.
- `isic`: The International Standard Industrial Classification code (e.g., `A9999_999_999`). This acts as the primary key for linking goods across the simulation.
- **Classification Columns:** `isic_section`, `isic_division`, `isic_group`, `isic_class`, `sub_class_a`, `sub_class_b`, `sub_class_c`, `sub_class_nf` to fully categorize the good according to ISIC standards.

### 2. `productions.csv`
Defines the production methods (recipes) used by producers to create goods.

**Key Columns:**
- `id`, `name`, `descriptive_name`, `id_number`, `isic`: Identifiers matching the producer/production method.
- `producer`: The ID of the economic agent performing the production.
- `produce`: The `id_number` of the good being produced.
- `produce_name`: The name of the good being produced.
- `production_inputs`: A JSON-formatted dictionary mapping input `isic` codes to the monetary cost/quantity required per production run (e.g., `{"A1397_975_51074": 12, "A3790_132_63": 33}`).
- `production_added_values`: A JSON-formatted dictionary mapping value-added components (e.g., `wages`, `surplus`, `tariffs`) to their respective costs.
- `production_rate`: The capacity or rate at which this production method operates.
- `production_quantity`: The total quantity produced.
- `price`: The final output price resulting from the inputs and added values.
- **Other Columns:** Efficiencies (`production_material_efficiency`, `production_labour_efficiency`, etc.), contact information, and address information.

---

## Optional Configuration Files

These files define shocks, policies, and dynamic changes applied to the core economy during simulations.

### 3. `tax_policies.csv`
Used to define taxation and fiscal policy scenarios for the Circular Flow Model.

**Key Columns:**
- `example_id`: Unique identifier for the policy scenario.
- `change_type`: The type of scenario (typically `tax_policy`).
- `title` & `description`: Metadata explaining the scenario's objective.
- `final_demand`: A semicolon-separated list of floating-point values representing final demand for each sector.
- `income_tax_rate_before` & `income_tax_rate_after`: Income tax rates (e.g., `0.1` for 10%).
- `corporate_tax_rate_before` & `corporate_tax_rate_after`: Corporate tax rates (e.g., `0.25` for 25%).
- `income_tax_applies_to`: Specifies which value-added component is subject to income tax (e.g., `wages`).
- `iterations`: The number of periods the simulation should iterate to show dynamic adjustment.

### 4. `tech_changes.csv`
Defines structural shifts, automation, productivity improvements, and input substitutions over time.

**Key Columns:**
- *Inherits scenario configuration columns similar to `tax_policies.csv` (e.g., `final_demand`, `iterations`, tax rates).*
- `tech_change_id`: A string identifier for the specific technological change (e.g., `energy_efficiency_30pct`).
- `sequence_number`: Determines the order in which multi-level changes are applied.
- `method`: The specific programmatic mutation to apply. Examples:
  - `add_input_change`: Modify a specific input requirement across all sectors.
  - `add_sector_change`: Improve overall productivity of a specific sector.
  - `add_coefficient_change`: Substitute or adjust a specific input-output relationship.
- `sector_idx` & `input_sector_idx`: ISIC codes targeting the specific sectors affected by the method.
- `change_type_param`: The mathematical operation to apply (e.g., `multiply`, `add`).
- `value`: The magnitude of the change (e.g., `0.7` to represent a 30% reduction when using `multiply`).
