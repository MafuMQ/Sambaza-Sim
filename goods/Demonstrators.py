from Evaluators import *
from Investment import *
import random
import faker
import shutil
import contextlib
import io
import pandas as pd

fake = faker.Faker()

def demonstrate_demand_shock_csv(table_type: str, table_name: str, trim_table: bool, final_demand: np.ndarray, output: np.ndarray, shock_vector: list):
    """
    Demonstrates the effect of a demand shock on the IO table.
    Args:
        table_type (str): 'csv' or 'sqlite'
        table_name (str): Path or name of the IO table
        trim_table (bool): Whether to trim extras from the IO table
        final_demand (np.ndarray): Final demand vector for each sector
        output (np.ndarray): Output vector for each sector
        shock_vector (list): List of demand shocks to apply to each sector (length = number of sectors)
    """
    A = load_io_table(table_type, table_name, trim_table)
    leontief_inv = create_leontief_inverse(A, output)
    new_final_demand = final_demand + np.array(shock_vector)
    new_output = leontief_inv @ new_final_demand

    print("Original Final Demand:", final_demand)
    print("Shock Vector:", shock_vector)
    print("New Final Demand:", new_final_demand)
    print("Original Output:", output)
    print("New Output after Demand Shock:", new_output)
    return new_output

def demonstrate_demand_shock_db(final_demand: np.ndarray, shock_vector: list):
    """
    Demonstrates the effect of a demand shock on the IO table.
    Args:
        table_type (str): 'csv' or 'sqlite'
        table_name (str): Path or name of the IO table
        trim_table (bool): Whether to trim extras from the IO table
        final_demand (np.ndarray): Final demand vector for each sector
        output (np.ndarray): Output vector for each sector
        shock_vector (list): List of demand shocks to apply to each sector (length = number of sectors)
    """
    A, VAvector, _ = evaluate_indicies_production_inputs_to_matrix()
    leontief_inv = create_leontief_inverse(A, Value_Added=VAvector)
    output = leontief_inv @ final_demand
    new_final_demand = final_demand + np.array(shock_vector)
    new_output = leontief_inv @ new_final_demand

    print("Original Final Demand:", final_demand)
    print("Shock Vector:", shock_vector)
    print("New Final Demand:", new_final_demand)
    print("Original Output:", output)
    print("New Output after Demand Shock:", new_output)
    print("Difference (Absolute):", new_output - output)
    print("Difference (Percentage):", (new_output - output) / output * 100)
    return new_output

def demonstrate_technological_improvement_csv(
    table_type: str,
    base_table_name: str,
    improved_table_name: str,
    trim_table: bool,
    final_demand: np.ndarray,
    output: np.ndarray
):
    """
    Demonstrates the effect of a technological improvement by comparing two IO tables:
    the base table and the improved (technology) table.
    Args:
        table_type (str): 'csv' or 'sqlite'
        base_table_name (str): Path or name of the base IO table
        improved_table_name (str): Path or name of the improved IO table
        trim_table (bool): Whether to trim extras from the IO table
        final_demand (np.ndarray): Final demand vector for each sector
        output (np.ndarray): Output vector for each sector (should match both tables)
    """
    # Load both IO tables
    A_base = load_io_table(table_type, base_table_name, trim_table)
    A_improved = load_io_table(table_type, improved_table_name, trim_table)

    # Compute Leontief inverses
    leontief_inv_base = create_leontief_inverse(A_base, output)
    leontief_inv_improved = create_leontief_inverse(A_improved, output)

    # Calculate outputs required to meet the same final demand
    output_base = leontief_inv_base @ final_demand
    output_improved = leontief_inv_improved @ final_demand

    print("\n--- Technological Improvement (Table Comparison) ---")
    print("Original Output:", output_base)
    print("Output with Improved Technology (same final demand):", output_improved)
    print("Difference:", output_improved - output_base)
    return output_improved

def test_technological_change(final_demand: np.ndarray, target_produce: Optional[Investment] = None):
    def print_matrix_with_totals(matrix, row_labels, col_labels, title):
        df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
        df['Row Total'] = df.sum(axis=1)
        df.loc['Col Total'] = df.sum(axis=0)
        print(f"\n{title}:\n", df)

    def calculate_va_ratios(va_matrix, sector_count):
        va_ratios = np.zeros_like(va_matrix)
        for sector_idx in range(sector_count):
            col_sum = np.sum(va_matrix[:, sector_idx])
            if col_sum > 0:
                va_ratios[:, sector_idx] = va_matrix[:, sector_idx] / col_sum
        return va_ratios

    def calculate_va_coefficients(leontief_inverse, output):
        identity = np.eye(len(leontief_inverse))
        A_matrix = identity - np.linalg.inv(leontief_inverse)
        output_diag = np.diag(output)
        intermediate_inputs = np.sum(A_matrix @ output_diag, axis=0)
        value_added_by_sector = output - intermediate_inputs
        value_added_coefficients = value_added_by_sector / output
        print("\nValue Added by Sector Calculation:", value_added_by_sector)
        print("\nValue Added Coefficients: ", value_added_coefficients)
        return value_added_by_sector, value_added_coefficients

    def print_va_matrix_with_totals(matrix, value_added_types, sector_names, title):
        df = pd.DataFrame(matrix, index=value_added_types, columns=sector_names)
        df['Row Total'] = df.sum(axis=1)
        df.loc['Col Total'] = df.sum(axis=0)
        print(f"\n{title}:\n", df)

    # --- Main Refactored Logic ---
    goods_db = GoodsDatabase()
    goods_list = goods_db.get_all_goods()
    sector_names = [good.isic for good in goods_list]
    value_added_types = ["wages", "surplus", "taxes", "mixed_income"]
    sector_count = len(sector_names)

    # Copy database for demo
    shutil.copy2("data.db", "dataDEMO.db")

    # Suppress print output during investment application and matrix evaluation
    with contextlib.redirect_stdout(io.StringIO()):
        # Before technological change
        Flows_before, va_vector_before, va_matrix_before = evaluate_indicies_production_inputs_to_matrix(demoDB=True)
        leontief_inv_before = create_leontief_inverse(Flows_before, Value_Added=va_vector_before)
        output_before = leontief_inv_before @ final_demand

        # Apply investment (technological change)
        investment = test_create_indice_investment(random.choice(sector_names)) if target_produce is None else target_produce # pyright: ignore[reportArgumentType]
        investment.apply_investment()

        # After technological change
        Flows_after, va_vector_after, va_matrix_after = evaluate_indicies_production_inputs_to_matrix(demoDB=True)
        leontief_inv_after = create_leontief_inverse(Flows_after, Value_Added=va_vector_after)
        output_after = leontief_inv_after @ final_demand

    va_ratios_before = calculate_va_ratios(va_matrix_before, sector_count)
    va_ratios_after = calculate_va_ratios(va_matrix_after, sector_count)

    # Calculate new value added vectors and matrices
    va_vector_before_new, _ = calculate_va_coefficients(leontief_inv_before, output_before)
    va_vector_after_new, _ = calculate_va_coefficients(leontief_inv_after, output_after)
    va_matrix_before_new = va_ratios_before * va_vector_before_new.reshape(1, -1)
    va_matrix_after_new = va_ratios_after * va_vector_after_new.reshape(1, -1)

    print(f'{"-"*100}\n{"-"*100}')
    print_matrix_with_totals(Flows_before, sector_names, sector_names, "Original Flow Matrix (A)")
    print_matrix_with_totals(leontief_inv_before, sector_names, sector_names, "Original Leontief Inverse")

    resolved_flow_before = leontief_inv_before * final_demand
    print_matrix_with_totals(resolved_flow_before, sector_names, sector_names, "Resolved Flow Using Original Technology (FD & VA Inclusive)")
    print("Final Demand:", final_demand)
    print("Total Output: ", output_before)
    print("Value Added: ", va_vector_before_new)
    print("Total Input: ", np.sum(resolved_flow_before, axis=0))

    print("\nNew Flow Matrix:")
    print_matrix_with_totals(Flows_after, sector_names, sector_names, "New Flow Matrix (A)")
    print_matrix_with_totals(leontief_inv_after, sector_names, sector_names, "New Leontief Inverse")
    resolved_flow_after = leontief_inv_after * final_demand
    print_matrix_with_totals(resolved_flow_after, sector_names, sector_names, "Resolved Flow Using Updated Technology (FD & VA Inclusive)")
    print("Final Demand:", final_demand)
    print("Total Output: ", output_after)
    print("Value Added: ", va_vector_after_new)
    print("Total Input: ", np.sum(resolved_flow_after, axis=0))

    # Show absolute and relative change in resolved flows (moved here for correct order)
    resolved_flow_change_abs = resolved_flow_after - resolved_flow_before
    with np.errstate(divide='ignore', invalid='ignore'):
        resolved_flow_change_rel = np.where(resolved_flow_before != 0, (resolved_flow_after - resolved_flow_before) / resolved_flow_before * 100, 0)
    print_matrix_with_totals(resolved_flow_change_abs, sector_names, sector_names, "Change in Resolved Flow (Absolute)")
    print_matrix_with_totals(resolved_flow_change_rel, sector_names, sector_names, "Change in Resolved Flow (Percentage)")

    print("Final Demand:", final_demand)
    print("Original Output:", output_before)
    print("Technological Change:", investment)
    print("New Output After Technological Change Implementation:", output_after)
    print("Difference (Absolute):", output_after - output_before)
    print("Difference (Percentage):", (output_after - output_before) / output_before * 100)

    print("-"*100)
    # Show change in value added matrix with headings and totals
    print_va_matrix_with_totals(va_matrix_before_new, value_added_types, sector_names, "Value Added Matrix BEFORE Technological Change")
    print_va_matrix_with_totals(va_matrix_after_new, value_added_types, sector_names, "Value Added Matrix AFTER Technological Change")
    print_va_matrix_with_totals(va_matrix_after_new - va_matrix_before_new, value_added_types, sector_names, "Change in Value Added Matrix (Absolute)")
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_change = np.where(va_matrix_before_new != 0, (va_matrix_after_new - va_matrix_before_new) / va_matrix_before_new * 100, 0)
    print_va_matrix_with_totals(percent_change, value_added_types, sector_names, "Change in Value Added Matrix (Percentage)")

    # Verification: Output = Intermediate Inputs + Value Added
    print(f'{"-"*100}\n{"-"*100}')
    print("\nVerification - Output should equal Intermediate Inputs + Value Added:")
    A_before = np.eye(len(leontief_inv_before)) - np.linalg.inv(leontief_inv_before)
    intermediate_inputs_before = A_before @ output_before  # 1D vector
    A_After = np.eye(len(leontief_inv_after)) - np.linalg.inv(leontief_inv_after)
    intermediate_inputs_after = A_After @ output_after  # 1D vector
    total_inputs_before = np.sum(intermediate_inputs_before, axis=0)
    total_inputs_after = np.sum(intermediate_inputs_after, axis=0)

    print("Before - Total Output:", output_before)
    print("Before - Total Input:", intermediate_inputs_before)
    print(f"Before - Final Demand Check (should be {np.sum(final_demand)}):", (np.sum(output_before) - np.sum(total_inputs_before)).round(10)) #TODO how does this work
    print("Before - Sum check (should be ~0):", (np.sum(output_before) - np.sum(np.sum(leontief_inv_before * final_demand, axis=0))).round(10))

    print("\nAfter - Total Output:", output_after)
    print("After - Total Input:", intermediate_inputs_after)
    print(f"After - Final Demand Check (should be {np.sum(final_demand)}):", (np.sum(output_after) - np.sum(total_inputs_after)).round(10))
    print("After - Sum check (should be ~0):", (np.sum(output_after) - np.sum(np.sum(leontief_inv_after * final_demand, axis=0))).round(10))

def test_create_indice_investment(produce_isic:str) -> Investment:
    gdb = GoodsDatabase()
    existing_goods = gdb.get_all_goods()
    existing_goods_isic = [good.isic for good in existing_goods]
    value_added_types = ["wages","surplus","taxes","mixed_income"]
    # inputs = {random.choice(existing_goods_isic): fake.random_int(min=1, max=100) for _ in range(3)} looks more realistic if there isn't a row/column with only zeros
    inputs = {isic: fake.random_int(min=1, max=100) for isic in existing_goods_isic}
    added_values = {value_type: fake.random_int(min=1, max=50) for value_type in value_added_types}
    investment = Investment(id=fake.random_int(min=1, max=1000), name=fake.word(), type_of_investment=InvestmentType.INDICE, produce_isic=produce_isic, implementation_cost=fake.random_number(digits=5))
    investment.set_investment_metrics(inputs, added_values)
    # investment.price_history = indice.price_history
    # investment.quantity = indice.quantity

    return investment


def demonstrate_technological_improvement_db():
    pass