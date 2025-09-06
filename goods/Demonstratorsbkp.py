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

def test_technological_change(final_demand: np.ndarray, target_produce: Optional[str] = None):
    gdb = GoodsDatabase()
    existing_goods = gdb.get_all_goods()
    existing_goods_isic = [good.isic for good in existing_goods]
    value_added_types = ["wages", "surplus", "taxes", "mixed_income"]  # Ensure order is known
    shutil.copy2("data.db", "dataDEMO.db")

    with contextlib.redirect_stdout(io.StringIO()): # Suppress print output
        A_Before, VA_Vector_Before, VA_Flows_Matrix_Before = evaluate_indicies_production_inputs_to_matrix(demoDB=True)
        leontief_inv_before = create_leontief_inverse(A_Before, Value_Added=VA_Vector_Before)
        output_before = leontief_inv_before @ final_demand
        # Apply the investment
        target_produce = random.choice(existing_goods_isic) if target_produce is None else target_produce # pyright: ignore[reportAssignmentType]
        investment = test_create_indice_investment(target_produce) # pyright: ignore[reportArgumentType]
        investment.apply_investment()
        # Get the new output
        A_After, VA_Vector_After, VA_Flows_Matrix_After = evaluate_indicies_production_inputs_to_matrix(demoDB=True)
        leontief_inv_after = create_leontief_inverse(A_After, Value_Added=VA_Vector_After)
        output_after = leontief_inv_after @ final_demand

        # Calculate original output levels (these are the reference output levels from the original data)
        original_output = np.sum(A_Before, axis=0) + np.sum(VA_Flows_Matrix_Before, axis=0)
        
        # Calculate VA ratios from original flows
        def VARatioMatrix(VA_Matrix):
            va_ratios = np.zeros_like(VA_Matrix)
            for sector_idx in range(len(existing_goods_isic)):
                col_sum = np.sum(VA_Matrix[:, sector_idx])
                if col_sum > 0:  # Avoid division by zero
                    va_ratios[:, sector_idx] = VA_Matrix[:, sector_idx] / col_sum
            return va_ratios

    print("Flows")
    print(A_Before)

    print("Value Added Ratios")
    print(VARatioMatrix(VA_Flows_Matrix_Before))

    # --- NEW VALUE ADDED CALCULATION ---
    # TODO validate that all total value added values are not negative (>0)
    # TODO ^ because we dictate FD from VA, maybe its not possible for the above situation to happen, could be caused by regarding FD at this point
    # TODO ^ let us discover total VA from resolving leontief inverse from final demand
    
    # For BEFORE
    total_output_before = np.sum(A_Before, axis=1) + final_demand
    print("Sum A rows: ",np.sum(A_Before, axis=1))
    print("Final Demand:", final_demand)
    print("Total Output Before:", total_output_before)
    intermediate_input_before = np.sum(A_Before, axis=0)
    print("Intermediate Input Before:", intermediate_input_before)
    total_va_before = total_output_before - intermediate_input_before  # This is just np.sum(VA_Matrix_Before, axis=0) #required Value Added to match demand
    print("Total VA Before:", total_va_before)
    VA_Matrix_Before_new = VARatioMatrix(VA_Flows_Matrix_Before) * total_va_before  # shape: (va_types, sectors)

    # For AFTER
    total_output_after = np.sum(A_After, axis=1) + final_demand
    print("Sum A rows: ",np.sum(A_After, axis=1))
    print("Final Demand:", final_demand)
    print("Total Output After:", total_output_after)
    intermediate_input_after = np.sum(A_After, axis=0)
    print("Intermediate Input After:", intermediate_input_after)
    total_va_after = total_output_after - intermediate_input_after  # This is just np.sum(VA_Matrix_After, axis=0) #required Value Added to match demand
    print("Total VA After:", total_va_after)
    VA_Matrix_After_new = VARatioMatrix(VA_Flows_Matrix_After) * total_va_after  # shape: (va_types, sectors)

    print("\n\nOriginal Flow Matrix:\n",A_Before)
    print("\nOriginal Leontief Inverse:\n",leontief_inv_before)
    print("Resolved Flow Using Original Technology:\n",leontief_inv_before * final_demand)
    print("\n\nNew Flow Matrix:\n",A_After)
    print("\nNew Leontief Inverse:\n",leontief_inv_after)
    print("Resolved Flow Using Updated Technology:\n", leontief_inv_after * final_demand)

    print("Final Demand:", final_demand)
    print("Original Output:", output_before)
    print("Technological Change:", investment)
    print("New Output After Technological Change Implementation:", output_after)
    print("Difference (Absolute):", output_after - output_before)
    print("Difference (Percentage):", (output_after - output_before) / output_before * 100)

    # Show change in value added matrix with headings and totals
    sector_names = existing_goods_isic
    def print_va_matrix_with_totals(matrix, title):
        df = pd.DataFrame(matrix, index=value_added_types, columns=sector_names)
        df['Row Total'] = df.sum(axis=1)
        df.loc['Col Total'] = df.sum(axis=0)
        print(f"\n{title}:\n", df)

    print_va_matrix_with_totals(VA_Matrix_Before_new, "Value Added Matrix BEFORE Technological Change")
    print_va_matrix_with_totals(VA_Matrix_After_new, "Value Added Matrix AFTER Technological Change")
    print_va_matrix_with_totals(VA_Matrix_After_new - VA_Matrix_Before_new, "Change in Value Added Matrix (Absolute)")
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_change = np.where(VA_Matrix_Before_new != 0, (VA_Matrix_After_new - VA_Matrix_Before_new) / VA_Matrix_Before_new * 100, 0)
    print_va_matrix_with_totals(percent_change, "Change in Value Added Matrix (Percentage)")

    # Verification: Check that Output = Intermediate Inputs + Value Added
    print("\nVerification - Output should equal Intermediate Inputs + Value Added:")
    intermediate_inputs_before = A_Before @ output_before
    intermediate_inputs_after = A_After @ output_after
    total_va_before = np.sum(VA_Matrix_Before_new, axis=0)
    total_va_after = np.sum(VA_Matrix_After_new, axis=0)

    print("Before - Output:", output_before)
    print("Before - Intermediate Inputs:", intermediate_inputs_before)
    print("Before - Total VA per sector:", total_va_before)
    print("Before - Sum check (should be ~0):", output_before - intermediate_inputs_before - total_va_before)
    
    print("\nAfter - Output:", output_after)
    print("After - Intermediate Inputs:", intermediate_inputs_after)
    print("After - Total VA per sector:", total_va_after)
    print("After - Sum check (should be ~0):", output_after - intermediate_inputs_after - total_va_after)
    

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

test_technological_change(np.array([40, 70, 95, 110, 75]))