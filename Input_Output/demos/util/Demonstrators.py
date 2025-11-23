from Input_Output.util.Evaluators import *
from Input_Output.models.entities.Investment import *
from Input_Output.demos.util.Setup_Data import test_create_indice_investment, test_create_indice_investment_random
import random
import faker
import shutil
import contextlib
import io
import pandas as pd

fake = faker.Faker()

def perform_detailed_model_verification(
    sector_names, 
    output_before, output_after,
    intermediate_inputs_before, intermediate_inputs_after,
    value_added_before, value_added_after,
    final_demand,
    leontief_inv_before, leontief_inv_after,
    A_before, A_after,
    va_matrix_before_new, va_matrix_after_new,
    va_vector_before_new, va_vector_after_new
):
    """
    Perform detailed verification of input-output model consistency.
    
    Args:
        sector_names: List of sector names
        output_before/after: Output vectors before/after technological change
        intermediate_inputs_before/after: Intermediate input vectors
        value_added_before/after: Value added vectors
        final_demand: Final demand vector
        leontief_inv_before/after: Leontief inverse matrices
        A_before/after: Input coefficient matrices
        va_matrix_before_new/after_new: Value added matrices
        va_vector_before_new/after_new: Value added vectors
    """
    print(f'\n{"-"*100}')
    print("DETAILED MODEL VERIFICATION")
    print(f'{"-"*100}')
    
    # 1. Sector-by-sector accounting identity checks
    print("\n1. SECTOR-BY-SECTOR ACCOUNTING IDENTITY CHECKS:")
    print("   X_i = A_i * X + VA_i (for each sector i)")
    
    for i, sector in enumerate(sector_names):
        # Before technological change
        calculated_output_before = intermediate_inputs_before[i] + value_added_before[i]
        error_before = abs(output_before[i] - calculated_output_before)
        
        # After technological change
        calculated_output_after = intermediate_inputs_after[i] + value_added_after[i]
        error_after = abs(output_after[i] - calculated_output_after)
        
        print(f"   {sector}:")
        print(f"     Before: X={output_before[i]:.3f}, A*X+VA={calculated_output_before:.3f}, Error={error_before:.6f}")
        print(f"     After:  X={output_after[i]:.3f}, A*X+VA={calculated_output_after:.3f}, Error={error_after:.6f}")
    
    # 2. Total economy checks
    print("\n2. TOTAL ECONOMY CHECKS:")
    total_output_before = np.sum(output_before)
    total_intermediate_before = np.sum(intermediate_inputs_before)
    total_va_before = np.sum(value_added_before)
    total_fd = np.sum(final_demand)
    
    total_output_after = np.sum(output_after)
    total_intermediate_after = np.sum(intermediate_inputs_after)
    total_va_after = np.sum(value_added_after)
    
    print(f"   Before Technological Change:")
    print(f"     Total Output: {total_output_before:.3f}")
    print(f"     Total Intermediate Inputs: {total_intermediate_before:.3f}")
    print(f"     Total Value Added: {total_va_before:.3f}")
    print(f"     Total Final Demand: {total_fd:.3f}")
    print(f"     Check (Output = Intermediate + VA): {total_output_before:.3f} = {total_intermediate_before:.3f} + {total_va_before:.3f}")
    print(f"     Error: {abs(total_output_before - (total_intermediate_before + total_va_before)):.6f}")
    print(f"     Check (Output - Intermediate = FD): {total_output_before - total_intermediate_before:.3f} = {total_fd:.3f}")
    print(f"     Error: {abs((total_output_before - total_intermediate_before) - total_fd):.6f}")
    
    print(f"   After Technological Change:")
    print(f"     Total Output: {total_output_after:.3f}")
    print(f"     Total Intermediate Inputs: {total_intermediate_after:.3f}")
    print(f"     Total Value Added: {total_va_after:.3f}")
    print(f"     Total Final Demand: {total_fd:.3f}")
    print(f"     Check (Output = Intermediate + VA): {total_output_after:.3f} = {total_intermediate_after:.3f} + {total_va_after:.3f}")
    print(f"     Error: {abs(total_output_after - (total_intermediate_after + total_va_after)):.6f}")
    print(f"     Check (Output - Intermediate = FD): {total_output_after - total_intermediate_after:.3f} = {total_fd:.3f}")
    print(f"     Error: {abs((total_output_after - total_intermediate_after) - total_fd):.6f}")
    
    # 3. Leontief inverse verification
    print("\n3. LEONTIEF INVERSE VERIFICATION:")
    print("   X = (I - A)^(-1) * F")
    
    # Before
    calculated_output_leontief_before = leontief_inv_before @ final_demand
    leontief_error_before = np.max(np.abs(output_before - calculated_output_leontief_before))
    print(f"   Before: Max error between X and L*F: {leontief_error_before:.6f}")
    
    # After
    calculated_output_leontief_after = leontief_inv_after @ final_demand
    leontief_error_after = np.max(np.abs(output_after - calculated_output_leontief_after))
    print(f"   After:  Max error between X and L*F: {leontief_error_after:.6f}")
    
    # 4. Matrix consistency checks
    print("\n4. MATRIX CONSISTENCY CHECKS:")
    # Check that (I - A) * L = I
    identity_before = np.eye(len(A_before))
    product_before = (identity_before - A_before) @ leontief_inv_before
    identity_error_before = np.max(np.abs(product_before - identity_before))
    
    identity_after = np.eye(len(A_after))
    product_after = (identity_after - A_after) @ leontief_inv_after
    identity_error_after = np.max(np.abs(product_after - identity_after))
    
    print(f"   Before: Max error in (I-A)*L = I: {identity_error_before:.6f}")
    print(f"   After:  Max error in (I-A)*L = I: {identity_error_after:.6f}")
    
    # 5. Value added matrix consistency
    print("\n5. VALUE ADDED MATRIX CONSISTENCY:")
    va_matrix_sum_before = np.sum(va_matrix_before_new, axis=0)
    va_vector_error_before = np.max(np.abs(va_matrix_sum_before - va_vector_before_new))
    
    va_matrix_sum_after = np.sum(va_matrix_after_new, axis=0)
    va_vector_error_after = np.max(np.abs(va_matrix_sum_after - va_vector_after_new))
    
    print(f"   Before: Max error between sum of VA matrix columns and VA vector: {va_vector_error_before:.6f}")
    print(f"   After:  Max error between sum of VA matrix columns and VA vector: {va_vector_error_after:.6f}")
    
    # 6. Overall model assessment
    print(f"\n6. OVERALL MODEL ASSESSMENT:")
    total_errors = [leontief_error_before, leontief_error_after, identity_error_before, identity_error_after, 
                   va_vector_error_before, va_vector_error_after]
    max_error = max(total_errors)
    
    if max_error < 1e-10:
        status = "EXCELLENT"
    elif max_error < 1e-6:
        status = "GOOD"
    elif max_error < 1e-3:
        status = "ACCEPTABLE"
    else:
        status = "POOR - CHECK MODEL"
    
    print(f"   Maximum error across all checks: {max_error:.2e}")
    print(f"   Model consistency status: {status}")
    
    print(f'{"-"*100}')

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

def demonstrate_technological_change_random(final_demand: np.ndarray, target_production_tech_change: Optional[Investment] = None):
    def print_vector_with_labels(vector, labels, title):
        df = pd.DataFrame(vector.reshape(-1, 1), index=labels, columns=[title])
        print(f"\n{title}:\n", df)
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

    # Suppress print output during investment application and matrix evaluation
    with contextlib.redirect_stdout(io.StringIO()):
        # --- Main Refactored Logic ---
        goods_db = GoodsDatabase()
        goods_list = goods_db.get_all_goods()
        sector_names = [good.isic for good in goods_list]
        value_added_types = ["wages", "surplus", "taxes", "mixed_income"]
        sector_count = len(sector_names)

        # Copy database for demo
        shutil.copy2("data.db", "dataDEMO.db")
        # Before technological change
        Flows_before, va_vector_before, va_matrix_before = evaluate_indicies_production_inputs_to_matrix(demoDB=True)
        leontief_inv_before = create_leontief_inverse(Flows_before, Value_Added=va_vector_before)
        output_before = leontief_inv_before @ final_demand

        # Apply investment (technological change)
        investment = test_create_indice_investment_random(random.choice(sector_names)) if target_production_tech_change is None else target_production_tech_change # pyright: ignore[reportArgumentType]
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

    print("Final Demand:", final_demand)
    print("Technological Change Applied:", investment)

    print(f'{"*"*100}\n{"*"*100}')
    print(f'\n{"-"*100}')
    print("BEFORE TECHNOLOGICAL CHANGE")
    print(f'{"-"*100}')
    print_matrix_with_totals(Flows_before, sector_names, sector_names, "Original Flow Matrix (A)")
    print_va_matrix_with_totals(va_matrix_before, value_added_types, sector_names, "Original Value Added Flow Matrix")
    print_matrix_with_totals(leontief_inv_before, sector_names, sector_names, "Original Leontief Inverse")

    # Calculate intermediate inputs (A*X), value added, and check identities
    A_before = np.eye(len(leontief_inv_before)) - np.linalg.inv(leontief_inv_before)
    intermediate_inputs_before = A_before @ output_before
    value_added_before = output_before - intermediate_inputs_before
    resolved_flow_before = leontief_inv_before @ final_demand
    resolved_flow_before_view = leontief_inv_before * final_demand

    print("Final Demand:", final_demand)
    print("Output (X):", output_before)
    print("Intermediate Inputs (A*X):", intermediate_inputs_before)
    print_matrix_with_totals(resolved_flow_before_view, sector_names, sector_names, "Resolved Flow Using Original Technology (FD & VA Inclusive)")
    print_va_matrix_with_totals(va_matrix_before_new, value_added_types, sector_names, "Resolved Value Added Matrix BEFORE Technological Change")
    print("Value Added (X - A*X):", value_added_before)
    print("Check: Output == Intermediate Inputs + Value Added:", np.allclose(output_before, intermediate_inputs_before + value_added_before))
    print("Check: Output - Intermediate Inputs == Final Demand:", np.allclose(output_before - intermediate_inputs_before, final_demand))

    print(f'\n{"-"*100}')
    print("AFTER TECHNOLOGICAL CHANGE")
    print(f'{"-"*100}')
    print_matrix_with_totals(Flows_after, sector_names, sector_names, "New Flow Matrix (A)")
    print_va_matrix_with_totals(va_matrix_after, value_added_types, sector_names, "Value Added Matrix Flow AFTER Technological Change")
    print_matrix_with_totals(leontief_inv_after, sector_names, sector_names, "New Leontief Inverse")

    A_after = np.eye(len(leontief_inv_after)) - np.linalg.inv(leontief_inv_after)
    intermediate_inputs_after = A_after @ output_after
    value_added_after = output_after - intermediate_inputs_after
    resolved_flow_after = leontief_inv_after @ final_demand
    resolved_flow_after_view = leontief_inv_after * final_demand

    print("Final Demand:", final_demand)
    print("Output (X):", output_after)
    print("Intermediate Inputs (A*X):", intermediate_inputs_after)
    print_matrix_with_totals(resolved_flow_after_view, sector_names, sector_names, "Resolved Flow Using Updated Technology (FD & VA Inclusive)")
    print_va_matrix_with_totals(va_matrix_after_new, value_added_types, sector_names, "Resolved Value Added Matrix AFTER Technological Change")
    print("Value Added (X - A*X):", value_added_after)
    print("Check: Output == Intermediate Inputs + Value Added:", np.allclose(output_after, intermediate_inputs_after + value_added_after))
    print("Check: Output - Intermediate Inputs == Final Demand:", np.allclose(output_after - intermediate_inputs_after, final_demand))

    # Show absolute and relative change in resolved flows (moved here for correct order)
    print(f'\n{"-"*100}')
    print("BEFORE & AFTER TECHNOLOGICAL CHANGE COMPARISON")
    print(f'{"-"*100}')

    #! resolved_flow_before = leontief_inv_before @ final_demand
    #! resolved_flow_after = leontief_inv_after @ final_demand
    resolved_flow_change_abs = resolved_flow_after - resolved_flow_before
    with np.errstate(divide='ignore', invalid='ignore'):
        resolved_flow_change_rel = np.where(resolved_flow_before != 0, (resolved_flow_after - resolved_flow_before) / resolved_flow_before * 100, 0)
    # Print as matrix if 2D, else as vector
    if resolved_flow_change_abs.ndim == 2 and resolved_flow_change_abs.shape[0] == resolved_flow_change_abs.shape[1]:
        print_matrix_with_totals(resolved_flow_change_abs, sector_names, sector_names, "Change in Resolved Flow (Absolute)")
        print_matrix_with_totals(resolved_flow_change_rel, sector_names, sector_names, "Change in Resolved Flow (Percentage)")
    else:
        print_vector_with_labels(resolved_flow_change_abs, sector_names, "Change in Resolved Flow (Absolute)")
        print_vector_with_labels(resolved_flow_change_rel, sector_names, "Change in Resolved Flow (Percentage)")

    print("Technological Change:", investment)
    print("Difference in Output (Absolute):", output_after - output_before)
    print("Difference in Output (Percentage):", (output_after - output_before) / output_before * 100)

    print("-"*100)
    # Show change in value added matrix with headings and totals
    print_va_matrix_with_totals(va_matrix_after_new - va_matrix_before_new, value_added_types, sector_names, "Change in Value Added Matrix (Absolute)")
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_change = np.where(va_matrix_before_new != 0, (va_matrix_after_new - va_matrix_before_new) / va_matrix_before_new * 100, 0)
    print_va_matrix_with_totals(percent_change, value_added_types, sector_names, "Change in Value Added Matrix (Percentage)")

    # Call detailed verification function
    perform_detailed_model_verification(
        sector_names, 
        output_before, output_after,
        intermediate_inputs_before, intermediate_inputs_after,
        value_added_before, value_added_after,
        final_demand,
        leontief_inv_before, leontief_inv_after,
        A_before, A_after,
        va_matrix_before_new, va_matrix_after_new,
        va_vector_before_new, va_vector_after_new
    )

def demonstrate_technological_change(final_demand: np.ndarray, target_production_tech_change: Optional[Investment] = None, improvement_percentage: Optional[float] = 2.0, improvement_type: str = "total_cost", target_va_type: str | None = None, investment_cost: float | None = None):
    goods_db = GoodsDatabase()
    goods_list = goods_db.get_all_goods()
    sector_names = [good.isic for good in goods_list]
    # target_production_tech_change = random.choice(sector_names) if target_production_tech_change is None else target_production_tech_change # pyright: ignore[reportAssignmentType]
    # Prefer the first for ease of analysis
    target_production_tech_change = sector_names[0] if target_production_tech_change is None else target_production_tech_change # pyright: ignore[reportAssignmentType]
    investment = test_create_indice_investment(produce_isic=target_production_tech_change,improvement_percentage=improvement_percentage,improvement_type=improvement_type,investment_cost=investment_cost) # pyright: ignore[reportArgumentType]
    demonstrate_technological_change_random(final_demand=final_demand, target_production_tech_change=investment)

# TODO break down value added & final demand

# there is two types of breakdowns, raw breakdowns with values, and simply just totals with percentages
# regard the known transformation constraints i.e. consumer spending cannot be higher than total wages etc.

# final demand directs production, therefore absolute value added will depend on the amount of final demand,
# we are assuming all of value added is immediately spent as a final demand, as we did so in the transformation
# models. If money is saved, e.g. by the employees, there is actaully less household consumption (C), the figures
# still remain in the model though, representing a future reserve for a future purchase. Not conserving this may
# be used to show/demonstrate inflation/deflation. How?. 

# TODO apply and evaluate investment/change (also be able to apply tax and tarrifs) 1, without breakdown 2, with breakdown
# TODO rank investments based on evaluated result*
# TODO friendly for business and soverign fiscal

# We can evaluate/imply value added from the table from the total output and intermediate inputs
# For now, we demonstrate improvements on the technology, what it has on the output and economy
# First the raw inputs, capital saving improvements
# Then, we demonstrate labour saving improvements (nothing will change for now, we demonstrate this on second section)

# we must then break down added value and final demand, to demonstrate the impact of both capital and labour saving improvements
# finally, we simulate saving and investment
# we'll start by "applying" the change to an io table