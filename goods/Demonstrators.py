from Evaluators import *
from Investment import *
import random
import faker
import shutil
import contextlib
import io

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
    shutil.copy2("data.db", "dataDEMO.db")

    with contextlib.redirect_stdout(io.StringIO()): # Suppress print output
        A, VAvector, _ = evaluate_indicies_production_inputs_to_matrix(demoDB=True)
        leontief_inv = create_leontief_inverse(A, Value_Added=VAvector)
        target_produce = random.choice(existing_goods_isic) if target_produce is None else target_produce # pyright: ignore[reportAssignmentType]
        investment = test_create_indice_investment(target_produce) # pyright: ignore[reportArgumentType]

        output = leontief_inv @ final_demand
        # Apply the investment
        investment.apply_investment()
        # Get the new output
        A, VAvector, _ = evaluate_indicies_production_inputs_to_matrix(demoDB=True)
        leontief_inv = create_leontief_inverse(A, Value_Added=VAvector)
        new_output = leontief_inv @ final_demand

    print("Final Demand:", final_demand)
    print("Original Output:", output)
    print("Investment:", investment)
    print("New Output After Investment Implementation:", new_output)
    print("Difference (Absolute):", new_output - output)
    print("Difference (Percentage):", (new_output - output) / output * 100)
    

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