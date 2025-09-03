from Evaluators import *
# --- Demonstrations Functions ---

def demonstrate_demand_shock(table_type: str, table_name: str, trim_table: bool, final_demand: np.ndarray, output: np.ndarray, shock_vector: list):
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

def demonstrate_technological_improvement(
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

def test_csv_import_and_creation_of_leontieff_inverses():
    # Manually provide final demand and output vectors for 'one.csv'
    final_demand = np.array([40, 70, 95])
    output = np.array([100, 150, 150]) #this is neccessary if the table is not normalized
    print(create_leontief_inverse(load_io_table('csv', 'io1.csv', True), Total_Output=output))
    print(create_leontief_inverse(load_io_table('csv', 'io1.csv', True), Final_Demand=final_demand))
    print(create_leontief_inverse(load_io_table('csv', 'io1.csv', True), Value_Added=np.array([65,50,90])))

def test_shocks_and_improvents():
    # Manually provide final demand and output vectors for 'one.csv'
    final_demand = np.array([40, 70, 95])
    output = np.array([100, 150, 150]) #this is neccessary if the table is not normalized
    demonstrate_demand_shock('csv', 'io1.csv', True, final_demand, output, [10, 0, -5])

    # To use technological improvement, provide two tables: base and improved
    demonstrate_technological_improvement('csv', 'io1.csv', 'io2.csv', True, final_demand, output)

# --- Example Usage ---

if __name__ == "__main__":
    test_csv_import_and_creation_of_leontieff_inverses()
    test_shocks_and_improvents()

# TODO break down value added & final demand

# there is two types of breakdowns, raw breakdowns with values, and simply just totals with percentages
# regard the known transformation constraints i.e. consumer spending cannot be higher than total wages etc.

# final demand directs production, therefore absolute value added will depend on the amount of final demand,
# we are assuming all of value added is immediately spent as a final demand, as we did so in the transformation
# models. If money is saved, e.g. by the employees, there is actaully less household consumption (C), the figures
# still remain in the model though, representing a future reserve for a future purchase. Not conserving this may
# be used to show/demonstrate inflation/deflation. How?. 

# TODO apply and evaluate investment/change (also be able to apply tax and tarrifs) 1, without breakdown 2, with breakdown
# TODO rank investments based on evaluated result
# TODO friendly for business and soverign fiscal

# We can evaluate/imply value added from the table from the total output and intermediate inputs
# For now, we demonstrate improvements on the technology, what it has on the output and economy
# First the raw inputs, capital saving improvements
# Then, we demonstrate labour saving improvements (nothing will change for now, we demonstrate this on second section)

# we must then break down added value and final demand, to demonstrate the impact of both capital and labour saving improvements
# finally, we simulate saving and investment
# we'll start by "applying" the change to an io table