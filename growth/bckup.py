import numpy as np
import pandas as pd

# --- IO Table Loading Functions ---

def load_io_table_from_csv(csv_path: str, trim_extras: bool = True) -> np.ndarray:
    """
    Loads the intermediate input matrix (A) from the CSV file.
    If trim_extras is True (default), removes the last two rows and columns (e.g., Value Added, Total Input, Final Demand, Total Output).
    If trim_extras is False, loads the full matrix as-is.
    """
    df = pd.read_csv(csv_path, index_col=0)
    if trim_extras:
        # Remove the last two rows and columns (Value Added, Total Input, Final Demand, Total Output)
        A = df.iloc[:-2, :-2].fillna(0).values
    else:
        A = df.fillna(0).values
    return A

def load_io_table_from_sqlite(name: str) -> np.ndarray:
    # Placeholder for future implementation
    return np.array([])

def load_io_table(table_type: str, name: str, trim_extras: bool = True) -> np.ndarray:
    if table_type == 'csv':
        return load_io_table_from_csv(name, trim_extras)
    elif table_type == 'sqlite':
        return load_io_table_from_sqlite(name) 
    else:
        raise ValueError(f"Unknown table type: {table_type}")

# --- Economic Analysis Functions ---

def create_leontief_inverse(A: np.ndarray, output: np.ndarray) -> np.ndarray:
    """
    Creates the Leontief inverse matrix from the intermediate input matrix and output vector.
    Args:
        A (np.ndarray): Intermediate input matrix (n x n)
        output (np.ndarray): Output vector (n,)
    Returns:
        np.ndarray: Leontief inverse matrix (n x n)
    """
    n = A.shape[0]
    A_coeff = A / output.reshape(-1, 1)
    leontief_inv = np.linalg.inv(np.eye(n) - A_coeff)
    return leontief_inv

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

def demonstrate_technological_improvement(table_type: str, table_name: str, trim_table: bool, final_demand: np.ndarray, output: np.ndarray, input_reduction: float):
    """
    Demonstrates the effect of a technological improvement that reduces required inputs per unit output.
    Args:
        table_type (str): 'csv' or 'sqlite'
        table_name (str): Path or name of the IO table
        trim_table (bool): Whether to trim extras from the IO table
        final_demand (np.ndarray): Final demand vector for each sector
        output (np.ndarray): Output vector for each sector
        input_reduction (float): Fraction by which to reduce each input (e.g., 0.1 for 10% reduction)
    """
    A = load_io_table(table_type, table_name, trim_table)
    # Apply technological improvement: reduce all intermediate inputs by input_reduction fraction
    A_improved = A * (1 - input_reduction)
    leontief_inv_improved = create_leontief_inverse(A_improved, output)
    # Output required to meet the same final demand with improved technology
    new_output = leontief_inv_improved @ final_demand

    print(f"\n--- Technological Improvement: {int(input_reduction*100)}% input reduction ---")
    print("Original Output:", output)
    print("Output with Improved Technology (same final demand):", new_output)
    print("Difference:", new_output - output)
    return new_output

# --- Example Usage ---

if __name__ == "__main__":
    # Manually provide final demand and output vectors for 'one.csv'
    final_demand = np.array([40, 70, 95])
    output = np.array([100, 150, 150])
    demonstrate_demand_shock('csv', 'one.csv', True, final_demand, output, [10, 0, -5])

    # Demonstrate technological improvement: 10% reduction in all intermediate inputs
    demonstrate_technological_improvement('csv', 'one.csv', True, final_demand, output, input_reduction=0.1)