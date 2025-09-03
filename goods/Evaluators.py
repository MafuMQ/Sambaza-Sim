import logging
from typing import List, Dict
from Good import GoodsDatabase
from Good_Indice import GoodsIndiceDatabase
from Production import *
import sympy as sp
import typing as t
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_goods_isic():
    gdb = GoodsDatabase()
    all_goods = gdb.get_all_goods()
    for good in all_goods:
        result = evaluate_good_isic(good.isic)  # pyright: ignore[reportArgumentType]
        gdb.update_good(int(good.id), **result) # pyright: ignore[reportArgumentType] #TO:DO Look this up

def evaluate_good_isic(isic: str) -> dict:
    try:
        isic_core,subclass,non_fungible = isic.split('_')
        isic_section = isic_core[0]
        isic_division = isic_core[1:3]
        isic_group = isic_core[3:4]
        isic_class = isic_core[4:]
        return {
            "isic_section": isic_section,
            "isic_division": isic_division,
            "isic_group": isic_group,
            "isic_class": isic_class,
            "sub_class_a": subclass[0] if len(subclass) > 0 else None,
            "sub_class_b": subclass[1] if len(subclass) > 1 else None,
            "sub_class_c": subclass[2] if len(subclass) > 2 else None,
            "sub_class_nf": non_fungible if non_fungible else None
        }
    except ValueError as e:
        logger.error(f"Invalid ISIC format '{isic}': {e}")
        return {
            "isic_section": None,
            "isic_division": None,
            "isic_group": None,
            "isic_class": None,
            "sub_class_a": None,
            "sub_class_b": None,
            "sub_class_c": None,
            "sub_class_nf": None
        }

# print(evaluate_good_isic("A1173-366-7178"))
# evaluate_goods_isic()

def evaluate_productions_price():
    ptdb = ProductionsDatabase()
    productions = ptdb.get_all_productions()
    for production in productions:
        total_inputs_cost = sum([quantity for quantity in production.production_inputs.values()])
        total_value_added = sum([quantity for quantity in production.production_added_values.values()])
        ptdb.update_production(production_id=int(production.id), total_inputs_cost=total_inputs_cost, total_value_added=total_value_added, price=total_inputs_cost+total_value_added)  # pyright: ignore[reportArgumentType]

def evaluate_indicies():
    gidb = GoodsIndiceDatabase()
    ptdb = ProductionsDatabase()
    indices = gidb.get_all_good_indices()
    
    for indice in indices:
        productions:List[Production] = ptdb.get_all_productions_by_good(int(indice.id_number))  # pyright: ignore[reportArgumentType]

        cheapest_production = min(productions, key=lambda p: float(p.price)) # type: ignore
        production_added_values = cheapest_production.production_added_values
        gidb.update_good_indice(int(indice.id_number), # pyright: ignore[reportArgumentType]
                                production_inputs=cheapest_production.production_inputs, 
                                production_added_values=production_added_values,
                                total_inputs_cost=cheapest_production.total_inputs_cost, 
                                total_value_added=cheapest_production.total_value_added, 
                                price=cheapest_production.price)  

def evaluate_indicies_production_inputs_to_matrix(demoDB = False):
    symbols, equations = evaluate_production_inputs_to_equations("indicies",demoDB=demoDB)
    VA_symbols, VA_equations = evaluate_production_inputs_to_equations("indicies", Value_Added=True, demoDB=demoDB)

    A: sp.Matrix = evaluate_coefficients(symbols, equations)
    VA: sp.Matrix = evaluate_coefficients(VA_symbols, VA_equations)
    # Assume production added value is the last column in A (if you appended it in your equations)
    # If not, an adjustment may be needed in how value added is included in your equations/matrix
    if A.shape[1] > 1:
        value_added_vector = A[:, -1]
        A_trimmed: sp.Matrix = A[:, :-1] # pyright: ignore[reportAssignmentType]
    else:
        value_added_vector = A
        A_trimmed = sp.Matrix([])

    # Convert SymPy matrices to NumPy arrays here
    A_trimmed_np = np.array(A_trimmed).astype(float)
    value_added_vector_np = np.array(value_added_vector).astype(float).flatten()
    VA_np = np.array(VA).astype(float)

    #Trim Va_np
    VA_np = VA_np[:, :4]

    return A_trimmed_np.T, value_added_vector_np, VA_np.T  # Transpose A_trimmed before returning TODO, find out why i had to invert VA, and the extra zeros

def evaluate_production_inputs_to_equations(which: str = "indicies", Value_Added: bool = False, demoDB: bool = False):
    """Evaluate production inputs to equations."""

    if which == "productions":
        ptdb = ProductionsDatabase() if not demoDB else ProductionsDatabase(database_url="sqlite:///dataDEMO.db")
        productions = ptdb.get_all_productions()
    elif which == "indicies":
        indc = GoodsIndiceDatabase() if not demoDB else GoodsIndiceDatabase(database_url="sqlite:///dataDEMO.db")
        productions = indc.get_all_good_indices()
    else:
        logger.error(f"Unknown type: {which}")
        raise Exception(f"Unknown type: {which}")
    
    symbols = []
    equations = []

    for production in productions:
        result = evaluate_production_inputs_to_equation(production,"sympy") if not Value_Added else evaluate_production_inputs_to_equation(production,"sympy", Value_Added=Value_Added)
        if result:
            symbols.extend(result[0])
            equations.append(result[1])

    if not Value_Added:
        # Remove duplicates while preserving order
        symbols = list(dict.fromkeys(symbols))
        # Ensure VA is only at the end
        va_symbol = sp.symbols("VA")
        symbols = [s for s in symbols if s != va_symbol]
        symbols.append(va_symbol)
        return list(symbols), equations
    else:
        # Remove duplicates while preserving order
        symbols = list(dict.fromkeys(symbols))
        # TODO make an order of the symbols
    return list(symbols), equations

def evaluate_production_inputs_to_equation(production: Production, format: str, Value_Added: bool = False):
    if format == "string" and not Value_Added:
        return production_inputs_to_string(production)
    elif format == "string" and Value_Added:
        print(production_inputs_to_string(production, Value_Added=Value_Added))
        return production_inputs_to_sympy(production, Value_Added=Value_Added)
    elif format == "sympy" and not Value_Added:
        print(production_inputs_to_string(production))
        return production_inputs_to_sympy(production)
    elif format == "sympy" and Value_Added:
        print(production_inputs_to_string(production, Value_Added=Value_Added))
        return production_inputs_to_sympy(production, Value_Added=Value_Added)
    else:
        logger.error(f"Unknown format: {format}")
        return None

def production_inputs_to_string(production: Production, Value_Added: bool = False) -> str:
    inputs = production.production_inputs
    value_added_items = production.production_added_values
    equals = " + ".join([f"{value} * {key}" for key, value in inputs.items()])
    if not Value_Added:
        va_term = f"{production.total_value_added} * VA"
    else:
        va_term = " + ".join([f"{value} * {key}" for key, value in value_added_items.items()])
    equation = f"{production.isic} = {equals} + {va_term}"
    return equation

def production_inputs_to_sympy(production: Production, Value_Added: bool = False):
    inputs = production.production_inputs
    symbols = {key: sp.symbols(key) for key in inputs.keys()}
    output_symbol = sp.symbols(str(production.isic))
    if not Value_Added:
        value_added_symbol = sp.symbols("VA")  # Use the same VA symbol for all equations
        va_coeff = getattr(production, "total_value_added", 1)
        # Build the left side: sum of inputs + value_added * VA
        left_expr = sp.Add(*[inputs[key] * symbols[key] for key in inputs.keys()]) + va_coeff * value_added_symbol
        equation = sp.Eq(left_expr, output_symbol)
        return list(symbols.values()) + [value_added_symbol, output_symbol], equation
    else:
        value_added_inputs = production.production_added_values
        value_added_symbols = {key: sp.symbols(key) for key in value_added_inputs.keys()}
        va_coeff = getattr(production, "total_value_added", 1)
        # Build the left side: sum of inputs + value_added * VA
        left_expr = sp.Add(*[value_added_inputs[key] * value_added_symbols[key] for key in value_added_inputs.keys()])
        equation = sp.Eq(left_expr, output_symbol)
        return list(value_added_symbols.values()) + [output_symbol], equation

def evaluate_coefficients(symbols, expr) -> sp.Matrix:
    A, b = sp.linear_eq_to_matrix(expr, symbols)
    # Replace all -1's with 0's
    A = A.applyfunc(lambda x: 0 if x == -1 else x)
    print("Symbols (variables):", symbols)
    print("Expressions:", expr)
    print("Flow Coefficient matrix A (with -1 replaced by 0):")
    sp.pprint(A)
    # print("Right-hand side vector b:")
    # sp.pprint(b)
    return A

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
    # TODO: Implement loading from SQLite database, as well as for the FinalDemand and ValueAdded classes
    return np.array([])

def load_io_table(table_type: str, name: str, trim_extras: bool = True) -> np.ndarray:
    if table_type == 'csv':
        return load_io_table_from_csv(name, trim_extras)
    elif table_type == 'sqlite':
        return load_io_table_from_sqlite(name) 
    else:
        raise ValueError(f"Unknown table type: {table_type}")

# --- Economic Analysis Functions ---

def create_leontief_inverse(Z: np.ndarray, Total_Output: np.ndarray = None, Final_Demand: np.ndarray = None, Value_Added: np.ndarray = None) -> np.ndarray: # pyright: ignore[reportArgumentType]
    """
    Creates the Leontief inverse matrix from the intermediate transaction matrix Z.
    
    The technical coefficient matrix A is defined as A_ij = Z_ij / x_j,
    where x_j is the total output of sector j.
    
    One of Total_Output, Final_Demand, or Value_Added must be provided
    to determine x_j.
    """
    n = Z.shape[0]

    if Total_Output is not None:
        output = Total_Output
    elif Final_Demand is not None:
        # Total output = sum of intermediate sales (row sum) + final demand
        output = Z.sum(axis=1) + Final_Demand  # x_i = sales from i to all + final
    elif Value_Added is not None:
        # Total output = intermediate inputs (column sum) + value added
        output = Z.sum(axis=0) + Value_Added  # x_j = inputs used by j + VA_j
    else:
        raise ValueError("At least one of Total_Output, Final_Demand, or Value_Added must be provided.")

    # Ensure output is a 1D array of length n
    if output.shape != (n,):
        raise ValueError(f"Output must be ({n},), got {output.shape}")

    # Avoid division by zero
    output_safe = np.where(output == 0, 1e-10, output)

    # Compute technical coefficients: A_ij = Z_ij / x_j
    A_coeff = Z / output_safe  # Broadcasting over columns via numpy broadcasting (Z / x_j)

    # Return Leontief inverse
    try:
        # print("Technical Coefficient matrix A:")
        # sp.pprint(A_coeff)
        leontief_inverse = np.linalg.inv(np.eye(n) - A_coeff)
        # print("Leontief Inverse matrix (I - A)^-1:")
        # sp.pprint(leontief_inverse)
        print("\nLeontief Inverse Matrix (I - A)^-1 Created Successfully")
        return leontief_inverse
    except np.linalg.LinAlgError:
        print("Singular matrix: (I - A) is not invertible.")
        return np.eye(n)  # fallback
