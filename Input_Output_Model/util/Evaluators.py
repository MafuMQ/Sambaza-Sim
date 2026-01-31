import logging
from typing import List, Dict
from Input_Output_Model.models.entities.Good import GoodsDatabase
from Input_Output_Model.models.entities.SupplyCurve import SupplyCurveDatabase
from Input_Output_Model.models.entities.Production import *
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
    """
    Calculate production prices from monetary input costs and value added.
    
    Database stores MONETARY costs in production_inputs (e.g., $45 worth of steel, $20 worth of coal).
    This function simply sums them:
    
    Total Input Cost = Sum(input_cost_i) for each input
    Total Price = Total Input Cost + Value Added
    
    No iteration needed since inputs are already in monetary terms.
    """
    ptdb = ProductionsDatabase()
    productions = ptdb.get_all_productions()
    
    logger.info("Calculating production prices from monetary input costs...")
    
    for production in productions:
        # Sum monetary input costs (already in dollars)
        input_cost = sum([float(cost) for cost in production.production_inputs.values()])
        
        # Sum monetary value added components (already in dollars)
        total_va = sum([float(va) for va in production.production_added_values.values()])
        
        # Production price = sum of input costs + value added
        final_price = input_cost + total_va
        
        # Update database
        ptdb.update_production(
            production_id=int(production.id), 
            total_inputs_cost=input_cost, 
            total_value_added=total_va, 
            price=final_price
        )  # pyright: ignore[reportArgumentType]
    
    logger.info(f"Price calculation complete for {len(productions)} productions.")

def build_supply_curves(with_functions: bool = False):
    """Build supply curves for all goods from production methods."""
    if with_functions:
        build_supply_curves_with_tiers()
    else:
        build_supply_curves_simple()

def build_supply_curves_simple():
    """Build simple supply curves using cheapest production for each good."""
    scdb = SupplyCurveDatabase()
    ptdb = ProductionsDatabase()
    supply_curves = scdb.get_all_supply_curves()
    
    for curve in supply_curves:
        productions: List[Production] = ptdb.get_all_productions_by_good(int(curve.id_number))  # pyright: ignore[reportArgumentType]

        cheapest_production = min(productions, key=lambda p: float(p.price))  # type: ignore
        production_added_values = cheapest_production.production_added_values
        scdb.update_supply_curve(
            int(curve.id_number),  # pyright: ignore[reportArgumentType]
            production_inputs=cheapest_production.production_inputs,
            production_added_values=production_added_values,
            total_inputs_cost=cheapest_production.total_inputs_cost,
            total_value_added=cheapest_production.total_value_added,
            price=cheapest_production.price
        )  

def build_tiered_supply_curve(functionDicts: List[Dict]) -> Dict:
    """
    Transforms a list of production dictionaries into tiered supply curve profiles
    sorted by Merit Order (Cheapest Price First).
    
    Returns a dictionary where each key (price, total_inputs_cost, etc.) contains 
    a JSON-compatible structure defining that specific curve.
    """
    # 1. Sort by Price (Merit Order)
    # We sort all attributes based on the 'price' to ensure the market clears 
    # the cheapest options first.
    sorted_productions = sorted(functionDicts, key=lambda x: x['price'])
    
    # 2. Initialize the tier lists
    price_tiers = []
    input_cost_tiers = []
    value_added_tiers = []
    
    # 3. Build the tiers
    for p in sorted_productions:
        # Ensure cap is valid (handle None or -1 if necessary, though logic assumes finite here)
        cap = p['cap']
        
        # Append to Price Curve
        price_tiers.append({
            "cap": cap, 
            "price": p['price']
        })
        
        # Append to Input Cost Curve
        # Note: We use the key "price" inside the tier because the NumPy loader 
        # expects generic keys: 'cap' and 'price' (value).
        input_cost_tiers.append({
            "cap": cap, 
            "price": p['total_inputs_cost']
        })
        
        # Append to Value Added Curve
        value_added_tiers.append({
            "cap": cap, 
            "price": p['total_value_added']
        })

    # 4. Return the structures
    return {
        "price": {"tiers": price_tiers},
        "total_inputs_cost": {"tiers": input_cost_tiers},
        "total_value_added": {"tiers": value_added_tiers}
    }

def build_supply_curve_from_productions(productions: List[Production]) -> Dict:
    """Build tiered supply curve from a list of production methods."""
    functionDicts = []
    for production in productions:
        functionDict = {}
        functionDict["cap"] = production.production_quantity
        
        # Ensure these are floats for calculation/sorting
        functionDict["total_inputs_cost"] = float(production.total_inputs_cost)
        functionDict["total_value_added"] = float(production.total_value_added)
        functionDict["price"] = float(production.price)
        
        functionDicts.append(functionDict)

    return build_tiered_supply_curve(functionDicts)

def build_supply_curves_with_tiers():
    """Build tiered supply curves for all goods from their production methods."""
    scdb = SupplyCurveDatabase()
    ptdb = ProductionsDatabase()
    supply_curves = scdb.get_all_supply_curves()
    
    for curve in supply_curves:
        productions: List[Production] = ptdb.get_all_productions_by_good(int(curve.id_number))  # pyright: ignore[reportArgumentType]

        supply_curve_data = build_supply_curve_from_productions(productions)  # type: ignore
        scdb.update_supply_curve(
            int(curve.id_number),  # pyright: ignore[reportArgumentType]
            total_inputs_cost=supply_curve_data["total_inputs_cost"]["tiers"],
            total_value_added=supply_curve_data["total_value_added"]["tiers"],
            price=supply_curve_data["price"]["tiers"]
        )
        
# --- Input-Output Matrix Functions ---
import numpy as np

def build_io_matrix(demoDB=False, loggingLevel=logging.INFO) -> t.Tuple[np.ndarray, np.ndarray, dict]:
    """
    Builds the MONETARY Input-Output Coefficient Matrix from DB.
    
    Database stores MONETARY costs in production_inputs (e.g., $45 worth of steel).
    These costs are not yet normalized per dollar of output.
    
    This function builds monetary technical coefficients:
    A_mon[i,j] = dollars of input i per dollar of output j
    
    Returns:
        A_mon (np.ndarray): Monetary technical coefficients.
                           A[i,j] = $ of input i per $ of output j
        VA_mon (np.ndarray): Monetary value added per $ of output
        isic_map (dict): Mapping from ISIC string -> Matrix Index.
    """
    logger.setLevel(loggingLevel)
    # 1. Initialize Databases
    if demoDB:
        scdb = SupplyCurveDatabase(database_url="sqlite:///dataDEMO.db")
        ptdb = ProductionsDatabase(database_url="sqlite:///dataDEMO.db")
    else:
        scdb = SupplyCurveDatabase()
        ptdb = ProductionsDatabase()

    # 2. Build the Index Map (ISIC -> Row/Col ID)
    # We need a fixed order for the matrix.
    supply_curves = scdb.get_all_supply_curves()
    
    # Sort to ensure deterministic matrix order (e.g., by ID or ISIC)
    sorted_curves = sorted(supply_curves, key=lambda x: x.isic) 
    
    n = len(sorted_curves)
    isic_map = {curve.isic: i for i, curve in enumerate(sorted_curves)}
    
    # 3. Initialize the Monetary Coefficient Matrix (A_mon)
    # Rows = Inputs, Columns = Outputs
    # A[i,j] = dollars of input i per dollar of output j
    A_mon = np.zeros((n, n))
    
    # Initialize Value Added Vector (dollars of VA per dollar of output)
    VA_mon = np.zeros(n) 

    # 4. Fill Matrix from Production Recipes
    for output_good in sorted_curves:
        col_idx = isic_map[output_good.isic]
        
        # Fetch production method for this good
        prods = ptdb.get_all_productions_by_good(int(output_good.id_number))
        
        if not prods:
            continue
        
        # Use the cheapest production method (could be domestic or import)
        # This represents the marginal/active production technology
        production = min(prods, key=lambda p: float(p.price) if p.price else float('inf'))
        
        if production.name == "IMPORT":
            logger.info(f"Using IMPORT production for {output_good.name} (ISIC: {output_good.isic})") 

        # --- THE CORE MATH ---
        # production_inputs already stores MONETARY costs (e.g., $45 worth of steel)
        # We need to normalize to: A_mon[i,j] = $ of input i per $ of output j
        #
        # Strategy: Divide each monetary input cost by total output price
        
        output_price = float(production.price) if production.price else 1.0
        if output_price <= 0: output_price = 1.0
        
        # A. Fill Intermediate Input Coefficients (monetary)
        # For each input: input_cost / output_price
        for input_isic, input_cost_monetary in production.production_inputs.items():
            if input_isic in isic_map:
                row_idx = isic_map[input_isic]
                
                # Monetary coefficient: $ of input / $ of output
                coeff = float(input_cost_monetary) / output_price
                A_mon[row_idx, col_idx] = coeff
            else:
                logger.warning(f"Input ISIC {input_isic} not found in Goods Index.")

        # B. Fill Value Added Coefficient (monetary)
        # total_value_added is already in monetary terms
        total_va_monetary = float(production.total_value_added) if production.total_value_added else 0.0
        VA_mon[col_idx] = total_va_monetary / output_price

    logger.info(f"Generated Monetary Coefficient Matrix with shape {A_mon.shape}")
    print("\nMonetary Input-Output Coefficient Matrix A:")
    print("(Each element = dollars of input per dollar of output)")
    print(A_mon)
    print("\nMonetary Value Added Coefficients:")
    print(VA_mon)
    print("\nISIC to Matrix Index Map:")
    print(isic_map)
    return A_mon, VA_mon, isic_map

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
