import logging
from typing import List, Dict
from Input_Output_Model.models.entities.Good import GoodsDatabase
from Input_Output_Model.models.entities.Good_Indice import GoodsIndiceDatabase
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
    Calculate production prices iteratively based on actual input costs.
    
    This ensures that prices reflect the true cost of production:
    Price = (Sum of input quantities * input prices) + Value Added
    
    Uses iterative refinement since prices depend on each other.
    """
    ptdb = ProductionsDatabase()
    gidb = GoodsIndiceDatabase()
    productions = ptdb.get_all_productions()
    
    # Build a map from ISIC to current price estimate
    # Start with initial estimates based on physical quantities only
    isic_to_price = {}
    
    # First pass: Set initial prices for all goods (sum of physical inputs + VA)
    logger.info("Initial price estimation (physical quantities + VA)...")
    for production in productions:
        total_inputs_qty = sum([quantity for quantity in production.production_inputs.values()])
        total_value_added = sum([quantity for quantity in production.production_added_values.values()])
        
        # Initial price estimate ignoring input prices
        initial_price = total_inputs_qty + total_value_added
        
        if production.isic not in isic_to_price or initial_price < isic_to_price[production.isic]:
            isic_to_price[production.isic] = max(1.0, initial_price)  # Minimum price of 1
    
    # Iteratively refine prices based on input costs
    max_iterations = 10
    convergence_threshold = 0.01  # 1% change
    
    for iteration in range(max_iterations):
        max_change = 0.0
        new_prices = {}
        
        logger.info(f"Price iteration {iteration + 1}...")
        
        for production in productions:
            # Calculate true input cost using current price estimates
            input_cost = 0.0
            for input_isic, input_qty in production.production_inputs.items():
                input_price = isic_to_price.get(input_isic, 1.0)
                input_cost += float(input_qty) * input_price
            
            # Calculate total value added
            total_va = sum([quantity for quantity in production.production_added_values.values()])
            
            # Production price = input costs + value added
            calculated_price = input_cost + total_va
            
            # Track the cheapest production for each good
            if production.isic not in new_prices or calculated_price < new_prices[production.isic]:
                new_prices[production.isic] = max(1.0, calculated_price)
        
        # Check convergence
        for isic in new_prices:
            old_price = isic_to_price.get(isic, 0.0)
            new_price = new_prices[isic]
            if old_price > 0:
                change = abs(new_price - old_price) / old_price
                max_change = max(max_change, change)
        
        isic_to_price = new_prices
        
        logger.info(f"  Max price change: {max_change * 100:.2f}%")
        
        if max_change < convergence_threshold:
            logger.info(f"  Prices converged after {iteration + 1} iterations")
            break
    
    # Final pass: Update all productions with calculated prices
    logger.info("Updating production prices in database...")
    for production in productions:
        # Calculate final input cost
        input_cost = 0.0
        for input_isic, input_qty in production.production_inputs.items():
            input_price = isic_to_price.get(input_isic, 1.0)
            input_cost += float(input_qty) * input_price
        
        # Calculate total value added
        total_va = sum([quantity for quantity in production.production_added_values.values()])
        
        # Final price
        final_price = input_cost + total_va
        
        # Update database
        ptdb.update_production(
            production_id=int(production.id), 
            total_inputs_cost=input_cost, 
            total_value_added=total_va, 
            price=final_price
        )  # pyright: ignore[reportArgumentType]
    
    logger.info(f"Price calculation complete. Final ISIC prices: {isic_to_price}")

def evaluate_indicies(with_functions: bool = False):
    if with_functions:
        evaluate_indicies_with_functions()
    else:
        evaluate_indicies_without_functions()

def evaluate_indicies_without_functions():
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

# TODO: Complete this function
def handle_evaluate_indicie_function(functionDicts: List[Dict]) -> Dict:
    """
    Transforms a list of production dictionaries into tiered supply function profiles
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

# Update this function to ensure it passes the data correctly
def evaluate_indicie_function(productions: List[Production]) -> Dict:
    functionDicts = []
    for production in productions:
        functionDict = {}
        functionDict["cap"] = production.production_quantity
        
        # Ensure these are floats for calculation/sorting
        functionDict["total_inputs_cost"] = float(production.total_inputs_cost)
        functionDict["total_value_added"] = float(production.total_value_added)
        functionDict["price"] = float(production.price)
        
        functionDicts.append(functionDict)

    return handle_evaluate_indicie_function(functionDicts)

# TODO: Complete this function
def evaluate_indicies_with_functions():
    gidb = GoodsIndiceDatabase()
    ptdb = ProductionsDatabase()
    indices = gidb.get_all_good_indices()
    
    for indice in indices:
        productions:List[Production] = ptdb.get_all_productions_by_good(int(indice.id_number))  # pyright: ignore[reportArgumentType]

        production_function = evaluate_indicie_function(productions) # type: ignore
        gidb.update_good_indice(int(indice.id_number), # pyright: ignore[reportArgumentType]
                                total_inputs_cost=production_function["total_inputs_cost"]["tiers"], 
                                total_value_added=production_function["total_value_added"]["tiers"], 
                                price=production_function["price"]["tiers"])
        
# --- Input-Output Matrix Functions ---
import numpy as np

def evaluate_indicies_production_inputs_to_matrix(demoDB=False):
    """
    New Entry Point: Builds the Physical Input-Output Matrix directly from DB.
    Returns:
        A_phys (np.ndarray): The physical technical coefficients (Input/Output).
        VA_phys (np.ndarray): The physical value added (e.g. Labor/Capital per unit).
        isic_map (dict): Mapping from ISIC string -> Matrix Index.
    """
    # 1. Initialize Databases
    if demoDB:
        gidb = GoodsIndiceDatabase(database_url="sqlite:///dataDEMO.db")
        ptdb = ProductionsDatabase(database_url="sqlite:///dataDEMO.db")
    else:
        gidb = GoodsIndiceDatabase()
        ptdb = ProductionsDatabase()

    # 2. Build the Index Map (ISIC -> Row/Col ID)
    # We need a fixed order for the matrix.
    indices = gidb.get_all_good_indices()
    
    # Sort to ensure deterministic matrix order (e.g., by ID or ISIC)
    sorted_indices = sorted(indices, key=lambda x: x.isic) 
    
    n = len(sorted_indices)
    isic_map = {idx.isic: i for i, idx in enumerate(sorted_indices)}
    
    # 3. Initialize the Physical Matrix (A_phys)
    # Rows = Inputs, Columns = Outputs
    A_phys = np.zeros((n, n))
    
    # Initialize Value Added Vector (One row for Total VA, or multiple if you track them)
    # For now, let's assume one aggregate 'Total Value Added' row
    VA_phys = np.zeros(n) 

    # 4. Fill Matrix from Production Recipes
    for output_good in sorted_indices:
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
        # We need coefficients: How much input is needed for 1.0 unit of output?
        total_qty = float(production.production_quantity)
        if total_qty <= 0: total_qty = 1.0 # Prevent division by zero
        
        # A. Fill Intermediate Inputs (The Matrix)
        for input_isic, input_qty in production.production_inputs.items():
            if input_isic in isic_map:
                row_idx = isic_map[input_isic]
                
                # Coefficient = Input Quantity / Total Output Quantity
                coeff = float(input_qty) / total_qty
                A_phys[row_idx, col_idx] = coeff
            else:
                logger.warning(f"Input ISIC {input_isic} not found in Goods Index.")

        # B. Fill Value Added (The Vector)
        # We handle Value Added as a physical ratio too (e.g. $VA per unit output)
        total_va = float(production.total_value_added)
        VA_phys[col_idx] = total_va / total_qty

    logger.info(f"Generated Physical Matrix with shape {A_phys.shape}")
    print("\nPhysical Input-Output Matrix A (Inputs/Outputs):")
    print(A_phys)
    print("\nPhysical Value Added Vector VA (per unit output):")
    print(VA_phys)
    print("\nISIC to Matrix Index Map:")
    print(isic_map)
    return A_phys, VA_phys, isic_map

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
