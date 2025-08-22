import logging
from typing import List, Dict
from Good import GoodsDatabase
from Good_Indice import GoodsIndiceDatabase
from Production import *
import sympy as sp
import typing as t

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
        price = sum([quantity for quantity in production.production_inputs.values()]) + production.production_added_value
        ptdb.update_production(int(production.id), price=price)  # pyright: ignore[reportArgumentType]

def evaluate_indicies():
    gidb = GoodsIndiceDatabase()
    ptdb = ProductionsDatabase()
    indices = gidb.get_all_good_indices()
    
    for indice in indices:
        productions:List[Production] = ptdb.get_all_productions_by_good(int(indice.id_number))  # pyright: ignore[reportArgumentType]

        cheapest_production = min(productions, key=lambda p: p.price) # type: ignore
        price = cheapest_production.price
        inputs = cheapest_production.production_inputs
        gidb.update_good_indice(int(indice.id_number), production_inputs=inputs, price=price)  # pyright: ignore[reportArgumentType]

def evaluate_production_inputs_to_equations():
    """Evaluate production inputs to equations."""
    ptdb = ProductionsDatabase()
    productions = ptdb.get_all_productions()
    symbols = set()
    equations = []

    for production in productions:
        result = evaluate_production_inputs_to_equation(production,"sympy")
        if result:
            symbols.update(result[0])
            equations.append(result[1])

    evaluate_coefficients(list(symbols), equations)

def evaluate_production_inputs_to_equation(production: Production, format: str):
    if format == "string":
        return production_inputs_to_string(production)
    elif format == "sympy":
        return production_inputs_to_sympy(production)
    else:
        logger.error(f"Unknown format: {format}")
        return None

def production_inputs_to_string(production: Production) -> str:
    inputs = production.production_inputs
    equals = " + ".join([f"{value} * {key}" for key, value in inputs.items()])
    equation = f"{production.isic} = {equals}"
    return equation

def production_inputs_to_sympy(production: Production):
    inputs = production.production_inputs
    symbols = {key: sp.symbols(key) for key in inputs.keys()}
    output_symbol = sp.symbols(str(production.isic))
    equation = sp.Eq(
        sp.Add(*[inputs[key] * symbols[key] for key in inputs.keys()]),
        output_symbol
    )
    return list(symbols.values()) + [output_symbol], equation

def evaluate_coefficients(symbols, expr):
    A, b = sp.linear_eq_to_matrix(expr, symbols)
    print("Symbols (variables):", symbols)
    print("Expressions:", expr)
    print("Coefficient matrix A:")
    sp.pprint(A)
    # print("Right-hand side vector b:")
    # sp.pprint(b)

evaluate_production_inputs_to_equations()

