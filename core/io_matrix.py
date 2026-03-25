"""
Input-Output matrix builder
============================

Functions to build supply curves and the monetary I-O coefficient matrix
from database records. Extracted from the original Evaluators.py.
"""

from __future__ import annotations

import logging
import typing as t
from typing import Dict, List

import numpy as np

from db.repositories.good_repo import GoodsDatabase
from db.repositories.production_repo import Production, ProductionsDatabase
from db.repositories.supply_curve_repo import SupplyCurveDatabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price / cost evaluation
# ---------------------------------------------------------------------------


def evaluate_productions_price() -> None:
    """
    Calculate production prices from monetary input costs and value added.

    Database stores MONETARY costs in ``production_inputs``
    (e.g., $45 worth of steel, $20 worth of coal).  This function sums them:

    * Total Input Cost = Sum(input_cost_i)
    * Total Price = Total Input Cost + Value Added

    No iteration is needed because inputs are already in monetary terms.
    """
    ptdb = ProductionsDatabase()
    productions = ptdb.get_all_productions()

    logger.info("Calculating production prices from monetary input costs…")

    for production in productions:
        input_cost = sum(float(c) for c in production.production_inputs.values())
        total_va = sum(float(v) for v in production.production_added_values.values())
        final_price = input_cost + total_va

        ptdb.update_production(
            production_id=int(production.id),  # type: ignore[arg-type]
            total_inputs_cost=input_cost,
            total_value_added=total_va,
            price=final_price,
        )

    logger.info(f"Price calculation complete for {len(productions)} productions.")


# ---------------------------------------------------------------------------
# Supply curve builders
# ---------------------------------------------------------------------------


def build_supply_curves(with_functions: bool = False) -> None:
    """Build supply curves for all goods from their production methods.

    Parameters
    ----------
    with_functions:
        If ``True``, build tiered (merit-order) supply curves; otherwise use
        the simple single-step approach.
    """
    if with_functions:
        build_supply_curves_with_tiers()
    else:
        build_supply_curves_simple()


def build_supply_curves_simple() -> None:
    """Build simple supply curves using the cheapest production for each good."""
    scdb = SupplyCurveDatabase()
    ptdb = ProductionsDatabase()
    supply_curves = scdb.get_all_supply_curves()

    for curve in supply_curves:
        productions: List[Production] = ptdb.get_all_productions_by_good(
            int(curve.id_number)  # type: ignore[arg-type]
        )
        cheapest = min(productions, key=lambda p: float(p.price))  # type: ignore[arg-type]
        scdb.update_supply_curve(
            int(curve.id_number),  # type: ignore[arg-type]
            production_inputs=cheapest.production_inputs,
            production_added_values=cheapest.production_added_values,
            total_inputs_cost=cheapest.total_inputs_cost,
            total_value_added=cheapest.total_value_added,
            price=cheapest.price,
        )


def build_tiered_supply_curve(function_dicts: List[Dict]) -> Dict:
    """
    Transform a list of production dicts into tiered supply curve profiles
    sorted by merit order (cheapest price first).

    Parameters
    ----------
    function_dicts:
        Each element must have keys: ``cap``, ``price``,
        ``total_inputs_cost``, ``total_value_added``.

    Returns
    -------
    dict
        ``{"price": {"tiers": [...]}, "total_inputs_cost": {"tiers": [...]},
          "total_value_added": {"tiers": [...]}}``
    """
    sorted_productions = sorted(function_dicts, key=lambda x: x["price"])

    price_tiers: List[Dict] = []
    input_cost_tiers: List[Dict] = []
    value_added_tiers: List[Dict] = []

    for p in sorted_productions:
        cap = p["cap"]
        price_tiers.append({"cap": cap, "price": p["price"]})
        input_cost_tiers.append({"cap": cap, "price": p["total_inputs_cost"]})
        value_added_tiers.append({"cap": cap, "price": p["total_value_added"]})

    return {
        "price": {"tiers": price_tiers},
        "total_inputs_cost": {"tiers": input_cost_tiers},
        "total_value_added": {"tiers": value_added_tiers},
    }


def build_supply_curve_from_productions(productions: List[Production]) -> Dict:
    """Build a tiered supply curve dict from a list of Production objects."""
    function_dicts = [
        {
            "cap": p.production_quantity,
            "total_inputs_cost": float(p.total_inputs_cost),
            "total_value_added": float(p.total_value_added),
            "price": float(p.price),
        }
        for p in productions
    ]
    return build_tiered_supply_curve(function_dicts)


def build_supply_curves_with_tiers() -> None:
    """Build tiered supply curves for all goods from their production methods."""
    scdb = SupplyCurveDatabase()
    ptdb = ProductionsDatabase()

    for curve in scdb.get_all_supply_curves():
        productions: List[Production] = ptdb.get_all_productions_by_good(
            int(curve.id_number)  # type: ignore[arg-type]
        )
        data = build_supply_curve_from_productions(productions)  # type: ignore[arg-type]
        scdb.update_supply_curve(
            int(curve.id_number),  # type: ignore[arg-type]
            total_inputs_cost=data["total_inputs_cost"]["tiers"],
            total_value_added=data["total_value_added"]["tiers"],
            price=data["price"]["tiers"],
        )


# ---------------------------------------------------------------------------
# I-O matrix builder
# ---------------------------------------------------------------------------


def build_io_matrix(
    demoDB: bool = False,
    loggingLevel: int = logging.INFO,
) -> t.Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build the monetary Input-Output Coefficient Matrix from the database.

    Database stores MONETARY costs in ``production_inputs``.  The function
    computes:

    * ``A_mon[i, j]`` = dollars of input *i* per dollar of output *j*
    * ``VA_mon[j]``   = dollars of value added per dollar of output *j*

    Parameters
    ----------
    demoDB:
        If ``True`` use the demo database (``dataDEMO.db``); otherwise use
        the production database (``data.db``).
    loggingLevel:
        Python logging level for this operation.

    Returns
    -------
    (A_mon, VA_mon, isic_map)
        Monetary coefficient matrix, value-added vector, and the mapping
        from ISIC code → matrix index.
    """
    logger.setLevel(loggingLevel)

    db_url = "sqlite:///dataDEMO.db" if demoDB else "sqlite:///data.db"
    scdb = SupplyCurveDatabase(database_url=db_url)
    ptdb = ProductionsDatabase(database_url=db_url)

    supply_curves = scdb.get_all_supply_curves()
    sorted_curves = sorted(supply_curves, key=lambda x: x.isic)

    n = len(sorted_curves)
    isic_map: Dict[str, int] = {curve.isic: i for i, curve in enumerate(sorted_curves)}

    A_mon = np.zeros((n, n))
    VA_mon = np.zeros(n)

    for output_good in sorted_curves:
        col_idx = isic_map[output_good.isic]
        prods = ptdb.get_all_productions_by_good(int(output_good.id_number))  # type: ignore[arg-type]

        if not prods:
            continue

        # Marginal / active technology = cheapest production method
        production = min(
            prods,
            key=lambda p: float(p.price) if p.price else float("inf"),
        )

        if production.name == "IMPORT":
            logger.info(
                f"Using IMPORT production for {output_good.name} (ISIC: {output_good.isic})"
            )

        output_price = float(production.price) if production.price else 1.0
        if output_price <= 0:
            output_price = 1.0

        for input_isic, input_cost in production.production_inputs.items():
            if input_isic in isic_map:
                row_idx = isic_map[input_isic]
                A_mon[row_idx, col_idx] = float(input_cost) / output_price
            else:
                logger.warning(f"Input ISIC {input_isic} not found in goods index.")

        total_va = float(production.total_value_added) if production.total_value_added else 0.0
        VA_mon[col_idx] = total_va / output_price

    logger.info(f"Generated monetary coefficient matrix with shape {A_mon.shape}")
    print("\nMonetary Input-Output Coefficient Matrix A:")
    print("(Each element = dollars of input per dollar of output)")
    print(A_mon)
    print("\nMonetary Value Added Coefficients:")
    print(VA_mon)
    print("\nISIC to Matrix Index Map:")
    print(isic_map)
    return A_mon, VA_mon, isic_map


# ---------------------------------------------------------------------------
# Leontief inverse (kept here; also used via core.solvers.leontief)
# ---------------------------------------------------------------------------


def create_leontief_inverse(
    Z: np.ndarray,
    Total_Output: np.ndarray = None,
    Final_Demand: np.ndarray = None,
    Value_Added: np.ndarray = None,
) -> np.ndarray:
    """
    Create the Leontief inverse from an intermediate transaction matrix *Z*.

    Exactly one of ``Total_Output``, ``Final_Demand``, or ``Value_Added``
    must be supplied to determine sector total outputs.
    """
    n = Z.shape[0]

    if Total_Output is not None:
        output = Total_Output
    elif Final_Demand is not None:
        output = Z.sum(axis=1) + Final_Demand
    elif Value_Added is not None:
        output = Z.sum(axis=0) + Value_Added
    else:
        raise ValueError(
            "At least one of Total_Output, Final_Demand, or Value_Added must be provided."
        )

    if output.shape != (n,):
        raise ValueError(f"Output must be ({n},), got {output.shape}")

    output_safe = np.where(output == 0, 1e-10, output)
    A_coeff = Z / output_safe

    try:
        leontief_inverse = np.linalg.inv(np.eye(n) - A_coeff)
        print("\nLeontief Inverse Matrix (I - A)^-1 Created Successfully")
        return leontief_inverse
    except np.linalg.LinAlgError:
        print("Singular matrix: (I - A) is not invertible.")
        return np.eye(n)
