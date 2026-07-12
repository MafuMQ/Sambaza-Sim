import typing as t
from typing import Dict, List
from pipeline.db.repositories.production_repo import Production, ProductionsDatabase
from pipeline.db.repositories.supply_curve_repo import SupplyCurveDatabase

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


