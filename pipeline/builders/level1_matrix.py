import logging
import typing as t
from typing import Dict
import numpy as np

from pipeline.db.repositories.production_repo import ProductionsDatabase
from pipeline.db.repositories.supply_curve_repo import SupplyCurveDatabase

logger = logging.getLogger(__name__)

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


