import logging
from pipeline.db.repositories.production_repo import ProductionsDatabase

logger = logging.getLogger(__name__)

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


