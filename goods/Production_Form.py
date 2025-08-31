import Production
import logging
import types
from sqlalchemy import create_engine, Column, Integer, String

producer = str()

def producer_log_in(producer_name: str) -> None:
    """Log in a producer by name."""
    global producer
    producer = producer_name
    logging.info(f"Producer {producer} logged in.")
     
def add_production():

    connection = Production.ProductionsDatabase()

    connection.add_production(
        name=producer, id_number=int(1), producer=int(1), produce=int(1)
    )

def update_production(production_id: int, **kwargs) -> bool:
    """
    Update a production's details by its ID.
    Returns True if update was successful, False otherwise.
    """
    connection = Production.ProductionsDatabase()
    try:
        with connection.get_session() as session:
            production = session.query(Production.Production).filter_by(id=production_id).first()
            if not production:
                logging.warning(f"No production found with ID: {production_id}")
                return False
            for key, value in kwargs.items():
                if hasattr(production, key):
                    setattr(production, key, value)
            session.commit()
            logging.info(f"Updated production with ID: {production_id}")
            return True
    except Exception as e:
        logging.error(f"Failed to update production with ID {production_id}: {e}")
        return False