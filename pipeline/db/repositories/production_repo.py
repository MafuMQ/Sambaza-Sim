"""
Production repository
======================

ORM model and database manager for production methods.
"""

from sqlalchemy import Column, Integer, String
from sqlalchemy.types import JSON
from sqlalchemy.orm import declarative_base
from typing import Optional, List
import logging

from pipeline.db.base import DatabaseBase

logger = logging.getLogger(__name__)
Base = declarative_base()


class Production(Base):
    """ORM model representing a production method."""

    __tablename__ = "productions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    descriptive_name = Column(String)
    id_number = Column(Integer, nullable=False)  # FK to Good.id_number
    isic = Column(String, nullable=False)
    producer = Column(Integer, nullable=False)
    produce = Column(Integer, nullable=False)
    produce_name = Column(String, nullable=False)

    production_inputs = Column(JSON)
    production_added_values = Column(JSON)
    production_rate = Column(Integer)
    production_quantity = Column(Integer)
    production_material_efficiency = Column(Integer)
    production_labour_efficiency = Column(Integer)
    production_energy_efficiency = Column(Integer)

    # Contact
    contact_name = Column(String)
    contact_email = Column(String)
    contact_phone = Column(Integer)
    contact_phone2 = Column(Integer)
    contact_website = Column(String)

    # Address
    address = Column(String)
    address_street = Column(String)
    address_city = Column(String)
    address_country = Column(String)
    address_postal_code = Column(String)

    total_inputs_cost = Column(Integer, nullable=True)
    total_value_added = Column(Integer, nullable=True)
    price = Column(String, nullable=True)

    def __repr__(self) -> str:
        return f"<Production(name={self.name}, id_number={self.id_number})>"

    def __str__(self) -> str:
        return f"Production(name={self.name}, id_number={self.id_number})"


class ProductionsDatabase(DatabaseBase):
    """Database manager for production methods."""

    def __init__(self, database_url: str = "sqlite:///data.db", echo: bool = False):
        super().__init__(database_url, echo, Base)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def add_production(
        self, name: str, id_number: int, producer: int, produce: int, **kwargs
    ) -> Optional[Production]:
        """Add a new production method."""
        try:
            with self.get_session() as session:
                production = Production(
                    name=name,
                    id_number=id_number,
                    producer=producer,
                    produce=produce,
                    **kwargs,
                )
                session.add(production)
                session.flush()
                logger.info(f"Added production: {production}")
                return production
        except Exception as exc:
            logger.error(f"Failed to add production '{name}': {exc}")
            return None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all_productions(self, limit: Optional[int] = None) -> List[Production]:
        """Retrieve all production methods with optional limit."""
        try:
            with self.get_session() as session:
                query = session.query(Production)
                if limit is not None:
                    query = query.limit(limit)
                productions = query.all()
                logger.info(f"Retrieved {len(productions)} productions")
                return productions
        except Exception as exc:
            logger.error(f"Failed to retrieve productions: {exc}")
            return []

    def get_all_productions_by_producer(self, producer_id: int) -> List[Production]:
        """Retrieve all production methods by a specific producer."""
        try:
            with self.get_session() as session:
                productions = session.query(Production).filter_by(producer=producer_id).all()
                logger.info(
                    f"Retrieved {len(productions)} productions for producer ID {producer_id}"
                )
                return productions
        except Exception as exc:
            logger.error(
                f"Failed to retrieve productions for producer ID {producer_id}: {exc}"
            )
            return []

    def get_all_productions_by_good(self, good_id: int) -> List[Production]:
        """Retrieve all production methods for a specific good."""
        try:
            with self.get_session() as session:
                productions = session.query(Production).filter_by(produce=good_id).all()
                logger.info(
                    f"Retrieved {len(productions)} productions for good ID {good_id}"
                )
                return productions
        except Exception as exc:
            logger.error(
                f"Failed to retrieve productions for good ID {good_id}: {exc}"
            )
            return []

    def get_production_by_id(self, production_id: int) -> Optional[Production]:
        """Retrieve a production method by its ID."""
        try:
            with self.get_session() as session:
                production = session.query(Production).filter_by(id=production_id).first()
                if production:
                    logger.info(f"Retrieved production: {production}")
                else:
                    logger.warning(f"No production found with ID: {production_id}")
                return production
        except Exception as exc:
            logger.error(f"Failed to retrieve production by ID {production_id}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_production(self, production_id: int, **kwargs) -> bool:
        """Update a production method's details. Returns True on success."""
        try:
            with self.get_session() as session:
                production = session.query(Production).filter_by(id=production_id).first()
                if not production:
                    logger.warning(f"No production found with ID: {production_id}")
                    return False
                for key, value in kwargs.items():
                    if hasattr(production, key):
                        setattr(production, key, value)
                session.commit()
                logger.info(f"Updated production with ID: {production_id}")
                return True
        except Exception as exc:
            logger.error(f"Failed to update production with ID {production_id}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_production(self, production_id: int) -> bool:
        """Delete a production method by its ID."""
        try:
            with self.get_session() as session:
                production = session.query(Production).filter_by(id=production_id).first()
                if production:
                    session.delete(production)
                    session.commit()
                    logger.info(f"Deleted production with ID: {production_id}")
                    return True
                logger.warning(f"No production found with ID: {production_id}")
                return False
        except Exception as exc:
            logger.error(f"Failed to delete production with ID {production_id}: {exc}")
            return False


