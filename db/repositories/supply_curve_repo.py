"""
SupplyCurve repository
=======================

ORM model and database manager for supply curves.
"""

from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy.orm import declarative_base
from typing import Optional, List
import logging

from db.base import DatabaseBase

logger = logging.getLogger(__name__)
Base = declarative_base()


class SupplyCurve(Base):
    """ORM model representing a tiered supply curve for a good."""

    __tablename__ = "supply_curves"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    id_number = Column(Integer, unique=True, nullable=False)  # FK to Good.id_number
    isic = Column(String, nullable=False)

    production_inputs = Column(JSON)
    production_added_values = Column(JSON)
    total_inputs_cost = Column(JSON, nullable=True)
    total_value_added = Column(JSON, nullable=True)
    price = Column(JSON, nullable=True)
    price_history = Column(JSON, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<SupplyCurve(id={self.id}, name='{self.name}', "
            f"id_number={self.id_number}, isic='{self.isic}', "
            f"price={self.price}, "
            f"production_added_values={self.production_added_values})>"
        )

    def __str__(self) -> str:
        return (
            f"SupplyCurve(name='{self.name}', id_number={self.id_number}, "
            f"isic={self.isic}, price={self.price}, "
            f"production_added_values={self.production_added_values})"
        )


class SupplyCurveDatabase(DatabaseBase):
    """Database manager for supply curves."""

    def __init__(self, database_url: str = "sqlite:///data.db", echo: bool = False):
        super().__init__(database_url, echo, Base)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def add_supply_curve(self, name: str, id_number: int, isic: str) -> None:
        """Add a new supply curve to the database."""
        try:
            with self.get_session() as session:
                supply_curve = SupplyCurve(name=name, id_number=id_number, isic=isic)
                session.add(supply_curve)
                session.flush()
                logger.info(f"Added supply curve: {supply_curve}")
        except Exception as exc:
            logger.error(f"Failed to add supply curve: {exc}")
            raise

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all_supply_curves(self, limit: Optional[int] = None) -> List[SupplyCurve]:
        """Retrieve all supply curves with optional limit."""
        try:
            with self.get_session() as session:
                query = session.query(SupplyCurve)
                if limit is not None:
                    query = query.limit(limit)
                supply_curves = query.all()
                logger.info(f"Retrieved {len(supply_curves)} supply curves")
                return supply_curves
        except Exception as exc:
            logger.error(f"Failed to retrieve supply curves: {exc}")
            return []

    def get_supply_curve_by_id(self, id_number: int) -> Optional[SupplyCurve]:
        """Retrieve a supply curve by its id_number."""
        try:
            with self.get_session() as session:
                supply_curve = (
                    session.query(SupplyCurve).filter_by(id_number=id_number).first()
                )
                if supply_curve:
                    logger.info(f"Retrieved supply curve: {supply_curve}")
                else:
                    logger.warning(f"No supply curve found with id_number: {id_number}")
                return supply_curve
        except Exception as exc:
            logger.error(
                f"Failed to retrieve supply curve by id_number {id_number}: {exc}"
            )
            return None

    def get_supply_curves_by_id_number(self, id_number: int) -> List[SupplyCurve]:
        """Retrieve all supply curves for a specific good id_number."""
        try:
            with self.get_session() as session:
                supply_curves = (
                    session.query(SupplyCurve).filter_by(good_id_number=id_number).all()
                )
                logger.info(
                    f"Retrieved {len(supply_curves)} supply curves for good ID {id_number}"
                )
                return supply_curves
        except Exception as exc:
            logger.error(
                f"Failed to retrieve supply curves for good ID {id_number}: {exc}"
            )
            return []

    def get_supply_curves_by_isic(self, isic: str) -> List[SupplyCurve]:
        """Retrieve all supply curves for a specific ISIC code."""
        try:
            with self.get_session() as session:
                supply_curves = (
                    session.query(SupplyCurve).filter_by(isic=isic).all()
                )
                logger.info(
                    f"Retrieved {len(supply_curves)} supply curves for ISIC {isic}"
                )
                return supply_curves
        except Exception as exc:
            logger.error(
                f"Failed to retrieve supply curves for ISIC {isic}: {exc}"
            )
            return []

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_supply_curve(
        self, id_number: int, **kwargs
    ) -> Optional[SupplyCurve]:
        """Update a supply curve's details by id_number."""
        try:
            with self.get_session() as session:
                supply_curve = (
                    session.query(SupplyCurve).filter_by(id_number=id_number).first()
                )
                if supply_curve:
                    for key, value in kwargs.items():
                        setattr(supply_curve, key, value)
                    session.commit()
                    logger.info(f"Updated supply curve: {supply_curve}")
                    return supply_curve
                logger.warning(f"No supply curve found with id_number: {id_number}")
                return None
        except Exception as exc:
            logger.error(
                f"Failed to update supply curve with id_number {id_number}: {exc}"
            )
            return None

    def update_supply_curve_by_isic(
        self, isic: str, **kwargs
    ) -> Optional[SupplyCurve]:
        """Update a supply curve by ISIC code (expects a unique match)."""
        try:
            with self.get_session() as session:
                supply_curve = (
                    session.query(SupplyCurve).filter_by(isic=isic).first()
                )
                if supply_curve:
                    for key, value in kwargs.items():
                        setattr(supply_curve, key, value)
                    session.commit()
                    logger.info(f"Updated supply curve for ISIC {isic}")
                    return supply_curve
                logger.warning(f"No supply curve found with ISIC: {isic}")
                return None
        except Exception as exc:
            logger.error(
                f"Failed to update supply curve with ISIC {isic}: {exc}"
            )
            return None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_supply_curve(self, id_number: int) -> bool:
        """Delete a supply curve by its id_number."""
        try:
            with self.get_session() as session:
                supply_curve = (
                    session.query(SupplyCurve).filter_by(id_number=id_number).first()
                )
                if supply_curve:
                    session.delete(supply_curve)
                    session.commit()
                    logger.info(f"Deleted supply curve with id_number: {id_number}")
                    return True
                logger.warning(
                    f"No supply curve found with id_number: {id_number}"
                )
                return False
        except Exception as exc:
            logger.error(
                f"Failed to delete supply curve with id_number {id_number}: {exc}"
            )
            return False
