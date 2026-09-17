"""
Good repository
================

ORM model and database manager for goods.
"""

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from typing import Optional, List
import logging

from pipeline.db.base import DatabaseBase

logger = logging.getLogger(__name__)
Base = declarative_base()


class Good(Base):
    """ORM model representing a good with ISIC classification details."""

    __tablename__ = "goods"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    descriptive_name = Column(String)
    id_number = Column(Integer, unique=True, nullable=False)
    isic = Column(String, nullable=False)
    isic_section = Column(String)
    isic_division = Column(Integer)
    isic_group = Column(Integer)
    isic_class = Column(Integer)
    sub_class_a = Column(Integer)
    sub_class_b = Column(Integer)
    sub_class_c = Column(Integer)
    sub_class_nf = Column(Integer)

    def __repr__(self) -> str:
        return f"<Good(id={self.id}, name='{self.name}', isic='{self.isic}')>"

    def __str__(self) -> str:
        return f"{self.name} ({self.descriptive_name or 'No description'})"


class GoodsDatabase(DatabaseBase):
    """Database manager for goods."""

    def __init__(self, database_url: str = "sqlite:///data.db", echo: bool = False):
        super().__init__(database_url, echo, Base)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def add_good(
        self,
        name: str,
        descriptive_name: Optional[str] = None,
        id_number: int = 0,
        isic: str = "",
        **kwargs,
    ) -> Optional[Good]:
        """Add a new good to the database."""
        try:
            with self.get_session() as session:
                good = Good(
                    name=name,
                    descriptive_name=descriptive_name,
                    id_number=id_number,
                    isic=isic,
                    **kwargs,
                )
                session.add(good)
                session.flush()
                logger.info(f"Added good: {good}")
                return good
        except Exception as exc:
            logger.error(f"Failed to add good: {exc}")
            return None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all_goods(self, limit: Optional[int] = None) -> List[Good]:
        """Retrieve all goods with optional limit."""
        try:
            with self.get_session() as session:
                query = session.query(Good)
                if limit:
                    query = query.limit(limit)
                goods = query.all()
                for good in goods:
                    # Ensure attributes loaded before session closes
                    _ = (
                        good.id, good.name, good.descriptive_name, good.id_number,
                        good.isic, good.isic_section, good.isic_division,
                        good.isic_group, good.isic_class, good.sub_class_a,
                        good.sub_class_b, good.sub_class_c, good.sub_class_nf,
                    )
                return goods
        except Exception as exc:
            logger.error(f"Failed to get all goods: {exc}")
            return []

    def get_goods_by_subclass(self, subclass_value: int) -> List[Good]:
        """Retrieve goods where ANY subclass column matches the given value."""
        try:
            with self.get_session() as session:
                goods = session.query(Good).filter(
                    (Good.sub_class_a == subclass_value)
                    | (Good.sub_class_b == subclass_value)
                    | (Good.sub_class_c == subclass_value)
                    | (Good.sub_class_nf == subclass_value)
                ).all()
                for good in goods:
                    _ = (
                        good.id, good.name, good.descriptive_name, good.id_number,
                        good.isic, good.isic_section, good.isic_division,
                        good.isic_group, good.isic_class, good.sub_class_a,
                        good.sub_class_b, good.sub_class_c, good.sub_class_nf,
                    )
                return goods
        except Exception as exc:
            logger.error(f"Failed to get goods by any subclass {subclass_value}: {exc}")
            return []

    def get_goods_by_specific_subclass(
        self, subclass_value: int, subclass_column: str
    ) -> List[Good]:
        """Retrieve goods by a specific subclass column."""
        valid_columns = ["sub_class_a", "sub_class_b", "sub_class_c", "sub_class_nf"]
        if subclass_column not in valid_columns:
            logger.error(
                f"Invalid subclass column: {subclass_column}. Valid: {valid_columns}"
            )
            return []
        try:
            with self.get_session() as session:
                column = getattr(Good, subclass_column)
                goods = session.query(Good).filter(column == subclass_value).all()
                for good in goods:
                    _ = (
                        good.id, good.name, good.descriptive_name, good.id_number,
                        good.isic, good.isic_section, good.isic_division,
                        good.isic_group, good.isic_class, good.sub_class_a,
                        good.sub_class_b, good.sub_class_c, good.sub_class_nf,
                    )
                logger.info(
                    f"Found {len(goods)} goods with {subclass_column} = {subclass_value}"
                )
                return goods
        except Exception as exc:
            logger.error(
                f"Failed to get goods by {subclass_column} = {subclass_value}: {exc}"
            )
            return []

    def search_goods(self, search_term: str) -> List[Good]:
        """Search goods by name or descriptive name."""
        try:
            with self.get_session() as session:
                return session.query(Good).filter(
                    Good.name.contains(search_term)
                    | Good.descriptive_name.contains(search_term)
                ).all()
        except Exception as exc:
            logger.error(f"Failed to search goods with term '{search_term}': {exc}")
            return []

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_good(self, good_id: int, **kwargs) -> Optional[Good]:
        """Update a good's attributes."""
        try:
            with self.get_session() as session:
                good = session.query(Good).filter(Good.id == good_id).first()
                if not good:
                    logger.warning(f"Good with ID {good_id} not found")
                    return None
                for key, value in kwargs.items():
                    if hasattr(good, key):
                        setattr(good, key, value)
                session.add(good)
                session.flush()
                logger.info(f"Updated good: {good}")
                return good
        except Exception as exc:
            logger.error(f"Failed to update good {good_id}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_good(self, good_id: int) -> bool:
        """Delete a good by its ID."""
        try:
            with self.get_session() as session:
                good = session.query(Good).filter(Good.id == good_id).first()
                if not good:
                    logger.warning(f"Good with ID {good_id} not found")
                    return False
                session.delete(good)
                logger.info(f"Deleted good: {good}")
                return True
        except Exception as exc:
            logger.error(f"Failed to delete good {good_id}: {exc}")
            return False


