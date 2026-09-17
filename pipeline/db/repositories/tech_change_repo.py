"""
TechChange repository
======================

ORM model and database manager for technological change configurations
used by the demo and dashboard examples.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, Text
from sqlalchemy.types import JSON
from sqlalchemy.orm import declarative_base
from typing import Optional, List, Dict, Any
import logging
import json as json_lib

from pipeline.db.base import DatabaseBase

logger = logging.getLogger(__name__)
Base = declarative_base()


class TechChange(Base):
    """ORM model representing a technological change configuration."""

    __tablename__ = "tech_changes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    example_id = Column(Integer, unique=True, nullable=False, index=True)
    change_type = Column(String, nullable=False, index=True)  # 'tax_policy' | 'tech_change'

    title = Column(String, nullable=False)
    description = Column(Text)

    # Final demand
    final_demand = Column(JSON)

    # Tax / demand parameters
    total_demand = Column(Float, nullable=True)
    uniform_demand = Column(Float, nullable=True)
    demand_vector = Column(JSON, nullable=True)
    proportions = Column(JSON, nullable=True)
    wage_proportions = Column(JSON, nullable=True)
    surplus_proportions = Column(JSON, nullable=True)
    government_proportions = Column(JSON, nullable=True)

    income_tax_rate_before = Column(Float, nullable=True)
    income_tax_rate_after = Column(Float, nullable=True)
    corporate_tax_rate_before = Column(Float, nullable=True)
    corporate_tax_rate_after = Column(Float, nullable=True)
    income_tax_applies_to = Column(String, nullable=True)

    consumption_rate = Column(Float, nullable=True, default=1.0)
    iterations = Column(Integer, nullable=True, default=1)

    # Tech-change parameters
    tech_change_function_name = Column(String, nullable=True)
    tech_change_params = Column(JSON, nullable=True)
    use_multi_level = Column(Boolean, default=False)
    solver_type = Column(String, nullable=True, default="leontief")

    target_isic = Column(String, nullable=True)
    demand_shock = Column(Float, nullable=True)
    is_tech_comparison = Column(Boolean, default=False)

    def __repr__(self) -> str:
        return (
            f"<TechChange(id={self.id}, example_id={self.example_id}, "
            f"type='{self.change_type}', title='{self.title}')>"
        )

    def __str__(self) -> str:
        return f"Example {self.example_id}: {self.title}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary compatible with the demo config structure."""
        result = {
            "title": self.title,
            "description": self.description.split("\n") if self.description else [],
            "params": {},
        }

        common_fields = ["iterations", "consumption_rate"]

        tax_policy_fields = [
            "total_demand", "uniform_demand", "demand_vector", "proportions",
            "target_isic", "demand_shock",
            "wage_proportions", "surplus_proportions", "government_proportions",
            "income_tax_rate_before", "income_tax_rate_after",
            "corporate_tax_rate_before", "corporate_tax_rate_after",
            "income_tax_applies_to",
        ]

        tech_change_fields = [
            "final_demand", "use_multi_level", "solver_type", "is_tech_comparison",
            "tech_change_function_name", "tech_change_params",
            "income_tax_rate_before", "income_tax_rate_after",
            "corporate_tax_rate_before", "corporate_tax_rate_after",
            "income_tax_applies_to",
            "wage_proportions", "surplus_proportions", "government_proportions",
        ]

        allowed_fields = (
            common_fields + tax_policy_fields
            if self.change_type == "tax_policy"
            else common_fields + tech_change_fields
        )

        for field in allowed_fields:
            value = getattr(self, field, None)
            if value is not None:
                result["params"][field] = value

        return result


class TechChangeDatabase(DatabaseBase):
    """Database manager for tech change configurations."""

    def __init__(self, database_url: str = "sqlite:///data.db", echo: bool = False):
        super().__init__(database_url, echo, Base)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def add_tech_change(
        self,
        example_id: int,
        change_type: str,
        title: str,
        description: str = "",
        **kwargs,
    ) -> Optional[TechChange]:
        """Add a new tech change configuration."""
        try:
            with self.get_session() as session:
                tech_change = TechChange(
                    example_id=example_id,
                    change_type=change_type,
                    title=title,
                    description=description,
                    **kwargs,
                )
                session.add(tech_change)
                session.flush()
                logger.info(f"Added tech change: {tech_change}")
                return tech_change
        except Exception as exc:
            logger.error(f"Failed to add tech change: {exc}")
            return None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all_tech_changes(self, limit: Optional[int] = None) -> List[TechChange]:
        """Retrieve all tech change configurations ordered by example_id."""
        try:
            with self.get_session() as session:
                query = session.query(TechChange).order_by(TechChange.example_id)
                if limit:
                    query = query.limit(limit)
                tech_changes = query.all()
                logger.info(f"Retrieved {len(tech_changes)} tech changes")
                return tech_changes
        except Exception as exc:
            logger.error(f"Failed to get all tech changes: {exc}")
            return []

    def get_tech_change_by_id(self, example_id: int) -> Optional[TechChange]:
        """Retrieve a specific tech change by example_id."""
        try:
            with self.get_session() as session:
                tc = session.query(TechChange).filter_by(example_id=example_id).first()
                if tc:
                    logger.info(f"Retrieved tech change: {tc}")
                return tc
        except Exception as exc:
            logger.error(f"Failed to get tech change {example_id}: {exc}")
            return None

    def get_tech_changes_by_type(self, change_type: str) -> List[TechChange]:
        """Retrieve all tech changes of a specific type."""
        try:
            with self.get_session() as session:
                tech_changes = (
                    session.query(TechChange)
                    .filter_by(change_type=change_type)
                    .order_by(TechChange.example_id)
                    .all()
                )
                logger.info(
                    f"Retrieved {len(tech_changes)} tech changes of type '{change_type}'"
                )
                return tech_changes
        except Exception as exc:
            logger.error(f"Failed to get tech changes by type: {exc}")
            return []

    # ------------------------------------------------------------------
    # Delete / Clear
    # ------------------------------------------------------------------

    def delete_tech_change(self, example_id: int) -> bool:
        """Delete a tech change by example_id."""
        try:
            with self.get_session() as session:
                tc = session.query(TechChange).filter_by(example_id=example_id).first()
                if tc:
                    session.delete(tc)
                    logger.info(f"Deleted tech change: {example_id}")
                    return True
                return False
        except Exception as exc:
            logger.error(f"Failed to delete tech change {example_id}: {exc}")
            return False

    def clear_all(self) -> bool:
        """Delete all tech change records."""
        try:
            with self.get_session() as session:
                count = session.query(TechChange).delete()
                logger.info(f"Cleared {count} tech change records")
                return True
        except Exception as exc:
            logger.error(f"Failed to clear tech changes: {exc}")
            return False

    # ------------------------------------------------------------------
    # Bulk load
    # ------------------------------------------------------------------

    def load_from_dict(self, examples_dict: Dict[int, Dict]) -> int:
        """
        Load tech changes from a dictionary (e.g. the EXAMPLES dict from demo.py).

        Returns the number of records successfully loaded.
        """
        count = 0
        for example_id, config in examples_dict.items():
            try:
                change_type = "tax_policy" if example_id <= 6 else "tech_change"

                description = config.get("description", [])
                if isinstance(description, list):
                    description = "\n".join(description)

                params = config.get("params", {})

                if "tech_change_config" in params:
                    tech_config = params["tech_change_config"]
                    params["tech_change_function_name"] = tech_config.get("name", "")
                    params.pop("tech_change_config", None)

                self.add_tech_change(
                    example_id=example_id,
                    change_type=change_type,
                    title=config["title"],
                    description=description,
                    **params,
                )
                count += 1
            except Exception as exc:
                logger.error(f"Failed to load example {example_id}: {exc}")

        logger.info(f"Loaded {count} tech changes into database")
        return count


# Convenience factory function
def get_tech_change_db(
    database_url: str = "sqlite:///data.db",
) -> TechChangeDatabase:
    """Return a TechChangeDatabase instance."""
    return TechChangeDatabase(database_url=database_url)


