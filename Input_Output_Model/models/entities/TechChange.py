from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text
from sqlalchemy.types import JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import logging
import json as json_lib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base class for ORM models
Base = declarative_base()

class TechChange(Base):
    """Model representing a technological change configuration for economic simulations."""
    __tablename__ = 'tech_changes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    example_id = Column(Integer, unique=True, nullable=False, index=True)  # Example number (1-15)
    change_type = Column(String, nullable=False, index=True)  # 'tax_policy' or 'tech_change'
    
    # Basic metadata
    title = Column(String, nullable=False)
    description = Column(Text)  # Multi-line descriptions stored as single text
    
    # Final demand specification (stored as JSON array)
    final_demand = Column(JSON)  # [250.0, 200.0, 300.0, ...]
    
    # Tax policy parameters
    total_demand = Column(Float, nullable=True)
    uniform_demand = Column(Float, nullable=True)
    demand_vector = Column(JSON, nullable=True)  # Custom demand vector
    proportions = Column(JSON, nullable=True)  # General proportions
    consumption_proportions = Column(JSON, nullable=True)
    investment_proportions = Column(JSON, nullable=True)
    government_proportions = Column(JSON, nullable=True)
    
    income_tax_rate_before = Column(Float, nullable=True)
    income_tax_rate_after = Column(Float, nullable=True)
    corporate_tax_rate_before = Column(Float, nullable=True)
    corporate_tax_rate_after = Column(Float, nullable=True)
    income_tax_applies_to = Column(String, nullable=True)  # 'wages', 'bonusWages', 'both'
    
    consumption_rate = Column(Float, nullable=True, default=1.0)
    iterations = Column(Integer, nullable=True, default=1)
    
    # Technological change parameters
    tech_change_function_name = Column(String, nullable=True)  # Name of builder function
    tech_change_params = Column(JSON, nullable=True)  # Additional parameters for builder
    use_multi_level = Column(Boolean, default=False)
    solver_type = Column(String, nullable=True, default='leontief')  # 'leontief' or 'supply_curves'
    
    # Special case: sector-specific demand shock
    target_isic = Column(String, nullable=True)
    demand_shock = Column(Float, nullable=True)
    
    # Flag for tech comparison mode
    is_tech_comparison = Column(Boolean, default=False)
    
    def __repr__(self) -> str:
        return f"<TechChange(id={self.id}, example_id={self.example_id}, type='{self.change_type}', title='{self.title}')>"

    def __str__(self) -> str:
        return f"Example {self.example_id}: {self.title}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with demo.py config structure."""
        result = {
            'title': self.title,
            'description': self.description.split('\n') if self.description else [],
            'params': {}
        }
        
        # Define which fields apply to which change types
        common_fields = ['iterations', 'consumption_rate']
        
        tax_policy_fields = [
            'total_demand', 'uniform_demand', 'demand_vector', 'proportions',
            'target_isic', 'demand_shock',
            'consumption_proportions', 'investment_proportions', 'government_proportions',
            'income_tax_rate_before', 'income_tax_rate_after',
            'corporate_tax_rate_before', 'corporate_tax_rate_after',
            'income_tax_applies_to'
        ]
        
        tech_change_fields = [
            'final_demand', 'use_multi_level', 'solver_type',
            'is_tech_comparison', 'tech_change_function_name', 'tech_change_params',
            # Include tax policy fields for combined examples
            'income_tax_rate_before', 'income_tax_rate_after',
            'corporate_tax_rate_before', 'corporate_tax_rate_after',
            'income_tax_applies_to',
            'consumption_proportions', 'investment_proportions', 'government_proportions'
        ]
        
        # Determine which fields to include based on change type
        if self.change_type == 'tax_policy':
            allowed_fields = common_fields + tax_policy_fields
        else:  # tech_change
            allowed_fields = common_fields + tech_change_fields
        
        # Add only non-null values from allowed fields
        for field in allowed_fields:
            value = getattr(self, field, None)
            if value is not None:
                result['params'][field] = value
        
        return result


class TechChangeDatabase:
    """Database manager for technological change configurations with proper session handling."""
    
    def __init__(self, database_url: str = "sqlite:///data.db", echo: bool = False):
        """Initialize database connection and create tables."""
        self.engine = create_engine(database_url, echo=echo)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        logger.info(f"TechChange database initialized: {database_url}")

    @contextmanager
    def get_session(self):
        """Context manager for database sessions with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def add_tech_change(self, example_id: int, change_type: str, title: str, 
                       description: str = "", **kwargs) -> Optional[TechChange]:
        """Add a new tech change configuration to the database."""
        try:
            with self.get_session() as session:
                tech_change = TechChange(
                    example_id=example_id,
                    change_type=change_type,
                    title=title,
                    description=description,
                    **kwargs
                )
                session.add(tech_change)
                session.flush()
                logger.info(f"Added tech change: {tech_change}")
                return tech_change
        except Exception as e:
            logger.error(f"Failed to add tech change: {e}")
            return None

    def get_all_tech_changes(self, limit: Optional[int] = None) -> List[TechChange]:
        """Retrieve all tech change configurations with optional limit."""
        try:
            with self.get_session() as session:
                query = session.query(TechChange).order_by(TechChange.example_id)
                if limit:
                    query = query.limit(limit)
                tech_changes = query.all()
                logger.info(f"Retrieved {len(tech_changes)} tech changes")
                return tech_changes
        except Exception as e:
            logger.error(f"Failed to get all tech changes: {e}")
            return []

    def get_tech_change_by_id(self, example_id: int) -> Optional[TechChange]:
        """Retrieve a specific tech change by example ID."""
        try:
            with self.get_session() as session:
                tech_change = session.query(TechChange).filter_by(example_id=example_id).first()
                if tech_change:
                    logger.info(f"Retrieved tech change: {tech_change}")
                return tech_change
        except Exception as e:
            logger.error(f"Failed to get tech change {example_id}: {e}")
            return None
    
    def get_tech_changes_by_type(self, change_type: str) -> List[TechChange]:
        """Retrieve all tech changes of a specific type ('tax_policy' or 'tech_change')."""
        try:
            with self.get_session() as session:
                tech_changes = session.query(TechChange).filter_by(
                    change_type=change_type
                ).order_by(TechChange.example_id).all()
                logger.info(f"Retrieved {len(tech_changes)} tech changes of type '{change_type}'")
                return tech_changes
        except Exception as e:
            logger.error(f"Failed to get tech changes by type: {e}")
            return []

    def delete_tech_change(self, example_id: int) -> bool:
        """Delete a tech change by example ID."""
        try:
            with self.get_session() as session:
                tech_change = session.query(TechChange).filter_by(example_id=example_id).first()
                if tech_change:
                    session.delete(tech_change)
                    logger.info(f"Deleted tech change: {example_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete tech change {example_id}: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all tech change records from database."""
        try:
            with self.get_session() as session:
                count = session.query(TechChange).delete()
                logger.info(f"Cleared {count} tech change records")
                return True
        except Exception as e:
            logger.error(f"Failed to clear tech changes: {e}")
            return False

    def load_from_dict(self, examples_dict: Dict[int, Dict]) -> int:
        """
        Load tech changes from a dictionary (like EXAMPLES from demo.py).
        
        Parameters:
        -----------
        examples_dict : dict
            Dictionary mapping example_id -> config dict with 'title', 'description', 'params'
        
        Returns:
        --------
        int : Number of tech changes successfully loaded
        """
        count = 0
        for example_id, config in examples_dict.items():
            try:
                # Determine change type
                if example_id <= 6:
                    change_type = 'tax_policy'
                else:
                    change_type = 'tech_change'
                
                # Extract description (join list if needed)
                description = config.get('description', [])
                if isinstance(description, list):
                    description = '\n'.join(description)
                
                # Extract params
                params = config.get('params', {})
                
                # Handle tech_change_config for tech change examples
                if 'tech_change_config' in params:
                    tech_config = params['tech_change_config']
                    params['tech_change_function_name'] = tech_config.get('name', '')
                    # Don't store the builder function itself
                    params.pop('tech_change_config', None)
                
                # Add to database
                self.add_tech_change(
                    example_id=example_id,
                    change_type=change_type,
                    title=config['title'],
                    description=description,
                    **params
                )
                count += 1
            except Exception as e:
                logger.error(f"Failed to load example {example_id}: {e}")
        
        logger.info(f"Loaded {count} tech changes into database")
        return count


# Convenience function for getting database instance
def get_tech_change_db(database_url: str = "sqlite:///data.db") -> TechChangeDatabase:
    """Get a TechChangeDatabase instance."""
    return TechChangeDatabase(database_url=database_url)
