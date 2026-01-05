from sqlalchemy import JSON, create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from contextlib import contextmanager
from typing import Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base class for ORM models
Base = declarative_base()

class SupplyCurve(Base):
    """Model representing a tiered supply curve for a good."""
    __tablename__ = 'supply_curves'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    id_number = Column(Integer, unique=True, nullable=False) # ID number of the good this index refers to, FOREIGN KEY to Good.id_number
    isic = Column(String, nullable=False)  # International Standard Industrial Classification code

    # Index fields
    production_inputs = Column(JSON)  # Inputs required for production in JSON format
    production_added_values = Column(JSON)  # Value added for the good
    total_inputs_cost = Column(JSON, nullable=True)  # Total input price used in production
    total_value_added = Column(JSON, nullable=True)  # Total value added by the production method
    price = Column(JSON, nullable=True)  # Price of the good
    price_history = Column(JSON, nullable=True)  # Historical price data in JSON format

    def __repr__(self) -> str:
        return f"<SupplyCurve(id={self.id}, name='{self.name}', id_number={self.id_number}, isic='{self.isic}', price={self.price}, production_added_values={self.production_added_values})>"

    def __str__(self) -> str:
        return f"SupplyCurve(name='{self.name}', id_number={self.id_number}, isic={self.isic}, price={self.price}, production_added_values={self.production_added_values})"

class SupplyCurveDatabase:
    """Database manager for Supply Curves with proper session handling."""
    
    def __init__(self, database_url: str = "sqlite:///data.db", echo: bool = False):
        """Initialize database connection and create tables."""
        self.engine = create_engine(database_url, echo=echo)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        logger.info(f"Database initialized: {database_url}")

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
            session.close()  # Ensure the session is closed after use

    def add_supply_curve(self, name: str, id_number: int, isic: str):
        """Add a new supply curve to the database with error handling."""
        try:
            with self.get_session() as session:
                supply_curve = SupplyCurve(
                    name=name,
                    id_number=id_number,
                    isic=isic,
                )
                session.add(supply_curve)
                session.flush()
                logger.info(f"Added supply curve: {supply_curve}")
        except Exception as e:
            logger.error(f"Failed to add supply curve: {e}")
            raise

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
        except Exception as e:
            logger.error(f"Failed to retrieve supply curves: {e}")
            return []
        
    def get_supply_curve_by_id(self, id_number: int) -> Optional[SupplyCurve]:
        """Retrieve a supply curve by its ID number."""
        try:
            with self.get_session() as session:
                supply_curve = session.query(SupplyCurve).filter_by(id_number=id_number).first()
                if supply_curve:
                    logger.info(f"Retrieved supply curve: {supply_curve}")
                else:
                    logger.warning(f"No supply curve found with id_number: {id_number}")
                return supply_curve
        except Exception as e:
            logger.error(f"Failed to retrieve supply curve by id_number {id_number}: {e}")
            return None
        
    def get_supply_curves_by_id_number(self, id_number: int) -> List[SupplyCurve]:
        """Retrieve all supply curves for a specific good ID number."""
        try:
            with self.get_session() as session:
                supply_curves = session.query(SupplyCurve).filter_by(good_id_number=id_number).all()
                logger.info(f"Retrieved {len(supply_curves)} supply curves for good ID {id_number}")
                return supply_curves
        except Exception as e:
            logger.error(f"Failed to retrieve supply curves for good ID {id_number}: {e}")
            return []
        
    def get_supply_curves_by_isic(self, isic: int) -> List[SupplyCurve]:
        """Retrieve all supply curves for a specific ISIC code."""
        try:
            with self.get_session() as session:
                supply_curves = session.query(SupplyCurve).filter_by(isic=isic).all()
                logger.info(f"Retrieved {len(supply_curves)} supply curves for ISIC {isic}")
                return supply_curves
        except Exception as e:
            logger.error(f"Failed to retrieve supply curves for ISIC {isic}: {e}")
            return []
        
    def update_supply_curve(self, id_number: int, **kwargs) -> Optional[SupplyCurve]:
        """Update a supply curve's details."""
        try:
            with self.get_session() as session:
                supply_curve = session.query(SupplyCurve).filter_by(id_number=id_number).first()
                if supply_curve:
                    for key, value in kwargs.items():
                        setattr(supply_curve, key, value)
                    session.commit()
                    logger.info(f"Updated supply curve: {supply_curve}")
                    return supply_curve
                else:
                    logger.warning(f"No supply curve found with id_number: {id_number}")
                    return None
        except Exception as e:
            logger.error(f"Failed to update supply curve with id_number {id_number}: {e}")
            return None
        
    def update_supply_curve_by_isic(self, isic: str, **kwargs) -> Optional[SupplyCurve]:
        """Update a supply curve's details by ISIC code (only one match expected)."""
        try:
            with self.get_session() as session:
                supply_curve = session.query(SupplyCurve).filter_by(isic=isic).first()
                if supply_curve:
                    for key, value in kwargs.items():
                        setattr(supply_curve, key, value)
                    session.commit()
                    logger.info(f"Updated supply curve for ISIC {isic}: {supply_curve}")
                    return supply_curve
                else:
                    logger.warning(f"No supply curve found with ISIC: {isic}")
                    return None
        except Exception as e:
            logger.error(f"Failed to update supply curve with ISIC {isic}: {e}")
            return None
        
    def delete_supply_curve(self, id_number: int) -> bool:
        """Delete a supply curve by its ID number."""
        try:
            with self.get_session() as session:
                supply_curve = session.query(SupplyCurve).filter_by(id_number=id_number).first()
                if supply_curve:
                    session.delete(supply_curve)
                    session.commit()
                    logger.info(f"Deleted supply curve with id_number: {id_number}")
                    return True
                else:
                    logger.warning(f"No supply curve found with id_number: {id_number}")
                    return False
        except Exception as e:
            logger.error(f"Failed to delete supply curve with id_number {id_number}: {e}")
            return False