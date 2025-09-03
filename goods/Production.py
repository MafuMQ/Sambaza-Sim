from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.types import JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from contextlib import contextmanager
from typing import Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base class for ORM models
Base = declarative_base()

class Production(Base):
    """Model representing a Production Method."""
    __tablename__ = 'productions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    descriptive_name = Column(String)  # Using snake_case for consistency
    id_number = Column(Integer, unique=False, nullable=False) # ID number of the good this index refers to, FOREIGN KEY to Good.id_number
    isic = Column(String, nullable=False)
    producer = Column(Integer, nullable=False)  # Producer name or ID
    produce = Column(Integer, nullable=False)  # What the production method produces
    produce_name = Column(String, nullable=False)  # Name of the produced item

    production_inputs = Column(JSON)  # Inputs required for production in JSON format
    production_added_values = Column(JSON)  # value added components in JSON format
    production_rate = Column(Integer)  # Rate of production
    production_material_efficiency = Column(Integer)  # Efficiency of the production method
    production_labour_efficiency = Column(Integer)  # Labour efficiency of the production method
    production_energy_efficiency = Column(Integer)  # Energy efficiency of the production

    # Contact fields
    contact_name = Column(String)
    contact_email = Column(String)
    contact_phone = Column(Integer)
    contact_phone2 = Column(Integer)  # Optional second phone number
    contact_website = Column(String)

    # Address fields
    address = Column(String)
    address_street = Column(String)  # Optional street address
    address_city = Column(String)
    address_country = Column(String)
    address_postal_code = Column(String)

    total_inputs_cost = Column(Integer, nullable=True)  # Total input price used in production
    total_value_added = Column(Integer, nullable=True)  # Total value added by the production method
    price = Column(String, nullable=True) # Null price means it is not applicable, i.e. in-house barter production

    def __repr__(self) -> str:
        return f"<Production(name={self.name}, id_number={self.id_number})>"

    def __str__(self) -> str:
        return f"Production(name={self.name}, id_number={self.id_number})"


class ProductionsDatabase:
    """Database manager for Productions with proper session handling."""
    
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
            session.close()

    def add_production(self, name: str, id_number: int, producer: int, produce: int, **kwargs) -> Optional[Production]:
        """Add a new producer to the database with error handling."""
        try:
            with self.get_session() as session:
                production = Production(name=name, id_number=id_number, producer=producer, produce=produce, **kwargs)
                session.add(production)
                session.flush()  # Ensure ID is generated
                logger.info(f"Added production: {production}")
                return production
        except Exception as e:
            logger.error(f"Failed to add production '{name}': {e}")
            return None
        
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
        except Exception as e:
            logger.error(f"Failed to retrieve productions: {e}")
            return []
        
    def get_all_productions_by_producer(self, producer_id: int) -> List[Production]:
        """Retrieve all production methods by a specific producer."""
        try:
            with self.get_session() as session:
                productions = session.query(Production).filter_by(producer=producer_id).all()
                logger.info(f"Retrieved {len(productions)} productions for producer ID {producer_id}")
                return productions
        except Exception as e:
            logger.error(f"Failed to retrieve productions for producer ID {producer_id}: {e}")
            return []
        
    def get_all_productions_by_good(self, good_id: int) -> List[Production]:
        """Retrieve all production methods for a specific good."""
        try:
            with self.get_session() as session:
                productions = session.query(Production).filter_by(produce=good_id).all()
                logger.info(f"Retrieved {len(productions)} productions for good ID {good_id}")
                return productions
        except Exception as e:
            logger.error(f"Failed to retrieve productions for good ID {good_id}: {e}")
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
        except Exception as e:
            logger.error(f"Failed to retrieve production by ID {production_id}: {e}")
            return None
        
    def update_production(self, production_id: int, **kwargs) -> bool:
        """
        Update a production's details by its ID.
        Returns True if update was successful, False otherwise.
        """
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
        except Exception as e:
            logger.error(f"Failed to update production with ID {production_id}: {e}")
            return False
        
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
                else:
                    logger.warning(f"No production found with ID: {production_id}")
                    return False
        except Exception as e:
            logger.error(f"Failed to delete production with ID {production_id}: {e}")
            return False