from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from contextlib import contextmanager
from typing import Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base class for ORM models
Base = declarative_base()

class Producer(Base):
    """Model representing a Producer."""
    __tablename__ = 'producers'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    descriptive_name = Column(String)  # Using snake_case for consistency
    id_number = Column(Integer, unique=True)
    # Contact fields
    contact_name = Column(String)
    contact_email = Column(String)
    contact_phone = Column(String)
    contact_phone2 = Column(String)  # Optional second phone number
    contact_website = Column(String)
    # Address fields
    address = Column(String)
    address_street = Column(String)  # Optional street address
    address_city = Column(String)
    address_country = Column(String)
    address_postal_code = Column(String)

    def __repr__(self) -> str:
        return f"<Producer(name={self.name}, id_number={self.id_number})>"

    def __str__(self) -> str:
        return f"Producer(name={self.name}, id_number={self.id_number})"

class ProducersDatabase:
    """Database manager for Producers with proper session handling."""
    
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

    def add_producer(self, name: str, descriptive_name: Optional[str] = None,
                     id_number: Optional[int] = None, **kwargs) -> Optional[Producer]:
        """Add a new producer to the database with error handling."""
        try:
            with self.get_session() as session:
                producer = Producer(name=name, descriptive_name=descriptive_name,
                                    id_number=id_number, **kwargs)
                session.add(producer)
                session.flush()  # Ensure ID is generated
                logger.info(f"Added producer: {producer}")
                return producer
        except Exception as e:
            logger.error(f"Failed to add producer '{name}': {e}")
            return None
        
    def get_all_producers(self, limit: Optional[int] = None) -> List[Producer]:
        """Retrieve all producers with optional limit."""
        try:
            with self.get_session() as session:
                query = session.query(Producer)
                if limit is not None:
                    query = query.limit(limit)
                producers = query.all()
                logger.info(f"Retrieved {len(producers)} producers")
                return producers
        except Exception as e:
            logger.error(f"Failed to retrieve producers: {e}")
            return []
        
    def get_producer_by_id(self, producer_id: int) -> Optional[Producer]:
        """Retrieve a producer by ID."""
        try:
            with self.get_session() as session:
                producer = session.query(Producer).filter_by(id=producer_id).first()
                if producer:
                    logger.info(f"Found producer: {producer}")
                else:
                    logger.warning(f"No producer found with ID: {producer_id}")
                return producer
        except Exception as e:
            logger.error(f"Failed to retrieve producer by ID {producer_id}: {e}")
            return None
        
    def delete_producer(self, producer_id: int) -> bool:
        """Delete a producer by ID."""
        try:
            with self.get_session() as session:
                producer = session.query(Producer).filter_by(id=producer_id).first()
                if producer:
                    session.delete(producer)
                    logger.info(f"Deleted producer: {producer}")
                    return True
                else:
                    logger.warning(f"No producer found with ID: {producer_id}")
                    return False
        except Exception as e:
            logger.error(f"Failed to delete producer with ID {producer_id}: {e}")
            return False
        
    def update_producer(self, producer_id: int, **kwargs) -> Optional[Producer]:
        """Update a producer's details."""
        try:
            with self.get_session() as session:
                producer = session.query(Producer).filter_by(id=producer_id).first()
                if not producer:
                    logger.warning(f"No producer found with ID: {producer_id}")
                    return None
                for key, value in kwargs.items():
                    setattr(producer, key, value)
                session.commit()
                logger.info(f"Updated producer: {producer}")
                return producer
        except Exception as e:
            logger.error(f"Failed to update producer with ID {producer_id}: {e}")
            return None
        
    def get_producer_by_name(self, name: str) -> Optional[Producer]:
        """Retrieve a producer by name."""
        try:
            with self.get_session() as session:
                producer = session.query(Producer).filter_by(name=name).first()
                if producer:
                    logger.info(f"Found producer: {producer}")
                else:
                    logger.warning(f"No producer found with name: {name}")
                return producer
        except Exception as e:
            logger.error(f"Failed to retrieve producer by name '{name}': {e}")
            return None
        
    def get_producer_by_id_number(self, id_number: int) -> Optional[Producer]:
        """Retrieve a producer by ID number."""
        try:
            with self.get_session() as session:
                producer = session.query(Producer).filter_by(id_number=id_number).first()
                if producer:
                    logger.info(f"Found producer: {producer}")
                else:
                    logger.warning(f"No producer found with ID number: {id_number}")
                return producer
        except Exception as e:
            logger.error(f"Failed to retrieve producer by ID number {id_number}: {e}")
            return None
        
    def get_producer_from_city(self, city: str) -> Optional[Producer]:
        """Retrieve a producer by city."""
        try:
            with self.get_session() as session:
                producer = session.query(Producer).filter_by(address_city=city).first()
                if producer:
                    logger.info(f"Found producer in city '{city}': {producer}")
                else:
                    logger.warning(f"No producer found in city: {city}")
                return producer
        except Exception as e:
            logger.error(f"Failed to retrieve producer by city '{city}': {e}")
            return None
        
    def get_producer_from_country(self, country: str) -> Optional[Producer]:
        """Retrieve a producer by country."""
        try:
            with self.get_session() as session:
                producer = session.query(Producer).filter_by(address_country=country).first()
                if producer:
                    logger.info(f"Found producer in country '{country}': {producer}")
                else:
                    logger.warning(f"No producer found in country: {country}")
                return producer
        except Exception as e:
            logger.error(f"Failed to retrieve producer by country '{country}': {e}")
            return None
        
    def get_producer_from_postal_code(self, postal_code: str) -> Optional[Producer]:
        """Retrieve a producer by postal code."""
        try:
            with self.get_session() as session:
                producer = session.query(Producer).filter_by(address_postal_code=postal_code).first()
                if producer:
                    logger.info(f"Found producer with postal code '{postal_code}': {producer}")
                else:
                    logger.warning(f"No producer found with postal code: {postal_code}")
                return producer
        except Exception as e:
            logger.error(f"Failed to retrieve producer by postal code '{postal_code}': {e}")
            return None
        
    def search_producers(self, search_term: str) -> List[Producer]:
        """Search for producers by name or descriptive name."""
        try:
            with self.get_session() as session:
                producers = session.query(Producer).filter(
                    (Producer.name.ilike(f"%{search_term}%")) |
                    (Producer.descriptive_name.ilike(f"%{search_term}%"))
                ).all()
                logger.info(f"Found {len(producers)} producers matching '{search_term}'")
                return producers
        except Exception as e:
            logger.error(f"Failed to search producers with term '{search_term}': {e}")
            return []