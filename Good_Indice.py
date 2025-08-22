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

class GoodIndice(Base):
    """Model representing an entry in the Good Indices."""
    __tablename__ = 'goodsIndices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    id_number = Column(Integer, unique=True, nullable=False) # ID number of the good this index refers to, FOREIGN KEY to Good.id_number
    isic = Column(String, nullable=False)  # International Standard Industrial Classification code

    # Index fields
    production_inputs = Column(JSON)  # Inputs required for production in JSON format
    price = Column(Integer, nullable=True)  # Price of the good
    price_history = Column(String)  # Historical price data in JSON format
    quantity = Column(Integer, nullable=False)  # Quantity of the good

    def __repr__(self) -> str:
        return f"<GoodIndice(id={self.id}, name='{self.name}', id_number={self.id_number}, isic='{self.isic}', price={self.price})>"

    def __str__(self) -> str:
        return f"GoodIndice(name='{self.name}', id_number={self.id_number}, isic={self.isic}, price={self.price}, quantity={self.quantity})"
    
class GoodsIndiceDatabase:
    """Database manager for Good Indices with proper session handling."""
    
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

    def add_good_indice(self, name: str, id_number: int, isic: str, quantity: int, price_history: Optional[str] = None):
        """Add a new good indice to the database with error handling."""
        try:
            with self.get_session() as session:
                good_indice = GoodIndice(name=name, id_number=id_number, isic=isic, quantity=quantity, price_history=price_history)
                session.add(good_indice)
                session.flush()  # Ensure ID is generated
                logger.info(f"Added good indice: {good_indice}")
        except Exception as e:
            logger.error(f"Failed to add good indice: {e}")
            raise

    def get_all_good_indices(self, limit: Optional[int] = None) -> List[GoodIndice]:
        """Retrieve all good indices with optional limit."""
        try:
            with self.get_session() as session:
                query = session.query(GoodIndice)
                if limit is not None:
                    query = query.limit(limit)
                good_indices = query.all()
                logger.info(f"Retrieved {len(good_indices)} good indices")
                return good_indices
        except Exception as e:
            logger.error(f"Failed to retrieve good indices: {e}")
            return []
        
    def get_good_indice_by_id(self, id_number: int) -> Optional[GoodIndice]:
        """Retrieve a good indice by its ID number."""
        try:
            with self.get_session() as session:
                good_indice = session.query(GoodIndice).filter_by(id_number=id_number).first()
                if good_indice:
                    logger.info(f"Retrieved good indice: {good_indice}")
                else:
                    logger.warning(f"No good indice found with id_number: {id_number}")
                return good_indice
        except Exception as e:
            logger.error(f"Failed to retrieve good indice by id_number {id_number}: {e}")
            return None
        
    def get_good_indices_by_id_number(self, id_number: int) -> List[GoodIndice]:
        """Retrieve all good indices for a specific good ID number."""
        try:
            with self.get_session() as session:
                good_indices = session.query(GoodIndice).filter_by(good_id_number=id_number).all()
                logger.info(f"Retrieved {len(good_indices)} good indices for good ID {id_number}")
                return good_indices
        except Exception as e:
            logger.error(f"Failed to retrieve good indices for good ID {id_number}: {e}")
            return []
        
    def get_good_indices_by_isic(self, isic: int) -> List[GoodIndice]:
        """Retrieve all good indices for a specific ISIC code."""
        try:
            with self.get_session() as session:
                good_indices = session.query(GoodIndice).filter_by(isic=isic).all()
                logger.info(f"Retrieved {len(good_indices)} good indices for ISIC {isic}")
                return good_indices
        except Exception as e:
            logger.error(f"Failed to retrieve good indices for ISIC {isic}: {e}")
            return []
        
    def update_good_indice(self, id_number: int, **kwargs) -> Optional[GoodIndice]:
        """Update a good indice's details."""
        try:
            with self.get_session() as session:
                good_indice = session.query(GoodIndice).filter_by(id_number=id_number).first()
                if good_indice:
                    for key, value in kwargs.items():
                        setattr(good_indice, key, value)
                    session.commit()
                    logger.info(f"Updated good indice: {good_indice}")
                    return good_indice
                else:
                    logger.warning(f"No good indice found with id_number: {id_number}")
                    return None
        except Exception as e:
            logger.error(f"Failed to update good indice with id_number {id_number}: {e}")
            return None
        
    def delete_good_indice(self, id_number: int) -> bool:
        """Delete a good indice by its ID number."""
        try:
            with self.get_session() as session:
                good_indice = session.query(GoodIndice).filter_by(id_number=id_number).first()
                if good_indice:
                    session.delete(good_indice)
                    session.commit()
                    logger.info(f"Deleted good indice with id_number: {id_number}")
                    return True
                else:
                    logger.warning(f"No good indice found with id_number: {id_number}")
                    return False
        except Exception as e:
            logger.error(f"Failed to delete good indice with id_number {id_number}: {e}")
            return False