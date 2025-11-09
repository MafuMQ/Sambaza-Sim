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

class Good(Base):
    """Model representing a good with ISIC classification details."""
    __tablename__ = 'goods'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    descriptive_name = Column(String)  # Using snake_case for consistency
    id_number = Column(Integer, unique=True, nullable=False)
    isic = Column(String, nullable=False)  # International Standard Industrial Classification code
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


class GoodsDatabase:
    """Database manager for goods with proper session handling."""
    
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

    def add_good(self, name: str, descriptive_name: Optional[str] = None, id_number: int = 0, isic: str = "", **kwargs) -> Optional[Good]:
        """Add a new good to the database with error handling."""
        try:
            with self.get_session() as session:
                good = Good(name=name, descriptive_name=descriptive_name, id_number=id_number, isic=isic, **kwargs)
                session.add(good)
                session.flush()  # Ensure ID is generated
                logger.info(f"Added good: {good}")
                return good
        except Exception as e:
            logger.error(f"Failed to add good: {e}")
            return None

    def get_all_goods(self, limit: Optional[int] = None) -> List[Good]:
        """Retrieve all goods with optional limit."""
        try:
            with self.get_session() as session:
                query = session.query(Good)
                if limit:
                    query = query.limit(limit)
                goods = query.all()
                # Ensure all attributes are loaded before returning
                for good in goods:
                    _ = (good.id, good.name, good.descriptive_name, good.id_number, 
                         good.isic, good.isic_section, good.isic_division, 
                         good.isic_group, good.isic_class, good.sub_class_a,
                         good.sub_class_b, good.sub_class_c, good.sub_class_nf)
                return goods
        except Exception as e:
            logger.error(f"Failed to get all goods: {e}")
            return []

    def get_goods_by_subclass(self, subclass_value: int) -> List[Good]:
        """Retrieve goods where ANY subclass matches the given value."""
        try:
            with self.get_session() as session:
                goods = session.query(Good).filter(
                    (Good.sub_class_a == subclass_value) |
                    (Good.sub_class_b == subclass_value) |
                    (Good.sub_class_c == subclass_value) |
                    (Good.sub_class_nf == subclass_value)
                ).all()
                # Ensure all attributes are loaded
                for good in goods:
                    _ = (good.id, good.name, good.descriptive_name, good.id_number, 
                         good.isic, good.isic_section, good.isic_division, 
                         good.isic_group, good.isic_class, good.sub_class_a,
                         good.sub_class_b, good.sub_class_c, good.sub_class_nf)
                return goods
        except Exception as e:
            logger.error(f"Failed to get goods by any subclass {subclass_value}: {e}")
            return []
        
    def get_goods_by_specific_subclass(self, subclass_value: int, subclass_column: str) -> List[Good]:
        """
        Retrieve goods by searching a specific subclass column.
        
        Args:
            subclass_value: The value to search for
            subclass_column: Column name to search ("sub_class_a", "sub_class_b", "sub_class_c", "sub_class_nf")
            
        Returns:
            List of Good objects matching the criteria
        """
        valid_columns = ["sub_class_a", "sub_class_b", "sub_class_c", "sub_class_nf"]
        
        if subclass_column not in valid_columns:
            logger.error(f"Invalid subclass column: {subclass_column}. Valid options: {valid_columns}")
            return []
        
        try:
            with self.get_session() as session:
                # Use getattr to dynamically access the column
                column = getattr(Good, subclass_column)
                goods = session.query(Good).filter(column == subclass_value).all()
                
                # Ensure all attributes are loaded
                for good in goods:
                    _ = (good.id, good.name, good.descriptive_name, good.id_number,
                        good.isic, good.isic_section, good.isic_division,
                        good.isic_group, good.isic_class, good.sub_class_a,
                        good.sub_class_b, good.sub_class_c, good.sub_class_nf)
                
                logger.info(f"Found {len(goods)} goods with {subclass_column} = {subclass_value}")
                return goods
                
        except Exception as e:
            logger.error(f"Failed to get goods by {subclass_column} = {subclass_value}: {e}")
            return []

    def update_good(self, good_id: int, **kwargs) -> Optional[Good]:
        """Update a good's attributes (generic version)."""
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
        except Exception as e:
            logger.error(f"Failed to update good {good_id}: {e}")
            return None

    # def update_good(self,good_id: int,
    #     name: Optional[str] = None, descriptive_name: Optional[str] = None, id_number: Optional[int] = None, isic: Optional[str] = None, isic_section: Optional[str] = None, isic_division: Optional[int] = None, isic_group: Optional[int] = None,isic_class: Optional[int] = None, sub_class_a: Optional[int] = None,sub_class_b: Optional[int] = None, sub_class_c: Optional[int] = None, sub_class_nf: Optional[int] = None,
    # ) -> Optional[Good]:
    #     """
    #     Update all attributes of a good explicitly.
    #     Only non-None arguments will be updated.
    #     """
    #     try:
    #         with self.get_session() as session:
    #             good = session.query(Good).filter(Good.id == good_id).first()
    #             if not good:
    #                 logger.warning(f"Good with ID {good_id} not found")
    #                 return None

    #             if name is not None:
    #                 good.name = name
    #             if descriptive_name is not None:
    #                 good.descriptive_name = descriptive_name
    #             if id_number is not None:
    #                 good.id_number = id_number
    #             if isic is not None:
    #                 good.isic = isic
    #             if isic_section is not None:
    #                 good.isic_section = isic_section
    #             if isic_division is not None:
    #                 good.isic_division = isic_division
    #             if isic_group is not None:
    #                 good.isic_group = isic_group
    #             if isic_class is not None:
    #                 good.isic_class = isic_class
    #             if sub_class_a is not None:
    #                 good.sub_class_a = sub_class_a
    #             if sub_class_b is not None:
    #                 good.sub_class_b = sub_class_b
    #             if sub_class_c is not None:
    #                 good.sub_class_c = sub_class_c
    #             if sub_class_nf is not None:
    #                 good.sub_class_nf = sub_class_nf

    #             logger.info(f"Updated good (full): {good}")
    #             return good
    #     except Exception as e:
    #         logger.error(f"Failed to fully update good {good_id}: {e}")
    #         return None

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
        except Exception as e:
            logger.error(f"Failed to delete good {good_id}: {e}")
            return False

    def search_goods(self, search_term: str) -> List[Good]:
        """Search goods by name or descriptive name."""
        try:
            with self.get_session() as session:
                return session.query(Good).filter(
                    (Good.name.contains(search_term)) |
                    (Good.descriptive_name.contains(search_term))
                ).all()
        except Exception as e:
            logger.error(f"Failed to search goods with term '{search_term}': {e}")
            return []
