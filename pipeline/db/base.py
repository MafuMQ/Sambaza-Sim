"""
Shared database base class
===========================

Provides a DatabaseBase mixin that eliminates the copy-pasted
__init__ / get_session boilerplate that was duplicated across all
four entity database classes in the original codebase.

Usage
-----
    from pipeline.db.base import DatabaseBase

    class GoodsDatabase(DatabaseBase):
        def __init__(self, database_url="sqlite:///data.db", echo=False):
            super().__init__(database_url, echo, Base)
"""

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class DatabaseBase:
    """
    Shared base for all repository classes.

    Handles engine creation, table initialisation, and session management
    so that each repository only needs to implement its own CRUD methods.
    """

    def __init__(self, database_url: str, echo: bool, declarative_base):
        """
        Initialise the database connection.

        Parameters
        ----------
        database_url : str
            SQLAlchemy-compatible URL, e.g. ``"sqlite:///data.db"``.
        echo : bool
            If True, SQL statements are echoed to stdout.
        declarative_base :
            The SQLAlchemy ``Base`` specific to the entity (created by
            ``declarative_base()`` in each repository module).
        """
        self.declarative_base = declarative_base
        self.engine = create_engine(database_url, echo=echo)
        self.declarative_base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        logger.debug(f"Database initialised: {database_url}")

    def clear_all_tables(self):
        """Drops and recreates all tables associated with this declarative base."""
        self.declarative_base.metadata.drop_all(self.engine)
        self.declarative_base.metadata.create_all(self.engine)

    @contextmanager
    def get_session(self):
        """
        Context manager that yields a session and handles commit / rollback / close.

        Usage::

            with self.get_session() as session:
                results = session.query(MyModel).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Database error: {exc}")
            raise
        finally:
            session.close()


