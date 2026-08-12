"""
database.py
-----------
Database engine and session management.

Uses SQLAlchemy's modern (2.0-style) engine/session pattern. Kept deliberately
small: a single engine, a session factory, and a helper to initialize tables.
Callers obtain a session via `get_session()` as a context manager so
transactions are always closed/rolled back correctly.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import config
from models import Base

logger = logging.getLogger(__name__)

# `check_same_thread` is only relevant for SQLite; harmless to set otherwise
# but we guard it so this engine works cleanly if DATABASE_URL is later
# pointed at Postgres/MySQL without code changes.
_connect_args = {"check_same_thread": False} if config.database_url.startswith("sqlite") else {}

engine = create_engine(config.database_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create all tables if they do not already exist."""
    logger.info("Initializing database schema at %s", config.database_url)
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of database operations.

    Usage:
        with get_session() as session:
            session.add(obj)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
