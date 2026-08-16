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
    _run_column_migrations()


def _run_column_migrations() -> None:
    """
    create_all() only creates missing TABLES, never adds columns to a
    table that already exists — so a column added to a model after the
    table was first created (like KBArticle.retrieval_embedding_json)
    silently never appears on an already-deployed database, and the next
    read/write against it fails with "no such column". This adds any
    columns listed below if missing, so existing deployments pick up
    schema changes automatically on restart instead of crashing.

    Deliberately minimal (SQLite ADD COLUMN only, no data backfill here —
    application code handles backfilling actual values, e.g.
    kb_service.find_relevant_articles self-heals retrieval_embedding_json
    lazily). Add an entry here whenever a new nullable/defaulted column is
    added to an existing table.
    """
    from sqlalchemy import inspect, text

    migrations = [
        ("kb_article", "retrieval_embedding_json", "TEXT NOT NULL DEFAULT '[]'"),
    ]

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table, column, ddl_type in migrations:
            if table not in existing_tables:
                continue  # table itself is new — create_all() already handled it
            existing_columns = {c["name"] for c in inspector.get_columns(table)}
            if column in existing_columns:
                continue
            logger.info("Migrating schema: adding %s.%s", table, column)
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))


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