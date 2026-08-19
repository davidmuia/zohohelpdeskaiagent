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

# Some Postgres providers (Neon included, historically also Heroku) hand
# out connection strings starting with "postgres://" rather than
# "postgresql://" — the former is rejected outright by some
# SQLAlchemy/psycopg combinations. Defensive fixup so pasting a provider's
# connection string directly into DATABASE_URL just works, rather than
# failing with a cryptic dialect error the first time someone deploys
# against real Postgres.
_database_url = config.database_url
if _database_url.startswith("postgres://"):
    _database_url = _database_url.replace("postgres://", "postgresql://", 1)

# `check_same_thread` is only relevant for SQLite; harmless to set otherwise
# but we guard it so this engine works cleanly if DATABASE_URL is later
# pointed at Postgres/MySQL without code changes.
_connect_args = {"check_same_thread": False} if _database_url.startswith("sqlite") else {}

# pool_pre_ping / pool_recycle matter specifically because of Neon's free
# tier: Neon's compute auto-suspends after its own idle window, completely
# independently of whether this Render service is awake. When that
# happens, any connection SQLAlchemy is holding in its pool goes dead —
# the next request to reuse it throws an OperationalError ("SSL connection
# has been closed unexpectedly" or similar) on the very first query, which
# surfaces as a bare 500 even on routes with no AI involvement at all
# (e.g. GET /admin/kb/articles). pool_pre_ping issues a cheap SELECT 1
# before handing out a pooled connection and transparently reconnects if
# it's dead; pool_recycle proactively retires connections before Neon's
# own suspend window is likely to have kicked in, so we replace them on
# our terms rather than mid-request. Harmless no-ops for SQLite.
engine = create_engine(
    _database_url,
    connect_args=_connect_args,
    future=True,
    pool_pre_ping=True,
    pool_recycle=300,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create all tables if they do not already exist."""
    logger.info("Initializing database schema at %s", _database_url)
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
