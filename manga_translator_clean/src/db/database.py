"""
Database connection setup for the manga translation tool.

Uses SQLite — simple, local, no external dependencies.
All state lives in a single .db file next to the project root.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "manga_translator.db"


def get_engine(db_path: str | Path | None = None):
    """Create a SQLAlchemy engine for the given (or default) SQLite path."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False)


def init_db(db_path: str | Path | None = None):
    """Create all tables if they don't exist yet, and apply any missing column migrations."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    _migrate(engine)
    return engine


def _migrate(engine):
    """
    Idempotent column migrations for SQLite.
    SQLAlchemy's create_all() only creates missing *tables*, not missing *columns*
    on existing tables. This function adds any new columns safely.
    """
    _ADD_COLUMNS = [
        # (table, column, sql_type)
        ("bubbles", "quality_score", "REAL"),
        ("bubbles", "notes",         "TEXT"),
    ]

    with engine.connect() as conn:
        for table, col, dtype in _ADD_COLUMNS:
            rows = conn.execute(
                __import__("sqlalchemy").text(f"PRAGMA table_info({table})")
            ).fetchall()
            existing = {r[1] for r in rows}
            if col not in existing:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE {table} ADD COLUMN {col} {dtype}"
                    )
                )
                conn.commit()


def get_session(engine=None) -> Session:
    """Return a new session bound to *engine* (or the default DB)."""
    if engine is None:
        engine = get_engine()
    factory = sessionmaker(bind=engine)
    return factory()
