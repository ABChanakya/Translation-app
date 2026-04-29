"""Shared FastAPI dependencies (DB session, GPU semaphore)."""

from __future__ import annotations

import asyncio
from typing import Generator

from sqlalchemy.orm import Session

from src.db.database import get_engine, get_session, init_db

# Ensure tables exist on import
_engine = init_db()

# One GPU inference at a time
gpu_sem = asyncio.Semaphore(1)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and closes it after."""
    session = get_session(_engine)
    try:
        yield session
    finally:
        session.close()
