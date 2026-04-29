#!/usr/bin/env python3
"""
Initialize the manga translator database.

Usage:
    python tools/init_db.py              # default path: data/manga_translator.db
    python tools/init_db.py --db /tmp/test.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.database import init_db, DEFAULT_DB_PATH


def main():
    parser = argparse.ArgumentParser(description="Initialize the manga translator database")
    parser.add_argument("--db", type=str, default=None, help="Database file path")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    print(f"Initializing database at: {db_path}")

    engine = init_db(db_path)

    # Verify tables
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Created {len(tables)} tables: {', '.join(tables)}")

    for table in tables:
        cols = [c["name"] for c in inspector.get_columns(table)]
        print(f"  {table}: {', '.join(cols)}")

    print("\nDatabase ready.")


if __name__ == "__main__":
    main()
