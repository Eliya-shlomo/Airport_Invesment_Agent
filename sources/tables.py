"""Read the processed tables. The only place tools touch the database file."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from sources import config


def read_tables(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Every table of the database as a data frame, by table name."""
    return {name: con.execute(f"SELECT * FROM {name}").df() for (name,) in con.execute("SHOW TABLES").fetchall()}


def load_tables(db_path: Path = config.DB_PATH) -> dict[str, pd.DataFrame]:
    """Load the processed database (a few MB; well under a second)."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found; run `uv run python -m sources.build` first")
    with duckdb.connect(str(db_path), read_only=True) as con:
        return read_tables(con)
