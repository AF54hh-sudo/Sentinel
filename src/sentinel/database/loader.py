"""Transactional loading of validated data frames into the business schema."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Connection, Engine, func, insert, select

from sentinel.database.schema import (
    TABLES_IN_DELETE_ORDER,
    TABLES_IN_LOAD_ORDER,
    create_schema,
)

DATE_COLUMNS = {
    "customers": ("signup_date",),
    "subscriptions": ("start_date", "renewal_date", "end_date"),
    "sales": ("sale_date",),
    "cloud_costs": ("date",),
    "support_tickets": ("created_date", "resolved_date"),
}


class DatabaseNotEmptyError(RuntimeError):
    """Raised when a seed would overwrite existing records without permission."""


def _python_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Decimal | date):
        return value
    return value


def dataframe_records(frame: pd.DataFrame, table_name: str) -> list[dict[str, Any]]:
    """Convert a frame to driver-safe records while preserving nullable values."""
    prepared = frame.copy()
    for column in DATE_COLUMNS.get(table_name, ()):
        prepared[column] = pd.to_datetime(prepared[column], errors="raise")
    return [
        {column: _python_value(value) for column, value in row.items()}
        for row in prepared.to_dict(orient="records")
    ]


def _existing_counts(connection: Connection) -> dict[str, int]:
    return {
        table.name: int(connection.scalar(select(func.count()).select_from(table)) or 0)
        for table in TABLES_IN_LOAD_ORDER
    }


def seed_tables(
    engine: Engine,
    frames: dict[str, pd.DataFrame],
    *,
    replace: bool = False,
    batch_size: int = 2_000,
) -> dict[str, int]:
    """Load all five tables atomically, optionally replacing existing records."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    expected = {table.name for table in TABLES_IN_LOAD_ORDER}
    missing = expected - frames.keys()
    if missing:
        raise ValueError(f"Missing frames for tables: {sorted(missing)}")

    create_schema(engine)
    with engine.begin() as connection:
        existing = _existing_counts(connection)
        if any(existing.values()) and not replace:
            occupied = {name: count for name, count in existing.items() if count}
            raise DatabaseNotEmptyError(
                f"Database already contains Sentinel records: {occupied}. "
                "Use replace=True only when replacement is intended."
            )
        if replace:
            for table in TABLES_IN_DELETE_ORDER:
                connection.execute(table.delete())

        loaded: dict[str, int] = {}
        for table in TABLES_IN_LOAD_ORDER:
            records = dataframe_records(frames[table.name], table.name)
            for offset in range(0, len(records), batch_size):
                connection.execute(insert(table), records[offset : offset + batch_size])
            loaded[table.name] = len(records)
    return loaded


def read_database_tables(engine: Engine) -> dict[str, pd.DataFrame]:
    """Extract all five business tables in dependency order for analytical work."""
    with engine.connect() as connection:
        return {
            table.name: pd.read_sql(select(table), connection)
            for table in TABLES_IN_LOAD_ORDER
        }
