"""Database engine construction and connectivity checks."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url


def create_database_engine(
    database_url: str,
    *,
    echo: bool = False,
    connect_timeout: int = 10,
) -> Engine:
    """Create a pooled PostgreSQL engine without opening a connection immediately."""
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("Sentinel requires a PostgreSQL database URL")
    connect_args: dict[str, Any] = {"connect_timeout": connect_timeout}
    return create_engine(
        url,
        echo=echo,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


def redacted_database_url(database_url: str | URL) -> str:
    """Render a connection URL without exposing its password."""
    return make_url(str(database_url)).render_as_string(hide_password=True)


def verify_connection(engine: Engine) -> dict[str, str]:
    """Run a minimal connection check and return server metadata."""
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT current_database() AS database_name, "
                "current_user AS user_name, version() AS server_version"
            )
        ).mappings().one()
    return {key: str(value) for key, value in row.items()}
