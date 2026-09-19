"""Create the Phase 3 PostgreSQL schema and transactionally load generated CSVs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sqlalchemy.exc import OperationalError, SQLAlchemyError

from sentinel.config import get_settings
from sentinel.data.loader import read_csv_dataset
from sentinel.data.validation import validate_dataset
from sentinel.database.connection import (
    create_database_engine,
    redacted_database_url,
    verify_connection,
)
from sentinel.database.loader import DatabaseNotEmptyError, seed_tables
from sentinel.database.queries import foreign_key_violation_counts, table_row_counts
from sentinel.logging import configure_logging

LOGGER = logging.getLogger("sentinel.seed_database")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and seed Sentinel's five PostgreSQL business tables."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly delete existing Sentinel rows before reloading them.",
    )
    parser.add_argument("--batch-size", type=int, default=2_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    data_dir = args.data_dir or settings.data_dir
    frames = read_csv_dataset(data_dir)
    report = validate_dataset(frames)
    report.raise_for_errors()

    engine = create_database_engine(settings.database_url)
    LOGGER.info("Connecting to %s", redacted_database_url(settings.database_url))
    try:
        server = verify_connection(engine)
        loaded = seed_tables(
            engine,
            frames,
            replace=args.replace,
            batch_size=args.batch_size,
        )
        counts = table_row_counts(engine)
        orphan_counts = foreign_key_violation_counts(engine)
    except DatabaseNotEmptyError as error:
        LOGGER.error("%s", error)
        return 3
    except OperationalError:
        LOGGER.error(
            "PostgreSQL is unavailable. Confirm that the server is running, sentinel_db exists, "
            "and SENTINEL_DATABASE_URL is correct."
        )
        return 2
    except SQLAlchemyError:
        LOGGER.exception("PostgreSQL schema/load operation failed and was rolled back")
        return 1
    finally:
        engine.dispose()

    if loaded != counts or any(orphan_counts.values()):
        LOGGER.error("Post-load verification failed")
        return 1
    print(
        json.dumps(
            {
                "server": server,
                "loaded_rows": loaded,
                "database_rows": counts,
                "foreign_key_violations": orphan_counts,
                "validation_warnings": report.warnings,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
