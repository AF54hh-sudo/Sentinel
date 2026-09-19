"""Audit data files, internal documents, installed metadata, and PostgreSQL branding."""

from __future__ import annotations

import re
from importlib.metadata import metadata
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from sentinel.config import get_settings
from sentinel.database.connection import create_database_engine

OLD_BRAND = re.compile(r"nova[\s_-]*tech|novatech|nova client|\bnova\b", re.IGNORECASE)
TEXT_TYPE_NAMES = {"CHAR", "TEXT", "VARCHAR"}


def generated_file_matches() -> list[str]:
    matches: list[str] = []
    audited_files = [
        *Path("data/generated").glob("*"),
        *Path("data/documents").glob("*"),
    ]
    for path in audited_files:
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md"}:
            continue
        if OLD_BRAND.search(path.read_text(encoding="utf-8")):
            matches.append(str(path))
    return matches


def database_matches() -> dict[str, int] | None:
    engine = create_database_engine(get_settings().database_url)
    findings: dict[str, int] = {}
    try:
        inspector = inspect(engine)
        with engine.connect() as connection:
            for table_name in inspector.get_table_names():
                for column in inspector.get_columns(table_name):
                    if column["type"].__class__.__name__.upper() not in TEXT_TYPE_NAMES:
                        continue
                    column_name = column["name"]
                    statement = text(
                        f'SELECT COUNT(*) FROM "{table_name}" '
                        f'WHERE CAST("{column_name}" AS TEXT) ~* :pattern'
                    )
                    count = int(
                        connection.scalar(statement, {"pattern": OLD_BRAND.pattern}) or 0
                    )
                    if count:
                        findings[f"{table_name}.{column_name}"] = count
    except SQLAlchemyError:
        return None
    finally:
        engine.dispose()
    return findings


def main() -> int:
    file_findings = generated_file_matches()
    database_findings = database_matches()
    installed_summary = metadata("sentinel-decision-intelligence")["Summary"] or ""
    metadata_has_old_brand = bool(OLD_BRAND.search(installed_summary))
    print(f"generated_file_matches={file_findings}")
    print(
        f"database_matches={database_findings}"
        if database_findings is not None
        else "database_matches=unavailable (PostgreSQL audit skipped)"
    )
    print(f"installed_summary={installed_summary}")
    if file_findings or (database_findings or {}) or metadata_has_old_brand:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
