import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from sentinel.data.generator import AMXTechDataGenerator, GenerationConfig
from sentinel.database.loader import DatabaseNotEmptyError, seed_tables
from sentinel.database.queries import (
    AnalyticsFilter,
    foreign_key_violation_counts,
    table_row_counts,
    validate_read_only_sql,
)
from sentinel.database.schema import metadata


def test_schema_contains_exactly_five_business_tables():
    assert set(metadata.tables) == {
        "customers",
        "subscriptions",
        "sales",
        "cloud_costs",
        "support_tickets",
    }
    assert metadata.tables["subscriptions"].c.customer_id.foreign_keys
    assert metadata.tables["sales"].c.subscription_id.foreign_keys
    assert metadata.tables["support_tickets"].c.customer_id.foreign_keys


def test_schema_compiles_for_postgresql():
    dialect = postgresql.dialect()
    for table in metadata.sorted_tables:
        assert str(CreateTable(table).compile(dialect=dialect)).startswith("\nCREATE TABLE")
        for index in table.indexes:
            assert "CREATE INDEX" in str(CreateIndex(index).compile(dialect=dialect))


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM customers",
        "DROP TABLE customers",
        "UPDATE customers SET customer_status = 'inactive'",
        "WITH removed AS (DELETE FROM customers RETURNING *) SELECT * FROM removed",
        "SELECT * FROM customers; DROP TABLE customers",
        "SELECT * INTO copied_customers FROM customers",
    ],
)
def test_destructive_sql_is_blocked(statement):
    with pytest.raises(ValueError):
        validate_read_only_sql(statement)


def test_single_select_is_allowed():
    assert validate_read_only_sql(" SELECT customer_id FROM customers; ") == (
        "SELECT customer_id FROM customers"
    )


def test_analytics_filter_validates_without_database():
    filters = AnalyticsFilter("2024-01-01", "2024-03-31", region="Europe")
    assert str(filters.start_date) == "2024-01-01"
    with pytest.raises(ValueError, match="start_date"):
        AnalyticsFilter("2024-04-01", "2024-03-31")
    with pytest.raises(ValueError, match="Unknown region"):
        AnalyticsFilter("2024-01-01", "2024-03-31", region="Unknown")
    with pytest.raises(ValueError, match="Unknown plan_type"):
        AnalyticsFilter("2024-01-01", "2024-03-31", plan_type="Unlimited")


def test_transactional_loader_and_smoke_queries():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    frames = AMXTechDataGenerator(
        GenerationConfig(n_customers=80, n_subscriptions=120, n_tickets=160)
    ).generate()
    loaded = seed_tables(engine, frames, batch_size=50)
    assert table_row_counts(engine) == loaded
    assert not any(foreign_key_violation_counts(engine).values())

    with pytest.raises(DatabaseNotEmptyError):
        seed_tables(engine, frames)

    assert seed_tables(engine, frames, replace=True) == loaded
    engine.dispose()
