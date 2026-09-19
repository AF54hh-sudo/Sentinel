import pandas as pd
import pytest
import yaml

from sentinel.data.generator import AMXTechDataGenerator, GenerationConfig
from sentinel.data.loader import read_csv_dataset, write_csv_dataset
from sentinel.data.validation import validate_dataset


@pytest.fixture(scope="module")
def tables():
    return AMXTechDataGenerator(GenerationConfig()).generate()


def test_expected_table_sizes(tables):
    assert len(tables["customers"]) == 5_000
    assert len(tables["subscriptions"]) == 8_000
    assert 30_000 <= len(tables["sales"]) <= 65_000
    assert len(tables["cloud_costs"]) == 2_924
    assert len(tables["support_tickets"]) == 10_003


def test_amx_tech_branding_is_generated_and_recorded(tables):
    assert tables["customers"].customer_name.str.startswith("AMX Client ").all()
    assert not tables["customers"].customer_name.str.contains("nova", case=False).any()
    with open("data/ground_truth/scenarios.yaml", encoding="utf-8") as stream:
        scenarios = yaml.safe_load(stream)
    assert scenarios["metadata"]["company"] == "AMX Tech"


def test_contract_and_relational_integrity(tables):
    report = validate_dataset(tables)
    assert report.is_valid, report.errors
    assert any("duplicate business records" in warning for warning in report.warnings)


def test_generation_is_deterministic():
    config = GenerationConfig(n_customers=100, n_subscriptions=150, n_tickets=200)
    first = AMXTechDataGenerator(config).generate()
    second = AMXTechDataGenerator(config).generate()
    for table in first:
        pd.testing.assert_frame_equal(first[table], second[table])


def test_csv_dataset_round_trip_preserves_tables_and_iso_dates(tmp_path):
    source = {
        "sample": pd.DataFrame(
            {
                "record_id": ["ROW-001", "ROW-002"],
                "event_date": pd.to_datetime(["2024-01-02", "2024-02-03"]),
                "value": [10.5, 20.25],
            }
        )
    }

    write_csv_dataset(source, tmp_path)
    loaded = read_csv_dataset(tmp_path)

    assert set(loaded) == {"sample"}
    assert loaded["sample"].event_date.tolist() == ["2024-01-02", "2024-02-03"]
    assert loaded["sample"].value.tolist() == [10.5, 20.25]


def test_revenue_and_cost_identities(tables):
    sales = tables["sales"]
    costs = tables["cloud_costs"]
    assert ((sales.gross_revenue - sales.discount_amount - sales.net_revenue).abs() <= 0.02).all()
    cost_difference = (
        costs.compute_cost + costs.storage_cost + costs.network_cost + costs.gpu_cost - costs.total_cost
    ).abs()
    assert (cost_difference <= 0.03).all()


def test_validation_rejects_denormalized_sales_mismatch(tables):
    from sentinel.data.validation import validate_dataset

    altered = {name: frame.copy() for name, frame in tables.items()}
    altered["sales"].loc[altered["sales"].index[0], "plan_type"] = "Invalid plan"
    report = validate_dataset(altered)
    assert not report.is_valid
    assert "sales.plan_type does not match its subscription" in report.errors


def test_intentional_q2_discount_and_gpu_patterns(tables):
    sales = tables["sales"].assign(date=lambda x: pd.to_datetime(x.sale_date))
    enterprise = sales[sales.plan_type == "Enterprise"].copy()
    enterprise["discount_rate"] = enterprise.discount_amount / enterprise.gross_revenue
    q2 = enterprise[enterprise.date.between("2024-04-01", "2024-06-30")].discount_rate.mean()
    q1 = enterprise[enterprise.date.between("2024-01-01", "2024-03-31")].discount_rate.mean()
    assert q2 > q1 + 0.07

    costs = tables["cloud_costs"].assign(date=lambda x: pd.to_datetime(x.date))
    anomaly = costs[costs.date.between("2024-05-13", "2024-05-19")].gpu_cost.mean()
    baseline = costs[costs.date.between("2024-02-01", "2024-03-31")].gpu_cost.mean()
    assert anomaly > baseline * 2.5


def test_q2_profitability_scenario_is_discoverable(tables):
    sales = tables["sales"].assign(date=lambda x: pd.to_datetime(x.sale_date))
    costs = tables["cloud_costs"].assign(date=lambda x: pd.to_datetime(x.date))

    def quarter_values(start, end):
        revenue = sales[sales.date.between(start, end)].net_revenue.sum()
        infrastructure = costs[costs.date.between(start, end)].total_cost.sum()
        return revenue, infrastructure, revenue - infrastructure

    q1_revenue, q1_cost, q1_contribution = quarter_values("2024-01-01", "2024-03-31")
    q2_revenue, q2_cost, q2_contribution = quarter_values("2024-04-01", "2024-06-30")
    assert q2_revenue > q1_revenue
    assert q2_revenue / q1_revenue < 1.05
    assert q2_cost > q1_cost * 1.15
    assert q2_contribution < q1_contribution


def test_europe_q2_cancellation_scenario_is_discoverable(tables):
    subscriptions = tables["subscriptions"].merge(
        tables["customers"][["customer_id", "region"]], on="customer_id", how="left"
    )
    subscriptions["end_date"] = pd.to_datetime(subscriptions.end_date)
    europe_cancelled = subscriptions[
        subscriptions.region.eq("Europe")
        & subscriptions.subscription_status.eq("cancelled")
    ]
    q1 = europe_cancelled.end_date.between("2024-01-01", "2024-03-31").sum()
    q2 = europe_cancelled.end_date.between("2024-04-01", "2024-06-30").sum()
    assert q2 > q1 * 5


def test_europe_support_signal_is_associative(tables):
    tickets = tables["support_tickets"].merge(
        tables["customers"][["customer_id", "region"]], on="customer_id", how="left"
    )
    tickets["created_date"] = pd.to_datetime(tickets.created_date)
    europe_q2 = tickets[(tickets.region == "Europe") & tickets.created_date.between("2024-04-01", "2024-06-30")]
    europe_q1 = tickets[(tickets.region == "Europe") & tickets.created_date.between("2024-01-01", "2024-03-31")]
    assert europe_q2.resolution_hours.mean() > europe_q1.resolution_hours.mean() * 1.5
    assert europe_q2.satisfaction_score.mean() < europe_q1.satisfaction_score.mean()
