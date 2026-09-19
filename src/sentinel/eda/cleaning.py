"""Explicit and auditable cleaning policy for the Phase 2 synthetic data."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from sentinel.data.validation import ValidationReport, validate_dataset

DATE_COLUMNS = {
    "customers": ("signup_date",),
    "subscriptions": ("start_date", "renewal_date", "end_date"),
    "sales": ("sale_date",),
    "cloud_costs": ("date",),
    "support_tickets": ("created_date", "resolved_date"),
}


@dataclass
class CleaningReport:
    rows_before: dict[str, int] = field(default_factory=dict)
    rows_after: dict[str, int] = field(default_factory=dict)
    duplicate_events_removed: int = 0
    industries_filled: int = 0
    satisfaction_values_retained_missing: int = 0
    validation: ValidationReport | None = None


def convert_date_columns(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    converted = {name: frame.copy() for name, frame in tables.items()}
    for table_name, columns in DATE_COLUMNS.items():
        for column in columns:
            converted[table_name][column] = pd.to_datetime(
                converted[table_name][column], errors="raise"
            )
    return converted


def clean_dataset(
    tables: dict[str, pd.DataFrame],
    *,
    unknown_industry: str = "Unknown",
) -> tuple[dict[str, pd.DataFrame], CleaningReport]:
    """Apply the documented Phase 4 policy and validate the result.

    Missing ticket satisfaction is retained because a missing survey is not equivalent to
    neutral satisfaction. Open-ticket resolution dates and active-subscription end dates are
    structural nulls. Duplicate ticket events are identified without their surrogate key.
    """
    cleaned = convert_date_columns(tables)
    report = CleaningReport(rows_before={name: len(frame) for name, frame in cleaned.items()})

    industries_missing = int(cleaned["customers"].industry.isna().sum())
    cleaned["customers"]["industry"] = cleaned["customers"].industry.fillna(unknown_industry)
    report.industries_filled = industries_missing

    tickets = cleaned["support_tickets"]
    business_columns = [column for column in tickets.columns if column != "ticket_id"]
    duplicate_mask = tickets.duplicated(business_columns, keep="first")
    report.duplicate_events_removed = int(duplicate_mask.sum())
    cleaned["support_tickets"] = tickets.loc[~duplicate_mask].reset_index(drop=True)
    report.satisfaction_values_retained_missing = int(
        cleaned["support_tickets"].satisfaction_score.isna().sum()
    )

    report.rows_after = {name: len(frame) for name, frame in cleaned.items()}
    report.validation = validate_dataset(cleaned)
    report.validation.raise_for_errors()
    return cleaned, report
