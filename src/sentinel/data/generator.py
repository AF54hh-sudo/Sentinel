"""Deterministic, relational synthetic data for fictional SaaS company AMX Tech."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REGIONS = ["India", "North America", "Europe", "APAC"]
COUNTRIES = {
    "India": ["India"],
    "North America": ["United States", "Canada"],
    "Europe": ["United Kingdom", "Germany", "France", "Netherlands"],
    "APAC": ["Singapore", "Australia", "Japan"],
}
PLANS = ["Basic", "Professional", "Enterprise"]


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 42
    n_customers: int = 5_000
    n_subscriptions: int = 8_000
    n_tickets: int = 10_000
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    inject_imperfections: bool = True


class AMXTechDataGenerator:
    """Generate five related tables with controlled, discoverable business patterns."""

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.start = pd.Timestamp(self.config.start_date)
        self.end = pd.Timestamp(self.config.end_date)

    def generate(self) -> dict[str, pd.DataFrame]:
        customers = self._customers()
        subscriptions = self._subscriptions(customers)
        tickets = self._support_tickets(customers, subscriptions)
        sales = self._sales(customers, subscriptions)
        costs = self._cloud_costs()
        customers = self._derive_customer_status(customers, subscriptions)
        return {
            "customers": customers,
            "subscriptions": subscriptions,
            "sales": sales,
            "cloud_costs": costs,
            "support_tickets": tickets,
        }

    def _customers(self) -> pd.DataFrame:
        n = self.config.n_customers
        region = self.rng.choice(REGIONS, n, p=[0.30, 0.29, 0.25, 0.16])
        size = self.rng.choice(["Small", "Medium", "Enterprise"], n, p=[0.52, 0.33, 0.15])
        # AMX Tech is established before the observation window, avoiding artificial ramp-up.
        customer_history_start = self.start - pd.DateOffset(years=3)
        signup_days = self.rng.integers(
            0, (self.end - customer_history_start).days - 60, n
        )
        frame = pd.DataFrame({
            "customer_id": [f"CUS{i:06d}" for i in range(1, n + 1)],
            "customer_name": [f"AMX Client {i:04d}" for i in range(1, n + 1)],
            "region": region,
            "country": [self.rng.choice(COUNTRIES[value]) for value in region],
            "industry": self.rng.choice(
                ["Technology", "Finance", "Retail", "Healthcare", "Manufacturing"], n
            ),
            "company_size": size,
            "acquisition_channel": self.rng.choice(
                ["Direct", "Partner", "Organic", "Paid Search", "Referral"],
                n,
                p=[0.25, 0.17, 0.24, 0.21, 0.13],
            ),
            "signup_date": customer_history_start + pd.to_timedelta(signup_days, unit="D"),
            "customer_status": "active",
        })
        if self.config.inject_imperfections:
            missing = self.rng.choice(frame.index, size=max(1, n // 100), replace=False)
            frame.loc[missing, "industry"] = pd.NA
        return frame

    def _subscriptions(self, customers: pd.DataFrame) -> pd.DataFrame:
        n = self.config.n_subscriptions
        customer_idx = np.concatenate([
            np.arange(len(customers)),
            self.rng.choice(len(customers), n - len(customers), replace=True),
        ])
        self.rng.shuffle(customer_idx)
        chosen = customers.iloc[customer_idx].reset_index(drop=True)
        plan_prob = {
            "Small": [0.69, 0.28, 0.03],
            "Medium": [0.23, 0.62, 0.15],
            "Enterprise": [0.04, 0.31, 0.65],
        }
        plans = np.array([self.rng.choice(PLANS, p=plan_prob[s]) for s in chosen.company_size])
        signup = pd.to_datetime(chosen.signup_date)
        earliest_subscription = pd.Series(
            np.maximum(signup.to_numpy(), (self.start - pd.DateOffset(years=2)).to_datetime64())
        )
        latest_subscription = self.end - pd.Timedelta(60, unit="D")
        max_offsets = (latest_subscription - earliest_subscription).dt.days.clip(lower=1).to_numpy()
        start_offsets = np.array([self.rng.integers(0, max(1, days)) for days in max_offsets])
        starts = earliest_subscription + pd.to_timedelta(start_offsets, unit="D")
        seats = np.where(
            plans == "Basic", self.rng.integers(2, 16, n),
            np.where(plans == "Professional", self.rng.integers(10, 80, n), self.rng.integers(60, 420, n)),
        )
        base_price = np.select([plans == "Basic", plans == "Professional"], [49.0, 129.0], default=299.0)
        monthly_price = np.round(base_price * seats * self.rng.normal(1, 0.035, n), 2)

        # Scenario 2/5: Professional customers, especially in Europe, have elevated churn risk.
        q2_2024_exposed = (
            (chosen.region.to_numpy() == "Europe")
            & (plans == "Professional")
            & (starts <= pd.Timestamp("2024-03-31"))
        )
        churn_probability = 0.07 + 0.07 * (plans == "Professional") + 0.05 * (plans == "Basic")
        churn_probability += 0.20 * q2_2024_exposed
        churn_probability += np.where(starts >= pd.Timestamp("2024-01-01"), 0.04, 0)
        cancelled = self.rng.random(n) < churn_probability
        expired = (~cancelled) & (self.rng.random(n) < 0.035)
        status = np.where(cancelled, "cancelled", np.where(expired, "expired", "active"))
        tenure_days = np.array([self.rng.integers(75, 520) for _ in range(n)])
        end_dates = starts + pd.to_timedelta(tenure_days, unit="D")
        targeted_q2_churn = cancelled & q2_2024_exposed
        q2_days = self.rng.integers(0, 91, int(targeted_q2_churn.sum()))
        end_dates.loc[targeted_q2_churn] = pd.Timestamp("2024-04-01") + pd.to_timedelta(
            q2_days, unit="D"
        )
        end_dates = pd.Series(end_dates).where(status != "active", pd.NaT)
        end_dates = end_dates.where(end_dates <= self.end, pd.NaT)
        status = np.where((status != "active") & end_dates.isna(), "active", status)
        renewal = starts + pd.to_timedelta(365, unit="D")
        return pd.DataFrame({
            "subscription_id": [f"SUB{i:06d}" for i in range(1, n + 1)],
            "customer_id": chosen.customer_id.to_numpy(),
            "plan_type": plans,
            "start_date": starts.to_numpy(),
            "renewal_date": renewal.to_numpy(),
            "end_date": end_dates.to_numpy(),
            "monthly_price": monthly_price,
            "seats": seats,
            "subscription_status": status,
        })

    def _sales(self, customers: pd.DataFrame, subscriptions: pd.DataFrame) -> pd.DataFrame:
        customer_regions = customers.set_index("customer_id").region
        rows: list[dict[str, Any]] = []
        sale_id = 1
        for sub in subscriptions.itertuples(index=False):
            last = min(pd.Timestamp(sub.end_date) if pd.notna(sub.end_date) else self.end, self.end)
            first = max(
                pd.Timestamp(sub.start_date).to_period("M").to_timestamp(), self.start
            )
            months = pd.date_range(first, last, freq="MS")
            # The sales table is a laptop-sized transaction sample, distributed across the
            # subscription lifetime rather than biased toward its latest months.
            if len(months) > 6:
                sampled_indices = np.unique(
                    np.linspace(0, len(months) - 1, 6, dtype=int)
                )
                months = months[sampled_indices]
            for month in months:
                seasonality = 1 + 0.045 * np.sin(2 * np.pi * month.month / 12)
                trend = 1 + 0.00042 * (month - self.start).days
                gross = sub.monthly_price * seasonality * trend * self.rng.normal(1, 0.045)
                base_discount = {"Basic": 0.025, "Professional": 0.065, "Enterprise": 0.105}[sub.plan_type]
                # Scenarios 1/4: Enterprise discounting increases in Q2 2024.
                if sub.plan_type == "Enterprise" and pd.Timestamp("2024-04-01") <= month <= pd.Timestamp("2024-06-30"):
                    base_discount += 0.095
                discount_rate = float(np.clip(self.rng.normal(base_discount, 0.018), 0, 0.32))
                discount = gross * discount_rate
                rows.append({
                    "sale_id": f"SAL{sale_id:07d}", "customer_id": sub.customer_id,
                    "subscription_id": sub.subscription_id, "sale_date": month,
                    "region": customer_regions[sub.customer_id], "plan_type": sub.plan_type,
                    "gross_revenue": round(gross, 2), "discount_amount": round(discount, 2),
                    "net_revenue": round(gross - discount, 2),
                    "sales_channel": self.rng.choice(["Online", "Direct Sales", "Partner"]),
                })
                sale_id += 1
        frame = pd.DataFrame(rows)
        # Normalize monthly totals to an established SaaS trajectory while preserving each
        # subscription's relative plan/seat value. Q2 2024 gross sales grow modestly; increased
        # Enterprise discounts keep net growth small enough for higher infrastructure cost to
        # reduce contribution.
        month_index = (
            (frame.sale_date.dt.year - self.start.year) * 12
            + frame.sale_date.dt.month
            - self.start.month
        )
        seasonal = 1 + 0.025 * np.sin(2 * np.pi * month_index / 12)
        target = 25_000_000 * (1 + 0.007 * month_index) * seasonal
        target *= np.where(
            frame.sale_date.between("2024-04-01", "2024-06-30"), 1.09, 1.0
        )
        actual = frame.groupby("sale_date").gross_revenue.transform("sum")
        scale = target / actual
        frame["gross_revenue"] = (frame.gross_revenue * scale).round(2)
        discount_rate = frame.discount_amount / (
            frame.gross_revenue / scale
        )
        frame["discount_amount"] = (frame.gross_revenue * discount_rate).round(2)
        frame["net_revenue"] = (frame.gross_revenue - frame.discount_amount).round(2)
        return frame

    def _cloud_costs(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        cost_id = 1
        for date in pd.date_range(self.start, self.end, freq="D"):
            trend = 1 + 0.00055 * (date - self.start).days
            for region in REGIONS:
                scale = {"India": 0.82, "North America": 1.25, "Europe": 1.05, "APAC": 0.74}[region]
                compute = 15_600 * scale * trend * self.rng.lognormal(0, 0.07)
                storage = 5_400 * scale * trend * self.rng.lognormal(0, 0.05)
                network = 3_750 * scale * trend * self.rng.lognormal(0, 0.08)
                gpu = 9_300 * scale * trend * self.rng.lognormal(0, 0.10)
                # Scenario 1 plus scenario 3: broad Q2 increase and a pronounced May anomaly.
                if pd.Timestamp("2024-04-01") <= date <= pd.Timestamp("2024-06-30"):
                    gpu *= 1.55
                if pd.Timestamp("2024-05-13") <= date <= pd.Timestamp("2024-05-19"):
                    gpu *= 2.15
                values = [compute, storage, network, gpu]
                rows.append({
                    "cost_id": f"CST{cost_id:06d}", "date": date, "region": region,
                    "service_type": "AI Platform", "compute_cost": round(compute, 2),
                    "storage_cost": round(storage, 2), "network_cost": round(network, 2),
                    "gpu_cost": round(gpu, 2), "total_cost": round(sum(values), 2),
                })
                cost_id += 1
        return pd.DataFrame(rows)

    def _support_tickets(self, customers: pd.DataFrame, subscriptions: pd.DataFrame) -> pd.DataFrame:
        n = self.config.n_tickets
        churned_customers = set(subscriptions.loc[subscriptions.subscription_status == "cancelled", "customer_id"])
        weights = customers.customer_id.isin(churned_customers).astype(float).to_numpy() + 0.75
        weights *= np.where(customers.company_size == "Enterprise", 1.8, 1.0)
        customer_idx = self.rng.choice(len(customers), n, p=weights / weights.sum())
        selected = customers.iloc[customer_idx].reset_index(drop=True)
        days = self.rng.integers(0, (self.end - self.start).days + 1, n)
        created = self.start + pd.to_timedelta(days, unit="D")
        churn_link = selected.customer_id.isin(churned_customers).to_numpy()
        europe_q2 = (selected.region.to_numpy() == "Europe") & (created >= pd.Timestamp("2024-04-01")) & (created <= pd.Timestamp("2024-06-30"))
        resolution = self.rng.lognormal(2.35, 0.62, n) * (1 + 0.65 * churn_link) * (1 + 1.15 * europe_q2)
        satisfaction = np.clip(5.15 - 0.055 * resolution + self.rng.normal(0, 0.65, n), 1, 5)
        resolved = created + pd.to_timedelta(resolution, unit="h")
        status = np.where(resolved <= self.end, "resolved", "open")
        resolved = pd.Series(resolved).where(status == "resolved", pd.NaT)
        frame = pd.DataFrame({
            "ticket_id": [f"TKT{i:07d}" for i in range(1, n + 1)],
            "customer_id": selected.customer_id.to_numpy(), "created_date": created,
            "resolved_date": resolved.to_numpy(),
            "ticket_category": self.rng.choice(["Technical", "Billing", "Account", "Integration", "Performance"], n),
            "severity": self.rng.choice(["Low", "Medium", "High", "Critical"], n, p=[0.26, 0.46, 0.23, 0.05]),
            "resolution_hours": np.round(resolution, 2), "satisfaction_score": np.round(satisfaction, 1),
            "status": status,
        })
        if self.config.inject_imperfections:
            missing = self.rng.choice(frame.index, size=max(1, n * 8 // 100), replace=False)
            frame.loc[missing, "satisfaction_score"] = np.nan
            # Duplicate business events retain unique primary keys so relational integrity remains valid.
            dup = frame.sample(3, random_state=self.config.seed).copy()
            dup["ticket_id"] = [f"TKT{n+i:07d}" for i in range(1, 4)]
            frame = pd.concat([frame, dup], ignore_index=True)
        return frame

    @staticmethod
    def _derive_customer_status(customers: pd.DataFrame, subscriptions: pd.DataFrame) -> pd.DataFrame:
        active_ids = set(subscriptions.loc[subscriptions.subscription_status == "active", "customer_id"])
        result = customers.copy()
        result["customer_status"] = np.where(result.customer_id.isin(active_ids), "active", "inactive")
        return result
