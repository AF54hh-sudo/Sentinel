# AMX Tech Data Dictionary

The Phase 2 dataset covers 2023-01-01 through 2024-12-31. Currency is represented in generic monetary units so presentation can localize it later. Dates are generated as UTC-neutral calendar values.

## customers

One row per customer. `customer_id` is the primary key. `region`, `country`, `industry`, `company_size`, and `acquisition_channel` are segmentation attributes. `customer_status` is derived from whether any related subscription remains active. About 1% of `industry` values are deliberately missing for later cleaning exercises.

## subscriptions

One row per subscription. `subscription_id` is the primary key and `customer_id` references `customers`. `monthly_price` is the contracted monthly value derived from plan and seats. `subscription_status` is `active`, `cancelled`, or `expired`; `end_date` is null for active subscriptions.

## sales

Monthly billing observations. `sale_id` is the primary key; customer and subscription identifiers are foreign keys. `gross_revenue - discount_amount = net_revenue`. Enterprise discounts are plan-dependent and contain a controlled Q2 2024 change.

## cloud_costs

Daily regional infrastructure costs. `cost_id` is the primary key. `total_cost` is the sum of compute, storage, network, and GPU cost. Q2 2024 contains an intentional GPU-cost increase and May contains a short, strong anomaly.

## support_tickets

Support events linked to customers. Resolution time follows a skewed positive distribution. Satisfaction declines as resolution time rises. Missing satisfaction values and three duplicate business events are deliberate, limited data-quality imperfections; duplicate rows retain distinct primary keys.

## Relationships

`customers.customer_id` is referenced by subscriptions, sales, and support tickets. `subscriptions.subscription_id` is referenced by sales. Cloud costs join analytically by region and date, not through a foreign key.

## Leakage boundary

`data/ground_truth/scenarios.yaml` documents injected mechanisms only for automated evaluation. It is not an application data source and must never be used to answer a business question.
