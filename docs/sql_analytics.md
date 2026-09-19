# Phase 5 SQL Analytics

Phase 5 provides a parameterized PostgreSQL analytical layer for AMX Tech. The implementation uses SQLAlchemy Core expressions so dates and categorical filters are bound parameters rather than interpolated user input. Dynamic grouping is limited to the allowlisted `region` and `plan_type` dimensions.

## Metric contracts

All date boundaries are inclusive.

| Metric | Definition |
|---|---|
| Gross revenue | Sum of `sales.gross_revenue` in the filtered sale period |
| Net revenue | Sum of `sales.net_revenue` in the filtered sale period |
| Discount rate | Sum of discounts divided by sum of gross revenue |
| Average revenue per customer | Net revenue divided by distinct billed customers |
| Infrastructure cost | Sum of compute, storage, network, and GPU cost in the filtered cost period |
| Estimated gross contribution | Net revenue minus recorded infrastructure cost |
| Cost-to-revenue ratio | Infrastructure cost divided by net revenue |
| Subscriptions at risk | Started by period end and not ended before period start |
| Period cancellations | Cancelled subscriptions whose end date falls within the period |
| Period churn rate | Period cancellations divided by subscriptions at risk |
| Renewals due | Renewal date falls within the period |
| Renewal rate | Due subscriptions retained through their renewal date divided by renewals due |
| Support metrics | Business-unique tickets created in the period; exact duplicate events are removed |

Estimated gross contribution is not accounting profit. AMX Tech has no payroll, marketing, tax, depreciation, or other operating-expense table.

## Supported filters

- Revenue: inclusive date range, region, and plan.
- Subscriptions: inclusive period, region, and plan.
- Support: inclusive created-date period and region.
- Cloud costs and contribution: inclusive date range and region.

Cloud costs cannot be filtered by plan because no defensible plan-level allocation exists. Support tickets also cannot be assigned to a plan because customers may have multiple subscriptions. The API rejects both requests rather than returning misleading results.

## Demonstration investigations

### Q1 versus Q2 2024 profitability

Live PostgreSQL results show:

- Net revenue rises from 74.63 million to 76.44 million, or 2.42%.
- Infrastructure cost rises 22.66%.
- Estimated gross contribution declines 2.57%.
- GPU cost and Enterprise discounting are visible as the primary and secondary pressures for later decomposition.

### Europe churn and support

Europe records five cancellations ending in Q1 and 161 ending in Q2. Average resolution time rises from 15.13 to 34.38 hours, while average reported satisfaction falls from 4.20 to 3.38. These results demonstrate association only; no SQL aggregation establishes causation.

## Reconciliation strategy

Live tests independently extract the five PostgreSQL tables into Pandas, apply the same documented filter boundaries, and compare results against SQL for:

- Full-period, Europe-filtered, and Enterprise-filtered revenue.
- Regional infrastructure costs and contribution.
- Period churn and renewal components.
- Deduplicated support metrics.
- Monthly revenue and cloud-cost series.
- Region and plan groupings.
- Q1/Q2 profitability and Europe support demonstrations.

This is not a comparison of two aliases over the same aggregate query: SQL aggregation happens inside PostgreSQL, while reconciliation aggregation happens independently in Pandas.

## Safety boundary

Application analytics call predefined query functions. The lower-level ad hoc helper accepts only one statement beginning with `SELECT` and rejects common mutation, DDL, locking, session-control, and multi-statement tokens. It is defense-in-depth, not a complete SQL parser. The future question layer should route users to predefined analytical functions rather than arbitrary SQL wherever possible.
