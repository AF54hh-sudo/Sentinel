# Phase 4 Data Validation and EDA Findings

This report describes the deterministic seed-42 dataset generated on 2026-08-13. All results come from Pandas calculations over the five business tables. The evaluation-only ground-truth file was not read by the EDA workflow.

## Data quality

- Raw row counts are 5,000 customers, 8,000 subscriptions, 43,277 sales observations, 2,924 daily regional cloud-cost records, and 10,003 support-ticket rows.
- Primary keys and all four foreign-key relationships validate successfully.
- Monetary identities reconcile within the documented rounding tolerance.
- Fifty missing customer industries are labeled `Unknown` during cleaning.
- Three duplicate support events are removed by comparing every business field except the unique ticket ID, leaving 10,000 events.
- Eight hundred missing satisfaction scores are retained. Treating a non-response as neutral satisfaction would introduce unsupported information.
- Null end dates for 7,147 active subscriptions and null resolution dates for 17 open tickets are structural, not cleaning errors.

## Descriptive business metrics

Across the full observation window:

| Metric | Value |
|---|---:|
| Gross revenue | 655,757,863.76 |
| Net revenue | 584,779,466.77 |
| Average discount rate | 10.82% |
| Infrastructure cost | 118,654,853.07 |
| Cost-to-revenue ratio | 20.29% |
| Estimated gross contribution | 466,124,613.70 |
| Subscription churn rate | 8.99% |
| Active subscriptions | 7,147 |
| Average resolution time | 15.29 hours |
| Average satisfaction | 4.22 / 5 |

Estimated gross contribution is defined narrowly as net revenue minus the four recorded infrastructure-cost components. It is not accounting profit because payroll, sales, taxes, and other operating costs are unavailable.

## Revenue and churn segments

- Enterprise contributes the most net revenue and carries the highest overall discount rate.
- Professional has the highest plan-level cancellation rate, while Enterprise has the lowest.
- India contributes the most net revenue in absolute terms.
- Europe has a higher full-period cancellation rate than India and North America. This is descriptive and does not control for plan mix or tenure.

## Q2 2024 profitability signal

From Q1 to Q2 2024:

- Net revenue increases from 74.63 million to 76.44 million, about 2.4%.
- Infrastructure cost increases from 14.77 million to 18.12 million, about 22.7%.
- Estimated gross contribution declines from 59.86 million to 58.32 million, about 2.6%.
- GPU cost increases from 4.06 million to 7.00 million.
- Q2 discounts total 16.25 million versus 8.13 million in Q1.

This supports a later investigation of infrastructure cost as the primary pressure and discounting as a secondary pressure. Formal attribution belongs to Phase 5 SQL analytics and Phase 6 statistical work.

## Europe support and cancellations

Europe’s average resolution time rises from 15.13 hours in Q1 2024 to 34.38 hours in Q2, while average reported satisfaction falls from 4.20 to 3.38. Europe records 161 cancellations ending in Q2 versus five in Q1. Customer-level Spearman exploration also shows churn positively associated with ticket volume and resolution time and negatively associated with satisfaction.

These patterns are associations in synthetic observational data. They do not establish that support issues caused cancellation; plan, tenure, selection into support, and other factors may confound the relationship.

## Cloud-cost pattern

May 2024 has the highest monthly GPU cost in the observation window. IQR and model-based anomaly classification are intentionally deferred to Phase 8.

## Limitations and next questions

- Sales observations are a deterministic, evenly distributed sample across subscription lifetimes, not an invoice ledger.
- Currency units are generic.
- Satisfaction is missing for 8% of support events and should not be imputed without a modeling-specific rationale.
- Descriptive churn rates do not yet adjust for exposure duration or competing subscription outcomes.
- Hypothesis tests, effect sizes, confidence intervals, model predictions, anomaly scores, and forecasts are outside Phase 4.
