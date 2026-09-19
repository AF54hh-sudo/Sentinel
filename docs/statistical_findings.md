# Phase 6 Statistical Findings

This report summarizes the deterministic AMX Tech seed-42 dataset. Statistical tests are exploratory demonstrations over synthetic observational data. They describe uncertainty and association; they do not prove causation or generalize to a real company.

## Analysis design

Support analyses use one row per customer rather than one row per ticket. For a customer who cancelled, only tickets created on or before the first cancellation date are included. Retained customers are observed through the end of the dataset. This prevents post-outcome support events from being used as if they predicted cancellation and reduces repeated-ticket pseudoreplication.

Customers without support events are retained for ticket-count summaries but cannot enter resolution-time or satisfaction comparisons. Churn is defined as having at least one cancelled subscription.

## Support resolution and churn status

### Business question

Do customers who later cancel have different pre-outcome support-resolution times than retained customers?

### Descriptive evidence

| Group | Customers with support | Mean hours | 95% CI | Median hours | Skewness |
|---|---:|---:|---:|---:|---:|
| Churned | 501 | 22.06 | 21.00–23.12 | 20.06 | 4.07 |
| Retained | 3,468 | 12.95 | 12.70–13.20 | 11.42 | 2.95 |

Resolution is strongly right-skewed and group variances differ. Therefore, the analysis reports both Welch’s unequal-variance mean test and a Mann–Whitney rank test.

### Welch’s t-test

- Null: mean pre-outcome resolution time is equal across churn groups.
- Alternative: the means differ.
- t = 16.42; p ≈ 1.05×10⁻⁴⁹.
- Hedges’ g = 1.108, a large standardized mean difference.

### Mann–Whitney U

- Null: the two resolution-time distributions are the same.
- Alternative: the distributions differ.
- U = 1,391,757; p ≈ 1.67×10⁻¹⁰⁵.
- Rank-biserial correlation = 0.602, indicating a substantial distributional separation in the direction of longer resolution for churned customers.

Both methods agree that churned customers in this dataset experienced materially longer pre-outcome support resolution. This remains an association: support demand, product complexity, plan, tenure, and other omitted factors may affect both resolution and cancellation.

## Resolution time and satisfaction

At the customer level, Spearman rho is −0.579 across 3,854 complete pairs, with a p-value below machine-reportable precision. This is a moderately strong negative monotonic association: longer resolution tends to accompany lower satisfaction.

Spearman is the primary measure because resolution is skewed and satisfaction is bounded. Pearson is included only as a linear sensitivity analysis, with its stronger assumptions stated in the machine-readable report. Neither correlation establishes that resolution time caused satisfaction to change.

## Plan and cancellation

The plan-by-cancellation table contains 8,000 subscriptions:

| Plan | Not cancelled | Cancelled | Observed cancellation rate |
|---|---:|---:|---:|
| Basic | 3,250 | 283 | 8.01% |
| Professional | 2,804 | 374 | 11.77% |
| Enterprise | 1,227 | 62 | 4.81% |

Pearson’s chi-square statistic is 61.67 with p ≈ 4.05×10⁻¹⁴. The association is statistically detectable, but Cramér’s V is only 0.088. The effect is small, demonstrating why the p-value alone is insufficient. Region, tenure, company size, and support factors are not controlled, and customers with multiple subscriptions may weaken the strict independence assumption.

## Europe churn-rate uncertainty

Using subscriptions exposed during each period as the denominator:

| Period | Cancellations | At risk | Rate | 95% Wilson interval |
|---|---:|---:|---:|---:|
| Q1 2024 | 5 | 1,137 | 0.44% | 0.19%–1.03% |
| Q2 2024 | 161 | 1,364 | 11.80% | 10.20%–13.62% |

The intervals quantify sampling uncertainty within the synthetic population-generating exercise. They do not incorporate uncertainty about generator assumptions or external validity.

## Global limitations

- The dataset is synthetic and deliberately contains controlled patterns.
- These are exploratory analyses without multiplicity adjustment.
- Statistical significance does not automatically imply material business importance.
- Confidence intervals describe sampling variation under model assumptions, not all sources of uncertainty.
- No test here identifies a causal effect.
