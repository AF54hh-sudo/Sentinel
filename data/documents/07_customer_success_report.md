---
document_id: DOC-007
title: Customer Success Health Review
document_date: 2024-11-12
category: customer_success
regions: [India, North America, Europe, APAC]
topics: [renewal, churn_risk, support, customer_health]
---
# Customer Success Health Review

## Risk signals, intervention use, and model boundaries

Customer Success uses a combination of account conversations and structured behavior to prioritize outreach. Relevant signals include plan, tenure, prior renewal history, support volume, average resolution time, satisfaction feedback, and recent revenue. No single signal determines the outcome. A customer with several tickets may be expanding successfully, while a quiet customer may already have disengaged. The team therefore treats risk scores as prioritization aids and expects account owners to review the surrounding evidence.

Professional-plan accounts received additional attention during the second half of 2024 because managers observed more difficult renewals in parts of the cohort. European queue pressure was a related operational concern, but the review avoids assuming that region, plan, or support alone caused cancellation. Interventions focused on aged-ticket review, product adoption conversations, and clear renewal ownership. Outcomes from those interventions were not randomized, so they cannot be used as unbiased proof of treatment effectiveness.

The churn model should be evaluated on a held-out cohort after preprocessing, model selection, tuning, and threshold selection are fixed using training data only. Precision and recall both matter: missed high-risk accounts reduce the opportunity to intervene, while excessive false positives waste specialist time. Displayed probabilities are not certainties and are not separately calibrated. Feature importance describes predictive contribution, not a causal mechanism or an instruction to change a specific account.

For customer-facing use, the team asks that rankings show customer ID, plan, region, probability, threshold status, and model limitations. Sensitive or unsupported narrative should not be invented. This health review supplies operational context for why particular features matter to a workflow, while the persisted model artifact and held-out metrics remain the sources for predictions and performance claims.
