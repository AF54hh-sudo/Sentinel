---
document_id: DOC-008
title: Operations and Data Quality Update
document_date: 2024-12-03
category: operations_update
regions: [India, North America, Europe, APAC]
topics: [data_quality, support_tickets, ownership, controls]
---
# Operations and Data Quality Update

## Source controls, known imperfections, and ownership

The operations review confirmed that the five governed business datasets remain customers, subscriptions, sales, cloud costs, and support tickets. Customer identifiers connect the first, second, third, and fifth datasets, while cloud costs support date and region analysis without a defensible plan-level allocation. Teams should preserve those grains when answering questions and should stop rather than inventing an unsupported join or allocation.

Known data imperfections are deliberately limited but visible. Some customer industries are missing and receive an explicit Unknown category during cleaning. Satisfaction surveys are not completed for every ticket, so missing responses remain missing. A small number of duplicated support events can appear under different ticket identifiers; the cleaning policy removes exact business-event duplicates before analysis. Structural null dates for active or open records are retained and validated against status.

Accounting checks require gross revenue minus discount amount to equal net revenue within rounding tolerance. Cloud-cost components must reconcile to total cost, and all monetary values must remain nonnegative. Subscription and ticket dates must follow their business sequence, while foreign keys must resolve to the appropriate parent record. These controls detect integrity problems but do not prove that every business field is complete or that synthetic patterns generalize to a real company.

Data owners asked Sentinel to surface errors, warnings, duplicate removals, and retained missing values separately. A clean validation result should not be described as perfect data quality. This document records policy and ownership context; machine-readable cleaning and validation reports provide the verified counts used by the analytical workflow.
