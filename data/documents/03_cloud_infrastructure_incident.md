---
document_id: DOC-003
title: Cloud Infrastructure Incident Review
document_date: 2024-05-24
category: incident_review
regions: [India, North America, Europe, APAC]
topics: [gpu_costs, anomaly, deployment, remediation]
---
# Cloud Infrastructure Incident Review

## GPU capacity event, response, and limitations

During the May deployment window, AMX Tech moved several model-serving workloads to a revised GPU scheduling configuration. The release completed without a prolonged customer-facing outage, but the scheduler did not consolidate jobs as expected. Reserved GPU workers stayed active after workloads had shifted, and utilization monitoring initially showed the capacity as available rather than unnecessary. Operations detected the mismatch during the next cost-and-capacity review and opened an incident because the pattern could increase cloud charges even when application traffic remained within the expected range.

The immediate response was to pause further migration batches, correct the scheduler policy, and drain idle workers in stages. Engineers also added an alert comparing reserved capacity with active workload assignments. The team avoided terminating all excess capacity at once because some long-running customer jobs could not be interrupted safely. By the end of the remediation window, workload placement was more consistent and unused reservations had been reduced. This chronology explains an operational mechanism but does not quantify the financial impact.

The incident commander asked analysts to test the daily GPU-cost series with both a transparent robust baseline and a model-based detector. A strong conclusion should require agreement between methods and should report the observed value, normal range, date, and detector limitations. The incident date must not be used to tune the detector or select its threshold. Regional records should first be reconciled to the company-wide daily series so duplicated or missing components do not create a false signal.

The review found no evidence that the event changed subscription prices or directly caused customer cancellations. It may provide context when structured cost evidence identifies an unusual May observation, but it cannot establish the size of that observation or its effect on quarterly contribution. Future actions include automated reservation expiry, clearer workload ownership, and a weekly exception report reviewed jointly by Infrastructure and Finance.
