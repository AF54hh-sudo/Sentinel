# Cloud-Cost Anomaly Findings

## Decision objective

Phase 8 asks whether AMX Tech experienced an unusual cloud-cost event, when it occurred, how far it departed from a normal local range, and whether a transparent statistical baseline agrees with a model-based detector. The analysis focuses on company-wide daily GPU cost from 2023-01-01 through 2024-12-31.

## Detection design

The 2,924 regional cost rows are summed to 731 reconciled daily observations. A centered 29-day median establishes the local cost level. Detection uses the ratio between observed GPU cost and this baseline so the broad Q2 increase is treated as a sustained level change rather than a short anomaly.

- Standard IQR baseline: values outside 1.5 interquartile ranges become review candidates.
- Strong IQR baseline: values outside 3.0 interquartile ranges become extreme outliers.
- Isolation Forest: 300 trees, seed 42, and a 1% anomaly-frequency prior score the log local-cost ratio.
- Final strong flag: the strong IQR and Isolation Forest methods must agree.

Ground-truth scenario dates are not used by either detector. They are supplied only after scoring for offline measurement.

## Method comparison

| Method | Flagged days | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Standard 1.5-IQR candidate | 13 | 0.538 | 1.000 | 0.700 |
| Isolation Forest | 8 | 0.875 | 1.000 | 0.933 |
| Conservative agreement | 7 | **1.000** | **1.000** | **1.000** |

The standard IQR rule catches every injected event day but also flags six ordinary high-variation days. Isolation Forest removes five of those false positives. Requiring model agreement with the 3.0-IQR outer fence removes the final false positive on this seed-42 synthetic dataset.

## Strong anomalies

| Date | Normal range | Observed GPU cost | Anomaly score |
|---|---:|---:|---:|
| 2024-05-13 | 62,751-82,208 | 150,767 | 0.096 |
| 2024-05-14 | 62,048-81,288 | 152,955 | 0.100 |
| 2024-05-15 | 62,048-81,288 | 162,979 | 0.111 |
| 2024-05-16 | 61,817-80,984 | 142,769 | 0.090 |
| 2024-05-17 | 61,817-80,984 | 162,650 | 0.112 |
| 2024-05-18 | 61,817-80,984 | 135,919 | 0.088 |
| 2024-05-19 | 62,048-81,288 | 143,863 | 0.092 |

All seven consecutive dates from 13-19 May 2024 are strong anomalies. Observed GPU cost is roughly 91%-128% above the robust local median and substantially exceeds each day's standard IQR normal range. The event is both large and persistent enough to warrant operational investigation.

## Business interpretation

The evidence supports the statement that AMX Tech experienced a short, strong company-wide GPU-cost anomaly in mid-May 2024. It does not identify the cause. Plausible follow-up checks include workload deployment history, runaway jobs, autoscaling behavior, provider billing changes, and region-level utilization. Those hypotheses require operational evidence that is not present in the dataset.

## Reproducible outputs

Run `python scripts/run_anomaly_detection.py --source database` when PostgreSQL is available, or use `--source csv` for the validated generated files. The command writes three git-ignored outputs:

- `models/cloud_cost_anomaly_detector.joblib`: fitted Isolation Forest plus baseline configuration and decision rule.
- `models/cloud_cost_anomaly_report.json`: method settings, counts, strong-event evidence, limitations, and evaluation-only metrics.
- `models/cloud_cost_anomaly_scores.csv`: all 731 daily normal ranges, observed values, flags, scores, and interpretations.

## Limitations

- Perfect recovery measures a known synthetic event and must not be generalized to real operations.
- The centered baseline is retrospective and cannot be used unchanged for same-day alerting.
- The 1% contamination prior is an analytical choice rather than an observed production incident rate.
- Company-wide aggregation may conceal anomalies isolated to a single region.
- The model is univariate and does not incorporate workload, utilization, deployment, or billing metadata.
