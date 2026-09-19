# Churn Model Evaluation

## Decision objective

Phase 7 ranks AMX Tech customers by the likelihood that a subscription already active on 2024-03-31 will be cancelled by 2024-12-31. The output is intended to prioritize retention review; it is not an automated cancellation decision or a causal estimate of intervention benefit.

## Cohort and leakage controls

The snapshot contains 3,288 eligible customers, including 319 future churners (9.70%). Features use only data available by 2024-03-31: customer attributes, active contract characteristics, historical revenue and discount behavior, renewal history, and historical support activity. Dataset-end customer or subscription status, subscription end dates as predictors, post-snapshot sales or tickets, and scenario ground truth are excluded.

The cohort is split 80/20 with stratification: 2,630 training customers and 658 held-out customers. Model selection, tuning, and operating-threshold selection use the training partition only. The held-out test is evaluated once after those choices are fixed.

## Candidate comparison

Average precision (AP) is the primary selection measure because only about one in ten eligible customers churns. ROC-AUC is also reported, while recall, precision, and F1 describe the default 0.50 operating point. Values below are five-fold stratified cross-validation means on the training partition.

| Candidate | AP | ROC-AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Dummy | 0.097 | 0.500 | 0.000 | 0.000 | 0.000 |
| Logistic regression | 0.524 | 0.888 | 0.780 | 0.346 | 0.479 |
| Random forest | 0.671 | 0.912 | 0.584 | 0.677 | 0.625 |
| Histogram gradient boosting | **0.698** | **0.927** | 0.682 | 0.611 | **0.644** |

Histogram gradient boosting was selected and tested across 12 randomized configurations. The winning configuration used learning rate 0.05, 120 iterations, 15 maximum leaf nodes, 10 minimum samples per leaf, and no L2 regularization. Its tuned cross-validation AP was 0.704.

## Held-out performance

The operating threshold, 0.591, was chosen from out-of-fold training predictions while requiring at least 70% recall. On 658 untouched test customers:

| Metric | Result |
|---|---:|
| Accuracy | 0.936 |
| Precision | 0.641 |
| Recall | 0.781 |
| F1 | 0.704 |
| ROC-AUC | 0.944 |
| Average precision | 0.774 |

The confusion matrix contains 566 true negatives, 28 false positives, 14 false negatives, and 50 true positives. The model therefore identifies 50 of 64 churners while sending 78 customers to the positive-risk queue. Its held-out AP is about eight times the dummy/prevalence baseline, indicating useful ranking separation in this synthetic setting.

At the unmodified 0.50 threshold the model reached 0.813 recall, 0.571 precision, and 0.671 F1. The selected threshold trades a small amount of recall for fewer false positives and a stronger F1 score while remaining above the stated recall floor.

## Interpretation and error analysis

Permutation importance on the held-out set ranked sales observation count, average support resolution time, region, trailing net revenue, total support tickets, historical renewals, days to renewal, and historical discount rate as the strongest predictive contributors. These values show how much held-out AP decreases when a feature is shuffled; they do not establish that changing a feature will change churn.

The 14 false negatives are the highest-cost errors if AMX Tech values missed retention opportunities, while the 28 false positives represent potentially unnecessary outreach. The persisted test-ranking CSV supports customer-level review, and the JSON report includes errors by region and plan plus example false positives and false negatives.

## Deployment artifact

`python scripts/train_churn_model.py` regenerates three git-ignored outputs:

- `models/churn_model.joblib`: versioned AMX Tech artifact with the fitted preprocessing/model pipeline, threshold, feature contract, and snapshot metadata.
- `models/churn_model_report.json`: complete machine-readable evaluation report.
- `models/churn_test_predictions.csv`: held-out predictions for honest evaluation and error analysis, not production scores.

The pipeline is refit on the full labeled cohort only after held-out evaluation. Production scoring must recreate the documented point-in-time feature contract; a future phase should add drift monitoring, probability calibration, and decision-cost validation before operational use.

## Limitations

- All observations are synthetic and intentionally contain learnable patterns.
- A random customer split estimates within-period generalization, not future temporal drift.
- The target concerns subscriptions active at the snapshot; later acquisitions are outside scope.
- Permutation importance can distribute or suppress importance among correlated predictors.
- The threshold reflects a generic recall floor, not an observed retention-contact cost function.
- Probability calibration and fairness diagnostics are not yet part of Phase 7.
