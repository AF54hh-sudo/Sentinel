# Methodology

Synthetic attributes are conditionally generated so plan, company size, seats, price, tickets, satisfaction, churn, discounts, and costs have plausible dependencies. Controlled scenario mechanisms are time- and segment-bounded. Randomness uses NumPy's seeded generator. Validation distinguishes integrity errors from intentional cleaning warnings.

## Phase 4 cleaning policy

CSV date fields are converted with strict parsing. Missing industries receive an explicit `Unknown` category. Duplicate support events are removed using business columns rather than primary keys. Missing survey responses and structural null dates remain missing. Source frames are never mutated by cleaning functions.

EDA reports dataset shape and types, missingness, numerical and categorical distributions, IQR outlier flags, monthly revenue, costs, churn and ticket trends, segment metrics, and Spearman customer-behavior correlations. Outputs are descriptive; causal and inferential language is reserved for later phases.

## Phase 5 SQL methodology

SQL metrics use bound parameters, inclusive date ranges, allowlisted segment dimensions, and explicit denominators. Period churn measures cancellations ending in the window over subscriptions exposed during the window. Renewal measures subscriptions retained through a renewal date falling in the window. Support metrics deduplicate exact business events before aggregation. Unsupported plan-level cost and ticket allocation is rejected. Every principal SQL result is reconciled against an independent Pandas aggregation over database-extracted rows.

## Phase 6 statistical methodology

Support inference aggregates to one row per customer and censors support events at first cancellation. Welch's t-test compares mean resolution without assuming equal variances; Mann-Whitney provides a nonparametric distributional comparison. Hedges' g and rank-biserial correlation report magnitude. Spearman is primary for skewed resolution and bounded satisfaction; Pearson is a secondary linear sensitivity analysis. Plan and cancellation use chi-square with Cramér's V. Means use t intervals and churn proportions use Wilson score intervals. Every test reports hypotheses, rationale, sample size, assumptions, effect size, interpretation, and limitations.

## Phase 7 churn methodology

The prediction cohort contains customers with at least one subscription active on 2024-03-31. The binary outcome is one only when one of those eligible subscriptions is cancelled after the snapshot and by 2024-12-31. Customer, active-subscription, sales, and support features are computed only from information available on or before the snapshot; end-of-dataset statuses, subscription end dates, later activity, and ground-truth scenario metadata are excluded from predictors.

An 80/20 stratified split reserves the test set. Candidate selection uses mean average precision from five-fold stratified cross-validation on the training partition because the positive class is imbalanced. All categorical imputation and one-hot encoding, numeric median imputation and scaling, and model fitting live inside Scikit-learn pipelines so folds cannot share fitted preprocessing. Dummy, logistic regression, random forest, and histogram gradient boosting models are compared. Only the strongest non-dummy candidate is moderately tuned with 12 randomized configurations.

The operating threshold is selected from out-of-fold training probabilities subject to at least 70% recall; the held-out test set is evaluated once after model, hyperparameters, and threshold are fixed. Accuracy, precision, recall, F1, ROC-AUC, average precision, confusion counts, segment errors, and permutation importance are reported. After evaluation, the selected pipeline is refit on the complete labeled cohort for deployment. Importance is predictive rather than causal, and the synthetic result is not evidence of production performance.

## Phase 8 anomaly methodology

Cloud costs are summed across regions to one reconciled company-wide observation per day. The primary metric is GPU cost. Each value is divided by a centered 29-day rolling median, which provides a robust retrospective local baseline and prevents a sustained Q2 level increase from being treated as a one-day event. Standard 1.5-IQR fences identify review candidates, while 3.0-IQR outer fences identify extreme local ratios.

Isolation Forest is fitted to the log local GPU-cost ratio with 300 trees, seed 42, and a 1% contamination prior. A final strong anomaly requires both an outer-IQR flag and an Isolation Forest flag. This consensus favors precision while preserving transparent baseline evidence. The scenario file is read only by the offline runner after all scores and flags are fixed; it is not an input to aggregation, baseline construction, model fitting, thresholds, or final classification.

The centered median uses observations on both sides of each date, so this is retrospective detection rather than a real-time alerting algorithm. Daily aggregation can also hide region-specific deviations, and detector agreement identifies unusual values rather than operational causes.

## Phase 9 forecasting methodology

Sales are aggregated to 24 complete monthly net-revenue observations from January 2023 through December 2024. The first 12 months initialize the models, January-June 2024 forms the validation period, and July-December 2024 is reserved as the later test period. Evaluation is expanding-window and one step ahead: each monthly prediction can use only values observed before that target month, after which the realized month becomes available for the next origin. This matches the business question of forecasting next month and updating monthly.

The required previous-month naïve forecast and three-month moving average are explicit baselines. The model candidate is additive damped-trend exponential smoothing with estimated initialization and no seasonal component. Although the series has a strong 12-month correlation, only two annual cycles are available, which is insufficient for stable seasonal Holt-Winters estimation. Validation WAPE selects the final method, with MAE as a tie-breaker; test MAE, RMSE, and WAPE are then reported without changing that selection.

After evaluation, the chosen method is refit on all 24 observations to generate three multi-step point forecasts. Prediction intervals are not reported because the short synthetic history does not support reliable interval calibration. Results should therefore be treated as planning estimates rather than guaranteed revenue.

## Phase 10 visualization methodology

Charts consume already computed analytical evidence and never become a second calculation layer. Monthly revenue, churn, cloud-cost, and segment frames come from reusable EDA functions; anomaly, churn-model, and forecast figures use the persisted reports and scored rows from their owning phases. Every chart validates required columns or report keys and copies inputs before display-specific transformation.

Time-series charts retain chronological axes and explicit units. Related quantities with different scales use aligned subplots rather than misleading dual axes. Cloud-cost components use a stacked composition view. The anomaly figure preserves the robust normal range and final strong flags. Churn-model evaluation shows both cross-validation baseline comparison and held-out confusion counts. Forecast evidence distinguishes observed values, one-step test predictions, and future point forecasts.

Figures share a restrained accessible palette, descriptive titles, axis labels, hover evidence, responsive HTML export, and stable trace names for Streamlit reuse. Visual emphasis communicates analytical status but does not create causal claims, confidence intervals, or predictions absent from source results.

## Phase 11 dashboard methodology

The dashboard is a thin composition layer. On load, the selected CSV or PostgreSQL source passes through the Phase 4 cleaning and validation boundary. Required churn, forecast, and anomaly artifacts are checked explicitly, and the app stops with an actionable message when evidence is missing or the selected database is unavailable. Cached loads reduce repeated work without creating a second source of truth.

Date, region, and plan controls are applied only where the underlying grain supports them. Revenue uses all three; churn uses region and plan before producing monthly exposed-subscription rates; costs use date and region because plan-level allocation is unsupported. Persisted company-wide forecast and anomaly results remain unchanged and are labelled accordingly. Held-out churn probabilities may be re-thresholded for exploration, but the training-derived operating threshold remains the default and the displayed cohort is clearly identified as held out.

Every chart comes from Phase 10. The UI neither fits models nor generates numerical analysis, and it never reads evaluation-only scenario metadata. This preserves the project's evidence hierarchy before AI assistance is introduced in later phases.

## Phase 12 question-understanding methodology

Question understanding uses a strict schema rather than free-form model text. The schema allowlists supported business intents, metrics, and analysis modes; validates ordered date ranges; rejects duplicate metrics and filters; requires explicit clarification state; bounds confidence; and forbids extra properties. Trusted dataset dates, regions, and plans are supplied separately from the untrusted user question.

The hosted model is instructed only to classify and normalize the question. It receives no analytical tools or business rows and is prohibited from answering, calculating, generating SQL, executing code, fabricating evidence, or claiming causation. When wording cannot be resolved safely, it must request clarification rather than invent scope. The application validates every parsed response again before storing or displaying it.

Provider, model, timeout, retry count, and API key are environment-configured. The API key is neither displayed nor included in normalized errors. Unit tests inject a fake Responses client so schema, prompt, refusal, invalid-output, and provider-failure paths are verified without network access. At the Phase 12 boundary, analytical execution was deferred; Phase 13 now consumes validated intents through the bounded analytical workflow.

## Phase 13 orchestration methodology

The orchestration layer is a fixed directed graph rather than an open-ended agent loop. A validated intent is mapped by ordinary Python logic to at most one of six approved branches: business analytics, statistics, churn ML, forecasting, anomaly detection, or data quality. Planning steps are deterministic templates attached to that selected branch; the LLM does not generate executable plans or tool arguments.

Each analytical branch delegates to existing Phase 4–9 functions and persisted artifacts. It returns a typed evidence object containing metrics, records, source identifiers, and inherited limitations. A dedicated validation node requires the evidence tool to equal the planned tool and requires non-empty metrics and sources before marking the result verified. The graph has a recursion limit of 12 even though every valid path is four nodes long.

Grain compatibility is checked before execution. Shared infrastructure costs cannot be allocated to plans, industries, or customers; company-wide forecasts and anomaly reports reject segment filters; precomputed statistical reports reject unsupported segmentation. Clarification and unsupported questions terminate without evidence. Tests inject validated intents or a fake intent client, exercise every branch, reconcile revenue to source facts, and assert the graph contains no RAG or arbitrary-code nodes.

## Phase 14 retrieval methodology

The retrieval corpus is intentionally small: ten fictional internal Markdown records covering quarterly reviews, infrastructure, European support, pricing, discount governance, customer success, operations, reliability, and forecasting use. YAML front matter is validated under a strict typed contract. Text cleaning preserves paragraphs, and chunking respects heading and paragraph boundaries while targeting approximately 400–700 tokens. Stable `DOC-nnn-Cnnn` IDs, SHA-256 hashes, title, date, section, page, regions, topics, category, and source path stay attached through embedding and retrieval.

The OpenAI embedding client is provider-injected, batched, and fixed-dimension. It rejects empty input, count/index mismatches, nonnumeric or nonfinite values, and dimension mismatches. PostgreSQL stores vectors in a pgvector column. Because the corpus is tiny, retrieval uses exact cosine distance with a configurable top-k and minimum similarity rather than approximate indexes, reranking, hybrid search, or another database. Results use the same embedding model as the query and return the original citation metadata and text.

Documents provide operational or policy context only. They never replace structured metrics, model evaluation, forecasts, tests, or anomaly scores, and Phase 14 does not combine them into a generated report. No runtime RAG module reads evaluation ground truth. Tests use fake embedding and vector-store boundaries; the live pgvector schema check is opt-in through the dedicated test database URL.

## Phase 15 reporting methodology

Reporting begins only after Phase 13 marks analytical evidence verified. A per-report registry assigns type-specific, sequential evidence IDs and preserves metrics, bounded records, original sources, and limitations. Up to five Phase 14 chunks may be registered with separate report IDs while retaining their source chunk IDs and citation metadata. Documents stay contextual and do not increase analytical confidence.

Every narrative unit is a typed claim with a unique claim ID and explicit evidence references. Local validation resolves every reference, requires verified evidence, checks that every numerical token exists in the cited evidence, prevents documents from acting as the sole source for analytical findings or business numbers, and rejects positive causal wording because no current source has a causal-study contract. Reports without completed analytical evidence are stopped.

The deterministic narrator copies evidence and provides a fully offline report path. The optional hosted narrator receives only the validated interpretation and immutable registry, has no tools, and returns a strict Pydantic structured output. Its narrative is accepted only after the same local validation. Key metrics, document metadata, confidence, limitations, and Markdown formatting are not delegated to the model. Confidence is a transparent High/Medium/Low/Insufficient policy based on evidence type and causal scope, never an invented probability.
