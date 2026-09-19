# Phase 15 Evidence-Based Reporting

Phase 15 converts one completed analytical workflow into a citation-validated AMX Tech business
report. It can add up to five already retrieved document chunks, but documents remain contextual and
never replace analytical metrics.

## Pipeline

```text
Completed Phase 13 workflow
          +
Optional Phase 14 chunks
          ↓
Typed evidence registry
          ↓
Deterministic or hosted structured narration
          ↓
Claim and citation validation
          ↓
Qualitative confidence policy
          ↓
BusinessReport + Markdown renderer
```

A report is unavailable when the workflow needs clarification, is unsupported, failed, or lacks
verified analytical evidence. There is no best-effort answer without evidence.

## Evidence registry

Every report creates a new immutable registry. IDs are assigned deterministically by evidence type:

| Evidence | ID example |
|---|---|
| Business analytics | `SQL-001` |
| Statistical inference | `STAT-001` |
| Churn model | `ML-001` |
| Revenue forecast | `FORECAST-001` |
| Cost anomaly | `ANOMALY-001` |
| Data quality | `DATA-001` |
| Retrieved document | `DOC-001` |

Analytical registration requires the Phase 13 verification flag. Each item retains its title,
summary, metrics, a bounded sample of records, complete record count, source references, and inherited
limitations. Document evidence retains the source chunk ID, title/date, section/page, source path,
similarity, metadata, and retrieved text. Duplicate document chunks are registered once.

## Claim validation

Every executive-summary sentence, finding, recommendation, and limitation is a typed claim with a
unique `CLM-nnn` ID and one or more evidence citations. Validation rejects:

- missing, duplicate, unknown, or unverified evidence IDs;
- numerical tokens absent from the cited evidence;
- document-only numerical assertions other than citation metadata such as dates/pages;
- document-only analytical findings;
- positive causal wording when the cited evidence has no causal-study support;
- executive summaries or reports without analytical evidence.

Negated causal limitations such as “does not establish cause” remain valid. Retrieved records are
treated as untrusted text in the hosted prompt and cannot override the reporting role.

## Narration modes

The deterministic narrator works without another API request. It copies the verified analytical
summary, exposes up to five evidence metrics as findings, adds document metadata as context, selects
a branch-specific cautious recommendation, and preserves inherited limitations.

The optional OpenAI narrator receives only the user question, its validated interpretation, and the
immutable evidence registry. It has no tools and returns the strict `ReportNarrative` Pydantic schema.
Its output passes the same local claim validator; invalid, refused, incomplete, or fabricated output
fails instead of falling back silently. Numerical metrics, supporting-document metadata, confidence,
limitations, and final Markdown formatting remain deterministic application responsibilities.

## Confidence

Confidence is qualitative, never a fabricated percentage:

- **High:** direct deterministic business or data-quality calculation without a causal request.
- **Medium:** causal wording over observational metrics, statistical association, held-out ML,
  short-history forecasting, or retrospective anomaly detection.
- **Low:** reserved for material unresolved evidence limitations.
- **Insufficient:** no completed verified analytical workflow.

Document similarity does not increase analytical confidence.

## Interfaces

After **Run verified analysis**, Streamlit exposes **Build evidence report**. It uses hosted narration
only when explicitly enabled and configured; otherwise it uses the deterministic narrator. Any
currently retrieved document results are included, and the report can be downloaded as Markdown.

CLI:

```powershell
python scripts/build_report.py "Why did Q2 profit fall?"
python scripts/build_report.py "Why did Q2 profit fall?" --with-documents
python scripts/build_report.py "Why did Q2 profit fall?" --deterministic
```

Question understanding still requires a configured intent client. `--deterministic` disables the
second, report-narration request; it does not bypass Phase 12 question understanding.
