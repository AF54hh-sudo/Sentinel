# Phase 12 Question Understanding

Phase 12 adds one narrow LLM responsibility: convert a free-form AMX Tech business question into a
validated structured intent. It does not answer the question or run an analytical workflow.

## Configuration

Copy `.env.example` to `.env` and set the hosted-provider values locally:

```dotenv
SENTINEL_LLM_PROVIDER=openai
SENTINEL_LLM_MODEL=gpt-5.4-nano
SENTINEL_LLM_API_KEY=your-local-secret
SENTINEL_LLM_TIMEOUT_SECONDS=30
SENTINEL_LLM_MAX_RETRIES=2
```

The API key must never be committed. Provider and model are explicit rather than silently selected.
Only OpenAI is implemented in Phase 12; the factory rejects unknown providers.

Install and run either interface:

```powershell
python -m pip install -e ".[dev,phase3,science,dashboard,llm]"
streamlit run app.py
python scripts/understand_question.py "Why did Q2 profit fall?"
```

## Structured intent contract

The hosted model must return a Pydantic `QuestionIntent` containing:

- one allowlisted business intent and analysis mode;
- explicit business metrics;
- a resolved or deliberately unresolved time period and comparison period;
- allowlisted region/plan filters plus optional industry or customer filters;
- whether the question asks for a cause;
- clarification status and one clarification question when necessary;
- a concise interpretation and intent confidence.

All models forbid extra properties. Dates must form complete ordered ranges, metrics and filters
must be unique, confidence is bounded to zero–one, and supported intents require at least one
metric. Invalid, missing, incomplete, or refused output is rejected before it can enter Sentinel.

## Safety and evidence boundary

Trusted dataset dates, regions, and plans are serialized separately from the untrusted question.
The system prompt explicitly prohibits answering, calculation, SQL generation, code execution,
fabricated tests, invented evidence, and causal claims. The OpenAI client receives no tools and no
business rows—only the question and compact dataset context.

The Phase 12 Streamlit result shows intent metadata only. Phase 13 separately consumes validated
intents through the bounded analytical workflow. API tests use injected fake response clients and
never require a network call or secret.
