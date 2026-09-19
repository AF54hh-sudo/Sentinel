"""Injected OpenAI structured-output client for evidence-only narration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from sentinel.config.settings import Settings, get_settings
from sentinel.evidence import EvidenceBundle
from sentinel.llm import LLMConfigurationError
from sentinel.reporting.prompts import SYSTEM_PROMPT, build_report_input
from sentinel.reporting.schemas import ReportNarrative
from sentinel.reporting.validation import validate_narrative


class ReportRequestError(RuntimeError):
    """Raised when the hosted narration request fails without exposing secrets."""


class ReportResponseError(RuntimeError):
    """Raised when model-authored narration violates the report contract."""


class ResponsesAPI(Protocol):
    def parse(self, **kwargs: Any) -> Any: ...


class HostedClient(Protocol):
    responses: ResponsesAPI


@dataclass(frozen=True)
class NarrativeResult:
    narrative: ReportNarrative
    provider: str
    model: str
    response_id: str | None


class ReportNarrator(Protocol):
    def narrate(
        self,
        question: str,
        interpretation: str,
        evidence: EvidenceBundle,
    ) -> NarrativeResult: ...


class OpenAIReportNarrator:
    """Narrate verified evidence through a schema without granting tools or raw business data."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 30,
        max_retries: int = 2,
        client: HostedClient | None = None,
        request_error_types: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("SENTINEL_LLM_API_KEY is not configured")
        if not model.strip():
            raise LLMConfigurationError("SENTINEL_LLM_MODEL is not configured")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if client is None:
            try:
                from openai import OpenAI, OpenAIError
            except ImportError as error:
                raise LLMConfigurationError(
                    "Install the Phase 15 dependency with: pip install -e .[llm]"
                ) from error
            client = OpenAI(
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )
            request_error_types = (OpenAIError,)
        self._client = client
        self._model = model.strip()
        self._request_error_types = request_error_types or (
            OSError,
            RuntimeError,
            TimeoutError,
        )

    @property
    def model(self) -> str:
        return self._model

    def narrate(
        self,
        question: str,
        interpretation: str,
        evidence: EvidenceBundle,
    ) -> NarrativeResult:
        report_input = build_report_input(question, interpretation, evidence)
        try:
            response = self._client.responses.parse(
                model=self._model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": report_input},
                ],
                text_format=ReportNarrative,
            )
        except self._request_error_types as error:
            raise ReportRequestError("OpenAI evidence-narration request failed") from error
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ReportResponseError(
                "OpenAI returned no structured report; the response may be incomplete or refused"
            )
        try:
            narrative = (
                parsed
                if isinstance(parsed, ReportNarrative)
                else ReportNarrative.model_validate(parsed)
            )
            validate_narrative(narrative, evidence)
        except (TypeError, ValueError) as error:
            raise ReportResponseError(
                "OpenAI returned a report that conflicts with the verified evidence registry"
            ) from error
        response_id = getattr(response, "id", None)
        return NarrativeResult(
            narrative=narrative,
            provider="openai",
            model=self._model,
            response_id=str(response_id) if response_id else None,
        )


def create_report_narrator(settings: Settings | None = None) -> OpenAIReportNarrator:
    configured = settings or get_settings()
    provider = configured.llm_provider.strip().lower()
    if provider != "openai":
        raise LLMConfigurationError(f"Unsupported LLM provider: {provider or '(empty)'}")
    return OpenAIReportNarrator(
        api_key=configured.llm_api_key,
        model=configured.llm_model,
        timeout_seconds=configured.llm_timeout_seconds,
        max_retries=configured.llm_max_retries,
    )
