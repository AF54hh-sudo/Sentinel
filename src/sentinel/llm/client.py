"""Configurable hosted-LLM boundary for Phase 12 structured intent extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from sentinel.config.settings import Settings, get_settings
from sentinel.llm.prompts import SYSTEM_PROMPT, build_question_input
from sentinel.llm.schemas import QuestionContext, QuestionIntent, validate_intent_context


class LLMConfigurationError(RuntimeError):
    """Raised when the hosted LLM has not been configured safely."""


class LLMRequestError(RuntimeError):
    """Raised when the provider request fails without exposing secrets."""


class LLMResponseError(RuntimeError):
    """Raised when a provider response does not contain the required intent."""


class ResponsesAPI(Protocol):
    def parse(self, **kwargs: Any) -> Any: ...


class HostedClient(Protocol):
    responses: ResponsesAPI


@dataclass(frozen=True)
class IntentResult:
    intent: QuestionIntent
    provider: str
    model: str
    response_id: str | None


class OpenAIIntentClient:
    """Use OpenAI Structured Outputs without granting the model analytical tools."""

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
                    "Install the Phase 12 dependency with: pip install -e .[llm]"
                ) from error
            client = OpenAI(
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )
            request_error_types = (OpenAIError,)

        self._client = client
        self._model = model.strip()
        self._request_error_types = request_error_types or (OSError, RuntimeError, TimeoutError)

    @property
    def model(self) -> str:
        return self._model

    def understand_question(
        self,
        question: str,
        context: QuestionContext,
    ) -> IntentResult:
        user_input = build_question_input(question, context)
        try:
            response = self._client.responses.parse(
                model=self._model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                text_format=QuestionIntent,
            )
        except self._request_error_types as error:
            raise LLMRequestError("OpenAI question-understanding request failed") from error

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise LLMResponseError(
                "OpenAI returned no structured intent; the response may be incomplete or refused"
            )
        try:
            intent = (
                parsed
                if isinstance(parsed, QuestionIntent)
                else QuestionIntent.model_validate(parsed)
            )
        except (TypeError, ValueError) as error:
            raise LLMResponseError("OpenAI returned an invalid structured intent") from error
        try:
            validate_intent_context(intent, context, question)
        except ValueError as error:
            raise LLMResponseError(
                "OpenAI structured intent conflicts with trusted dataset context"
            ) from error
        response_id = getattr(response, "id", None)
        return IntentResult(
            intent=intent,
            provider="openai",
            model=self._model,
            response_id=str(response_id) if response_id else None,
        )


def create_intent_client(settings: Settings | None = None) -> OpenAIIntentClient:
    """Create the configured provider without silently selecting credentials or models."""
    configured = settings or get_settings()
    provider = configured.llm_provider.strip().lower()
    if not provider:
        raise LLMConfigurationError("SENTINEL_LLM_PROVIDER is not configured")
    if provider != "openai":
        raise LLMConfigurationError(f"Unsupported LLM provider: {provider}")
    return OpenAIIntentClient(
        api_key=configured.llm_api_key,
        model=configured.llm_model,
        timeout_seconds=configured.llm_timeout_seconds,
        max_retries=configured.llm_max_retries,
    )
