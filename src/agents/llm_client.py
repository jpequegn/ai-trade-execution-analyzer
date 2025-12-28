"""LLM client wrapper with retry logic and observability.

This module provides a unified interface for interacting with LLM providers
(Anthropic Claude and OpenAI GPT) with built-in retry logic, error handling,
and Langfuse tracing integration.

Example:
    >>> from src.agents.llm_client import LLMClient
    >>> client = LLMClient()
    >>> response = client.complete([{"role": "user", "content": "Hello"}])
    >>> print(response.content)
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.config import AppConfig, LLMConfig, get_config

if TYPE_CHECKING:
    from anthropic import Anthropic
    from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """Structured response from an LLM call.

    Attributes:
        content: The generated text content.
        input_tokens: Number of tokens in the input.
        output_tokens: Number of tokens in the output.
        model: The model used for generation.
        latency_ms: Time taken for the call in milliseconds.
        provider: The LLM provider used.
    """

    content: str = Field(..., description="Generated text content")
    input_tokens: int = Field(..., ge=0, description="Input token count")
    output_tokens: int = Field(..., ge=0, description="Output token count")
    model: str = Field(..., description="Model identifier used")
    latency_ms: float = Field(..., ge=0, description="Call latency in milliseconds")
    provider: str = Field(default="anthropic", description="LLM provider")

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


class LLMError(Exception):
    """Base exception for LLM client errors."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limited by the provider."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """Raised when the request times out."""

    pass


class LLMProviderError(LLMError):
    """Raised when the provider returns an error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMClient:
    """Unified LLM client with retry logic and observability.

    This client supports both Anthropic Claude and OpenAI GPT models with
    automatic retry on transient errors and optional Langfuse tracing.

    Attributes:
        config: LLM configuration settings.
        provider: The LLM provider ("anthropic" or "openai").

    Example:
        >>> client = LLMClient()
        >>> response = client.complete([
        ...     {"role": "user", "content": "What is 2+2?"}
        ... ])
        >>> print(response.content)
        4
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        app_config: AppConfig | None = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            config: Optional LLM configuration. If not provided, loads from environment.
            app_config: Optional full app configuration.

        Raises:
            ConfigurationError: If required configuration is missing.
        """
        if app_config is not None:
            self._app_config: AppConfig | None = app_config
            self.config = app_config.llm
        elif config is not None:
            self._app_config = None
            self.config = config
        else:
            self._app_config = get_config()
            self.config = self._app_config.llm

        self.provider = self.config.provider
        self._client: Anthropic | OpenAI | None = None

    def _get_client(self) -> Anthropic | OpenAI:
        """Get or create the provider client.

        Returns:
            The provider client instance.

        Raises:
            ConfigurationError: If the API key is not set.
        """
        if self._client is not None:
            return self._client

        self.config.validate()

        if self.provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        else:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )

        return self._client

    def _calculate_backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """Calculate delay for exponential backoff with jitter.

        Args:
            attempt: Current retry attempt (0-indexed).
            retry_after: Optional delay suggested by provider.

        Returns:
            Delay in seconds before next retry.
        """
        if retry_after is not None:
            return min(retry_after, self.config.retry_max_delay)

        # Exponential backoff with jitter
        base_delay = self.config.retry_base_delay * (2**attempt)
        jitter = random.uniform(0, 0.1 * base_delay)
        return float(min(base_delay + jitter, self.config.retry_max_delay))

    def _call_anthropic(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Make a call to Anthropic Claude API.

        Args:
            messages: List of message dicts with "role" and "content".
            system: Optional system prompt.
            **kwargs: Additional parameters for the API call.

        Returns:
            LLMResponse with the generated content.

        Raises:
            LLMRateLimitError: If rate limited.
            LLMTimeoutError: If request times out.
            LLMProviderError: If provider returns an error.
        """
        from anthropic import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )

        client = self._get_client()
        start_time = time.perf_counter()

        try:
            # Build request parameters
            request_params: dict[str, Any] = {
                "model": kwargs.get("model", self.config.model),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "messages": messages,
            }

            if system:
                request_params["system"] = system

            if self.config.temperature > 0:
                request_params["temperature"] = self.config.temperature

            response = client.messages.create(**request_params)  # type: ignore[union-attr]
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return LLMResponse(
                content=response.content[0].text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=response.model,
                latency_ms=elapsed_ms,
                provider="anthropic",
            )

        except RateLimitError as e:
            raise LLMRateLimitError(
                f"Rate limited by Anthropic: {e}",
                retry_after=getattr(e, "retry_after", None),
            ) from e
        except APITimeoutError as e:
            raise LLMTimeoutError(f"Anthropic request timed out: {e}") from e
        except APIConnectionError as e:
            raise LLMProviderError(f"Connection error to Anthropic: {e}") from e
        except APIStatusError as e:
            raise LLMProviderError(
                f"Anthropic API error: {e}",
                status_code=e.status_code,
            ) from e

    def _call_openai(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Make a call to OpenAI API.

        Args:
            messages: List of message dicts with "role" and "content".
            system: Optional system prompt.
            **kwargs: Additional parameters for the API call.

        Returns:
            LLMResponse with the generated content.

        Raises:
            LLMRateLimitError: If rate limited.
            LLMTimeoutError: If request times out.
            LLMProviderError: If provider returns an error.
        """
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )

        client = self._get_client()
        start_time = time.perf_counter()

        try:
            # Build messages with optional system prompt
            full_messages = []
            if system:
                full_messages.append({"role": "system", "content": system})
            full_messages.extend(messages)

            response = client.chat.completions.create(  # type: ignore[union-attr]
                model=kwargs.get("model", self.config.model),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                messages=full_messages,  # type: ignore[arg-type]
                temperature=self.config.temperature,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            content = response.choices[0].message.content or ""
            usage = response.usage

            return LLMResponse(
                content=content,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                model=response.model,
                latency_ms=elapsed_ms,
                provider="openai",
            )

        except RateLimitError as e:
            raise LLMRateLimitError(f"Rate limited by OpenAI: {e}") from e
        except APITimeoutError as e:
            raise LLMTimeoutError(f"OpenAI request timed out: {e}") from e
        except APIConnectionError as e:
            raise LLMProviderError(f"Connection error to OpenAI: {e}") from e
        except APIStatusError as e:
            raise LLMProviderError(
                f"OpenAI API error: {e}",
                status_code=e.status_code,
            ) from e

    def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from the LLM with retry logic.

        This method handles retries for transient errors (rate limits, timeouts)
        using exponential backoff with jitter.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            system: Optional system prompt (handled differently by providers).
            **kwargs: Additional parameters passed to the provider API.

        Returns:
            LLMResponse containing the generated content and metadata.

        Raises:
            LLMError: If all retry attempts fail.
            ConfigurationError: If required configuration is missing.

        Example:
            >>> response = client.complete([
            ...     {"role": "user", "content": "Explain quantum computing"}
            ... ])
            >>> print(f"Response ({response.total_tokens} tokens): {response.content}")
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if self.provider == "anthropic":
                    return self._call_anthropic(messages, system, **kwargs)
                else:
                    return self._call_openai(messages, system, **kwargs)

            except LLMRateLimitError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self._calculate_backoff(attempt, e.retry_after)
                    logger.warning(
                        f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.config.max_retries})"
                    )
                    time.sleep(delay)
                continue

            except LLMTimeoutError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Request timed out, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.config.max_retries})"
                    )
                    time.sleep(delay)
                continue

            except LLMProviderError as e:
                # Only retry on 5xx errors
                if e.status_code and 500 <= e.status_code < 600:
                    last_error = e
                    if attempt < self.config.max_retries:
                        delay = self._calculate_backoff(attempt)
                        logger.warning(f"Server error ({e.status_code}), retrying in {delay:.1f}s")
                        time.sleep(delay)
                    continue
                raise

        # All retries exhausted
        raise LLMError(
            f"All {self.config.max_retries + 1} attempts failed: {last_error}"
        ) from last_error
