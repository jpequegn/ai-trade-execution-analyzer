"""Tests for LLM client module."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.llm_client import (
    LLMClient,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
)
from src.config import ConfigurationError, LLMConfig


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_valid_response_creation(self) -> None:
        """Test creating a valid LLMResponse."""
        response = LLMResponse(
            content="Hello, world!",
            input_tokens=10,
            output_tokens=5,
            model="claude-3-5-sonnet-20241022",
            latency_ms=150.5,
            provider="anthropic",
        )
        assert response.content == "Hello, world!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.latency_ms == 150.5
        assert response.provider == "anthropic"

    def test_total_tokens_property(self) -> None:
        """Test total_tokens property."""
        response = LLMResponse(
            content="Test",
            input_tokens=100,
            output_tokens=50,
            model="test-model",
            latency_ms=100.0,
        )
        assert response.total_tokens == 150

    def test_serialization(self) -> None:
        """Test JSON serialization."""
        response = LLMResponse(
            content="Test",
            input_tokens=10,
            output_tokens=5,
            model="test-model",
            latency_ms=100.0,
        )
        json_str = response.model_dump_json()
        assert "Test" in json_str
        assert "test-model" in json_str


class TestLLMClientInitialization:
    """Tests for LLMClient initialization."""

    def test_init_with_config(self) -> None:
        """Test initialization with explicit config."""
        config = LLMConfig(
            provider="anthropic",
            api_key="test-key",
            model="claude-3-haiku-20240307",
        )
        client = LLMClient(config=config)
        assert client.provider == "anthropic"
        assert client.config.model == "claude-3-haiku-20240307"

    def test_init_openai_provider(self) -> None:
        """Test initialization with OpenAI provider."""
        config = LLMConfig(
            provider="openai",
            api_key="test-key",
            model="gpt-4",
        )
        client = LLMClient(config=config)
        assert client.provider == "openai"

    def test_validation_on_get_client(self) -> None:
        """Test that validation happens when getting client."""
        config = LLMConfig(provider="anthropic", api_key="")
        client = LLMClient(config=config)

        with pytest.raises(ConfigurationError):
            client._get_client()


class TestLLMClientBackoff:
    """Tests for backoff calculation."""

    def test_backoff_exponential_growth(self) -> None:
        """Test that backoff grows exponentially."""
        config = LLMConfig(
            api_key="test",
            retry_base_delay=1.0,
            retry_max_delay=30.0,
        )
        client = LLMClient(config=config)

        delay0 = client._calculate_backoff(0)
        delay1 = client._calculate_backoff(1)
        delay2 = client._calculate_backoff(2)

        # Should grow roughly exponentially (with jitter)
        assert delay0 < delay1 < delay2
        assert delay0 < 2.0  # Base delay + jitter
        assert delay1 < 4.0  # 2x base + jitter
        assert delay2 < 8.0  # 4x base + jitter

    def test_backoff_respects_max_delay(self) -> None:
        """Test that backoff respects max delay."""
        config = LLMConfig(
            api_key="test",
            retry_base_delay=1.0,
            retry_max_delay=5.0,
        )
        client = LLMClient(config=config)

        delay = client._calculate_backoff(10)  # Very high attempt
        assert delay <= 5.0

    def test_backoff_uses_retry_after(self) -> None:
        """Test that retry_after from provider is respected."""
        config = LLMConfig(
            api_key="test",
            retry_max_delay=30.0,
        )
        client = LLMClient(config=config)

        delay = client._calculate_backoff(0, retry_after=10.0)
        assert delay == 10.0

    def test_backoff_caps_retry_after(self) -> None:
        """Test that retry_after is capped at max_delay."""
        config = LLMConfig(
            api_key="test",
            retry_max_delay=5.0,
        )
        client = LLMClient(config=config)

        delay = client._calculate_backoff(0, retry_after=100.0)
        assert delay == 5.0


class TestLLMClientAnthropicCalls:
    """Tests for Anthropic API calls."""

    @patch("anthropic.Anthropic")
    def test_successful_anthropic_call(self, mock_anthropic_class: MagicMock) -> None:
        """Test successful Anthropic API call."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello!")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_client.messages.create.return_value = mock_response

        # Execute
        config = LLMConfig(provider="anthropic", api_key="test-key")
        client = LLMClient(config=config)
        response = client.complete([{"role": "user", "content": "Hi"}])

        # Verify
        assert response.content == "Hello!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.provider == "anthropic"
        mock_client.messages.create.assert_called_once()

    @patch("anthropic.Anthropic")
    def test_anthropic_with_system_prompt(self, mock_anthropic_class: MagicMock) -> None:
        """Test Anthropic call with system prompt."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 10
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_client.messages.create.return_value = mock_response

        config = LLMConfig(provider="anthropic", api_key="test-key")
        client = LLMClient(config=config)
        client.complete(
            [{"role": "user", "content": "Hi"}],
            system="You are a helpful assistant",
        )

        # Verify system was passed
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant"


class TestLLMClientOpenAICalls:
    """Tests for OpenAI API calls."""

    @patch("openai.OpenAI")
    def test_successful_openai_call(self, mock_openai_class: MagicMock) -> None:
        """Test successful OpenAI API call."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        config = LLMConfig(provider="openai", api_key="test-key", model="gpt-4")
        client = LLMClient(config=config)
        response = client.complete([{"role": "user", "content": "Hi"}])

        assert response.content == "Hello!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.provider == "openai"

    @patch("openai.OpenAI")
    def test_openai_with_system_prompt(self, mock_openai_class: MagicMock) -> None:
        """Test OpenAI call with system prompt."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        config = LLMConfig(provider="openai", api_key="test-key")
        client = LLMClient(config=config)
        client.complete(
            [{"role": "user", "content": "Hi"}],
            system="You are a helpful assistant",
        )

        # Verify system message was prepended
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"


class TestLLMClientRetryLogic:
    """Tests for retry logic."""

    @patch("anthropic.Anthropic")
    @patch("time.sleep")
    def test_retry_on_rate_limit(
        self, mock_sleep: MagicMock, mock_anthropic_class: MagicMock
    ) -> None:
        """Test retry behavior on rate limit."""
        from anthropic import RateLimitError

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First call raises rate limit, second succeeds
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Success")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.model = "claude-3-5-sonnet-20241022"

        rate_limit_error = RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_client.messages.create.side_effect = [rate_limit_error, mock_response]

        config = LLMConfig(provider="anthropic", api_key="test-key", max_retries=2)
        client = LLMClient(config=config)
        response = client.complete([{"role": "user", "content": "Hi"}])

        assert response.content == "Success"
        assert mock_client.messages.create.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("anthropic.Anthropic")
    @patch("time.sleep")
    def test_retry_exhausted(self, mock_sleep: MagicMock, mock_anthropic_class: MagicMock) -> None:
        """Test behavior when all retries are exhausted."""
        from anthropic import RateLimitError

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        rate_limit_error = RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_client.messages.create.side_effect = rate_limit_error

        config = LLMConfig(provider="anthropic", api_key="test-key", max_retries=2)
        client = LLMClient(config=config)

        with pytest.raises(LLMError) as exc_info:
            client.complete([{"role": "user", "content": "Hi"}])

        assert "All 3 attempts failed" in str(exc_info.value)
        assert mock_client.messages.create.call_count == 3

    @patch("anthropic.Anthropic")
    def test_no_retry_on_client_error(self, mock_anthropic_class: MagicMock) -> None:
        """Test no retry on 4xx errors."""
        from anthropic import APIStatusError

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        client_error = APIStatusError(
            message="Bad request",
            response=MagicMock(status_code=400),
            body=None,
        )
        mock_client.messages.create.side_effect = client_error

        config = LLMConfig(provider="anthropic", api_key="test-key", max_retries=2)
        client = LLMClient(config=config)

        with pytest.raises(LLMProviderError):
            client.complete([{"role": "user", "content": "Hi"}])

        # Should only try once for 4xx errors
        assert mock_client.messages.create.call_count == 1


class TestLLMErrors:
    """Tests for LLM error classes."""

    def test_llm_rate_limit_error(self) -> None:
        """Test LLMRateLimitError."""
        error = LLMRateLimitError("Rate limited", retry_after=5.0)
        assert str(error) == "Rate limited"
        assert error.retry_after == 5.0

    def test_llm_timeout_error(self) -> None:
        """Test LLMTimeoutError."""
        error = LLMTimeoutError("Request timed out")
        assert str(error) == "Request timed out"

    def test_llm_provider_error(self) -> None:
        """Test LLMProviderError."""
        error = LLMProviderError("Server error", status_code=500)
        assert str(error) == "Server error"
        assert error.status_code == 500
