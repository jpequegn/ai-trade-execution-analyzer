"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from src.config import (
    AppConfig,
    ConfigurationError,
    LangfuseConfig,
    LLMConfig,
    get_config,
    reset_config,
)


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == "anthropic"
        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.max_tokens == 4096
        assert config.temperature == 0.0
        assert config.timeout == 60.0
        assert config.max_retries == 3

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4-turbo",
            api_key="test-key",
            max_tokens=2048,
            temperature=0.7,
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4-turbo"
        assert config.api_key == "test-key"
        assert config.max_tokens == 2048
        assert config.temperature == 0.7

    def test_validate_missing_anthropic_key(self) -> None:
        """Test validation fails when Anthropic key is missing."""
        config = LLMConfig(provider="anthropic", api_key="")
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "ANTHROPIC_API_KEY" in exc_info.value.missing_vars

    def test_validate_missing_openai_key(self) -> None:
        """Test validation fails when OpenAI key is missing."""
        config = LLMConfig(provider="openai", api_key="")
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "OPENAI_API_KEY" in exc_info.value.missing_vars

    def test_validate_with_key(self) -> None:
        """Test validation passes with API key."""
        config = LLMConfig(api_key="sk-test-key")
        config.validate()  # Should not raise


class TestLangfuseConfig:
    """Tests for LangfuseConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LangfuseConfig()
        assert config.enabled is True
        assert config.host == "https://cloud.langfuse.com"
        assert config.debug is False

    def test_validate_disabled(self) -> None:
        """Test validation passes when disabled."""
        config = LangfuseConfig(enabled=False)
        config.validate()  # Should not raise

    def test_validate_missing_keys(self) -> None:
        """Test validation fails when keys are missing."""
        config = LangfuseConfig(enabled=True, public_key="", secret_key="")
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "LANGFUSE_PUBLIC_KEY" in exc_info.value.missing_vars
        assert "LANGFUSE_SECRET_KEY" in exc_info.value.missing_vars

    def test_validate_with_keys(self) -> None:
        """Test validation passes with keys."""
        config = LangfuseConfig(
            enabled=True,
            public_key="pk-test",
            secret_key="sk-test",
        )
        config.validate()  # Should not raise


class TestAppConfig:
    """Tests for AppConfig."""

    def test_from_env_defaults(self) -> None:
        """Test loading from environment with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            reset_config()
            config = AppConfig.from_env()
            assert config.llm.provider == "anthropic"
            assert config.langfuse.enabled is True
            assert config.log_level == "INFO"

    def test_from_env_custom_values(self) -> None:
        """Test loading from environment with custom values."""
        env = {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "LLM_MODEL": "gpt-4",
            "LLM_MAX_TOKENS": "2048",
            "LANGFUSE_ENABLED": "false",
            "LOG_LEVEL": "DEBUG",
            "DEBUG": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config()
            config = AppConfig.from_env()
            assert config.llm.provider == "openai"
            assert config.llm.api_key == "sk-test"
            assert config.llm.model == "gpt-4"
            assert config.llm.max_tokens == 2048
            assert config.langfuse.enabled is False
            assert config.log_level == "DEBUG"
            assert config.debug is True

    def test_from_env_anthropic_key(self) -> None:
        """Test loading Anthropic key from environment."""
        env = {
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config()
            config = AppConfig.from_env()
            assert config.llm.api_key == "sk-ant-test"

    def test_validate_all_sections(self) -> None:
        """Test that validate checks all sections."""
        config = AppConfig(
            llm=LLMConfig(api_key=""),
            langfuse=LangfuseConfig(enabled=False),
        )
        with pytest.raises(ConfigurationError):
            config.validate()


class TestGetConfig:
    """Tests for get_config function."""

    def test_returns_same_instance(self) -> None:
        """Test that get_config returns cached instance."""
        reset_config()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=True):
            config1 = get_config()
            config2 = get_config()
            assert config1 is config2

    def test_reset_clears_cache(self) -> None:
        """Test that reset_config clears the cache."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test1"}, clear=True):
            config1 = get_config()

        reset_config()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test2"}, clear=True):
            config2 = get_config()
            assert config1 is not config2
