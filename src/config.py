"""Centralized configuration management.

This module provides configuration classes for the application,
loading values from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""

    def __init__(self, message: str, missing_vars: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_vars = missing_vars or []


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM client.

    Attributes:
        provider: LLM provider to use ("anthropic" or "openai").
        model: Model identifier to use.
        api_key: API key for the provider.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0.0-1.0).
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        retry_base_delay: Base delay for exponential backoff (seconds).
        retry_max_delay: Maximum delay between retries (seconds).
    """

    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If required configuration is missing.
        """
        missing = []
        if not self.api_key:
            key_name = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
            missing.append(key_name)

        if missing:
            raise ConfigurationError(
                f"Missing required configuration: {', '.join(missing)}",
                missing_vars=missing,
            )


@dataclass(frozen=True)
class LangfuseConfig:
    """Configuration for Langfuse observability.

    Attributes:
        enabled: Whether Langfuse tracing is enabled.
        public_key: Langfuse public key.
        secret_key: Langfuse secret key.
        host: Langfuse host URL.
        release: Application release/version tag.
        debug: Enable debug mode for verbose logging.
    """

    enabled: bool = True
    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    release: str = "0.1.0"
    debug: bool = False

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If Langfuse is enabled but keys are missing.
        """
        if not self.enabled:
            return

        missing = []
        if not self.public_key:
            missing.append("LANGFUSE_PUBLIC_KEY")
        if not self.secret_key:
            missing.append("LANGFUSE_SECRET_KEY")

        if missing:
            raise ConfigurationError(
                f"Langfuse enabled but missing keys: {', '.join(missing)}. "
                "Set LANGFUSE_ENABLED=false to disable.",
                missing_vars=missing,
            )


@dataclass
class AppConfig:
    """Main application configuration.

    Attributes:
        llm: LLM client configuration.
        langfuse: Langfuse observability configuration.
        log_level: Logging level.
        debug: Debug mode flag.
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    log_level: str = "INFO"
    debug: bool = False

    @classmethod
    def from_env(cls) -> AppConfig:
        """Create configuration from environment variables.

        Returns:
            AppConfig instance populated from environment.
        """
        provider_env = os.getenv("LLM_PROVIDER", "anthropic")
        provider: Literal["anthropic", "openai"] = (
            "anthropic" if provider_env != "openai" else "openai"
        )

        # Get appropriate API key based on provider
        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")

        llm_config = LLMConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022"),
            api_key=api_key,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            timeout=float(os.getenv("LLM_TIMEOUT", "60.0")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            retry_base_delay=float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0")),
            retry_max_delay=float(os.getenv("LLM_RETRY_MAX_DELAY", "30.0")),
        )

        langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
        langfuse_config = LangfuseConfig(
            enabled=langfuse_enabled,
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            release=os.getenv("APP_VERSION", "0.1.0"),
            debug=os.getenv("LANGFUSE_DEBUG", "false").lower() == "true",
        )

        return cls(
            llm=llm_config,
            langfuse=langfuse_config,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )

    def validate(self) -> None:
        """Validate all configuration sections.

        Raises:
            ConfigurationError: If any required configuration is missing.
        """
        self.llm.validate()
        self.langfuse.validate()


# Global configuration instance (lazy loaded)
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the global application configuration.

    Returns:
        AppConfig instance loaded from environment.
    """
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
