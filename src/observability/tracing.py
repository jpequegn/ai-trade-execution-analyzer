"""Langfuse tracing integration for LLM observability.

This module provides decorators and utilities for tracing LLM calls
and other operations using Langfuse.

Example:
    >>> from src.observability.tracing import get_tracer, traced
    >>> tracer = get_tracer()
    >>>
    >>> @traced(name="my_function")
    >>> def my_function():
    ...     return "result"
"""

from __future__ import annotations

import functools
import logging
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from langfuse import Langfuse
    from langfuse.client import StatefulTraceClient

from src.config import AppConfig, LangfuseConfig, get_config

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class TracingDisabled:
    """Dummy tracer when Langfuse is disabled.

    This class provides no-op implementations of tracing methods,
    allowing code to use tracing decorators without errors when
    Langfuse is not configured.
    """

    def trace(self, **kwargs: Any) -> DummyTrace:
        """Create a no-op trace."""
        return DummyTrace()

    def span(self, **kwargs: Any) -> DummySpan:
        """Create a no-op span."""
        return DummySpan()

    def generation(self, **kwargs: Any) -> DummyGeneration:
        """Create a no-op generation."""
        return DummyGeneration()

    def flush(self) -> None:
        """No-op flush."""
        pass

    def shutdown(self) -> None:
        """No-op shutdown."""
        pass


class DummyTrace:
    """Dummy trace object when tracing is disabled."""

    def span(self, **kwargs: Any) -> DummySpan:
        """Create a no-op span."""
        return DummySpan()

    def generation(self, **kwargs: Any) -> DummyGeneration:
        """Create a no-op generation."""
        return DummyGeneration()

    def update(self, **kwargs: Any) -> DummyTrace:
        """No-op update."""
        return self

    def end(self, **kwargs: Any) -> None:
        """No-op end."""
        pass


class DummySpan:
    """Dummy span object when tracing is disabled."""

    def span(self, **kwargs: Any) -> DummySpan:
        """Create a no-op nested span."""
        return DummySpan()

    def generation(self, **kwargs: Any) -> DummyGeneration:
        """Create a no-op generation."""
        return DummyGeneration()

    def update(self, **kwargs: Any) -> DummySpan:
        """No-op update."""
        return self

    def end(self, **kwargs: Any) -> None:
        """No-op end."""
        pass


class DummyGeneration:
    """Dummy generation object when tracing is disabled."""

    def update(self, **kwargs: Any) -> DummyGeneration:
        """No-op update."""
        return self

    def end(self, **kwargs: Any) -> None:
        """No-op end."""
        pass


class Tracer:
    """Wrapper for Langfuse client with convenience methods.

    This class provides a simplified interface for creating traces,
    spans, and generations, handling the case where Langfuse is disabled.

    Attributes:
        config: Langfuse configuration.
        enabled: Whether tracing is enabled.
    """

    def __init__(
        self,
        config: LangfuseConfig | None = None,
        app_config: AppConfig | None = None,
    ) -> None:
        """Initialize the tracer.

        Args:
            config: Optional Langfuse configuration.
            app_config: Optional full app configuration.
        """
        if app_config is not None:
            self.config = app_config.langfuse
        elif config is not None:
            self.config = config
        else:
            self.config = get_config().langfuse

        self.enabled = self.config.enabled
        self._client: Langfuse | TracingDisabled | None = None

    def _get_client(self) -> Langfuse | TracingDisabled:
        """Get or create the Langfuse client.

        Returns:
            Langfuse client or TracingDisabled dummy.
        """
        if self._client is not None:
            return self._client

        if not self.enabled:
            self._client = TracingDisabled()
            logger.debug("Langfuse tracing disabled")
            return self._client

        if not self.config.public_key or not self.config.secret_key:
            logger.warning(
                "Langfuse keys not configured, tracing disabled. "
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY."
            )
            self._client = TracingDisabled()
            return self._client

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.config.public_key,
                secret_key=self.config.secret_key,
                host=self.config.host,
                release=self.config.release,
                debug=self.config.debug,
            )
            logger.info(f"Langfuse tracing initialized (host: {self.config.host})")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse, tracing disabled: {e}")
            self._client = TracingDisabled()

        return self._client

    def trace(
        self,
        name: str,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> StatefulTraceClient | DummyTrace:
        """Create a new trace.

        Args:
            name: Name of the trace.
            session_id: Optional session identifier.
            user_id: Optional user identifier.
            tags: Optional list of tags.
            metadata: Optional metadata dictionary.
            **kwargs: Additional parameters for Langfuse.

        Returns:
            Trace object for adding spans/generations.
        """
        client = self._get_client()
        return client.trace(  # type: ignore[union-attr]
            name=name,
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
            metadata=metadata or {},
            **kwargs,
        )

    def generation(
        self,
        name: str,
        model: str,
        input: Any = None,
        output: Any = None,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DummyGeneration:
        """Create a standalone generation (for LLM calls).

        Args:
            name: Name of the generation.
            model: Model identifier.
            input: Input to the LLM.
            output: Output from the LLM.
            usage: Token usage dict with input/output/total.
            metadata: Optional metadata.
            **kwargs: Additional parameters.

        Returns:
            Generation object.
        """
        client = self._get_client()
        return client.generation(  # type: ignore[union-attr]
            name=name,
            model=model,
            input=input,
            output=output,
            usage=usage,
            metadata=metadata or {},
            **kwargs,
        )

    def flush(self) -> None:
        """Flush pending events to Langfuse."""
        client = self._get_client()
        client.flush()

    def shutdown(self) -> None:
        """Shutdown the Langfuse client."""
        if self._client is not None:
            self._client.shutdown()
            self._client = None


# Global tracer instance
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get the global tracer instance.

    Returns:
        Tracer instance.
    """
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def reset_tracer() -> None:
    """Reset the global tracer (useful for testing)."""
    global _tracer
    if _tracer is not None:
        _tracer.shutdown()
    _tracer = None


def generate_trace_id() -> str:
    """Generate a unique trace ID.

    Returns:
        UUID string for trace identification.
    """
    return str(uuid.uuid4())


def traced(
    name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to trace a function execution.

    Creates a span for the decorated function, capturing input/output
    and timing information.

    Args:
        name: Optional name for the span (defaults to function name).
        tags: Optional tags for the span.
        metadata: Optional metadata for the span.

    Returns:
        Decorator function.

    Example:
        >>> @traced(name="process_trade", tags=["trading"])
        >>> def process_trade(trade_id: str) -> dict:
        ...     return {"status": "processed"}
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            span_name = name or func.__name__

            trace = tracer.trace(
                name=span_name,
                tags=tags or [],
                metadata={
                    **(metadata or {}),
                    "function": func.__name__,
                    "module": func.__module__,
                },
            )

            try:
                result = func(*args, **kwargs)
                trace.update(output=str(result)[:1000])  # Truncate output
                return result
            except Exception as e:
                trace.update(metadata={"error": str(e), "error_type": type(e).__name__})
                raise
            finally:
                trace.end()

        return wrapper

    return decorator


@contextmanager
def trace_context(
    name: str,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Context manager for creating traces.

    Provides a trace context that automatically ends when exiting.

    Args:
        name: Name of the trace.
        session_id: Optional session identifier.
        user_id: Optional user identifier.
        tags: Optional tags.
        metadata: Optional metadata.

    Yields:
        Trace object for adding spans/generations.

    Example:
        >>> with trace_context("analysis_session", session_id="sess123") as trace:
        ...     span = trace.span(name="parse_data")
        ...     # do work
        ...     span.end()
    """
    tracer = get_tracer()
    trace = tracer.trace(
        name=name,
        session_id=session_id,
        user_id=user_id,
        tags=tags or [],
        metadata=metadata or {},
    )

    try:
        yield trace
    except Exception as e:
        trace.update(metadata={"error": str(e), "error_type": type(e).__name__})
        raise
    finally:
        trace.end()
        tracer.flush()
