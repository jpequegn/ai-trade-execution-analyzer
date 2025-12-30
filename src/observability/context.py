"""Trace context management for hierarchical observability.

This module provides a TraceContext class for managing hierarchical traces
with detailed metadata, timing, and error tracking.

Example:
    >>> from src.observability.context import TraceContext
    >>> with TraceContext("analysis_pipeline") as ctx:
    ...     with ctx.span("parse_fix") as span:
    ...         span.set_attribute("message_type", "ExecutionReport")
    ...         # do parsing work
    ...     with ctx.span("llm_call") as span:
    ...         span.set_attribute("model", "claude-3-sonnet")
    ...         # do LLM call
    >>> print(ctx.get_summary())
"""

from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

from pydantic import BaseModel, Field


class SpanStatus(str, Enum):
    """Status of a span execution."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class ErrorCategory(str, Enum):
    """Categories of errors for classification."""

    PARSE_ERROR = "parse_error"
    LLM_ERROR = "llm_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class SpanData(BaseModel):
    """Data for a single span in a trace.

    Attributes:
        span_id: Unique identifier for this span.
        parent_id: ID of the parent span (None for root).
        name: Name of the span.
        start_time: When the span started.
        end_time: When the span ended (None if still running).
        status: Current status of the span.
        attributes: Key-value attributes for the span.
        events: List of events that occurred during the span.
        error: Error information if status is ERROR.
    """

    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_id: str | None = None
    name: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None

    @property
    def duration_ms(self) -> float | None:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
        }


class TraceData(BaseModel):
    """Data for a complete trace.

    Attributes:
        trace_id: Unique identifier for the trace.
        name: Name of the trace.
        start_time: When the trace started.
        end_time: When the trace ended.
        spans: List of spans in the trace.
        metadata: Additional trace metadata.
        session_id: Optional session identifier.
        user_id: Optional user identifier.
        tags: Optional tags for the trace.
    """

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None
    spans: list[SpanData] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    user_id: str | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Get total trace duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def has_errors(self) -> bool:
        """Check if any span has an error."""
        return any(span.status == SpanStatus.ERROR for span in self.spans)

    @property
    def error_count(self) -> int:
        """Count spans with errors."""
        return sum(1 for span in self.spans if span.status == SpanStatus.ERROR)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "has_errors": self.has_errors,
            "error_count": self.error_count,
            "spans": [span.to_dict() for span in self.spans],
            "metadata": self.metadata,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tags": self.tags,
        }


@dataclass
class Span:
    """Active span context manager.

    Provides a context manager interface for creating and managing spans
    within a trace context.

    Attributes:
        data: The underlying span data.
        context: Reference to the parent TraceContext.
    """

    data: SpanData
    context: TraceContext

    def set_attribute(self, key: str, value: Any) -> Span:
        """Set an attribute on the span.

        Args:
            key: Attribute name.
            value: Attribute value.

        Returns:
            Self for chaining.
        """
        self.data.attributes[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> Span:
        """Set multiple attributes on the span.

        Args:
            attributes: Dictionary of attributes.

        Returns:
            Self for chaining.
        """
        self.data.attributes.update(attributes)
        return self

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Add an event to the span.

        Args:
            name: Event name.
            attributes: Optional event attributes.

        Returns:
            Self for chaining.
        """
        self.data.events.append(
            {
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "attributes": attributes or {},
            }
        )
        return self

    def record_error(
        self,
        exception: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Record an error on the span.

        Args:
            exception: The exception that occurred.
            category: Error category for classification.
            attributes: Additional error attributes.

        Returns:
            Self for chaining.
        """
        self.data.status = SpanStatus.ERROR
        self.data.error = {
            "type": type(exception).__name__,
            "message": str(exception),
            "category": category.value,
            "attributes": attributes or {},
            "timestamp": datetime.now().isoformat(),
        }
        # Also add as event
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "error.category": category.value,
            },
        )
        return self

    def set_ok(self) -> Span:
        """Mark the span as successful.

        Returns:
            Self for chaining.
        """
        self.data.status = SpanStatus.OK
        return self

    def end(self) -> None:
        """End the span."""
        self.data.end_time = datetime.now()
        if self.data.status == SpanStatus.UNSET:
            self.data.status = SpanStatus.OK

    def __enter__(self) -> Span:
        """Enter span context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit span context, recording any exception."""
        if exc_val is not None and isinstance(exc_val, Exception):
            category = _categorize_exception(exc_val)
            self.record_error(exc_val, category)
        self.end()


def _categorize_exception(exc: Exception) -> ErrorCategory:
    """Categorize an exception for error tracking.

    Args:
        exc: The exception to categorize.

    Returns:
        Appropriate error category.
    """
    exc_name = type(exc).__name__.lower()

    if "timeout" in exc_name:
        return ErrorCategory.TIMEOUT_ERROR
    if "rate" in exc_name or "limit" in exc_name:
        return ErrorCategory.RATE_LIMIT_ERROR
    if "auth" in exc_name or "permission" in exc_name:
        return ErrorCategory.AUTHENTICATION_ERROR
    if "network" in exc_name or "connection" in exc_name:
        return ErrorCategory.NETWORK_ERROR
    if "parse" in exc_name or "decode" in exc_name:
        return ErrorCategory.PARSE_ERROR
    if "validation" in exc_name or "invalid" in exc_name:
        return ErrorCategory.VALIDATION_ERROR
    if "config" in exc_name:
        return ErrorCategory.CONFIGURATION_ERROR
    if "api" in exc_name or "llm" in exc_name:
        return ErrorCategory.LLM_ERROR

    return ErrorCategory.UNKNOWN_ERROR


@dataclass
class TraceContext:
    """Context manager for hierarchical traces.

    Provides a fluent interface for creating traces with nested spans,
    attributes, and error tracking.

    Attributes:
        name: Name of the trace.
        session_id: Optional session identifier.
        user_id: Optional user identifier.
        tags: Optional tags for the trace.
        metadata: Optional metadata dictionary.

    Example:
        >>> with TraceContext("process_trade", session_id="sess123") as ctx:
        ...     ctx.set_attribute("trade_id", "T123")
        ...     with ctx.span("validate") as span:
        ...         span.set_attribute("valid", True)
        ...     with ctx.span("analyze") as span:
        ...         result = do_analysis()
        ...         span.set_attribute("score", result.score)
    """

    name: str
    session_id: str | None = None
    user_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    _trace_data: TraceData = field(init=False)
    _current_span_id: str | None = field(default=None, init=False)
    _span_stack: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize trace data."""
        self._trace_data = TraceData(
            name=self.name,
            session_id=self.session_id,
            user_id=self.user_id,
            tags=self.tags.copy(),
            metadata=self.metadata.copy(),
        )

    def set_attribute(self, key: str, value: Any) -> TraceContext:
        """Set a trace-level attribute.

        Args:
            key: Attribute name.
            value: Attribute value.

        Returns:
            Self for chaining.
        """
        self._trace_data.metadata[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> TraceContext:
        """Set multiple trace-level attributes.

        Args:
            attributes: Dictionary of attributes.

        Returns:
            Self for chaining.
        """
        self._trace_data.metadata.update(attributes)
        return self

    def add_tag(self, tag: str) -> TraceContext:
        """Add a tag to the trace.

        Args:
            tag: Tag to add.

        Returns:
            Self for chaining.
        """
        if tag not in self._trace_data.tags:
            self._trace_data.tags.append(tag)
        return self

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Create a new span within this trace.

        Args:
            name: Name of the span.
            attributes: Optional initial attributes.

        Yields:
            Span context manager.
        """
        # Determine parent
        parent_id = self._span_stack[-1] if self._span_stack else None

        # Create span data
        span_data = SpanData(
            name=name,
            parent_id=parent_id,
            attributes=attributes or {},
        )
        self._trace_data.spans.append(span_data)
        self._span_stack.append(span_data.span_id)

        # Create span wrapper
        span = Span(data=span_data, context=self)

        try:
            yield span
        except Exception as e:
            category = _categorize_exception(e)
            span.record_error(e, category)
            raise
        finally:
            span.end()
            self._span_stack.pop()

    @property
    def trace_id(self) -> str:
        """Get the trace ID."""
        return self._trace_data.trace_id

    @property
    def data(self) -> TraceData:
        """Get the trace data."""
        return self._trace_data

    def end(self) -> None:
        """End the trace."""
        self._trace_data.end_time = datetime.now()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the trace.

        Returns:
            Dictionary with trace summary.
        """
        return {
            "trace_id": self._trace_data.trace_id,
            "name": self._trace_data.name,
            "duration_ms": self._trace_data.duration_ms,
            "span_count": len(self._trace_data.spans),
            "has_errors": self._trace_data.has_errors,
            "error_count": self._trace_data.error_count,
            "tags": self._trace_data.tags,
        }

    def __enter__(self) -> TraceContext:
        """Enter trace context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit trace context."""
        self.end()


# Thread-local storage for active context
_active_context: threading.local = threading.local()


def get_current_context() -> TraceContext | None:
    """Get the current active trace context.

    Returns:
        The active TraceContext or None.
    """
    return getattr(_active_context, "context", None)


def set_current_context(context: TraceContext | None) -> None:
    """Set the current active trace context.

    Args:
        context: The TraceContext to set as active.
    """
    _active_context.context = context


@contextmanager
def trace(
    name: str,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[TraceContext, None, None]:
    """Create a new trace context.

    Convenience function for creating traces. Sets the trace as the
    active context for the duration of the context manager.

    Args:
        name: Name of the trace.
        session_id: Optional session identifier.
        user_id: Optional user identifier.
        tags: Optional tags.
        metadata: Optional metadata.

    Yields:
        TraceContext for the trace.

    Example:
        >>> with trace("analysis", session_id="sess1") as ctx:
        ...     with ctx.span("process") as span:
        ...         span.set_attribute("items", 5)
    """
    ctx = TraceContext(
        name=name,
        session_id=session_id,
        user_id=user_id,
        tags=tags or [],
        metadata=metadata or {},
    )

    previous_context = get_current_context()
    set_current_context(ctx)

    try:
        yield ctx
    finally:
        ctx.end()
        set_current_context(previous_context)
