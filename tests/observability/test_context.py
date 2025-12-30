"""Tests for trace context module."""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from src.observability.context import (
    ErrorCategory,
    Span,
    SpanData,
    SpanStatus,
    TraceContext,
    TraceData,
    _categorize_exception,
    get_current_context,
    set_current_context,
    trace,
)


class TestSpanStatus:
    """Tests for SpanStatus enum."""

    def test_status_values(self) -> None:
        """Test status value strings."""
        assert SpanStatus.UNSET.value == "unset"
        assert SpanStatus.OK.value == "ok"
        assert SpanStatus.ERROR.value == "error"


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_category_values(self) -> None:
        """Test category value strings."""
        assert ErrorCategory.PARSE_ERROR.value == "parse_error"
        assert ErrorCategory.LLM_ERROR.value == "llm_error"
        assert ErrorCategory.TIMEOUT_ERROR.value == "timeout_error"
        assert ErrorCategory.RATE_LIMIT_ERROR.value == "rate_limit_error"
        assert ErrorCategory.NETWORK_ERROR.value == "network_error"
        assert ErrorCategory.UNKNOWN_ERROR.value == "unknown_error"


class TestSpanData:
    """Tests for SpanData model."""

    def test_default_values(self) -> None:
        """Test default span data values."""
        span = SpanData(name="test_span")

        assert span.name == "test_span"
        assert span.span_id  # Should be auto-generated
        assert span.parent_id is None
        assert span.status == SpanStatus.UNSET
        assert span.attributes == {}
        assert span.events == []
        assert span.error is None

    def test_duration_calculation(self) -> None:
        """Test duration calculation."""
        span = SpanData(name="test")
        assert span.duration_ms is None  # Not ended

        span.end_time = datetime.now()
        duration = span.duration_ms
        assert duration is not None
        assert duration >= 0

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        span = SpanData(
            name="test",
            attributes={"key": "value"},
        )
        span.end_time = datetime.now()

        result = span.to_dict()

        assert result["name"] == "test"
        assert result["attributes"]["key"] == "value"
        assert result["status"] == "unset"
        assert result["duration_ms"] is not None


class TestTraceData:
    """Tests for TraceData model."""

    def test_default_values(self) -> None:
        """Test default trace data values."""
        trace_data = TraceData(name="test_trace")

        assert trace_data.name == "test_trace"
        assert trace_data.trace_id  # Should be auto-generated
        assert trace_data.spans == []
        assert trace_data.metadata == {}
        assert trace_data.tags == []
        assert not trace_data.has_errors
        assert trace_data.error_count == 0

    def test_has_errors(self) -> None:
        """Test error detection."""
        trace_data = TraceData(name="test")
        assert not trace_data.has_errors

        # Add error span
        error_span = SpanData(name="error_span", status=SpanStatus.ERROR)
        trace_data.spans.append(error_span)

        assert trace_data.has_errors
        assert trace_data.error_count == 1

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        trace_data = TraceData(
            name="test",
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
        )
        trace_data.end_time = datetime.now()

        result = trace_data.to_dict()

        assert result["name"] == "test"
        assert result["tags"] == ["tag1", "tag2"]
        assert result["metadata"]["key"] == "value"
        assert result["has_errors"] is False
        assert result["error_count"] == 0


class TestSpan:
    """Tests for Span class."""

    def test_set_attribute(self) -> None:
        """Test setting attributes."""
        ctx = TraceContext(name="test")
        span_data = SpanData(name="test_span")
        span = Span(data=span_data, context=ctx)

        result = span.set_attribute("key", "value")

        assert result is span  # Chaining
        assert span.data.attributes["key"] == "value"

    def test_set_attributes(self) -> None:
        """Test setting multiple attributes."""
        ctx = TraceContext(name="test")
        span_data = SpanData(name="test_span")
        span = Span(data=span_data, context=ctx)

        span.set_attributes({"key1": "value1", "key2": "value2"})

        assert span.data.attributes["key1"] == "value1"
        assert span.data.attributes["key2"] == "value2"

    def test_add_event(self) -> None:
        """Test adding events."""
        ctx = TraceContext(name="test")
        span_data = SpanData(name="test_span")
        span = Span(data=span_data, context=ctx)

        span.add_event("test_event", {"attr": "value"})

        assert len(span.data.events) == 1
        assert span.data.events[0]["name"] == "test_event"
        assert span.data.events[0]["attributes"]["attr"] == "value"

    def test_record_error(self) -> None:
        """Test recording errors."""
        ctx = TraceContext(name="test")
        span_data = SpanData(name="test_span")
        span = Span(data=span_data, context=ctx)

        exc = ValueError("test error")
        span.record_error(exc, ErrorCategory.VALIDATION_ERROR)

        assert span.data.status == SpanStatus.ERROR
        assert span.data.error is not None
        assert span.data.error["type"] == "ValueError"
        assert span.data.error["message"] == "test error"
        assert span.data.error["category"] == "validation_error"

    def test_set_ok(self) -> None:
        """Test setting OK status."""
        ctx = TraceContext(name="test")
        span_data = SpanData(name="test_span")
        span = Span(data=span_data, context=ctx)

        span.set_ok()

        assert span.data.status == SpanStatus.OK

    def test_end(self) -> None:
        """Test ending span."""
        ctx = TraceContext(name="test")
        span_data = SpanData(name="test_span")
        span = Span(data=span_data, context=ctx)

        span.end()

        assert span.data.end_time is not None
        assert span.data.status == SpanStatus.OK  # Default on end

    def test_context_manager(self) -> None:
        """Test span as context manager."""
        ctx = TraceContext(name="test")
        span_data = SpanData(name="test_span")
        span = Span(data=span_data, context=ctx)

        with span as s:
            s.set_attribute("key", "value")

        assert span.data.end_time is not None
        assert span.data.status == SpanStatus.OK

    def test_context_manager_with_exception(self) -> None:
        """Test span context manager handles exceptions."""
        ctx = TraceContext(name="test")
        span_data = SpanData(name="test_span")
        span = Span(data=span_data, context=ctx)

        with pytest.raises(ValueError), span:
            raise ValueError("test error")

        assert span.data.status == SpanStatus.ERROR
        assert span.data.error is not None


class TestCategorizeException:
    """Tests for exception categorization."""

    def test_timeout_error(self) -> None:
        """Test timeout error categorization."""

        class TimeoutError(Exception):
            pass

        assert _categorize_exception(TimeoutError()) == ErrorCategory.TIMEOUT_ERROR

    def test_rate_limit_error(self) -> None:
        """Test rate limit error categorization."""

        class RateLimitExceeded(Exception):
            pass

        assert _categorize_exception(RateLimitExceeded()) == ErrorCategory.RATE_LIMIT_ERROR

    def test_parse_error(self) -> None:
        """Test parse error categorization."""

        class ParseError(Exception):
            pass

        assert _categorize_exception(ParseError()) == ErrorCategory.PARSE_ERROR

    def test_network_error(self) -> None:
        """Test network error categorization."""

        class NetworkError(Exception):
            pass

        assert _categorize_exception(NetworkError()) == ErrorCategory.NETWORK_ERROR

    def test_unknown_error(self) -> None:
        """Test unknown error categorization."""

        class SomeOtherError(Exception):
            pass

        assert _categorize_exception(SomeOtherError()) == ErrorCategory.UNKNOWN_ERROR


class TestTraceContext:
    """Tests for TraceContext class."""

    def test_basic_initialization(self) -> None:
        """Test basic trace context initialization."""
        ctx = TraceContext(name="test_trace")

        assert ctx.name == "test_trace"
        assert ctx._trace_data.name == "test_trace"
        assert ctx.trace_id  # Should be set

    def test_with_metadata(self) -> None:
        """Test trace context with metadata."""
        ctx = TraceContext(
            name="test",
            session_id="sess123",
            user_id="user456",
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
        )

        assert ctx._trace_data.session_id == "sess123"
        assert ctx._trace_data.user_id == "user456"
        assert "tag1" in ctx._trace_data.tags
        assert ctx._trace_data.metadata["key"] == "value"

    def test_set_attribute(self) -> None:
        """Test setting trace-level attributes."""
        ctx = TraceContext(name="test")

        result = ctx.set_attribute("key", "value")

        assert result is ctx  # Chaining
        assert ctx._trace_data.metadata["key"] == "value"

    def test_add_tag(self) -> None:
        """Test adding tags."""
        ctx = TraceContext(name="test")

        ctx.add_tag("new_tag")

        assert "new_tag" in ctx._trace_data.tags

    def test_add_tag_no_duplicates(self) -> None:
        """Test tags are not duplicated."""
        ctx = TraceContext(name="test", tags=["tag1"])

        ctx.add_tag("tag1")

        assert ctx._trace_data.tags.count("tag1") == 1

    def test_span_creation(self) -> None:
        """Test creating spans."""
        ctx = TraceContext(name="test")

        with ctx.span("test_span") as span:
            span.set_attribute("key", "value")

        assert len(ctx._trace_data.spans) == 1
        assert ctx._trace_data.spans[0].name == "test_span"
        assert ctx._trace_data.spans[0].attributes["key"] == "value"

    def test_nested_spans(self) -> None:
        """Test nested span creation."""
        ctx = TraceContext(name="test")

        with ctx.span("parent") as parent:
            parent.set_attribute("level", "parent")
            with ctx.span("child") as child:
                child.set_attribute("level", "child")

        assert len(ctx._trace_data.spans) == 2
        parent_span = ctx._trace_data.spans[0]
        child_span = ctx._trace_data.spans[1]

        assert parent_span.parent_id is None
        assert child_span.parent_id == parent_span.span_id

    def test_get_summary(self) -> None:
        """Test getting trace summary."""
        ctx = TraceContext(name="test", tags=["tag1"])

        with ctx.span("span1"):
            pass

        ctx.end()
        summary = ctx.get_summary()

        assert summary["name"] == "test"
        assert summary["span_count"] == 1
        assert summary["tags"] == ["tag1"]
        assert summary["has_errors"] is False

    def test_context_manager(self) -> None:
        """Test trace context as context manager."""
        with TraceContext(name="test") as ctx:
            ctx.set_attribute("key", "value")

        assert ctx._trace_data.end_time is not None

    def test_span_with_exception(self) -> None:
        """Test span handles exceptions correctly."""
        ctx = TraceContext(name="test")

        with pytest.raises(ValueError), ctx.span("failing_span"):
            raise ValueError("test error")

        assert ctx._trace_data.spans[0].status == SpanStatus.ERROR
        assert ctx._trace_data.has_errors


class TestGlobalContext:
    """Tests for global context management."""

    def test_get_set_current_context(self) -> None:
        """Test getting and setting current context."""
        # Initially None
        set_current_context(None)
        assert get_current_context() is None

        # Set context
        ctx = TraceContext(name="test")
        set_current_context(ctx)
        assert get_current_context() is ctx

        # Clean up
        set_current_context(None)

    def test_trace_function(self) -> None:
        """Test trace convenience function."""
        with trace("test_trace", session_id="sess1", tags=["tag1"]) as ctx:
            assert get_current_context() is ctx
            ctx.set_attribute("key", "value")

        # Context should be cleared
        assert get_current_context() is None

    def test_nested_trace_preserves_previous(self) -> None:
        """Test nested traces preserve previous context."""
        with trace("outer") as outer:
            assert get_current_context() is outer
            with trace("inner") as inner:
                assert get_current_context() is inner
            # Should restore outer
            assert get_current_context() is outer

        assert get_current_context() is None


class TestSpanTiming:
    """Tests for span timing accuracy."""

    def test_span_duration(self) -> None:
        """Test span records reasonable duration."""
        ctx = TraceContext(name="test")

        with ctx.span("timed_span") as span:
            time.sleep(0.01)  # 10ms

        duration = span.data.duration_ms
        assert duration is not None
        assert duration >= 10  # At least 10ms
        assert duration < 100  # But not too long

    def test_nested_span_timing(self) -> None:
        """Test nested span timing is accurate."""
        ctx = TraceContext(name="test")

        with ctx.span("parent"):
            time.sleep(0.01)
            with ctx.span("child"):
                time.sleep(0.01)

        parent_span = ctx._trace_data.spans[0]
        child_span = ctx._trace_data.spans[1]

        assert parent_span.duration_ms is not None
        assert child_span.duration_ms is not None
        # Parent should be longer than child
        assert parent_span.duration_ms > child_span.duration_ms
