"""Tests for observability/tracing module."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import LangfuseConfig
from src.observability.tracing import (
    DummyGeneration,
    DummySpan,
    DummyTrace,
    Tracer,
    TracingDisabled,
    generate_trace_id,
    get_tracer,
    reset_tracer,
    trace_context,
    traced,
)


class TestTracingDisabled:
    """Tests for TracingDisabled dummy class."""

    def test_trace_returns_dummy(self) -> None:
        """Test that trace returns DummyTrace."""
        disabled = TracingDisabled()
        trace = disabled.trace(name="test")
        assert isinstance(trace, DummyTrace)

    def test_span_returns_dummy(self) -> None:
        """Test that span returns DummySpan."""
        disabled = TracingDisabled()
        span = disabled.span(name="test")
        assert isinstance(span, DummySpan)

    def test_generation_returns_dummy(self) -> None:
        """Test that generation returns DummyGeneration."""
        disabled = TracingDisabled()
        gen = disabled.generation(name="test")
        assert isinstance(gen, DummyGeneration)

    def test_flush_no_op(self) -> None:
        """Test that flush does nothing."""
        disabled = TracingDisabled()
        disabled.flush()  # Should not raise

    def test_shutdown_no_op(self) -> None:
        """Test that shutdown does nothing."""
        disabled = TracingDisabled()
        disabled.shutdown()  # Should not raise


class TestDummyTrace:
    """Tests for DummyTrace class."""

    def test_span_returns_dummy(self) -> None:
        """Test nested span creation."""
        trace = DummyTrace()
        span = trace.span(name="test")
        assert isinstance(span, DummySpan)

    def test_generation_returns_dummy(self) -> None:
        """Test generation creation."""
        trace = DummyTrace()
        gen = trace.generation(name="test")
        assert isinstance(gen, DummyGeneration)

    def test_update_returns_self(self) -> None:
        """Test that update returns self for chaining."""
        trace = DummyTrace()
        result = trace.update(output="test")
        assert result is trace

    def test_end_no_op(self) -> None:
        """Test that end does nothing."""
        trace = DummyTrace()
        trace.end()  # Should not raise


class TestDummySpan:
    """Tests for DummySpan class."""

    def test_nested_span(self) -> None:
        """Test nested span creation."""
        span = DummySpan()
        nested = span.span(name="nested")
        assert isinstance(nested, DummySpan)

    def test_generation_returns_dummy(self) -> None:
        """Test generation creation from span."""
        span = DummySpan()
        gen = span.generation(name="test")
        assert isinstance(gen, DummyGeneration)


class TestTracer:
    """Tests for Tracer class."""

    def test_disabled_tracer(self) -> None:
        """Test tracer when disabled."""
        config = LangfuseConfig(enabled=False)
        tracer = Tracer(config=config)
        assert tracer.enabled is False

        trace = tracer.trace(name="test")
        assert isinstance(trace, DummyTrace)

    def test_tracer_without_keys(self) -> None:
        """Test tracer with missing keys falls back to disabled."""
        config = LangfuseConfig(enabled=True, public_key="", secret_key="")
        tracer = Tracer(config=config)

        # Should fall back to disabled
        trace = tracer.trace(name="test")
        assert isinstance(trace, DummyTrace)

    @patch("langfuse.Langfuse")
    def test_tracer_with_valid_config(self, mock_langfuse_class: MagicMock) -> None:
        """Test tracer initialization with valid config."""
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client

        config = LangfuseConfig(
            enabled=True,
            public_key="pk-test",
            secret_key="sk-test",
            host="https://test.langfuse.com",
        )
        tracer = Tracer(config=config)

        # Force client initialization
        tracer._get_client()

        mock_langfuse_class.assert_called_once_with(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://test.langfuse.com",
            release="0.1.0",
            debug=False,
        )

    @patch("langfuse.Langfuse")
    def test_tracer_flush(self, mock_langfuse_class: MagicMock) -> None:
        """Test tracer flush."""
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client

        config = LangfuseConfig(enabled=True, public_key="pk-test", secret_key="sk-test")
        tracer = Tracer(config=config)
        tracer.flush()

        mock_client.flush.assert_called_once()

    @patch("langfuse.Langfuse")
    def test_tracer_shutdown(self, mock_langfuse_class: MagicMock) -> None:
        """Test tracer shutdown."""
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client

        config = LangfuseConfig(enabled=True, public_key="pk-test", secret_key="sk-test")
        tracer = Tracer(config=config)
        tracer._get_client()  # Initialize
        tracer.shutdown()

        mock_client.shutdown.assert_called_once()
        assert tracer._client is None

    def test_tracer_handles_init_error(self) -> None:
        """Test tracer handles Langfuse initialization errors gracefully."""
        with patch("langfuse.Langfuse") as mock_langfuse:
            mock_langfuse.side_effect = Exception("Connection failed")

            config = LangfuseConfig(enabled=True, public_key="pk-test", secret_key="sk-test")
            tracer = Tracer(config=config)

            # Should fall back to disabled
            trace = tracer.trace(name="test")
            assert isinstance(trace, DummyTrace)


class TestGlobalTracer:
    """Tests for global tracer functions."""

    def test_get_tracer_returns_singleton(self) -> None:
        """Test that get_tracer returns same instance."""
        reset_tracer()
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2

    def test_reset_tracer(self) -> None:
        """Test that reset_tracer clears the singleton."""
        reset_tracer()
        tracer1 = get_tracer()
        reset_tracer()
        tracer2 = get_tracer()
        assert tracer1 is not tracer2


class TestGenerateTraceId:
    """Tests for trace ID generation."""

    def test_generates_uuid(self) -> None:
        """Test that trace IDs are valid UUIDs."""
        import uuid

        trace_id = generate_trace_id()
        # Should be a valid UUID
        uuid.UUID(trace_id)

    def test_generates_unique_ids(self) -> None:
        """Test that trace IDs are unique."""
        ids = [generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestTracedDecorator:
    """Tests for @traced decorator."""

    def test_traced_function_executes(self) -> None:
        """Test that decorated function still executes."""
        reset_tracer()

        @traced(name="test_function")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10

    def test_traced_function_uses_function_name(self) -> None:
        """Test that decorator uses function name as default."""
        reset_tracer()

        @traced()
        def another_function() -> str:
            return "result"

        result = another_function()
        assert result == "result"

    def test_traced_function_propagates_exception(self) -> None:
        """Test that exceptions are propagated."""
        reset_tracer()

        @traced(name="failing_function")
        def failing_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    def test_traced_with_tags_and_metadata(self) -> None:
        """Test decorator with tags and metadata."""
        reset_tracer()

        @traced(name="tagged_function", tags=["test"], metadata={"key": "value"})
        def tagged_function() -> str:
            return "tagged"

        result = tagged_function()
        assert result == "tagged"


class TestTraceContext:
    """Tests for trace_context context manager."""

    def test_trace_context_yields_trace(self) -> None:
        """Test that context manager yields a trace."""
        reset_tracer()

        with trace_context("test_context") as trace:
            assert trace is not None
            # Should have trace methods
            assert hasattr(trace, "span")
            assert hasattr(trace, "end")

    def test_trace_context_with_session(self) -> None:
        """Test context manager with session ID."""
        reset_tracer()

        with trace_context(
            "test_context",
            session_id="sess123",
            user_id="user456",
            tags=["test"],
        ) as trace:
            assert trace is not None

    def test_trace_context_handles_exception(self) -> None:
        """Test that exceptions are propagated from context."""
        reset_tracer()

        with pytest.raises(RuntimeError, match="Test error"), trace_context("failing_context"):
            raise RuntimeError("Test error")
