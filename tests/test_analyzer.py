"""Tests for the TradeAnalyzer class."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.agents.analyzer import AnalysisError, TradeAnalyzer
from src.agents.llm_client import LLMError, LLMResponse
from src.agents.prompts import PromptVariant
from src.parsers.fix_parser import ExecutionReport
from src.parsers.models import AnalysisResult


@pytest.fixture
def sample_execution() -> ExecutionReport:
    """Create a sample execution report for testing."""
    return ExecutionReport(
        order_id="ORD001",
        symbol="AAPL",
        side="BUY",
        quantity=100.0,
        price=150.50,
        venue="NYSE",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        fill_type="FULL",
    )


@pytest.fixture
def sample_executions() -> list[ExecutionReport]:
    """Create multiple sample executions for batch testing."""
    return [
        ExecutionReport(
            order_id=f"ORD00{i}",
            symbol=["AAPL", "GOOGL", "MSFT"][i % 3],
            side="BUY" if i % 2 == 0 else "SELL",
            quantity=100.0 * (i + 1),
            price=150.50 + i * 10,
            venue=["NYSE", "NASDAQ", "BATS"][i % 3],
            timestamp=datetime(2024, 1, 15, 10, 30 + i, 0),
            fill_type="FULL",
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """Create a mock LLM response."""
    return LLMResponse(
        content='{"quality_score": 7, "observations": ["Good execution"], '
        '"issues": [], "recommendations": [], "confidence": 0.85}',
        input_tokens=100,
        output_tokens=50,
        model="claude-3-5-sonnet-20241022",
        latency_ms=500.0,
        provider="anthropic",
    )


@pytest.fixture
def mock_llm_client(mock_llm_response: LLMResponse) -> MagicMock:
    """Create a mock LLM client."""
    client = MagicMock()
    client.complete.return_value = mock_llm_response
    return client


class TestTradeAnalyzer:
    """Tests for TradeAnalyzer class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        with patch("src.agents.analyzer.LLMClient"):
            analyzer = TradeAnalyzer()
            assert analyzer.default_variant == PromptVariant.DETAILED
            assert analyzer.max_concurrent == 5

    def test_init_with_client(self, mock_llm_client: MagicMock) -> None:
        """Test initialization with custom client."""
        analyzer = TradeAnalyzer(client=mock_llm_client)
        assert analyzer.client is mock_llm_client

    def test_init_with_custom_settings(self, mock_llm_client: MagicMock) -> None:
        """Test initialization with custom settings."""
        analyzer = TradeAnalyzer(
            client=mock_llm_client,
            default_variant=PromptVariant.QUICK,
            max_concurrent=10,
        )
        assert analyzer.default_variant == PromptVariant.QUICK
        assert analyzer.max_concurrent == 10


class TestAnalyzeSingle:
    """Tests for single execution analysis."""

    def test_analyze_success(
        self,
        sample_execution: ExecutionReport,
        mock_llm_client: MagicMock,
    ) -> None:
        """Test successful single analysis."""
        analyzer = TradeAnalyzer(client=mock_llm_client)
        result = analyzer.analyze(sample_execution)

        assert isinstance(result, AnalysisResult)
        assert result.execution.order_id == "ORD001"
        assert result.analysis.quality_score == 7
        assert result.analysis.observations == ["Good execution"]
        assert result.tokens_used == 150
        assert result.latency_ms == 500.0
        assert result.analysis_id is not None

    def test_analyze_with_variant(
        self,
        sample_execution: ExecutionReport,
        mock_llm_client: MagicMock,
    ) -> None:
        """Test analysis with specific variant."""
        analyzer = TradeAnalyzer(client=mock_llm_client)
        result = analyzer.analyze(sample_execution, variant=PromptVariant.QUICK)

        assert isinstance(result, AnalysisResult)
        mock_llm_client.complete.assert_called_once()

    def test_analyze_with_session_id(
        self,
        sample_execution: ExecutionReport,
        mock_llm_client: MagicMock,
    ) -> None:
        """Test analysis with session ID for tracing."""
        analyzer = TradeAnalyzer(client=mock_llm_client)
        result = analyzer.analyze(sample_execution, session_id="test-session-123")

        assert isinstance(result, AnalysisResult)

    def test_analyze_parse_error_fallback(
        self,
        sample_execution: ExecutionReport,
        mock_llm_client: MagicMock,
    ) -> None:
        """Test fallback when response parsing fails."""
        # Return invalid JSON
        mock_llm_client.complete.return_value = LLMResponse(
            content="This is not valid JSON at all",
            input_tokens=100,
            output_tokens=50,
            model="claude-3-5-sonnet-20241022",
            latency_ms=500.0,
            provider="anthropic",
        )

        analyzer = TradeAnalyzer(client=mock_llm_client)
        result = analyzer.analyze(sample_execution)

        # Should return fallback analysis
        assert result.analysis.quality_score == 5
        assert result.analysis.confidence == 0.1
        assert "parsing failed" in result.analysis.observations[0].lower()

    def test_analyze_llm_error(
        self,
        sample_execution: ExecutionReport,
        mock_llm_client: MagicMock,
    ) -> None:
        """Test handling of LLM errors."""
        mock_llm_client.complete.side_effect = LLMError("API error")

        analyzer = TradeAnalyzer(client=mock_llm_client)

        with pytest.raises(AnalysisError) as exc_info:
            analyzer.analyze(sample_execution)

        assert exc_info.value.execution == sample_execution
        assert exc_info.value.cause is not None


class TestAnalyzeBatch:
    """Tests for batch analysis."""

    def test_analyze_batch_success(
        self,
        sample_executions: list[ExecutionReport],
        mock_llm_client: MagicMock,
    ) -> None:
        """Test successful batch analysis."""
        analyzer = TradeAnalyzer(client=mock_llm_client, max_concurrent=3)
        results = analyzer.analyze_batch(sample_executions)

        assert len(results) == 5
        assert all(isinstance(r, AnalysisResult) for r in results)
        assert mock_llm_client.complete.call_count == 5

    def test_analyze_batch_empty_list(self, mock_llm_client: MagicMock) -> None:
        """Test batch with empty list."""
        analyzer = TradeAnalyzer(client=mock_llm_client)
        results = analyzer.analyze_batch([])

        assert results == []
        mock_llm_client.complete.assert_not_called()

    def test_analyze_batch_with_errors_continue(
        self,
        sample_executions: list[ExecutionReport],
        mock_llm_client: MagicMock,
    ) -> None:
        """Test batch continues on individual errors."""
        # Make every other call fail
        call_count = 0

        def side_effect(*_args: object, **_kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise LLMError("API error")
            return LLMResponse(
                content='{"quality_score": 7, "observations": [], "issues": [], '
                '"recommendations": [], "confidence": 0.8}',
                input_tokens=100,
                output_tokens=50,
                model="claude-3-5-sonnet-20241022",
                latency_ms=500.0,
                provider="anthropic",
            )

        mock_llm_client.complete.side_effect = side_effect

        analyzer = TradeAnalyzer(client=mock_llm_client)
        results = analyzer.analyze_batch(sample_executions, continue_on_error=True)

        # Should have mix of results and errors
        assert len(results) == 5
        successes = [r for r in results if isinstance(r, AnalysisResult)]
        errors = [r for r in results if isinstance(r, AnalysisError)]
        assert len(successes) >= 1
        assert len(errors) >= 1

    def test_analyze_batch_with_errors_stop(
        self,
        sample_executions: list[ExecutionReport],
        mock_llm_client: MagicMock,
    ) -> None:
        """Test batch stops on first error when continue_on_error=False."""
        mock_llm_client.complete.side_effect = LLMError("API error")

        analyzer = TradeAnalyzer(client=mock_llm_client)

        with pytest.raises(AnalysisError):
            analyzer.analyze_batch(sample_executions, continue_on_error=False)


class TestAnalysisError:
    """Tests for AnalysisError exception."""

    def test_error_attributes(self, sample_execution: ExecutionReport) -> None:
        """Test error has expected attributes."""
        cause = ValueError("root cause")
        error = AnalysisError(
            "Analysis failed",
            execution=sample_execution,
            cause=cause,
        )

        assert str(error) == "Analysis failed"
        assert error.execution == sample_execution
        assert error.cause == cause

    def test_error_without_execution(self) -> None:
        """Test error without execution."""
        error = AnalysisError("Generic error")
        assert error.execution is None
        assert error.cause is None


class TestAsyncMethods:
    """Tests for async analysis methods."""

    @pytest.mark.asyncio
    async def test_analyze_async(
        self,
        sample_execution: ExecutionReport,
        mock_llm_client: MagicMock,
    ) -> None:
        """Test async single analysis."""
        analyzer = TradeAnalyzer(client=mock_llm_client)
        result = await analyzer.analyze_async(sample_execution)

        assert isinstance(result, AnalysisResult)
        assert result.execution.order_id == "ORD001"

    @pytest.mark.asyncio
    async def test_analyze_batch_async(
        self,
        sample_executions: list[ExecutionReport],
        mock_llm_client: MagicMock,
    ) -> None:
        """Test async batch analysis."""
        analyzer = TradeAnalyzer(client=mock_llm_client, max_concurrent=3)
        results = await analyzer.analyze_batch_async(sample_executions)

        assert len(results) == 5
        assert all(isinstance(r, AnalysisResult) for r in results)
