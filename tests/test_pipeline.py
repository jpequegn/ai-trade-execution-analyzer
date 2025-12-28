"""Tests for the pipeline module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.agents.llm_client import LLMResponse
from src.parsers.fix_parser import ExecutionReport
from src.parsers.models import AnalysisResult, TradeAnalysis
from src.pipeline import (
    SAMPLE_FIX_MESSAGE,
    analyze_execution,
    analyze_fix_message,
    analyze_fix_messages,
    format_result_as_json,
    format_result_for_display,
)


@pytest.fixture
def sample_fix_message() -> str:
    """Return the sample FIX message."""
    return SAMPLE_FIX_MESSAGE


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """Create a mock LLM response."""
    return LLMResponse(
        content='{"quality_score": 8, "observations": ["Executed during market hours", "Good fill rate"], '
        '"issues": ["Slightly above VWAP"], "recommendations": ["Consider dark pools"], "confidence": 0.9}',
        input_tokens=150,
        output_tokens=75,
        model="claude-3-5-sonnet-20241022",
        latency_ms=450.0,
        provider="anthropic",
    )


@pytest.fixture
def sample_analysis_result() -> AnalysisResult:
    """Create a sample analysis result."""
    execution = ExecutionReport(
        order_id="ORD001",
        symbol="AAPL",
        side="BUY",
        quantity=100.0,
        price=150.50,
        venue="NYSE",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        fill_type="FULL",
    )
    analysis = TradeAnalysis(
        quality_score=8,
        observations=["Executed during market hours", "Good fill rate"],
        issues=["Slightly above VWAP"],
        recommendations=["Consider dark pools"],
        confidence=0.9,
    )
    return AnalysisResult(
        execution=execution,
        analysis=analysis,
        raw_response='{"quality_score": 8}',
        tokens_used=225,
        latency_ms=450.0,
        model="claude-3-5-sonnet-20241022",
        analysis_id="test-id-123",
        analyzed_at=datetime(2024, 1, 15, 10, 31, 0),
    )


class TestAnalyzeFixMessage:
    """Tests for analyze_fix_message function."""

    def test_analyze_fix_message_success(
        self,
        sample_fix_message: str,
        mock_llm_response: LLMResponse,
    ) -> None:
        """Test successful FIX message analysis."""
        with patch("src.pipeline.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.complete.return_value = mock_llm_response
            mock_client_class.return_value = mock_client

            result = analyze_fix_message(sample_fix_message)

            assert isinstance(result, AnalysisResult)
            assert result.execution.order_id == "ORD001"
            assert result.execution.symbol == "AAPL"
            assert result.analysis.quality_score == 8
            assert result.tokens_used == 225

    def test_analyze_fix_message_with_variant(
        self,
        sample_fix_message: str,
        mock_llm_response: LLMResponse,
    ) -> None:
        """Test analysis with specific variant."""
        from src.agents.prompts import PromptVariant

        with patch("src.pipeline.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.complete.return_value = mock_llm_response
            mock_client_class.return_value = mock_client

            result = analyze_fix_message(
                sample_fix_message,
                variant=PromptVariant.QUICK,
            )

            assert isinstance(result, AnalysisResult)

    def test_analyze_fix_message_invalid_message(self) -> None:
        """Test with invalid FIX message."""
        from src.parsers.exceptions import FIXParseError

        with pytest.raises(FIXParseError):
            analyze_fix_message("invalid message")


class TestAnalyzeFixMessages:
    """Tests for batch analysis of FIX messages."""

    def test_analyze_fix_messages_success(
        self,
        mock_llm_response: LLMResponse,
    ) -> None:
        """Test successful batch analysis."""
        messages = [
            "8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|31=150.50|30=NYSE|60=20240115-10:30:00.000|39=2",
            "8=FIX.4.4|35=8|37=ORD002|55=GOOGL|54=2|32=50|31=175.25|30=NASDAQ|60=20240115-10:31:00.000|39=2",
        ]

        with patch("src.pipeline.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.complete.return_value = mock_llm_response
            mock_client_class.return_value = mock_client

            results = analyze_fix_messages(messages)

            assert len(results) == 2
            assert all(isinstance(r, AnalysisResult) for r in results)

    def test_analyze_fix_messages_with_parse_error(
        self,
        mock_llm_response: LLMResponse,
    ) -> None:
        """Test batch with some invalid messages."""
        messages = [
            "8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|31=150.50|30=NYSE|60=20240115-10:30:00.000|39=2",
            "invalid message",  # This will fail to parse
        ]

        with patch("src.pipeline.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.complete.return_value = mock_llm_response
            mock_client_class.return_value = mock_client

            # Should continue despite parse error
            results = analyze_fix_messages(messages, continue_on_error=True)

            # Only one message was parseable
            assert len(results) == 1


class TestAnalyzeExecution:
    """Tests for analyze_execution function."""

    def test_analyze_execution_success(
        self,
        mock_llm_response: LLMResponse,
    ) -> None:
        """Test analyzing a pre-parsed execution."""
        execution = ExecutionReport(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            price=150.50,
            venue="NYSE",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            fill_type="FULL",
        )

        with patch("src.pipeline.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.complete.return_value = mock_llm_response
            mock_client_class.return_value = mock_client

            result = analyze_execution(execution)

            assert isinstance(result, AnalysisResult)
            assert result.execution.order_id == "ORD001"


class TestFormatResultForDisplay:
    """Tests for display formatting."""

    def test_format_result_contains_all_sections(
        self,
        sample_analysis_result: AnalysisResult,
    ) -> None:
        """Test that formatted output contains all sections."""
        output = format_result_for_display(sample_analysis_result)

        assert "ORD001" in output
        assert "AAPL" in output
        assert "BUY" in output
        assert "100" in output
        assert "150.50" in output
        assert "NYSE" in output
        assert "8/10" in output
        assert "Executed during market hours" in output
        assert "Slightly above VWAP" in output
        assert "Consider dark pools" in output
        assert "claude-3-5-sonnet" in output
        assert "225" in output  # tokens

    def test_format_result_handles_empty_lists(self) -> None:
        """Test formatting with empty analysis lists."""
        execution = ExecutionReport(
            order_id="ORD002",
            symbol="MSFT",
            side="SELL",
            quantity=50.0,
            price=300.00,
            venue="NASDAQ",
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
            fill_type="PARTIAL",
        )
        analysis = TradeAnalysis(
            quality_score=5,
            observations=[],
            issues=[],
            recommendations=[],
            confidence=0.7,
        )
        result = AnalysisResult(
            execution=execution,
            analysis=analysis,
            analysis_id="test-id",
        )

        output = format_result_for_display(result)

        assert "ORD002" in output
        assert "5/10" in output
        # Should not have observation/issue/recommendation headers
        # when lists are empty (or have empty sections)


class TestFormatResultAsJson:
    """Tests for JSON formatting."""

    def test_format_result_as_json_valid(
        self,
        sample_analysis_result: AnalysisResult,
    ) -> None:
        """Test that JSON output is valid."""
        import json

        output = format_result_as_json(sample_analysis_result)
        parsed = json.loads(output)

        assert parsed["execution"]["order_id"] == "ORD001"
        assert parsed["execution"]["symbol"] == "AAPL"
        assert parsed["analysis"]["quality_score"] == 8
        assert parsed["analysis"]["observations"] == [
            "Executed during market hours",
            "Good fill rate",
        ]
        assert parsed["metadata"]["tokens_used"] == 225
        assert parsed["metadata"]["analysis_id"] == "test-id-123"

    def test_format_result_as_json_structure(
        self,
        sample_analysis_result: AnalysisResult,
    ) -> None:
        """Test JSON structure has expected keys."""
        import json

        output = format_result_as_json(sample_analysis_result)
        parsed = json.loads(output)

        assert "execution" in parsed
        assert "analysis" in parsed
        assert "metadata" in parsed

        # Execution fields
        assert "order_id" in parsed["execution"]
        assert "symbol" in parsed["execution"]
        assert "side" in parsed["execution"]
        assert "quantity" in parsed["execution"]
        assert "price" in parsed["execution"]
        assert "venue" in parsed["execution"]
        assert "timestamp" in parsed["execution"]

        # Analysis fields
        assert "quality_score" in parsed["analysis"]
        assert "confidence" in parsed["analysis"]
        assert "observations" in parsed["analysis"]
        assert "issues" in parsed["analysis"]
        assert "recommendations" in parsed["analysis"]


class TestSampleFixMessage:
    """Tests for sample FIX message constant."""

    def test_sample_message_is_valid(self) -> None:
        """Test that sample message can be parsed."""
        from src.parsers.fix_parser import parse_fix_message

        execution = parse_fix_message(SAMPLE_FIX_MESSAGE)

        assert execution.order_id == "ORD001"
        assert execution.symbol == "AAPL"
        assert execution.side == "BUY"
        assert execution.quantity == 100.0
        assert execution.price == 150.50
        assert execution.venue == "NYSE"
