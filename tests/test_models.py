"""Tests for Pydantic data models."""

from datetime import datetime

import pytest

from src.parsers import AnalysisResult, ExecutionReport, TradeAnalysis


class TestTradeAnalysis:
    """Tests for TradeAnalysis model."""

    def test_valid_analysis_creation(self) -> None:
        """Test creating a valid TradeAnalysis."""
        analysis = TradeAnalysis(
            quality_score=8,
            observations=["Trade executed at market open"],
            issues=["Minor price slippage"],
            recommendations=["Consider limit orders"],
            confidence=0.85,
        )
        assert analysis.quality_score == 8
        assert len(analysis.observations) == 1
        assert len(analysis.issues) == 1
        assert len(analysis.recommendations) == 1
        assert analysis.confidence == 0.85

    def test_minimal_analysis(self) -> None:
        """Test creating analysis with only required fields."""
        analysis = TradeAnalysis(quality_score=5)
        assert analysis.quality_score == 5
        assert analysis.observations == []
        assert analysis.issues == []
        assert analysis.recommendations == []
        assert analysis.confidence == 0.8  # Default

    def test_score_minimum_boundary(self) -> None:
        """Test quality score at minimum (1)."""
        analysis = TradeAnalysis(quality_score=1)
        assert analysis.quality_score == 1

    def test_score_maximum_boundary(self) -> None:
        """Test quality score at maximum (10)."""
        analysis = TradeAnalysis(quality_score=10)
        assert analysis.quality_score == 10

    def test_score_below_minimum_raises_error(self) -> None:
        """Test that quality score below 1 raises error."""
        with pytest.raises(ValueError):
            TradeAnalysis(quality_score=0)

    def test_score_above_maximum_raises_error(self) -> None:
        """Test that quality score above 10 raises error."""
        with pytest.raises(ValueError):
            TradeAnalysis(quality_score=11)

    def test_confidence_minimum_boundary(self) -> None:
        """Test confidence at minimum (0.0)."""
        analysis = TradeAnalysis(quality_score=5, confidence=0.0)
        assert analysis.confidence == 0.0

    def test_confidence_maximum_boundary(self) -> None:
        """Test confidence at maximum (1.0)."""
        analysis = TradeAnalysis(quality_score=5, confidence=1.0)
        assert analysis.confidence == 1.0

    def test_confidence_below_minimum_raises_error(self) -> None:
        """Test that confidence below 0.0 raises error."""
        with pytest.raises(ValueError):
            TradeAnalysis(quality_score=5, confidence=-0.1)

    def test_confidence_above_maximum_raises_error(self) -> None:
        """Test that confidence above 1.0 raises error."""
        with pytest.raises(ValueError):
            TradeAnalysis(quality_score=5, confidence=1.1)

    def test_none_lists_converted_to_empty(self) -> None:
        """Test that None values for lists are converted to empty lists."""
        analysis = TradeAnalysis(
            quality_score=5,
            observations=None,  # type: ignore[arg-type]
            issues=None,  # type: ignore[arg-type]
            recommendations=None,  # type: ignore[arg-type]
        )
        assert analysis.observations == []
        assert analysis.issues == []
        assert analysis.recommendations == []

    def test_serialization_roundtrip(self) -> None:
        """Test that model can serialize and deserialize correctly."""
        original = TradeAnalysis(
            quality_score=7,
            observations=["Obs1", "Obs2"],
            issues=["Issue1"],
            recommendations=["Rec1", "Rec2", "Rec3"],
            confidence=0.92,
        )
        json_str = original.model_dump_json()
        restored = TradeAnalysis.model_validate_json(json_str)
        assert restored == original


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    @pytest.fixture
    def sample_execution(self) -> ExecutionReport:
        """Create a sample ExecutionReport for testing."""
        return ExecutionReport(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            price=185.50,
            venue="NASDAQ",
            timestamp=datetime(2024, 1, 15, 9, 30, 5),
            fill_type="FULL",
        )

    @pytest.fixture
    def sample_analysis(self) -> TradeAnalysis:
        """Create a sample TradeAnalysis for testing."""
        return TradeAnalysis(
            quality_score=8,
            observations=["Fast execution at market open"],
            issues=[],
            recommendations=[],
            confidence=0.9,
        )

    def test_valid_result_creation(
        self, sample_execution: ExecutionReport, sample_analysis: TradeAnalysis
    ) -> None:
        """Test creating a valid AnalysisResult."""
        result = AnalysisResult(
            execution=sample_execution,
            analysis=sample_analysis,
            raw_response="Quality score: 8/10...",
            tokens_used=450,
            latency_ms=1200.5,
        )
        assert result.execution.symbol == "AAPL"
        assert result.analysis.quality_score == 8
        assert result.tokens_used == 450
        assert result.latency_ms == 1200.5
        assert result.from_cache is False
        assert result.model == "claude-3-5-sonnet-20241022"

    def test_minimal_result(
        self, sample_execution: ExecutionReport, sample_analysis: TradeAnalysis
    ) -> None:
        """Test creating result with only required fields."""
        result = AnalysisResult(
            execution=sample_execution,
            analysis=sample_analysis,
        )
        assert result.raw_response == ""
        assert result.tokens_used == 0
        assert result.latency_ms == 0.0
        assert result.from_cache is False
        assert result.analysis_id is None

    def test_from_cache_flag(
        self, sample_execution: ExecutionReport, sample_analysis: TradeAnalysis
    ) -> None:
        """Test from_cache flag."""
        result = AnalysisResult(
            execution=sample_execution,
            analysis=sample_analysis,
            from_cache=True,
            tokens_used=0,
        )
        assert result.from_cache is True

    def test_custom_model_name(
        self, sample_execution: ExecutionReport, sample_analysis: TradeAnalysis
    ) -> None:
        """Test custom model name."""
        result = AnalysisResult(
            execution=sample_execution,
            analysis=sample_analysis,
            model="gpt-4-turbo",
        )
        assert result.model == "gpt-4-turbo"

    def test_analysis_id(
        self, sample_execution: ExecutionReport, sample_analysis: TradeAnalysis
    ) -> None:
        """Test analysis_id field."""
        result = AnalysisResult(
            execution=sample_execution,
            analysis=sample_analysis,
            analysis_id="AN-2024-001",
        )
        assert result.analysis_id == "AN-2024-001"

    def test_analyzed_at_auto_set(
        self, sample_execution: ExecutionReport, sample_analysis: TradeAnalysis
    ) -> None:
        """Test that analyzed_at is automatically set."""
        before = datetime.now()
        result = AnalysisResult(
            execution=sample_execution,
            analysis=sample_analysis,
        )
        after = datetime.now()
        assert before <= result.analyzed_at <= after

    def test_negative_tokens_raises_error(
        self, sample_execution: ExecutionReport, sample_analysis: TradeAnalysis
    ) -> None:
        """Test that negative tokens_used raises error."""
        with pytest.raises(ValueError):
            AnalysisResult(
                execution=sample_execution,
                analysis=sample_analysis,
                tokens_used=-1,
            )

    def test_negative_latency_raises_error(
        self, sample_execution: ExecutionReport, sample_analysis: TradeAnalysis
    ) -> None:
        """Test that negative latency_ms raises error."""
        with pytest.raises(ValueError):
            AnalysisResult(
                execution=sample_execution,
                analysis=sample_analysis,
                latency_ms=-1.0,
            )


class TestModelIntegration:
    """Integration tests for model interactions."""

    def test_full_analysis_workflow(self) -> None:
        """Test complete workflow from execution to result."""
        # Parse execution
        execution = ExecutionReport(
            order_id="ORD-TEST-001",
            symbol="MSFT",
            side="SELL",
            quantity=500.0,
            price=402.75,
            venue="NASDAQ",
            timestamp=datetime(2024, 1, 15, 10, 15, 0),
            fill_type="FULL",
            exec_type="F",
            cum_qty=500.0,
            avg_px=402.75,
        )

        # Create analysis
        analysis = TradeAnalysis(
            quality_score=9,
            observations=[
                "Trade executed on primary exchange",
                "Full fill achieved",
                "Optimal timing during high liquidity",
            ],
            issues=[],
            recommendations=["None - excellent execution"],
            confidence=0.95,
        )

        # Combine into result
        result = AnalysisResult(
            execution=execution,
            analysis=analysis,
            raw_response="Analysis complete...",
            tokens_used=380,
            latency_ms=950.0,
            analysis_id="AN-2024-TEST",
        )

        # Verify structure
        assert result.execution.order_id == "ORD-TEST-001"
        assert result.analysis.quality_score == 9
        assert len(result.analysis.observations) == 3
        assert result.tokens_used == 380

    def test_json_serialization_full_result(self) -> None:
        """Test JSON serialization of complete AnalysisResult."""
        execution = ExecutionReport(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            price=185.50,
            venue="NASDAQ",
            timestamp=datetime(2024, 1, 15, 9, 30, 5),
            fill_type="FULL",
        )

        analysis = TradeAnalysis(
            quality_score=8,
            observations=["Good execution"],
            issues=[],
            recommendations=[],
            confidence=0.9,
        )

        result = AnalysisResult(
            execution=execution,
            analysis=analysis,
            tokens_used=100,
        )

        # Serialize and verify it's valid JSON
        json_str = result.model_dump_json()
        assert "AAPL" in json_str
        assert "quality_score" in json_str
        assert "8" in json_str
