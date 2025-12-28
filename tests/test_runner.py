"""Tests for the evaluation runner."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.agents.analyzer import AnalysisError
from src.evaluation.ground_truth import ExpertAnalysis, IssueCategory, IssueSeverity
from src.evaluation.runner import (
    EvalResults,
    EvaluationRunner,
    ProgressCallback,
    SingleEvalResult,
)
from src.parsers.fix_parser import ExecutionReport
from src.parsers.models import AnalysisResult, TradeAnalysis

# --- Fixtures ---


@pytest.fixture
def mock_analyzer() -> MagicMock:
    """Create a mock TradeAnalyzer."""
    analyzer = MagicMock()

    # Default successful analysis
    def mock_analyze(execution, **_kwargs):
        return AnalysisResult(
            execution=execution,
            analysis=TradeAnalysis(
                quality_score=7,
                issues=["minor timing issue"],
                observations=["Full fill achieved", f"Traded {execution.symbol}"],
                confidence=0.85,
            ),
            raw_response="Test response",
            tokens_used=100,
            latency_ms=500.0,
            model="test-model",
        )

    analyzer.analyze = MagicMock(side_effect=mock_analyze)
    return analyzer


@pytest.fixture
def sample_execution() -> ExecutionReport:
    """Create a sample execution report."""
    return ExecutionReport(
        order_id="ORD001",
        symbol="AAPL",
        side="BUY",
        quantity=100.0,
        price=185.50,
        venue="NYSE",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        fill_type="FULL",
    )


@pytest.fixture
def sample_expert_analysis() -> ExpertAnalysis:
    """Create a sample expert analysis."""
    return ExpertAnalysis(
        quality_score=7,
        key_issues=["timing issue"],
        expected_observations=["Full fill achieved", "Traded AAPL"],
        severity=IssueSeverity.LOW,
        category=IssueCategory.TIMING,
    )


# --- SingleEvalResult Tests ---


class TestSingleEvalResult:
    """Tests for SingleEvalResult model."""

    def test_creation(self) -> None:
        """Test basic result creation."""
        result = SingleEvalResult(
            sample_id="GT001",
            ai_analysis=TradeAnalysis(quality_score=7),
            ground_truth={"quality_score": 7},
            execution={"symbol": "AAPL"},
            metrics={"insight_accuracy": 0.8},
            overall_score=0.75,
            passed=True,
        )
        assert result.sample_id == "GT001"
        assert result.passed is True
        assert result.overall_score == 0.75

    def test_with_errors(self) -> None:
        """Test result with errors."""
        result = SingleEvalResult(
            sample_id="GT002",
            ai_analysis=TradeAnalysis(quality_score=1),
            ground_truth={},
            execution={},
            metrics={},
            overall_score=0.0,
            passed=False,
            errors=["Analysis failed", "Connection timeout"],
        )
        assert not result.passed
        assert len(result.errors) == 2


# --- EvalResults Tests ---


class TestEvalResults:
    """Tests for EvalResults model."""

    def test_empty_results(self) -> None:
        """Test empty results."""
        results = EvalResults()
        assert results.total_samples == 0
        assert results.pass_rate == 0.0

    def test_pass_rate_calculation(self) -> None:
        """Test pass rate calculation."""
        results = EvalResults(
            total_samples=10,
            passed=8,
            failed=2,
        )
        assert results.pass_rate == 0.8

    def test_success_rate_excludes_errors(self) -> None:
        """Test success rate excludes errors."""
        results = EvalResults(
            total_samples=10,
            passed=6,
            failed=2,
            error_count=2,
        )
        # Success rate = passed / (total - errors) = 6 / 8 = 0.75
        assert results.success_rate == 0.75

    def test_get_failed_samples(self) -> None:
        """Test filtering failed samples."""
        passed_result = SingleEvalResult(
            sample_id="GT001",
            ai_analysis=TradeAnalysis(quality_score=8),
            ground_truth={},
            execution={},
            metrics={},
            overall_score=0.9,
            passed=True,
        )
        failed_result = SingleEvalResult(
            sample_id="GT002",
            ai_analysis=TradeAnalysis(quality_score=3),
            ground_truth={},
            execution={},
            metrics={},
            overall_score=0.4,
            passed=False,
        )
        results = EvalResults(
            total_samples=2,
            passed=1,
            failed=1,
            results=[passed_result, failed_result],
        )
        failed = results.get_failed_samples()
        assert len(failed) == 1
        assert failed[0].sample_id == "GT002"

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        results = EvalResults(
            total_samples=5,
            passed=4,
            aggregate_metrics={"overall": 0.85},
        )
        json_str = results.to_json()
        assert '"total_samples": 5' in json_str
        assert '"passed": 4' in json_str


# --- ProgressCallback Tests ---


class TestProgressCallback:
    """Tests for ProgressCallback."""

    def test_initial_state(self) -> None:
        """Test initial progress state."""
        progress = ProgressCallback(total=10)
        assert progress.completed == 0
        assert progress.progress_pct == 0.0

    def test_update(self) -> None:
        """Test progress update."""
        progress = ProgressCallback(total=10)
        progress.update(passed=True)
        assert progress.completed == 1
        assert progress.passed == 1
        assert progress.progress_pct == 10.0

    def test_error_tracking(self) -> None:
        """Test error tracking."""
        progress = ProgressCallback(total=10)
        progress.update(passed=False, error=True)
        assert progress.errors == 1
        assert progress.failed == 0  # Errors don't count as failed

    def test_callback_invoked(self) -> None:
        """Test callback is invoked on update."""
        callback_called = []

        def callback(p):
            callback_called.append(p.completed)

        progress = ProgressCallback(total=10, callback=callback)
        progress.update(passed=True)
        progress.update(passed=False)

        assert callback_called == [1, 2]

    def test_samples_per_second(self) -> None:
        """Test rate calculation."""
        progress = ProgressCallback(total=10)
        progress.completed = 5
        # Rate depends on elapsed time, just verify it doesn't error
        rate = progress.samples_per_second
        assert isinstance(rate, float)


# --- EvaluationRunner Tests ---


class TestEvaluationRunner:
    """Tests for EvaluationRunner."""

    @patch("src.evaluation.runner.load_ground_truth")
    def test_initialization(self, mock_load, mock_analyzer) -> None:
        """Test runner initialization."""
        from src.evaluation.ground_truth import GroundTruthDataset

        mock_load.return_value = GroundTruthDataset(samples=[])

        runner = EvaluationRunner(
            analyzer=mock_analyzer,
            thresholds={"overall": 0.80},
            max_workers=2,
        )

        assert runner.max_workers == 2
        assert runner.thresholds["overall"] == 0.80
        mock_load.assert_called_once()

    @patch("src.evaluation.runner.load_ground_truth")
    def test_run_empty_dataset(self, mock_load, mock_analyzer) -> None:
        """Test running on empty dataset."""
        from src.evaluation.ground_truth import GroundTruthDataset

        mock_load.return_value = GroundTruthDataset(samples=[])

        runner = EvaluationRunner(analyzer=mock_analyzer)
        results = runner.run()

        assert results.total_samples == 0
        assert len(results.results) == 0

    @patch("src.evaluation.runner.load_ground_truth")
    def test_run_with_sample_filter(self, mock_load, mock_analyzer) -> None:
        """Test running with sample ID filter."""
        from src.evaluation.ground_truth import (
            AnnotationMetadata,
            GroundTruthDataset,
            GroundTruthSample,
        )

        sample1 = GroundTruthSample(
            id="GT001",
            fix_message="8=FIX.4.4|35=8|...",
            execution=ExecutionReport(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                quantity=100.0,
                price=185.50,
                venue="NYSE",
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                fill_type="FULL",
            ),
            expert_analysis=ExpertAnalysis(quality_score=7, key_issues=[]),
            metadata=AnnotationMetadata(),
        )
        sample2 = GroundTruthSample(
            id="GT002",
            fix_message="8=FIX.4.4|35=8|...",
            execution=ExecutionReport(
                order_id="ORD002",
                symbol="MSFT",
                side="SELL",
                quantity=50.0,
                price=400.00,
                venue="NASDAQ",
                timestamp=datetime(2024, 1, 15, 11, 0, 0),
                fill_type="FULL",
            ),
            expert_analysis=ExpertAnalysis(quality_score=8, key_issues=[]),
            metadata=AnnotationMetadata(),
        )

        mock_load.return_value = GroundTruthDataset(samples=[sample1, sample2])

        runner = EvaluationRunner(analyzer=mock_analyzer)
        results = runner.run(sample_ids=["GT001"])

        assert results.total_samples == 1
        assert results.results[0].sample_id == "GT001"

    @patch("src.evaluation.runner.load_ground_truth")
    def test_threshold_checking(self, mock_load, mock_analyzer) -> None:
        """Test threshold pass/fail logic."""
        from src.evaluation.ground_truth import GroundTruthDataset

        mock_load.return_value = GroundTruthDataset(samples=[])

        runner = EvaluationRunner(
            analyzer=mock_analyzer,
            thresholds={
                "overall": 0.80,
                "insight_accuracy": 0.70,
            },
        )

        # Should pass - all thresholds met
        assert runner._check_thresholds(
            {"insight_accuracy": 0.75, "factual_correctness": 0.95},
            overall=0.85,
        )

        # Should fail - overall below threshold
        assert not runner._check_thresholds(
            {"insight_accuracy": 0.75},
            overall=0.70,
        )

        # Should fail - metric below threshold
        assert not runner._check_thresholds(
            {"insight_accuracy": 0.60},
            overall=0.85,
        )

    @patch("src.evaluation.runner.load_ground_truth")
    def test_aggregate_metrics_calculation(self, mock_load, mock_analyzer) -> None:
        """Test aggregate metric calculation."""
        from src.evaluation.ground_truth import GroundTruthDataset

        mock_load.return_value = GroundTruthDataset(samples=[])

        runner = EvaluationRunner(analyzer=mock_analyzer)

        results = [
            SingleEvalResult(
                sample_id="GT001",
                ai_analysis=TradeAnalysis(quality_score=8),
                ground_truth={},
                execution={},
                metrics={"insight_accuracy": 0.80, "factual_correctness": 0.90},
                overall_score=0.85,
                passed=True,
            ),
            SingleEvalResult(
                sample_id="GT002",
                ai_analysis=TradeAnalysis(quality_score=7),
                ground_truth={},
                execution={},
                metrics={"insight_accuracy": 0.70, "factual_correctness": 1.00},
                overall_score=0.80,
                passed=True,
            ),
        ]

        aggregate = runner._calculate_aggregate_metrics(results)

        assert aggregate["insight_accuracy"] == 0.75  # (0.8 + 0.7) / 2
        assert aggregate["factual_correctness"] == 0.95  # (0.9 + 1.0) / 2
        assert aggregate["overall"] == 0.825  # (0.85 + 0.80) / 2

    @patch("src.evaluation.runner.load_ground_truth")
    def test_aggregate_metrics_excludes_errors(self, mock_load, mock_analyzer) -> None:
        """Test that error results are excluded from aggregation."""
        from src.evaluation.ground_truth import GroundTruthDataset

        mock_load.return_value = GroundTruthDataset(samples=[])

        runner = EvaluationRunner(analyzer=mock_analyzer)

        results = [
            SingleEvalResult(
                sample_id="GT001",
                ai_analysis=TradeAnalysis(quality_score=8),
                ground_truth={},
                execution={},
                metrics={"insight_accuracy": 0.80},
                overall_score=0.85,
                passed=True,
            ),
            SingleEvalResult(
                sample_id="GT002",
                ai_analysis=TradeAnalysis(quality_score=1),
                ground_truth={},
                execution={},
                metrics={},
                overall_score=0.0,
                passed=False,
                errors=["Analysis failed"],
            ),
        ]

        aggregate = runner._calculate_aggregate_metrics(results)

        # Should only include GT001's metrics
        assert aggregate["insight_accuracy"] == 0.80
        assert aggregate["overall"] == 0.85


# --- Integration Tests ---


class TestEvaluationRunnerIntegration:
    """Integration tests for EvaluationRunner."""

    @patch("src.evaluation.runner.load_ground_truth")
    def test_full_evaluation_flow(self, mock_load, mock_analyzer) -> None:
        """Test complete evaluation flow."""
        from src.evaluation.ground_truth import (
            AnnotationMetadata,
            GroundTruthDataset,
            GroundTruthSample,
        )

        sample = GroundTruthSample(
            id="GT001",
            fix_message="8=FIX.4.4|35=8|...",
            execution=ExecutionReport(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                quantity=100.0,
                price=185.50,
                venue="NYSE",
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                fill_type="FULL",
            ),
            expert_analysis=ExpertAnalysis(
                quality_score=7,
                key_issues=["timing issue"],
                expected_observations=["Full fill achieved"],
            ),
            metadata=AnnotationMetadata(),
        )

        mock_load.return_value = GroundTruthDataset(samples=[sample])

        runner = EvaluationRunner(
            analyzer=mock_analyzer,
            thresholds={"overall": 0.50},  # Low threshold for test
        )

        progress_updates = []
        results = runner.run(progress_callback=lambda p: progress_updates.append(p.completed))

        assert results.total_samples == 1
        assert len(results.results) == 1
        assert progress_updates == [1]

        # Verify result structure
        result = results.results[0]
        assert result.sample_id == "GT001"
        assert result.ai_analysis.quality_score == 7

    @patch("src.evaluation.runner.load_ground_truth")
    def test_handles_analysis_errors(self, mock_load, mock_analyzer) -> None:
        """Test handling of analysis errors."""
        from src.evaluation.ground_truth import (
            AnnotationMetadata,
            GroundTruthDataset,
            GroundTruthSample,
        )

        sample = GroundTruthSample(
            id="GT001",
            fix_message="8=FIX.4.4|35=8|...",
            execution=ExecutionReport(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                quantity=100.0,
                price=185.50,
                venue="NYSE",
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                fill_type="FULL",
            ),
            expert_analysis=ExpertAnalysis(quality_score=7, key_issues=[]),
            metadata=AnnotationMetadata(),
        )

        mock_load.return_value = GroundTruthDataset(samples=[sample])

        # Make analyzer raise an error
        mock_analyzer.analyze.side_effect = AnalysisError("API error")

        runner = EvaluationRunner(analyzer=mock_analyzer)
        results = runner.run()

        assert results.total_samples == 1
        assert results.error_count == 1
        assert results.results[0].passed is False
        assert len(results.results[0].errors) > 0
