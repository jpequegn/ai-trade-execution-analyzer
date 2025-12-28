"""Tests for the evaluation reporting module."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.evaluation.reporting import (
    compare_runs,
    format_comparison_report,
    generate_json_export,
    generate_markdown_report,
    save_report,
)
from src.evaluation.runner import EvalResults, SingleEvalResult
from src.parsers.models import TradeAnalysis

# --- Fixtures ---


@pytest.fixture
def sample_results() -> EvalResults:
    """Create sample evaluation results."""
    passed_result = SingleEvalResult(
        sample_id="GT001",
        ai_analysis=TradeAnalysis(
            quality_score=8,
            issues=["minor issue"],
            observations=["Good execution"],
        ),
        ground_truth={"quality_score": 8},
        execution={"symbol": "AAPL"},
        metrics={
            "insight_accuracy": 0.90,
            "factual_correctness": 1.00,
            "completeness": 0.85,
            "score_accuracy": 1.00,
        },
        overall_score=0.92,
        passed=True,
        latency_ms=450.0,
    )

    failed_result = SingleEvalResult(
        sample_id="GT002",
        ai_analysis=TradeAnalysis(
            quality_score=4,
            issues=[],
            observations=["Order placed"],
        ),
        ground_truth={"quality_score": 7},
        execution={"symbol": "MSFT"},
        metrics={
            "insight_accuracy": 0.40,
            "factual_correctness": 0.80,
            "completeness": 0.50,
            "score_accuracy": 0.60,
        },
        overall_score=0.55,
        passed=False,
        latency_ms=520.0,
    )

    return EvalResults(
        run_id="test-run-123",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        dataset_path="ground_truth.json",
        total_samples=2,
        passed=1,
        failed=1,
        error_count=0,
        aggregate_metrics={
            "insight_accuracy": 0.65,
            "factual_correctness": 0.90,
            "completeness": 0.675,
            "score_accuracy": 0.80,
            "overall": 0.735,
        },
        thresholds={
            "insight_accuracy": 0.70,
            "factual_correctness": 0.90,
            "completeness": 0.65,
            "score_accuracy": 0.70,
            "overall": 0.75,
        },
        results=[passed_result, failed_result],
        duration_seconds=15.5,
    )


@pytest.fixture
def baseline_results() -> EvalResults:
    """Create baseline results for comparison."""
    return EvalResults(
        run_id="baseline-run-456",
        timestamp=datetime(2024, 1, 10, 14, 0, 0),
        total_samples=2,
        passed=2,
        failed=0,
        aggregate_metrics={
            "insight_accuracy": 0.80,
            "factual_correctness": 0.95,
            "completeness": 0.75,
            "score_accuracy": 0.85,
            "overall": 0.82,
        },
        thresholds={
            "overall": 0.75,
        },
        results=[
            SingleEvalResult(
                sample_id="GT001",
                ai_analysis=TradeAnalysis(quality_score=8),
                ground_truth={},
                execution={},
                metrics={"insight_accuracy": 0.85},
                overall_score=0.88,
                passed=True,
            ),
            SingleEvalResult(
                sample_id="GT002",
                ai_analysis=TradeAnalysis(quality_score=7),
                ground_truth={},
                execution={},
                metrics={"insight_accuracy": 0.75},
                overall_score=0.76,
                passed=True,
            ),
        ],
    )


# --- Markdown Report Tests ---


class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report function."""

    def test_basic_report_structure(self, sample_results: EvalResults) -> None:
        """Test basic report has expected sections."""
        report = generate_markdown_report(sample_results)

        assert "# Evaluation Report" in report
        assert "## Summary" in report
        assert "## Aggregate Metrics" in report
        assert "## Quality Thresholds" in report

    def test_summary_section(self, sample_results: EvalResults) -> None:
        """Test summary section content."""
        report = generate_markdown_report(sample_results)

        assert "test-run-123" in report
        assert "Total Samples**: 2" in report
        assert "Passed**: 1" in report
        assert "Failed**: 1" in report
        assert "Duration**: 15.5s" in report

    def test_metrics_table(self, sample_results: EvalResults) -> None:
        """Test metrics table content."""
        report = generate_markdown_report(sample_results)

        assert "| Insight Accuracy |" in report
        assert "| Factual Correctness |" in report
        assert "| Overall |" in report
        # Check for pass/fail indicators
        assert "✅" in report or "❌" in report

    def test_failed_samples_section(self, sample_results: EvalResults) -> None:
        """Test failed samples are shown."""
        report = generate_markdown_report(sample_results, include_failed_details=True)

        assert "## Failed Samples" in report
        assert "GT002" in report
        assert "Low insight accuracy" in report or "insight_accuracy" in report

    def test_all_samples_option(self, sample_results: EvalResults) -> None:
        """Test all samples table option."""
        report = generate_markdown_report(sample_results, include_all_samples=True)

        assert "## All Sample Results" in report
        assert "GT001" in report
        assert "GT002" in report

    def test_max_failed_samples_limit(self) -> None:
        """Test limit on failed samples shown."""
        # Create results with many failed samples
        failed_results = [
            SingleEvalResult(
                sample_id=f"GT{i:03d}",
                ai_analysis=TradeAnalysis(quality_score=3),
                ground_truth={},
                execution={},
                metrics={"insight_accuracy": 0.3},
                overall_score=0.4,
                passed=False,
            )
            for i in range(25)
        ]

        results = EvalResults(
            total_samples=25,
            failed=25,
            thresholds={"overall": 0.75},
            results=failed_results,
        )

        report = generate_markdown_report(results, max_failed_samples=5)

        # Should mention remaining samples
        assert "more failed samples" in report


# --- JSON Export Tests ---


class TestGenerateJsonExport:
    """Tests for generate_json_export function."""

    def test_valid_json_output(self, sample_results: EvalResults) -> None:
        """Test output is valid JSON."""
        import json

        json_str = generate_json_export(sample_results)
        data = json.loads(json_str)

        assert data["run_id"] == "test-run-123"
        assert data["total_samples"] == 2
        assert data["passed"] == 1

    def test_includes_all_fields(self, sample_results: EvalResults) -> None:
        """Test all fields are included."""
        import json

        json_str = generate_json_export(sample_results)
        data = json.loads(json_str)

        assert "aggregate_metrics" in data
        assert "thresholds" in data
        assert "results" in data
        assert len(data["results"]) == 2


# --- Save Report Tests ---


class TestSaveReport:
    """Tests for save_report function."""

    def test_save_markdown(self, sample_results: EvalResults) -> None:
        """Test saving markdown report."""
        with TemporaryDirectory() as tmpdir:
            saved = save_report(
                sample_results,
                tmpdir,
                include_json=False,
                include_markdown=True,
            )

            assert "markdown" in saved
            assert saved["markdown"].exists()
            assert saved["markdown"].suffix == ".md"

            content = saved["markdown"].read_text()
            assert "# Evaluation Report" in content

    def test_save_json(self, sample_results: EvalResults) -> None:
        """Test saving JSON report."""
        with TemporaryDirectory() as tmpdir:
            saved = save_report(
                sample_results,
                tmpdir,
                include_json=True,
                include_markdown=False,
            )

            assert "json" in saved
            assert saved["json"].exists()
            assert saved["json"].suffix == ".json"

    def test_save_both_formats(self, sample_results: EvalResults) -> None:
        """Test saving both formats."""
        with TemporaryDirectory() as tmpdir:
            saved = save_report(sample_results, tmpdir)

            assert "json" in saved
            assert "markdown" in saved

    def test_custom_basename(self, sample_results: EvalResults) -> None:
        """Test custom basename."""
        with TemporaryDirectory() as tmpdir:
            saved = save_report(sample_results, tmpdir, basename="custom_report")

            assert "custom_report" in str(saved["markdown"])

    def test_creates_directory(self, sample_results: EvalResults) -> None:
        """Test directory creation."""
        with TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "reports"
            saved = save_report(sample_results, nested_path)

            assert nested_path.exists()
            assert saved["markdown"].exists()


# --- Comparison Tests ---


class TestCompareRuns:
    """Tests for compare_runs function."""

    def test_basic_comparison(
        self, sample_results: EvalResults, baseline_results: EvalResults
    ) -> None:
        """Test basic run comparison."""
        comparison = compare_runs(sample_results, baseline_results)

        assert comparison["current_run_id"] == "test-run-123"
        assert comparison["baseline_run_id"] == "baseline-run-456"
        assert "metric_deltas" in comparison

    def test_detects_regressions(
        self, sample_results: EvalResults, baseline_results: EvalResults
    ) -> None:
        """Test regression detection."""
        comparison = compare_runs(sample_results, baseline_results)

        # Sample results have lower metrics than baseline
        assert comparison["has_regressions"] is True
        assert len(comparison["regressions"]) > 0

    def test_detects_improvements(self) -> None:
        """Test improvement detection."""
        current = EvalResults(
            aggregate_metrics={"overall": 0.90},
            thresholds={},
        )
        baseline = EvalResults(
            aggregate_metrics={"overall": 0.70},
            thresholds={},
        )

        comparison = compare_runs(current, baseline)

        assert len(comparison["improvements"]) > 0
        assert comparison["has_regressions"] is False

    def test_metric_deltas(
        self, sample_results: EvalResults, baseline_results: EvalResults
    ) -> None:
        """Test metric delta calculation."""
        comparison = compare_runs(sample_results, baseline_results)

        deltas = comparison["metric_deltas"]
        assert "overall" in deltas

        overall_delta = deltas["overall"]
        assert "current" in overall_delta
        assert "baseline" in overall_delta
        assert "delta" in overall_delta
        assert "delta_pct" in overall_delta

    def test_sample_status_changes(
        self, sample_results: EvalResults, baseline_results: EvalResults
    ) -> None:
        """Test sample status change detection."""
        comparison = compare_runs(sample_results, baseline_results)

        sample_changes = comparison["sample_changes"]
        # GT002 changed from passed to failed
        assert "GT002" in sample_changes
        assert sample_changes["GT002"]["status_change"] == "regressed"


class TestFormatComparisonReport:
    """Tests for format_comparison_report function."""

    def test_basic_format(self, sample_results: EvalResults, baseline_results: EvalResults) -> None:
        """Test basic comparison report formatting."""
        comparison = compare_runs(sample_results, baseline_results)
        report = format_comparison_report(comparison)

        assert "# Evaluation Run Comparison" in report
        assert "## Status:" in report
        assert "## Metric Changes" in report

    def test_shows_regressions(
        self, sample_results: EvalResults, baseline_results: EvalResults
    ) -> None:
        """Test regression section in report."""
        comparison = compare_runs(sample_results, baseline_results)
        report = format_comparison_report(comparison)

        assert "## Regressions" in report
        assert "REGRESSIONS DETECTED" in report

    def test_shows_improvements(self) -> None:
        """Test improvements section in report."""
        current = EvalResults(
            run_id="current",
            aggregate_metrics={"overall": 0.90},
            thresholds={},
            results=[],
        )
        baseline = EvalResults(
            run_id="baseline",
            aggregate_metrics={"overall": 0.70},
            thresholds={},
            results=[],
        )

        comparison = compare_runs(current, baseline)
        report = format_comparison_report(comparison)

        assert "## Improvements" in report
        assert "No Regressions" in report

    def test_metric_table(self, sample_results: EvalResults, baseline_results: EvalResults) -> None:
        """Test metric comparison table."""
        comparison = compare_runs(sample_results, baseline_results)
        report = format_comparison_report(comparison)

        assert "| Metric | Baseline | Current | Delta | Change |" in report
        assert "📈" in report or "📉" in report  # Change indicators
