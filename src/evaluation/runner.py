"""Automated evaluation pipeline for AI trade analysis quality.

This module provides an evaluation runner that compares AI-generated
trade analyses against expert ground truth annotations.

Example:
    >>> from src.evaluation.runner import EvaluationRunner
    >>> from src.agents.analyzer import TradeAnalyzer
    >>> runner = EvaluationRunner(TradeAnalyzer())
    >>> results = runner.run()
    >>> print(f"Overall score: {results.aggregate_metrics['overall']:.2%}")
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.evaluation.ground_truth import (
    GroundTruthSample,
    load_ground_truth,
)
from src.evaluation.metrics import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_WEIGHTS,
    EvaluationResult,
    evaluate,
)
from src.parsers.models import TradeAnalysis

if TYPE_CHECKING:
    from src.agents.analyzer import TradeAnalyzer

logger = logging.getLogger(__name__)


# Default quality thresholds for pass/fail
DEFAULT_THRESHOLDS = {
    "insight_accuracy": 0.70,
    "factual_correctness": 0.90,
    "completeness": 0.65,
    "score_accuracy": 0.70,
    "overall": 0.75,
}


class SingleEvalResult(BaseModel):
    """Result of evaluating a single sample.

    Attributes:
        sample_id: ID of the evaluated sample (e.g., "GT001").
        ai_analysis: The AI-generated trade analysis.
        ground_truth: Expert analysis from ground truth.
        execution: The execution data.
        metrics: Individual metric scores.
        overall_score: Weighted overall quality score.
        passed: Whether the sample passed quality thresholds.
        errors: List of errors encountered during evaluation.
        latency_ms: Time taken for analysis in milliseconds.
    """

    sample_id: str
    ai_analysis: TradeAnalysis
    ground_truth: dict[str, object]
    execution: dict[str, object]
    metrics: dict[str, float]
    overall_score: float
    passed: bool
    errors: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class EvalResults(BaseModel):
    """Aggregated results from an evaluation run.

    Attributes:
        run_id: Unique identifier for this evaluation run.
        timestamp: When the evaluation was run.
        dataset_path: Path to the dataset used.
        total_samples: Total number of samples evaluated.
        passed: Number of samples that passed thresholds.
        failed: Number of samples that failed thresholds.
        error_count: Number of samples that encountered errors.
        aggregate_metrics: Mean metrics across all samples.
        thresholds: Quality thresholds used for pass/fail.
        results: Individual sample results.
        duration_seconds: Total evaluation duration.
    """

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    dataset_path: str | None = None
    total_samples: int = 0
    passed: int = 0
    failed: int = 0
    error_count: int = 0
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    results: list[SingleEvalResult] = Field(default_factory=list)
    duration_seconds: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage."""
        if self.total_samples == 0:
            return 0.0
        return self.passed / self.total_samples

    @property
    def success_rate(self) -> float:
        """Calculate success rate (excluding errors)."""
        successful = self.total_samples - self.error_count
        if successful == 0:
            return 0.0
        return self.passed / successful

    def get_failed_samples(self) -> list[SingleEvalResult]:
        """Get list of failed sample results."""
        return [r for r in self.results if not r.passed]

    def get_passed_samples(self) -> list[SingleEvalResult]:
        """Get list of passed sample results."""
        return [r for r in self.results if r.passed]

    def to_json(self, indent: int = 2) -> str:
        """Export results to JSON string."""
        return self.model_dump_json(indent=indent)

    def save_json(self, path: Path | str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        logger.info(f"Saved evaluation results to {path}")


@dataclass
class ProgressCallback:
    """Progress tracking callback for evaluation runs.

    Attributes:
        total: Total number of samples to evaluate.
        completed: Number of samples completed.
        passed: Number of samples passed.
        failed: Number of samples failed.
        errors: Number of samples with errors.
        callback: Optional callback function called on each update.
    """

    total: int = 0
    completed: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    callback: Callable[[ProgressCallback], None] | None = None
    start_time: datetime = field(default_factory=datetime.now)

    def update(self, passed: bool, error: bool = False) -> None:
        """Update progress with a new result."""
        self.completed += 1
        if error:
            self.errors += 1
        elif passed:
            self.passed += 1
        else:
            self.failed += 1

        if self.callback:
            self.callback(self)

    @property
    def progress_pct(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return self.completed / self.total * 100

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def samples_per_second(self) -> float:
        """Get processing rate."""
        if self.elapsed_seconds == 0:
            return 0.0
        return self.completed / self.elapsed_seconds

    @property
    def eta_seconds(self) -> float | None:
        """Estimate time remaining in seconds."""
        if self.samples_per_second == 0:
            return None
        remaining = self.total - self.completed
        return remaining / self.samples_per_second


class EvaluationRunner:
    """Runner for automated evaluation of AI trade analysis.

    This class orchestrates the evaluation pipeline:
    1. Loads ground truth dataset
    2. Runs AI analysis on each sample
    3. Compares against expert annotations
    4. Calculates metrics and pass/fail status
    5. Generates aggregated results

    Attributes:
        analyzer: TradeAnalyzer instance for AI analysis.
        dataset: Loaded ground truth dataset.
        thresholds: Quality thresholds for pass/fail.
        weights: Metric weights for overall score.
        match_threshold: Fuzzy matching threshold.
        max_workers: Maximum parallel workers.

    Example:
        >>> from src.evaluation.runner import EvaluationRunner
        >>> from src.agents.analyzer import TradeAnalyzer
        >>> runner = EvaluationRunner(TradeAnalyzer())
        >>> results = runner.run()
        >>> print(f"Pass rate: {results.pass_rate:.1%}")
    """

    def __init__(
        self,
        analyzer: TradeAnalyzer,
        dataset_path: Path | str | None = None,
        thresholds: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        max_workers: int = 4,
    ) -> None:
        """Initialize the evaluation runner.

        Args:
            analyzer: TradeAnalyzer instance for running AI analysis.
            dataset_path: Path to ground truth dataset. Uses default if None.
            thresholds: Quality thresholds for pass/fail determination.
            weights: Metric weights for overall score calculation.
            match_threshold: Threshold for fuzzy text matching.
            max_workers: Maximum parallel workers for evaluation.
        """
        self.analyzer = analyzer
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.dataset = load_ground_truth(self.dataset_path)
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.match_threshold = match_threshold
        self.max_workers = max_workers

        logger.info(
            f"Initialized EvaluationRunner with {len(self.dataset.samples)} samples, "
            f"max_workers={max_workers}"
        )

    def run(
        self,
        sample_ids: list[str] | None = None,
        progress_callback: Callable[[ProgressCallback], None] | None = None,
    ) -> EvalResults:
        """Run evaluation on dataset or subset.

        Args:
            sample_ids: Optional list of sample IDs to evaluate.
                If None, evaluates all samples in the dataset.
            progress_callback: Optional callback for progress updates.

        Returns:
            EvalResults with aggregated metrics and individual results.

        Example:
            >>> results = runner.run()
            >>> results = runner.run(sample_ids=["GT001", "GT002"])
        """
        start_time = datetime.now()
        run_id = str(uuid.uuid4())

        # Get samples to evaluate
        if sample_ids:
            samples = [s for s in self.dataset.samples if s.id in sample_ids]
            if len(samples) != len(sample_ids):
                found = {s.id for s in samples}
                missing = set(sample_ids) - found
                logger.warning(f"Sample IDs not found: {missing}")
        else:
            samples = self.dataset.samples

        if not samples:
            logger.warning("No samples to evaluate")
            return EvalResults(
                run_id=run_id,
                dataset_path=str(self.dataset_path) if self.dataset_path else None,
                thresholds=self.thresholds,
            )

        logger.info(f"Starting evaluation run {run_id} with {len(samples)} samples")

        # Initialize progress tracking
        progress = ProgressCallback(
            total=len(samples),
            callback=progress_callback,
        )

        # Run evaluations in parallel
        results: list[SingleEvalResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sample = {
                executor.submit(self.run_single, sample): sample for sample in samples
            }

            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    results.append(result)
                    progress.update(passed=result.passed, error=bool(result.errors))

                    if progress.completed % 10 == 0 or progress.completed == progress.total:
                        logger.info(
                            f"Progress: {progress.completed}/{progress.total} "
                            f"({progress.passed} passed, {progress.failed} failed, "
                            f"{progress.errors} errors)"
                        )

                except Exception as e:
                    logger.error(f"Failed to evaluate {sample.id}: {e}")
                    # Create error result
                    error_result = SingleEvalResult(
                        sample_id=sample.id,
                        ai_analysis=TradeAnalysis(quality_score=1),
                        ground_truth=sample.expert_analysis.model_dump(),
                        execution=sample.execution.model_dump(),
                        metrics={},
                        overall_score=0.0,
                        passed=False,
                        errors=[str(e)],
                    )
                    results.append(error_result)
                    progress.update(passed=False, error=True)

        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)

        # Build final results
        duration = (datetime.now() - start_time).total_seconds()

        eval_results = EvalResults(
            run_id=run_id,
            timestamp=start_time,
            dataset_path=str(self.dataset_path) if self.dataset_path else None,
            total_samples=len(results),
            passed=progress.passed,
            failed=progress.failed,
            error_count=progress.errors,
            aggregate_metrics=aggregate_metrics,
            thresholds=self.thresholds,
            results=results,
            duration_seconds=duration,
        )

        logger.info(
            f"Evaluation complete: {eval_results.passed}/{eval_results.total_samples} passed "
            f"({eval_results.pass_rate:.1%}), took {duration:.1f}s"
        )

        return eval_results

    def run_single(self, sample: GroundTruthSample) -> SingleEvalResult:
        """Evaluate a single ground truth sample.

        Args:
            sample: The ground truth sample to evaluate.

        Returns:
            SingleEvalResult with metrics and pass/fail status.
        """
        errors: list[str] = []
        start_time = datetime.now()

        try:
            # Run AI analysis
            analysis_result = self.analyzer.analyze(sample.execution)
            ai_analysis = analysis_result.analysis
            latency_ms = analysis_result.latency_ms

        except Exception as e:
            logger.warning(f"Analysis failed for {sample.id}: {e}")
            errors.append(f"Analysis error: {e}")
            # Return failed result with fallback analysis
            return SingleEvalResult(
                sample_id=sample.id,
                ai_analysis=TradeAnalysis(quality_score=1),
                ground_truth=sample.expert_analysis.model_dump(),
                execution=sample.execution.model_dump(),
                metrics={},
                overall_score=0.0,
                passed=False,
                errors=errors,
            )

        overall_score = 0.0
        metrics: dict[str, float] = {}
        passed = False

        try:
            # Calculate metrics
            eval_result: EvaluationResult = evaluate(
                ai_analysis=ai_analysis,
                ground_truth=sample.expert_analysis,
                execution=sample.execution,
                weights=self.weights,
                threshold=self.match_threshold,
            )

            # Extract metrics
            metrics = {
                "insight_accuracy": eval_result.insight_accuracy,
                "factual_correctness": eval_result.factual_correctness,
                "completeness": eval_result.completeness,
                "score_accuracy": eval_result.score_accuracy,
            }
            overall_score = eval_result.overall_score

            # Determine pass/fail
            passed = self._check_thresholds(metrics, overall_score)

        except Exception as e:
            logger.warning(f"Metric calculation failed for {sample.id}: {e}")
            errors.append(f"Metric error: {e}")

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return SingleEvalResult(
            sample_id=sample.id,
            ai_analysis=ai_analysis,
            ground_truth=sample.expert_analysis.model_dump(),
            execution=sample.execution.model_dump(),
            metrics=metrics,
            overall_score=overall_score,
            passed=passed,
            errors=errors,
            latency_ms=latency_ms if not errors else elapsed_ms,
        )

    def _check_thresholds(self, metrics: dict[str, float], overall: float) -> bool:
        """Check if metrics meet quality thresholds.

        Args:
            metrics: Individual metric scores.
            overall: Overall quality score.

        Returns:
            True if all thresholds are met.
        """
        # Check overall threshold
        if overall < self.thresholds.get("overall", 0.75):
            return False

        # Check individual metric thresholds
        for metric_name, score in metrics.items():
            threshold = self.thresholds.get(metric_name)
            if threshold is not None and score < threshold:
                return False

        return True

    def _calculate_aggregate_metrics(self, results: list[SingleEvalResult]) -> dict[str, float]:
        """Calculate aggregate metrics from individual results.

        Args:
            results: List of individual evaluation results.

        Returns:
            Dictionary of mean metric values.
        """
        if not results:
            return {}

        # Filter out error results
        valid_results = [r for r in results if not r.errors and r.metrics]

        if not valid_results:
            return {}

        # Initialize accumulators
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        for result in valid_results:
            for metric_name, score in result.metrics.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + score
                metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1

        # Calculate means
        aggregate: dict[str, float] = {}
        for metric_name in metric_sums:
            aggregate[metric_name] = metric_sums[metric_name] / metric_counts[metric_name]

        # Add overall score
        overall_sum = sum(r.overall_score for r in valid_results)
        aggregate["overall"] = overall_sum / len(valid_results)

        return aggregate
