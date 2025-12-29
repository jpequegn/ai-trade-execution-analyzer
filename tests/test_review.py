"""Tests for human review interface module."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.parsers.fix_parser import ExecutionReport
from src.parsers.models import AnalysisResult, TradeAnalysis
from src.review.models import (
    HumanFeedback,
    ReviewItem,
    ReviewSessionStats,
    ReviewStatus,
    SamplingStrategy,
)
from src.review.queue import (
    ReviewQueue,
    create_stratified_queue,
    get_review_queue,
)
from src.review.storage import FeedbackStore


# Fixtures
@pytest.fixture
def sample_execution() -> ExecutionReport:
    """Create a sample execution report."""
    return ExecutionReport(
        order_id="ORD001",
        symbol="AAPL",
        side="BUY",
        quantity=100.0,
        price=150.50,
        venue="NYSE",
        fill_type="FULL",
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_analysis() -> TradeAnalysis:
    """Create a sample trade analysis."""
    return TradeAnalysis(
        quality_score=7,
        observations=["Executed at market open", "Good fill rate"],
        issues=["Minor slippage detected"],
        recommendations=["Consider limit orders"],
        confidence=0.85,
    )


@pytest.fixture
def sample_result(
    sample_execution: ExecutionReport, sample_analysis: TradeAnalysis
) -> AnalysisResult:
    """Create a sample analysis result."""
    return AnalysisResult(
        execution=sample_execution,
        analysis=sample_analysis,
        analysis_id="AN001",
        analyzed_at=datetime.now(),
    )


@pytest.fixture
def sample_results() -> list[AnalysisResult]:
    """Create a list of sample analysis results with varying scores."""
    results = []
    for i in range(10):
        execution = ExecutionReport(
            order_id=f"ORD{i:03d}",
            symbol=f"SYM{i}",
            side="BUY" if i % 2 == 0 else "SELL",
            quantity=100.0 * (i + 1),
            price=100.0 + i * 10,
            venue="NYSE",
            fill_type="FULL",
            timestamp=datetime.now(),
        )
        analysis = TradeAnalysis(
            quality_score=min(10, max(1, i + 1)),
            confidence=0.1 * (i + 1),
            observations=[f"Observation {i}"],
            issues=[] if i > 5 else [f"Issue {i}"],
        )
        result = AnalysisResult(
            execution=execution,
            analysis=analysis,
            analysis_id=f"AN{i:03d}",
            analyzed_at=datetime.now(),
        )
        results.append(result)
    return results


@pytest.fixture
def temp_feedback_store() -> FeedbackStore:
    """Create a temporary feedback store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "feedback.json"
        yield FeedbackStore(path)


@pytest.fixture
def sample_feedback() -> HumanFeedback:
    """Create a sample human feedback."""
    return HumanFeedback(
        analysis_id="AN001",
        reviewer_id="expert_1",
        agrees_with_score=True,
        human_score=7,
        review_time_seconds=60,
    )


# Test HumanFeedback Model
class TestHumanFeedback:
    """Tests for HumanFeedback model."""

    def test_create_feedback(self) -> None:
        """Test creating basic feedback."""
        feedback = HumanFeedback(
            analysis_id="AN001",
            reviewer_id="expert_1",
            agrees_with_score=True,
        )
        assert feedback.analysis_id == "AN001"
        assert feedback.reviewer_id == "expert_1"
        assert feedback.agrees_with_score is True
        assert feedback.feedback_id is not None

    def test_feedback_with_corrections(self) -> None:
        """Test feedback with corrections."""
        feedback = HumanFeedback(
            analysis_id="AN001",
            reviewer_id="expert_1",
            agrees_with_score=False,
            score_correction=8,
            missing_issues=["Missed timing issue"],
            incorrect_issues=["False positive on slippage"],
        )
        assert feedback.has_corrections is True
        # correction_count = 1 (score_correction) + 1 (missing_issues) + 1 (incorrect_issues)
        assert feedback.correction_count == 3

    def test_feedback_without_corrections(self) -> None:
        """Test feedback without corrections."""
        feedback = HumanFeedback(
            analysis_id="AN001",
            reviewer_id="expert_1",
            agrees_with_score=True,
        )
        assert feedback.has_corrections is False
        assert feedback.correction_count == 0

    def test_feedback_score_validation(self) -> None:
        """Test score correction validation."""
        with pytest.raises(ValueError):
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert_1",
                agrees_with_score=False,
                score_correction=11,  # Invalid: > 10
            )

    def test_feedback_default_status(self) -> None:
        """Test default status is COMPLETED."""
        feedback = HumanFeedback(
            analysis_id="AN001",
            reviewer_id="expert_1",
            agrees_with_score=True,
        )
        assert feedback.status == ReviewStatus.COMPLETED


# Test ReviewItem Model
class TestReviewItem:
    """Tests for ReviewItem model."""

    def test_create_review_item(self) -> None:
        """Test creating a review item."""
        item = ReviewItem(
            analysis_id="AN001",
            order_id="ORD001",
            symbol="AAPL",
            ai_score=7,
            ai_confidence=0.85,
        )
        assert item.analysis_id == "AN001"
        assert item.is_pending is True
        assert item.is_reviewed is False

    def test_review_item_status_properties(self) -> None:
        """Test status property helpers."""
        item = ReviewItem(
            analysis_id="AN001",
            order_id="ORD001",
            symbol="AAPL",
            ai_score=7,
            status=ReviewStatus.COMPLETED,
        )
        assert item.is_pending is False
        assert item.is_reviewed is True


# Test ReviewSessionStats Model
class TestReviewSessionStats:
    """Tests for ReviewSessionStats model."""

    def test_create_session_stats(self) -> None:
        """Test creating session stats."""
        stats = ReviewSessionStats(reviewer_id="expert_1")
        assert stats.reviewer_id == "expert_1"
        assert stats.total_reviewed == 0
        assert stats.total_agreed == 0

    def test_update_stats(self) -> None:
        """Test updating stats with feedback."""
        stats = ReviewSessionStats(reviewer_id="expert_1")
        feedback = HumanFeedback(
            analysis_id="AN001",
            reviewer_id="expert_1",
            agrees_with_score=True,
            review_time_seconds=30,
        )
        stats.update(feedback)
        assert stats.total_reviewed == 1
        assert stats.total_agreed == 1

    def test_agreement_rate(self) -> None:
        """Test agreement rate calculation."""
        stats = ReviewSessionStats(reviewer_id="expert_1")
        stats.total_reviewed = 10
        stats.total_agreed = 7
        assert stats.agreement_rate == 0.7

    def test_agreement_rate_empty(self) -> None:
        """Test agreement rate with no reviews."""
        stats = ReviewSessionStats(reviewer_id="expert_1")
        assert stats.agreement_rate == 0.0


# Test SamplingStrategy Enum
class TestSamplingStrategy:
    """Tests for SamplingStrategy enum."""

    def test_all_strategies_exist(self) -> None:
        """Test all expected strategies exist."""
        assert SamplingStrategy.RANDOM.value == "random"
        assert SamplingStrategy.LOWEST_CONFIDENCE.value == "lowest_confidence"
        assert SamplingStrategy.HIGHEST_CONFIDENCE.value == "highest_confidence"
        assert SamplingStrategy.NEWEST.value == "newest"
        assert SamplingStrategy.OLDEST.value == "oldest"
        assert SamplingStrategy.LOW_SCORE.value == "low_score"
        assert SamplingStrategy.HIGH_SCORE.value == "high_score"

    def test_strategy_from_string(self) -> None:
        """Test creating strategy from string."""
        strategy = SamplingStrategy("lowest_confidence")
        assert strategy == SamplingStrategy.LOWEST_CONFIDENCE


# Test ReviewQueue
class TestReviewQueue:
    """Tests for ReviewQueue class."""

    def test_create_empty_queue(self) -> None:
        """Test creating an empty queue."""
        queue = ReviewQueue()
        assert len(queue) == 0
        assert queue.pending_count == 0
        assert queue.completed_count == 0

    def test_create_queue_with_items(self) -> None:
        """Test creating a queue with items."""
        items = [
            ReviewItem(analysis_id=f"AN{i:03d}", order_id=f"ORD{i:03d}", symbol="AAPL", ai_score=7)
            for i in range(5)
        ]
        queue = ReviewQueue(items=items)
        assert len(queue) == 5
        assert queue.pending_count == 5

    def test_get_next(self) -> None:
        """Test getting next item from queue."""
        items = [ReviewItem(analysis_id="AN001", order_id="ORD001", symbol="AAPL", ai_score=7)]
        queue = ReviewQueue(items=items)
        next_item = queue.get_next()
        assert next_item is not None
        assert next_item.analysis_id == "AN001"

    def test_get_next_empty_queue(self) -> None:
        """Test getting next from empty queue."""
        queue = ReviewQueue()
        assert queue.get_next() is None

    def test_mark_completed(self) -> None:
        """Test marking item as completed."""
        items = [ReviewItem(analysis_id="AN001", order_id="ORD001", symbol="AAPL", ai_score=7)]
        queue = ReviewQueue(items=items)
        result = queue.mark_completed("AN001")
        assert result is True
        assert queue.completed_count == 1
        assert queue.pending_count == 0

    def test_mark_skipped(self) -> None:
        """Test marking item as skipped."""
        items = [ReviewItem(analysis_id="AN001", order_id="ORD001", symbol="AAPL", ai_score=7)]
        queue = ReviewQueue(items=items)
        result = queue.mark_skipped("AN001")
        assert result is True

    def test_progress(self) -> None:
        """Test progress calculation."""
        items = [
            ReviewItem(analysis_id=f"AN{i:03d}", order_id=f"ORD{i:03d}", symbol="AAPL", ai_score=7)
            for i in range(10)
        ]
        queue = ReviewQueue(items=items)
        queue.mark_completed("AN000")
        queue.mark_completed("AN001")
        assert queue.progress == 20.0

    def test_get_by_id(self) -> None:
        """Test getting item by ID."""
        items = [
            ReviewItem(analysis_id="AN001", order_id="ORD001", symbol="AAPL", ai_score=7),
            ReviewItem(analysis_id="AN002", order_id="ORD002", symbol="GOOGL", ai_score=8),
        ]
        queue = ReviewQueue(items=items)
        item = queue.get_by_id("AN002")
        assert item is not None
        assert item.symbol == "GOOGL"

    def test_add_item(self) -> None:
        """Test adding item to queue."""
        queue = ReviewQueue()
        item = ReviewItem(analysis_id="AN001", order_id="ORD001", symbol="AAPL", ai_score=7)
        queue.add_item(item)
        assert len(queue) == 1

    def test_remove_item(self) -> None:
        """Test removing item from queue."""
        items = [ReviewItem(analysis_id="AN001", order_id="ORD001", symbol="AAPL", ai_score=7)]
        queue = ReviewQueue(items=items)
        result = queue.remove_item("AN001")
        assert result is True
        assert len(queue) == 0

    def test_reset(self) -> None:
        """Test resetting queue."""
        items = [ReviewItem(analysis_id="AN001", order_id="ORD001", symbol="AAPL", ai_score=7)]
        queue = ReviewQueue(items=items)
        queue.mark_completed("AN001")
        queue.reset()
        assert queue.pending_count == 1
        assert queue.completed_count == 0

    def test_queue_iteration(self) -> None:
        """Test iterating over queue."""
        items = [
            ReviewItem(analysis_id=f"AN{i:03d}", order_id=f"ORD{i:03d}", symbol="AAPL", ai_score=7)
            for i in range(3)
        ]
        queue = ReviewQueue(items=items)
        count = sum(1 for _ in queue)
        assert count == 3

    def test_to_summary(self) -> None:
        """Test generating queue summary."""
        items = [ReviewItem(analysis_id="AN001", order_id="ORD001", symbol="AAPL", ai_score=7)]
        queue = ReviewQueue(items=items, strategy=SamplingStrategy.RANDOM)
        summary = queue.to_summary()
        assert "random" in summary.lower()
        assert "Total items: 1" in summary


# Test get_review_queue function
class TestGetReviewQueue:
    """Tests for get_review_queue function."""

    def test_create_queue_from_results(self, sample_results: list[AnalysisResult]) -> None:
        """Test creating queue from results."""
        queue = get_review_queue(sample_results)
        assert len(queue) == len(sample_results)

    def test_queue_with_limit(self, sample_results: list[AnalysisResult]) -> None:
        """Test queue with limit."""
        queue = get_review_queue(sample_results, limit=5)
        assert len(queue) == 5

    def test_lowest_confidence_strategy(self, sample_results: list[AnalysisResult]) -> None:
        """Test lowest confidence strategy."""
        queue = get_review_queue(
            sample_results, strategy=SamplingStrategy.LOWEST_CONFIDENCE, limit=3
        )
        assert len(queue) == 3
        # Should have lowest confidence items first
        items = list(queue)
        assert items[0].ai_confidence <= items[1].ai_confidence

    def test_highest_confidence_strategy(self, sample_results: list[AnalysisResult]) -> None:
        """Test highest confidence strategy."""
        queue = get_review_queue(
            sample_results, strategy=SamplingStrategy.HIGHEST_CONFIDENCE, limit=3
        )
        assert len(queue) == 3
        items = list(queue)
        assert items[0].ai_confidence >= items[1].ai_confidence

    def test_exclude_ids(self, sample_results: list[AnalysisResult]) -> None:
        """Test excluding specific IDs."""
        exclude = {"AN000", "AN001", "AN002"}
        queue = get_review_queue(sample_results, exclude_ids=exclude)
        assert len(queue) == len(sample_results) - 3
        for item in queue:
            assert item.analysis_id not in exclude

    def test_string_strategy(self, sample_results: list[AnalysisResult]) -> None:
        """Test passing strategy as string."""
        queue = get_review_queue(sample_results, strategy="lowest_confidence")
        assert queue.strategy == SamplingStrategy.LOWEST_CONFIDENCE


# Test create_stratified_queue function
class TestCreateStratifiedQueue:
    """Tests for create_stratified_queue function."""

    def test_stratified_by_score(self, sample_results: list[AnalysisResult]) -> None:
        """Test stratified sampling by score."""
        queue = create_stratified_queue(sample_results, strata_key="score", samples_per_stratum=2)
        # Should have samples from different score ranges
        assert len(queue) > 0

    def test_stratified_by_confidence(self, sample_results: list[AnalysisResult]) -> None:
        """Test stratified sampling by confidence."""
        queue = create_stratified_queue(
            sample_results, strata_key="confidence", samples_per_stratum=2
        )
        assert len(queue) > 0


# Test FeedbackStore
class TestFeedbackStore:
    """Tests for FeedbackStore class."""

    def test_create_store(self) -> None:
        """Test creating a feedback store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            assert store.path == path
            assert store.count() == 0

    def test_save_feedback(self, sample_feedback: HumanFeedback) -> None:
        """Test saving feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            store.save(sample_feedback)
            assert store.count() == 1

    def test_get_all(self, sample_feedback: HumanFeedback) -> None:
        """Test getting all feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            store.save(sample_feedback)
            all_feedback = store.get_all()
            assert len(all_feedback) == 1
            assert all_feedback[0].analysis_id == sample_feedback.analysis_id

    def test_get_by_id(self, sample_feedback: HumanFeedback) -> None:
        """Test getting feedback by ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            store.save(sample_feedback)
            retrieved = store.get_by_id(sample_feedback.feedback_id)
            assert retrieved is not None
            assert retrieved.analysis_id == sample_feedback.analysis_id

    def test_get_by_analysis(self, sample_feedback: HumanFeedback) -> None:
        """Test getting feedback by analysis ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            store.save(sample_feedback)
            feedback_list = store.get_by_analysis("AN001")
            assert len(feedback_list) == 1

    def test_get_by_reviewer(self, sample_feedback: HumanFeedback) -> None:
        """Test getting feedback by reviewer ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            store.save(sample_feedback)
            feedback_list = store.get_by_reviewer("expert_1")
            assert len(feedback_list) == 1

    def test_delete_feedback(self, sample_feedback: HumanFeedback) -> None:
        """Test deleting feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            store.save(sample_feedback)
            result = store.delete(sample_feedback.feedback_id)
            assert result is True
            assert store.count() == 0

    def test_clear_feedback(self, sample_feedback: HumanFeedback) -> None:
        """Test clearing all feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            store.save(sample_feedback)
            store.save(sample_feedback)
            cleared = store.clear()
            assert cleared == 2
            assert store.count() == 0

    def test_save_batch(self) -> None:
        """Test saving multiple feedback items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            feedback_list = [
                HumanFeedback(
                    analysis_id=f"AN{i:03d}", reviewer_id="expert_1", agrees_with_score=True
                )
                for i in range(5)
            ]
            count = store.save_batch(feedback_list)
            assert count == 5
            assert store.count() == 5

    def test_get_corrections_only(self) -> None:
        """Test getting only feedback with corrections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)

            # Feedback without corrections
            store.save(
                HumanFeedback(
                    analysis_id="AN001",
                    reviewer_id="expert_1",
                    agrees_with_score=True,
                )
            )

            # Feedback with corrections
            store.save(
                HumanFeedback(
                    analysis_id="AN002",
                    reviewer_id="expert_1",
                    agrees_with_score=False,
                    missing_issues=["Missed issue"],
                )
            )

            corrections = store.get_corrections_only()
            assert len(corrections) == 1
            assert corrections[0].analysis_id == "AN002"

    def test_get_disagreements(self) -> None:
        """Test getting feedback where reviewer disagreed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)

            store.save(
                HumanFeedback(
                    analysis_id="AN001",
                    reviewer_id="expert_1",
                    agrees_with_score=True,
                )
            )

            store.save(
                HumanFeedback(
                    analysis_id="AN002",
                    reviewer_id="expert_1",
                    agrees_with_score=False,
                )
            )

            disagreements = store.get_disagreements()
            assert len(disagreements) == 1
            assert disagreements[0].analysis_id == "AN002"

    def test_export_csv(self, sample_feedback: HumanFeedback) -> None:
        """Test exporting to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "feedback.json"
            export_path = Path(tmpdir) / "export.csv"
            store = FeedbackStore(store_path)
            store.save(sample_feedback)
            count = store.export_csv(export_path)
            assert count == 1
            assert export_path.exists()

    def test_export_json(self, sample_feedback: HumanFeedback) -> None:
        """Test exporting to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "feedback.json"
            export_path = Path(tmpdir) / "export.json"
            store = FeedbackStore(store_path)
            store.save(sample_feedback)
            count = store.export_json(export_path)
            assert count == 1
            assert export_path.exists()

            # Verify JSON content
            with export_path.open() as f:
                data = json.load(f)
            assert len(data) == 1

    def test_save_session(self) -> None:
        """Test saving review session stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            stats = ReviewSessionStats(reviewer_id="expert_1")
            stats.total_reviewed = 5
            store.save_session(stats)
            sessions = store.get_sessions()
            assert len(sessions) == 1
            assert sessions[0].total_reviewed == 5

    def test_get_statistics(self) -> None:
        """Test getting statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)

            # Add some feedback
            for i in range(5):
                store.save(
                    HumanFeedback(
                        analysis_id=f"AN{i:03d}",
                        reviewer_id="expert_1" if i < 3 else "expert_2",
                        agrees_with_score=i % 2 == 0,
                        review_time_seconds=30,
                    )
                )

            stats = store.get_statistics()
            assert stats["total_feedback"] == 5
            assert stats["unique_reviewers"] == 2
            assert stats["unique_analyses"] == 5

    def test_iter_feedback(self, sample_feedback: HumanFeedback) -> None:
        """Test iterating over feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            store.save(sample_feedback)
            store.save(sample_feedback)

            count = sum(1 for _ in store.iter_feedback())
            assert count == 2
