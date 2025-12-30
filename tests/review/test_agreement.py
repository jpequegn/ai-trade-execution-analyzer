"""Tests for agreement metrics module."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.review.agreement import (
    AgreementCalculator,
    AgreementMetrics,
    AgreementTracker,
    calculate_agreement_rate,
)
from src.review.models import HumanFeedback


class TestAgreementMetrics:
    """Tests for AgreementMetrics model."""

    def test_default_values(self) -> None:
        """Test default metric values."""
        metrics = AgreementMetrics()

        assert metrics.total_reviews == 0
        assert metrics.score_agreement_rate == 0.0
        assert metrics.issue_agreement_rate == 0.0
        assert metrics.overall_agreement == 0.0
        assert metrics.trend == "stable"
        assert metrics.by_category == {}
        assert metrics.by_score_range == {}
        assert metrics.by_reviewer == {}
        assert metrics.time_series == []

    def test_to_summary(self) -> None:
        """Test summary generation."""
        metrics = AgreementMetrics(
            total_reviews=50,
            score_agreement_rate=0.87,
            issue_agreement_rate=0.82,
            overall_agreement=0.85,
            trend="stable",
            by_score_range={"high (8-10)": 0.90, "low (1-3)": 0.75},
            by_reviewer={"expert1": 0.88, "expert2": 0.82},
        )

        summary = metrics.to_summary()

        assert "Total Reviews: 50" in summary
        assert "Score Agreement: 87.0%" in summary
        assert "Issue Agreement: 82.0%" in summary
        assert "Overall Agreement: 85.0%" in summary
        assert "Trend: Stable" in summary
        assert "high (8-10): 90.0%" in summary
        assert "expert1: 88.0%" in summary


class TestAgreementCalculator:
    """Tests for AgreementCalculator class."""

    def test_calculate_score_agreement_empty(self) -> None:
        """Test score agreement with no feedback."""
        calculator = AgreementCalculator()
        result = calculator.calculate_score_agreement([])

        assert result == 0.0

    def test_calculate_score_agreement_all_agree(self) -> None:
        """Test score agreement when all reviewers agree."""
        feedbacks = [
            HumanFeedback(
                analysis_id=f"AN{i:03d}",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
            )
            for i in range(5)
        ]
        calculator = AgreementCalculator()

        result = calculator.calculate_score_agreement(feedbacks)

        assert result == 1.0

    def test_calculate_score_agreement_none_agree(self) -> None:
        """Test score agreement when no reviewers agree."""
        feedbacks = [
            HumanFeedback(
                analysis_id=f"AN{i:03d}",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
            )
            for i in range(5)
        ]
        calculator = AgreementCalculator()

        result = calculator.calculate_score_agreement(feedbacks)

        assert result == 0.0

    def test_calculate_score_agreement_mixed(self) -> None:
        """Test score agreement with mixed feedback."""
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
            ),
            HumanFeedback(
                analysis_id="AN002",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
            ),
            HumanFeedback(
                analysis_id="AN003",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=7,
            ),
            HumanFeedback(
                analysis_id="AN004",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=4,
                score_correction=5,
            ),
        ]
        calculator = AgreementCalculator()

        result = calculator.calculate_score_agreement(feedbacks)

        assert result == 0.5  # 2 out of 4 agree

    def test_calculate_issue_agreement_empty(self) -> None:
        """Test issue agreement with no feedback."""
        calculator = AgreementCalculator()
        result = calculator.calculate_issue_agreement([])

        assert result == 0.0

    def test_calculate_issue_agreement_no_corrections(self) -> None:
        """Test issue agreement when no issue corrections."""
        feedbacks = [
            HumanFeedback(
                analysis_id=f"AN{i:03d}",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
                missing_issues=[],
                incorrect_issues=[],
            )
            for i in range(5)
        ]
        calculator = AgreementCalculator()

        result = calculator.calculate_issue_agreement(feedbacks)

        assert result == 1.0

    def test_calculate_issue_agreement_with_corrections(self) -> None:
        """Test issue agreement when there are issue corrections."""
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
                missing_issues=["Timing issue"],
            ),
            HumanFeedback(
                analysis_id="AN002",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
            ),
            HumanFeedback(
                analysis_id="AN003",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
                incorrect_issues=["False positive"],
            ),
            HumanFeedback(
                analysis_id="AN004",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
            ),
        ]
        calculator = AgreementCalculator()

        result = calculator.calculate_issue_agreement(feedbacks)

        assert result == 0.5  # 2 out of 4 have no issue problems


class TestCalculateAgreementRate:
    """Tests for calculate_agreement_rate function."""

    def test_empty_feedbacks(self) -> None:
        """Test with empty feedback list."""
        metrics = calculate_agreement_rate([])

        assert metrics.total_reviews == 0
        assert metrics.score_agreement_rate == 0.0
        assert metrics.issue_agreement_rate == 0.0
        assert metrics.overall_agreement == 0.0

    def test_single_feedback(self) -> None:
        """Test with single feedback."""
        feedback = HumanFeedback(
            analysis_id="AN001",
            reviewer_id="expert1",
            agrees_with_score=True,
            human_score=8,
        )
        metrics = calculate_agreement_rate([feedback])

        assert metrics.total_reviews == 1
        assert metrics.score_agreement_rate == 1.0
        assert metrics.issue_agreement_rate == 1.0
        assert metrics.overall_agreement == 1.0

    def test_multiple_feedbacks(self) -> None:
        """Test with multiple feedbacks."""
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
            ),
            HumanFeedback(
                analysis_id="AN002",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
            ),
            HumanFeedback(
                analysis_id="AN003",
                reviewer_id="expert2",
                agrees_with_score=True,
                human_score=7,
                missing_issues=["Timing"],
            ),
        ]
        metrics = calculate_agreement_rate(feedbacks)

        assert metrics.total_reviews == 3
        assert metrics.score_agreement_rate == pytest.approx(2 / 3)
        assert metrics.issue_agreement_rate == pytest.approx(2 / 3)
        # Overall = 0.6 * (2/3) + 0.4 * (2/3) = 2/3
        assert metrics.overall_agreement == pytest.approx(2 / 3)

    def test_by_reviewer(self) -> None:
        """Test agreement calculation by reviewer."""
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
            ),
            HumanFeedback(
                analysis_id="AN002",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=7,
            ),
            HumanFeedback(
                analysis_id="AN003",
                reviewer_id="expert2",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
            ),
        ]
        metrics = calculate_agreement_rate(feedbacks)

        assert "expert1" in metrics.by_reviewer
        assert "expert2" in metrics.by_reviewer
        assert metrics.by_reviewer["expert1"] == 1.0  # Both agree
        assert metrics.by_reviewer["expert2"] == 0.0  # Disagreed

    def test_by_score_range(self) -> None:
        """Test agreement calculation by score range."""
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=9,  # high range
            ),
            HumanFeedback(
                analysis_id="AN002",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=2,  # low range
                score_correction=3,
            ),
        ]
        metrics = calculate_agreement_rate(feedbacks)

        assert "high (8-10)" in metrics.by_score_range
        assert "low (1-3)" in metrics.by_score_range
        assert metrics.by_score_range["high (8-10)"] == 1.0
        assert metrics.by_score_range["low (1-3)"] == 0.0

    def test_trend_detection_improving(self) -> None:
        """Test trend detection for improving agreement."""
        base_time = datetime.now()
        feedbacks = [
            # Earlier feedbacks - low agreement
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
                timestamp=base_time - timedelta(days=21),
            ),
            HumanFeedback(
                analysis_id="AN002",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=4,
                score_correction=5,
                timestamp=base_time - timedelta(days=14),
            ),
            # Recent feedbacks - high agreement
            HumanFeedback(
                analysis_id="AN003",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
                timestamp=base_time - timedelta(days=3),
            ),
            HumanFeedback(
                analysis_id="AN004",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=7,
                timestamp=base_time - timedelta(days=1),
            ),
        ]
        metrics = calculate_agreement_rate(feedbacks)

        assert metrics.trend == "improving"

    def test_trend_detection_declining(self) -> None:
        """Test trend detection for declining agreement."""
        base_time = datetime.now()
        feedbacks = [
            # Earlier feedbacks - high agreement
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
                timestamp=base_time - timedelta(days=21),
            ),
            HumanFeedback(
                analysis_id="AN002",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=7,
                timestamp=base_time - timedelta(days=14),
            ),
            # Recent feedbacks - low agreement
            HumanFeedback(
                analysis_id="AN003",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
                timestamp=base_time - timedelta(days=3),
            ),
            HumanFeedback(
                analysis_id="AN004",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=4,
                score_correction=5,
                timestamp=base_time - timedelta(days=1),
            ),
        ]
        metrics = calculate_agreement_rate(feedbacks)

        assert metrics.trend == "declining"


class TestAgreementTracker:
    """Tests for AgreementTracker class."""

    def test_initial_state(self) -> None:
        """Test initial tracker state."""
        tracker = AgreementTracker()

        assert tracker.target_agreement == 0.85
        assert tracker.alert_threshold == 0.80
        assert len(tracker.history) == 0

    def test_record_metrics(self) -> None:
        """Test recording metrics."""
        tracker = AgreementTracker()
        metrics = AgreementMetrics(
            total_reviews=10,
            overall_agreement=0.87,
        )

        tracker.record(metrics)

        assert len(tracker.history) == 1
        assert tracker.history[0] == metrics

    def test_is_below_target(self) -> None:
        """Test target threshold checking."""
        tracker = AgreementTracker(target_agreement=0.85)

        # No history
        assert not tracker.is_below_target()

        # Above target
        tracker.record(AgreementMetrics(overall_agreement=0.90))
        assert not tracker.is_below_target()

        # Below target
        tracker.record(AgreementMetrics(overall_agreement=0.80))
        assert tracker.is_below_target()

    def test_is_below_alert_threshold(self) -> None:
        """Test alert threshold checking."""
        tracker = AgreementTracker(alert_threshold=0.80)

        # No history
        assert not tracker.is_below_alert_threshold()

        # Above threshold
        tracker.record(AgreementMetrics(overall_agreement=0.85))
        assert not tracker.is_below_alert_threshold()

        # Below threshold
        tracker.record(AgreementMetrics(overall_agreement=0.75))
        assert tracker.is_below_alert_threshold()

    def test_get_improvement_areas(self) -> None:
        """Test improvement area identification."""
        tracker = AgreementTracker()

        # No history
        assert tracker.get_improvement_areas() == []

        # Record metrics with issues
        tracker.record(
            AgreementMetrics(
                score_agreement_rate=0.75,  # Below 0.85
                issue_agreement_rate=0.70,  # Below 0.80
                by_score_range={"low (1-3)": 0.60},  # Below 0.75
            )
        )

        areas = tracker.get_improvement_areas()

        assert len(areas) == 3
        assert any("Score agreement" in a for a in areas)
        assert any("Issue agreement" in a for a in areas)
        assert any("low (1-3)" in a for a in areas)

    def test_get_summary_report(self) -> None:
        """Test summary report generation."""
        tracker = AgreementTracker()
        tracker.record(
            AgreementMetrics(
                total_reviews=50,
                overall_agreement=0.82,
                score_agreement_rate=0.85,
                issue_agreement_rate=0.78,
            )
        )

        report = tracker.get_summary_report()

        assert "AGREEMENT TRACKING REPORT" in report
        assert "Total Reviews: 50" in report
        assert "BELOW TARGET" in report
        assert "Historical Records: 1" in report

    def test_get_summary_report_no_history(self) -> None:
        """Test summary report with no history."""
        tracker = AgreementTracker()
        report = tracker.get_summary_report()

        assert "No agreement data recorded yet" in report
