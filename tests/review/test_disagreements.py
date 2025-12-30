"""Tests for disagreement analysis module."""

from __future__ import annotations

from src.review.disagreements import (
    DisagreementPattern,
    DisagreementReport,
    DisagreementTracker,
    analyze_disagreements,
)
from src.review.models import HumanFeedback


class TestDisagreementPattern:
    """Tests for DisagreementPattern model."""

    def test_creation(self) -> None:
        """Test pattern creation."""
        pattern = DisagreementPattern(
            pattern_type="score_bias",
            description="AI scores 2 points higher",
            frequency=10,
            examples=["example1", "example2"],
            severity="high",
        )

        assert pattern.pattern_type == "score_bias"
        assert pattern.description == "AI scores 2 points higher"
        assert pattern.frequency == 10
        assert len(pattern.examples) == 2
        assert pattern.severity == "high"

    def test_default_severity(self) -> None:
        """Test default severity is medium."""
        pattern = DisagreementPattern(
            pattern_type="test",
            description="Test pattern",
        )

        assert pattern.severity == "medium"


class TestDisagreementReport:
    """Tests for DisagreementReport model."""

    def test_default_values(self) -> None:
        """Test default report values."""
        report = DisagreementReport()

        assert report.total_disagreements == 0
        assert report.common_missing_issues == []
        assert report.common_incorrect_issues == []
        assert report.score_bias == 0.0
        assert report.score_bias_description == "neutral"
        assert report.problematic_categories == {}
        assert report.patterns == []
        assert report.actionable_insights == []

    def test_to_summary(self) -> None:
        """Test summary generation."""
        report = DisagreementReport(
            total_disagreements=15,
            common_missing_issues=[("timing", 5), ("venue", 3)],
            common_incorrect_issues=[("slippage", 4)],
            score_bias=1.5,
            score_bias_description="slightly higher",
            patterns=[
                DisagreementPattern(
                    pattern_type="score_bias",
                    description="AI scores higher",
                    severity="high",
                )
            ],
            actionable_insights=["Review timing detection"],
        )

        summary = report.to_summary()

        assert "Total Disagreements: 15" in summary
        assert "timing (5 times)" in summary
        assert "slippage (4 times)" in summary
        assert "Score Bias: slightly higher" in summary
        assert "[HIGH] AI scores higher" in summary
        assert "Review timing detection" in summary


class TestAnalyzeDisagreements:
    """Tests for analyze_disagreements function."""

    def test_empty_feedbacks(self) -> None:
        """Test with empty feedback list."""
        report = analyze_disagreements([])

        assert report.total_disagreements == 0

    def test_no_disagreements(self) -> None:
        """Test when all feedback agrees."""
        feedbacks = [
            HumanFeedback(
                analysis_id=f"AN{i:03d}",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
            )
            for i in range(5)
        ]
        report = analyze_disagreements(feedbacks)

        assert report.total_disagreements == 0
        assert "excellent AI-human alignment" in report.actionable_insights[0]

    def test_with_disagreements(self) -> None:
        """Test with disagreement feedbacks."""
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
                missing_issues=["Timing issue not detected"],
            ),
            HumanFeedback(
                analysis_id="AN002",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=4,
                score_correction=5,
                missing_issues=["Timing slippage"],
            ),
            HumanFeedback(
                analysis_id="AN003",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
                incorrect_issues=["False venue issue"],
            ),
        ]
        report = analyze_disagreements(feedbacks)

        assert report.total_disagreements == 3
        # Should detect timing issues pattern
        assert any("timing" in issue.lower() for issue, _ in report.common_missing_issues)

    def test_issue_normalization(self) -> None:
        """Test that similar issues are grouped together."""
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
                missing_issues=["Timing issue"],
            ),
            HumanFeedback(
                analysis_id="AN002",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=4,
                score_correction=5,
                missing_issues=["Bad execution time"],
            ),
            HumanFeedback(
                analysis_id="AN003",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=6,
                score_correction=7,
                missing_issues=["Late execution"],
            ),
        ]
        report = analyze_disagreements(feedbacks, min_frequency=2)

        # All should be normalized to "timing"
        timing_count = sum(
            count for issue, count in report.common_missing_issues if issue == "timing"
        )
        assert timing_count >= 2

    def test_min_frequency_filter(self) -> None:
        """Test that patterns below min_frequency are excluded."""
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
                missing_issues=["Unique issue 1"],
            ),
            HumanFeedback(
                analysis_id="AN002",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=4,
                score_correction=5,
                missing_issues=["Unique issue 2"],
            ),
        ]
        report = analyze_disagreements(feedbacks, min_frequency=3)

        # No issues should meet the threshold
        assert len(report.common_missing_issues) == 0

    def test_score_bias_detection(self) -> None:
        """Test score bias detection."""
        # Feedback where humans score lower than corrections would suggest
        feedbacks = [
            HumanFeedback(
                analysis_id=f"AN{i:03d}",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=7,  # Corrected up = AI was lower
            )
            for i in range(5)
        ]
        report = analyze_disagreements(feedbacks)

        # Should detect some bias
        assert report.score_bias != 0.0

    def test_actionable_insights_generation(self) -> None:
        """Test actionable insights are generated."""
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
                missing_issues=["Timing issue"],
            ),
            HumanFeedback(
                analysis_id="AN002",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=4,
                score_correction=5,
                missing_issues=["Time-related problem"],
            ),
        ]
        report = analyze_disagreements(feedbacks)

        assert len(report.actionable_insights) > 0
        # Should have insight about timing
        assert any("timing" in insight.lower() for insight in report.actionable_insights)


class TestDisagreementTracker:
    """Tests for DisagreementTracker class."""

    def test_initial_state(self) -> None:
        """Test initial tracker state."""
        tracker = DisagreementTracker()

        assert len(tracker.history) == 0
        assert tracker.alert_threshold == 3

    def test_record_report(self) -> None:
        """Test recording reports."""
        tracker = DisagreementTracker()
        report = DisagreementReport(total_disagreements=10)

        tracker.record(report)

        assert len(tracker.history) == 1
        assert tracker.history[0] == report

    def test_has_critical_patterns_none(self) -> None:
        """Test no critical patterns."""
        tracker = DisagreementTracker()

        assert not tracker.has_critical_patterns()

        tracker.record(DisagreementReport())
        assert not tracker.has_critical_patterns()

    def test_has_critical_patterns(self) -> None:
        """Test detection of critical patterns."""
        tracker = DisagreementTracker()
        report = DisagreementReport(
            patterns=[
                DisagreementPattern(
                    pattern_type="score_bias",
                    description="Critical issue",
                    severity="high",
                )
            ]
        )
        tracker.record(report)

        assert tracker.has_critical_patterns()

    def test_get_trending_issues_insufficient_history(self) -> None:
        """Test trending issues with insufficient history."""
        tracker = DisagreementTracker()

        # No history
        assert tracker.get_trending_issues() == []

        # Only one report
        tracker.record(DisagreementReport())
        assert tracker.get_trending_issues() == []

    def test_get_trending_issues(self) -> None:
        """Test trending issues detection."""
        tracker = DisagreementTracker()

        # Previous report - no timing issues
        tracker.record(
            DisagreementReport(
                common_missing_issues=[("venue", 3)],
                patterns=[],
            )
        )

        # Current report - new timing issues
        tracker.record(
            DisagreementReport(
                common_missing_issues=[("venue", 3), ("timing", 5)],
                patterns=[
                    DisagreementPattern(
                        pattern_type="new_pattern",
                        description="New pattern",
                    )
                ],
            )
        )

        trending = tracker.get_trending_issues()

        assert len(trending) >= 1
        assert any("timing" in t for t in trending) or any("new_pattern" in t for t in trending)
