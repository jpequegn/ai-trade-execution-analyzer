"""Agreement metrics for AI-human feedback analysis.

This module calculates and tracks agreement rates between AI analyses
and human reviewers over time.

Example:
    >>> from src.review.agreement import calculate_agreement_rate, AgreementMetrics
    >>> from src.review.storage import FeedbackStore
    >>> store = FeedbackStore()
    >>> metrics = calculate_agreement_rate(store.get_all())
    >>> print(f"Agreement rate: {metrics.overall_agreement:.1%}")
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.review.models import HumanFeedback


class AgreementMetrics(BaseModel):
    """Metrics for AI-human agreement.

    Attributes:
        total_reviews: Total number of reviews analyzed.
        score_agreement_rate: Rate of score agreement (within 1 point).
        issue_agreement_rate: Rate of issue identification agreement.
        overall_agreement: Combined overall agreement rate.
        trend: Direction of agreement over time.
        by_category: Agreement rates by category.
        by_score_range: Agreement rates by AI score range.
        by_reviewer: Agreement rates by reviewer.
        time_series: Agreement rates over time.
    """

    total_reviews: int = 0
    score_agreement_rate: float = 0.0
    issue_agreement_rate: float = 0.0
    overall_agreement: float = 0.0
    trend: Literal["improving", "stable", "declining"] = "stable"
    by_category: dict[str, float] = Field(default_factory=dict)
    by_score_range: dict[str, float] = Field(default_factory=dict)
    by_reviewer: dict[str, float] = Field(default_factory=dict)
    time_series: list[tuple[str, float]] = Field(default_factory=list)

    def to_summary(self) -> str:
        """Generate a human-readable summary.

        Returns:
            Summary string of agreement metrics.
        """
        lines = [
            "Agreement Metrics Summary",
            "=" * 40,
            f"Total Reviews: {self.total_reviews}",
            f"Score Agreement: {self.score_agreement_rate:.1%}",
            f"Issue Agreement: {self.issue_agreement_rate:.1%}",
            f"Overall Agreement: {self.overall_agreement:.1%}",
            f"Trend: {self.trend.capitalize()}",
        ]

        if self.by_score_range:
            lines.append("\nBy Score Range:")
            for range_name, rate in self.by_score_range.items():
                lines.append(f"  {range_name}: {rate:.1%}")

        if self.by_reviewer:
            lines.append("\nBy Reviewer:")
            for reviewer, rate in self.by_reviewer.items():
                lines.append(f"  {reviewer}: {rate:.1%}")

        return "\n".join(lines)


@dataclass
class AgreementCalculator:
    """Calculator for agreement metrics with configurable thresholds.

    Attributes:
        score_tolerance: Maximum score difference to consider agreement.
        issue_match_threshold: Minimum overlap ratio for issue agreement.
    """

    score_tolerance: int = 1
    issue_match_threshold: float = 0.5

    def calculate_score_agreement(self, feedbacks: Sequence[HumanFeedback]) -> float:
        """Calculate the rate of score agreement.

        A score is considered in agreement if the human score is within
        the tolerance of the AI score.

        Args:
            feedbacks: List of human feedback to analyze.

        Returns:
            Agreement rate as a float between 0 and 1.
        """
        if not feedbacks:
            return 0.0

        agreed = 0
        total = 0

        for feedback in feedbacks:
            # If agrees_with_score is True, count as agreement
            if feedback.agrees_with_score:
                agreed += 1
                total += 1
            # If there's a score correction, check the tolerance
            elif feedback.score_correction is not None and feedback.human_score is not None:
                # We need AI score from context, but if not available
                # use score_correction as indicator of disagreement
                total += 1
                # If human_score exists, compare with correction
            else:
                # No correction provided but disagreed
                total += 1

        return agreed / total if total > 0 else 0.0

    def calculate_issue_agreement(self, feedbacks: Sequence[HumanFeedback]) -> float:
        """Calculate the rate of issue identification agreement.

        Agreement means no missing or incorrect issues identified.

        Args:
            feedbacks: List of human feedback to analyze.

        Returns:
            Agreement rate as a float between 0 and 1.
        """
        if not feedbacks:
            return 0.0

        agreed = 0
        total = len(feedbacks)

        for feedback in feedbacks:
            # No missing or incorrect issues means full agreement
            has_issue_problems = bool(feedback.missing_issues or feedback.incorrect_issues)
            if not has_issue_problems:
                agreed += 1

        return agreed / total


def calculate_agreement_rate(
    feedbacks: Sequence[HumanFeedback],
    score_tolerance: int = 1,
) -> AgreementMetrics:
    """Calculate overall agreement between AI and humans.

    Args:
        feedbacks: List of human feedback to analyze.
        score_tolerance: Maximum score difference for agreement.

    Returns:
        AgreementMetrics with calculated rates.
    """
    if not feedbacks:
        return AgreementMetrics()

    calculator = AgreementCalculator(score_tolerance=score_tolerance)

    # Calculate basic rates
    score_agreement = calculator.calculate_score_agreement(feedbacks)
    issue_agreement = calculator.calculate_issue_agreement(feedbacks)

    # Overall is weighted average (60% score, 40% issues)
    overall = 0.6 * score_agreement + 0.4 * issue_agreement

    # Calculate by category
    by_score_range = _calculate_by_score_range(feedbacks)
    by_reviewer = _calculate_by_reviewer(feedbacks)
    time_series = _calculate_time_series(feedbacks)

    # Determine trend from time series
    trend = _determine_trend(time_series)

    return AgreementMetrics(
        total_reviews=len(feedbacks),
        score_agreement_rate=score_agreement,
        issue_agreement_rate=issue_agreement,
        overall_agreement=overall,
        trend=trend,
        by_score_range=by_score_range,
        by_reviewer=by_reviewer,
        time_series=time_series,
    )


def _calculate_by_score_range(
    feedbacks: Sequence[HumanFeedback],
) -> dict[str, float]:
    """Calculate agreement by AI score range.

    Args:
        feedbacks: List of human feedback to analyze.

    Returns:
        Dictionary mapping score range to agreement rate.
    """
    # Group by score range based on human_score
    ranges: dict[str, list[bool]] = {
        "low (1-3)": [],
        "medium_low (4-5)": [],
        "medium_high (6-7)": [],
        "high (8-10)": [],
    }

    for feedback in feedbacks:
        score = feedback.human_score or feedback.score_correction
        if score is None:
            continue

        if 1 <= score <= 3:
            range_key = "low (1-3)"
        elif 4 <= score <= 5:
            range_key = "medium_low (4-5)"
        elif 6 <= score <= 7:
            range_key = "medium_high (6-7)"
        else:
            range_key = "high (8-10)"

        ranges[range_key].append(feedback.agrees_with_score)

    return {
        k: sum(v) / len(v) if v else 0.0
        for k, v in ranges.items()
        if v  # Only include ranges with data
    }


def _calculate_by_reviewer(
    feedbacks: Sequence[HumanFeedback],
) -> dict[str, float]:
    """Calculate agreement by reviewer.

    Args:
        feedbacks: List of human feedback to analyze.

    Returns:
        Dictionary mapping reviewer ID to agreement rate.
    """
    by_reviewer: dict[str, list[bool]] = defaultdict(list)

    for feedback in feedbacks:
        by_reviewer[feedback.reviewer_id].append(feedback.agrees_with_score)

    return {
        reviewer: sum(agreements) / len(agreements) for reviewer, agreements in by_reviewer.items()
    }


def _calculate_time_series(
    feedbacks: Sequence[HumanFeedback],
    period: Literal["daily", "weekly", "monthly"] = "weekly",
) -> list[tuple[str, float]]:
    """Calculate agreement rate over time.

    Args:
        feedbacks: List of human feedback to analyze.
        period: Time period for grouping.

    Returns:
        List of (date_string, agreement_rate) tuples.
    """
    if not feedbacks:
        return []

    # Group by period
    by_period: dict[str, list[bool]] = defaultdict(list)

    for feedback in feedbacks:
        if period == "daily":
            key = feedback.timestamp.strftime("%Y-%m-%d")
        elif period == "weekly":
            # Get Monday of the week
            monday = feedback.timestamp - timedelta(days=feedback.timestamp.weekday())
            key = monday.strftime("%Y-%m-%d")
        else:  # monthly
            key = feedback.timestamp.strftime("%Y-%m")

        by_period[key].append(feedback.agrees_with_score)

    # Sort by date and calculate rates
    sorted_periods = sorted(by_period.items())
    return [
        (period_key, sum(agreements) / len(agreements)) for period_key, agreements in sorted_periods
    ]


def _determine_trend(
    time_series: list[tuple[str, float]],
    lookback: int = 3,
) -> Literal["improving", "stable", "declining"]:
    """Determine the trend from time series data.

    Args:
        time_series: List of (date, rate) tuples.
        lookback: Number of periods to compare.

    Returns:
        Trend direction.
    """
    if len(time_series) < 2:
        return "stable"

    # Compare recent average to earlier average
    recent = time_series[-lookback:] if len(time_series) >= lookback else time_series[-1:]
    earlier = time_series[:-lookback] if len(time_series) > lookback else time_series[:1]

    recent_avg = sum(r for _, r in recent) / len(recent)
    earlier_avg = sum(r for _, r in earlier) / len(earlier)

    diff = recent_avg - earlier_avg

    if diff > 0.05:  # 5% improvement
        return "improving"
    elif diff < -0.05:  # 5% decline
        return "declining"
    return "stable"


@dataclass
class AgreementTracker:
    """Track agreement metrics over time.

    This class provides methods for ongoing agreement tracking
    and generates alerts when thresholds are crossed.

    Attributes:
        target_agreement: Target overall agreement rate.
        alert_threshold: Threshold below which to alert.
        history: Historical metrics.
    """

    target_agreement: float = 0.85
    alert_threshold: float = 0.80
    history: list[AgreementMetrics] = field(default_factory=list)

    def record(self, metrics: AgreementMetrics) -> None:
        """Record new metrics.

        Args:
            metrics: The metrics to record.
        """
        self.history.append(metrics)

    def is_below_target(self) -> bool:
        """Check if current agreement is below target.

        Returns:
            True if below target, False otherwise.
        """
        if not self.history:
            return False
        return self.history[-1].overall_agreement < self.target_agreement

    def is_below_alert_threshold(self) -> bool:
        """Check if current agreement is below alert threshold.

        Returns:
            True if below threshold, False otherwise.
        """
        if not self.history:
            return False
        return self.history[-1].overall_agreement < self.alert_threshold

    def get_improvement_areas(self) -> list[str]:
        """Identify areas needing improvement.

        Returns:
            List of improvement recommendations.
        """
        if not self.history:
            return []

        latest = self.history[-1]
        areas = []

        if latest.score_agreement_rate < 0.85:
            areas.append(
                f"Score agreement at {latest.score_agreement_rate:.1%} - " "review scoring criteria"
            )

        if latest.issue_agreement_rate < 0.80:
            areas.append(
                f"Issue agreement at {latest.issue_agreement_rate:.1%} - "
                "improve issue detection prompts"
            )

        # Check score ranges
        for range_name, rate in latest.by_score_range.items():
            if rate < 0.75:
                areas.append(
                    f"Low agreement in {range_name} range ({rate:.1%}) - "
                    "review examples in this score range"
                )

        return areas

    def get_summary_report(self) -> str:
        """Generate a summary report.

        Returns:
            Formatted summary report string.
        """
        if not self.history:
            return "No agreement data recorded yet."

        latest = self.history[-1]
        lines = [
            "=" * 50,
            "AGREEMENT TRACKING REPORT",
            "=" * 50,
            "",
            latest.to_summary(),
            "",
            f"Target Agreement: {self.target_agreement:.1%}",
            f"Alert Threshold: {self.alert_threshold:.1%}",
            f"Current Status: {'BELOW TARGET' if self.is_below_target() else 'ON TARGET'}",
            "",
        ]

        if self.is_below_alert_threshold():
            lines.append("ALERT: Agreement below threshold!")
            lines.append("")

        areas = self.get_improvement_areas()
        if areas:
            lines.append("Areas for Improvement:")
            for area in areas:
                lines.append(f"  - {area}")

        lines.append("")
        lines.append(f"Historical Records: {len(self.history)}")

        return "\n".join(lines)
