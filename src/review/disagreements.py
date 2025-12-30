"""Disagreement analysis for AI-human feedback.

This module identifies patterns and systematic disagreements between
AI analyses and human reviewers.

Example:
    >>> from src.review.disagreements import analyze_disagreements
    >>> from src.review.storage import FeedbackStore
    >>> store = FeedbackStore()
    >>> report = analyze_disagreements(store.get_all())
    >>> print(report.to_summary())
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.review.models import HumanFeedback


class DisagreementPattern(BaseModel):
    """A pattern of disagreement between AI and humans.

    Attributes:
        pattern_type: Type of disagreement pattern.
        description: Human-readable description.
        frequency: How often this pattern occurs.
        examples: Example instances of this pattern.
        severity: How significant this pattern is.
    """

    pattern_type: str
    description: str
    frequency: int = 0
    examples: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"


class DisagreementReport(BaseModel):
    """Report on AI-human disagreement analysis.

    Attributes:
        total_disagreements: Total number of disagreements analyzed.
        common_missing_issues: Issues AI frequently fails to identify.
        common_incorrect_issues: Issues AI frequently identifies incorrectly.
        score_bias: Direction and magnitude of AI scoring bias.
        problematic_categories: Categories with most disagreements.
        patterns: Identified disagreement patterns.
        actionable_insights: Specific recommendations.
    """

    total_disagreements: int = 0
    common_missing_issues: list[tuple[str, int]] = Field(default_factory=list)
    common_incorrect_issues: list[tuple[str, int]] = Field(default_factory=list)
    score_bias: float = 0.0  # Positive = AI scores higher, negative = lower
    score_bias_description: str = "neutral"
    problematic_categories: dict[str, int] = Field(default_factory=dict)
    patterns: list[DisagreementPattern] = Field(default_factory=list)
    actionable_insights: list[str] = Field(default_factory=list)

    def to_summary(self) -> str:
        """Generate a human-readable summary.

        Returns:
            Summary string of disagreement analysis.
        """
        lines = [
            "Disagreement Analysis Report",
            "=" * 50,
            f"Total Disagreements: {self.total_disagreements}",
            "",
        ]

        if self.score_bias != 0:
            lines.append(f"Score Bias: {self.score_bias_description}")
            lines.append(
                f"  AI tends to score {abs(self.score_bias):.1f} points "
                f"{'higher' if self.score_bias > 0 else 'lower'} than humans"
            )
            lines.append("")

        if self.common_missing_issues:
            lines.append("Most Commonly Missing Issues:")
            for issue, count in self.common_missing_issues[:5]:
                lines.append(f"  - {issue} ({count} times)")
            lines.append("")

        if self.common_incorrect_issues:
            lines.append("Most Commonly Incorrect Issues:")
            for issue, count in self.common_incorrect_issues[:5]:
                lines.append(f"  - {issue} ({count} times)")
            lines.append("")

        if self.patterns:
            lines.append("Identified Patterns:")
            for pattern in self.patterns:
                lines.append(f"  [{pattern.severity.upper()}] {pattern.description}")
            lines.append("")

        if self.actionable_insights:
            lines.append("Actionable Insights:")
            for insight in self.actionable_insights:
                lines.append(f"  - {insight}")

        return "\n".join(lines)


def analyze_disagreements(
    feedbacks: Sequence[HumanFeedback],
    min_frequency: int = 2,
) -> DisagreementReport:
    """Find patterns in AI-human disagreements.

    Args:
        feedbacks: List of human feedback to analyze.
        min_frequency: Minimum occurrences to report a pattern.

    Returns:
        DisagreementReport with identified patterns.
    """
    if not feedbacks:
        return DisagreementReport()

    # Filter to only disagreements
    disagreements = [f for f in feedbacks if not f.agrees_with_score or f.has_corrections]

    if not disagreements:
        return DisagreementReport(
            actionable_insights=["No disagreements found - excellent AI-human alignment!"]
        )

    # Analyze missing issues
    missing_counter: Counter[str] = Counter()
    for feedback in disagreements:
        for issue in feedback.missing_issues:
            missing_counter[_normalize_issue(issue)] += 1

    common_missing = [
        (issue, count) for issue, count in missing_counter.most_common(10) if count >= min_frequency
    ]

    # Analyze incorrect issues
    incorrect_counter: Counter[str] = Counter()
    for feedback in disagreements:
        for issue in feedback.incorrect_issues:
            incorrect_counter[_normalize_issue(issue)] += 1

    common_incorrect = [
        (issue, count)
        for issue, count in incorrect_counter.most_common(10)
        if count >= min_frequency
    ]

    # Calculate score bias
    score_bias, bias_description = _calculate_score_bias(disagreements)

    # Identify problematic categories
    problematic = _identify_problematic_categories(disagreements)

    # Detect patterns
    patterns = _detect_patterns(disagreements, min_frequency)

    # Generate actionable insights
    insights = _generate_insights(common_missing, common_incorrect, score_bias, patterns)

    return DisagreementReport(
        total_disagreements=len(disagreements),
        common_missing_issues=common_missing,
        common_incorrect_issues=common_incorrect,
        score_bias=score_bias,
        score_bias_description=bias_description,
        problematic_categories=problematic,
        patterns=patterns,
        actionable_insights=insights,
    )


def _normalize_issue(issue: str) -> str:
    """Normalize an issue string for grouping.

    Args:
        issue: Raw issue string.

    Returns:
        Normalized issue string.
    """
    # Convert to lowercase and strip whitespace
    normalized = issue.lower().strip()

    # Map common variations to standard names
    issue_mappings = {
        "timing": ["timing", "time", "execution time", "late", "early"],
        "venue selection": ["venue", "exchange", "market", "routing"],
        "slippage": ["slippage", "slip", "price impact", "market impact"],
        "fill quality": ["fill", "partial fill", "unfilled", "fill rate"],
        "price improvement": ["price improvement", "improvement", "better price"],
        "order size": ["size", "quantity", "volume", "order size"],
        "liquidity": ["liquidity", "liquid", "illiquid"],
    }

    for standard, variations in issue_mappings.items():
        if any(v in normalized for v in variations):
            return standard

    return normalized


def _calculate_score_bias(
    feedbacks: Sequence[HumanFeedback],
) -> tuple[float, str]:
    """Calculate AI scoring bias.

    Args:
        feedbacks: List of feedback with score corrections.

    Returns:
        Tuple of (bias value, description).
    """
    differences = []

    for feedback in feedbacks:
        if feedback.score_correction is not None and feedback.human_score is not None:
            # Positive means AI scored higher
            # We don't have AI score directly, so we infer from correction
            # If human corrected down, AI was higher
            diff = feedback.human_score - feedback.score_correction
            if diff != 0:
                differences.append(-diff)  # Negate to get AI - human direction

    if not differences:
        return 0.0, "neutral"

    avg_bias = sum(differences) / len(differences)

    if abs(avg_bias) < 0.5:
        description = "neutral"
    elif avg_bias > 1.5:
        description = "significantly higher"
    elif avg_bias > 0.5:
        description = "slightly higher"
    elif avg_bias < -1.5:
        description = "significantly lower"
    else:
        description = "slightly lower"

    return avg_bias, description


def _identify_problematic_categories(
    feedbacks: Sequence[HumanFeedback],
) -> dict[str, int]:
    """Identify categories with most disagreements.

    Args:
        feedbacks: List of feedback with disagreements.

    Returns:
        Dictionary mapping category to disagreement count.
    """
    category_counts: Counter[str] = Counter()

    for feedback in feedbacks:
        # Extract categories from missing and incorrect issues
        for issue in feedback.missing_issues + feedback.incorrect_issues:
            category = _normalize_issue(issue)
            category_counts[category] += 1

    # Return top categories
    return dict(category_counts.most_common(5))


def _detect_patterns(
    feedbacks: Sequence[HumanFeedback],
    min_frequency: int,
) -> list[DisagreementPattern]:
    """Detect systematic disagreement patterns.

    Args:
        feedbacks: List of feedback to analyze.
        min_frequency: Minimum occurrences for a pattern.

    Returns:
        List of identified patterns.
    """
    patterns = []

    # Pattern 1: Consistent score overestimation
    score_corrections = [
        f.score_correction - (f.human_score or f.score_correction)
        for f in feedbacks
        if f.score_correction is not None and f.human_score is not None
    ]

    if score_corrections:
        avg_correction = sum(score_corrections) / len(score_corrections)
        if abs(avg_correction) > 1.0 and len(score_corrections) >= min_frequency:
            direction = "higher" if avg_correction < 0 else "lower"
            patterns.append(
                DisagreementPattern(
                    pattern_type="score_bias",
                    description=f"AI consistently scores {abs(avg_correction):.1f} points "
                    f"{direction} than human reviewers",
                    frequency=len(score_corrections),
                    severity="high" if abs(avg_correction) > 2.0 else "medium",
                )
            )

    # Pattern 2: Missing specific issue types
    missing_by_type = _group_issues_by_type(
        [issue for f in feedbacks for issue in f.missing_issues]
    )

    for issue_type, issues in missing_by_type.items():
        if len(issues) >= min_frequency:
            patterns.append(
                DisagreementPattern(
                    pattern_type="missing_issue",
                    description=f"AI frequently misses {issue_type} issues",
                    frequency=len(issues),
                    examples=issues[:3],
                    severity="high" if len(issues) >= min_frequency * 2 else "medium",
                )
            )

    # Pattern 3: False positive issue types
    incorrect_by_type = _group_issues_by_type(
        [issue for f in feedbacks for issue in f.incorrect_issues]
    )

    for issue_type, issues in incorrect_by_type.items():
        if len(issues) >= min_frequency:
            patterns.append(
                DisagreementPattern(
                    pattern_type="false_positive",
                    description=f"AI incorrectly identifies {issue_type} issues",
                    frequency=len(issues),
                    examples=issues[:3],
                    severity="medium",
                )
            )

    # Pattern 4: Reviewer-specific disagreements
    by_reviewer: dict[str, list[HumanFeedback]] = defaultdict(list)
    for feedback in feedbacks:
        by_reviewer[feedback.reviewer_id].append(feedback)

    for reviewer, reviewer_feedbacks in by_reviewer.items():
        disagreement_rate = len(reviewer_feedbacks) / len(feedbacks)
        if disagreement_rate > 0.3 and len(reviewer_feedbacks) >= min_frequency:
            patterns.append(
                DisagreementPattern(
                    pattern_type="reviewer_specific",
                    description=f"Reviewer '{reviewer}' disagrees more frequently "
                    f"({disagreement_rate:.0%} of all disagreements)",
                    frequency=len(reviewer_feedbacks),
                    severity="low",
                )
            )

    return patterns


def _group_issues_by_type(issues: list[str]) -> dict[str, list[str]]:
    """Group issues by their normalized type.

    Args:
        issues: List of raw issue strings.

    Returns:
        Dictionary mapping issue type to list of original issues.
    """
    grouped: dict[str, list[str]] = defaultdict(list)

    for issue in issues:
        issue_type = _normalize_issue(issue)
        grouped[issue_type].append(issue)

    return grouped


def _generate_insights(
    common_missing: list[tuple[str, int]],
    common_incorrect: list[tuple[str, int]],
    score_bias: float,
    patterns: list[DisagreementPattern],
) -> list[str]:
    """Generate actionable insights from analysis.

    Args:
        common_missing: Most commonly missing issues.
        common_incorrect: Most commonly incorrect issues.
        score_bias: AI scoring bias value.
        patterns: Identified disagreement patterns.

    Returns:
        List of actionable insight strings.
    """
    insights = []

    # Insights from missing issues
    if common_missing:
        top_missing = common_missing[0][0]
        insights.append(
            f"AI frequently misses '{top_missing}' issues - "
            "consider adding more examples to the prompt"
        )

    # Insights from incorrect issues
    if common_incorrect:
        top_incorrect = common_incorrect[0][0]
        insights.append(
            f"AI incorrectly identifies '{top_incorrect}' issues - "
            "review detection criteria in the prompt"
        )

    # Insights from score bias
    if abs(score_bias) > 1.0:
        direction = "higher" if score_bias > 0 else "lower"
        insights.append(
            f"AI scores {abs(score_bias):.1f} points {direction} on average - "
            "calibrate scoring rubric"
        )

    # Insights from patterns
    for pattern in patterns:
        if pattern.severity == "high":
            insights.append(
                f"Critical pattern: {pattern.description} - " "prioritize addressing this issue"
            )

    if not insights:
        insights.append("No significant issues identified - maintain current approach")

    return insights


@dataclass
class DisagreementTracker:
    """Track disagreements over time.

    Attributes:
        history: Historical disagreement reports.
        alert_threshold: Number of patterns to trigger alert.
    """

    history: list[DisagreementReport] = field(default_factory=list)
    alert_threshold: int = 3

    def record(self, report: DisagreementReport) -> None:
        """Record a new disagreement report.

        Args:
            report: The report to record.
        """
        self.history.append(report)

    def has_critical_patterns(self) -> bool:
        """Check if there are critical disagreement patterns.

        Returns:
            True if critical patterns exist.
        """
        if not self.history:
            return False

        latest = self.history[-1]
        critical = [p for p in latest.patterns if p.severity == "high"]
        return len(critical) >= 1

    def get_trending_issues(self) -> list[str]:
        """Get issues that are trending worse over time.

        Returns:
            List of trending issue descriptions.
        """
        if len(self.history) < 2:
            return []

        current = self.history[-1]
        previous = self.history[-2]

        trending = []

        # Compare missing issues
        current_missing = {issue for issue, _ in current.common_missing_issues}
        previous_missing = {issue for issue, _ in previous.common_missing_issues}
        new_missing = current_missing - previous_missing

        for issue in new_missing:
            trending.append(f"New missing issue pattern: {issue}")

        # Compare patterns
        current_pattern_types = {p.pattern_type for p in current.patterns}
        previous_pattern_types = {p.pattern_type for p in previous.patterns}
        new_patterns = current_pattern_types - previous_pattern_types

        for pattern_type in new_patterns:
            trending.append(f"New disagreement pattern: {pattern_type}")

        return trending
