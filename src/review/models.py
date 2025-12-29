"""Data models for human review feedback.

This module defines the data structures for capturing human feedback
on AI-generated trade analyses.

Example:
    >>> from src.review.models import HumanFeedback
    >>> feedback = HumanFeedback(
    ...     analysis_id="AN001",
    ...     reviewer_id="reviewer_1",
    ...     agrees_with_score=True,
    ...     human_score=8,
    ... )
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class ReviewStatus(str, Enum):
    """Status of a review."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class SamplingStrategy(str, Enum):
    """Strategy for selecting analyses to review."""

    RANDOM = "random"
    LOWEST_CONFIDENCE = "lowest_confidence"
    HIGHEST_CONFIDENCE = "highest_confidence"
    NEWEST = "newest"
    OLDEST = "oldest"
    LOW_SCORE = "low_score"
    HIGH_SCORE = "high_score"


class HumanFeedback(BaseModel):
    """Human feedback on an AI-generated trade analysis.

    Captures the reviewer's assessment, corrections, and notes
    for use in improving the AI system.

    Attributes:
        feedback_id: Unique identifier for this feedback.
        analysis_id: ID of the analysis being reviewed.
        reviewer_id: ID of the human reviewer.
        timestamp: When the feedback was submitted.
        agrees_with_score: Whether reviewer agrees with AI's quality score.
        human_score: Reviewer's quality score (1-10).
        score_correction: Corrected score if different from AI.
        missing_issues: Issues the AI failed to identify.
        incorrect_issues: Issues the AI incorrectly identified.
        missing_observations: Observations the AI should have made.
        incorrect_observations: Incorrect observations made by AI.
        notes: Free-form reviewer notes.
        review_time_seconds: Time spent on review.
        status: Current status of the review.

    Example:
        >>> feedback = HumanFeedback(
        ...     analysis_id="AN001",
        ...     reviewer_id="expert_1",
        ...     agrees_with_score=False,
        ...     human_score=6,
        ...     score_correction=6,
        ...     missing_issues=["Timing slippage not detected"],
        ...     notes="AI missed the obvious timing issue",
        ...     review_time_seconds=45,
        ... )
    """

    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    analysis_id: str
    reviewer_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Agreement assessment
    agrees_with_score: bool
    human_score: Annotated[int, Field(ge=1, le=10)] | None = None
    score_correction: Annotated[int, Field(ge=1, le=10)] | None = None

    # Issue corrections
    missing_issues: list[str] = Field(default_factory=list)
    incorrect_issues: list[str] = Field(default_factory=list)

    # Observation corrections
    missing_observations: list[str] = Field(default_factory=list)
    incorrect_observations: list[str] = Field(default_factory=list)

    # Additional feedback
    notes: str | None = None
    review_time_seconds: int = 0

    # Review metadata
    status: ReviewStatus = ReviewStatus.COMPLETED

    model_config = {"use_enum_values": True}

    @field_validator("missing_issues", "incorrect_issues", mode="before")
    @classmethod
    def convert_none_to_list(cls, v: list[str] | None) -> list[str]:
        """Convert None to empty list."""
        return v if v is not None else []

    @field_validator("missing_observations", "incorrect_observations", mode="before")
    @classmethod
    def convert_none_to_list_obs(cls, v: list[str] | None) -> list[str]:
        """Convert None to empty list."""
        return v if v is not None else []

    @property
    def has_corrections(self) -> bool:
        """Check if feedback includes any corrections."""
        return bool(
            self.score_correction is not None
            or self.missing_issues
            or self.incorrect_issues
            or self.missing_observations
            or self.incorrect_observations
        )

    @property
    def correction_count(self) -> int:
        """Count total number of corrections."""
        count = 0
        if self.score_correction is not None:
            count += 1
        count += len(self.missing_issues)
        count += len(self.incorrect_issues)
        count += len(self.missing_observations)
        count += len(self.incorrect_observations)
        return count

    def to_summary(self) -> str:
        """Generate a summary string of the feedback.

        Returns:
            Human-readable summary of the feedback.
        """
        lines = [
            f"Feedback for analysis {self.analysis_id}",
            f"  Reviewer: {self.reviewer_id}",
            f"  Agrees with score: {'Yes' if self.agrees_with_score else 'No'}",
        ]

        if self.human_score is not None:
            lines.append(f"  Human score: {self.human_score}")

        if self.score_correction is not None:
            lines.append(f"  Score correction: {self.score_correction}")

        if self.missing_issues:
            lines.append(f"  Missing issues: {len(self.missing_issues)}")

        if self.incorrect_issues:
            lines.append(f"  Incorrect issues: {len(self.incorrect_issues)}")

        if self.notes:
            lines.append(f"  Notes: {self.notes[:50]}...")

        lines.append(f"  Review time: {self.review_time_seconds}s")

        return "\n".join(lines)


class ReviewItem(BaseModel):
    """An item in the review queue.

    Pairs an analysis with its review status and any existing feedback.

    Attributes:
        analysis_id: ID of the analysis to review.
        order_id: ID of the original order.
        symbol: Trading symbol.
        ai_score: AI-assigned quality score.
        ai_confidence: AI's confidence in its analysis.
        status: Current review status.
        feedback: Associated feedback if reviewed.
        added_at: When item was added to queue.
    """

    analysis_id: str
    order_id: str
    symbol: str
    ai_score: int
    ai_confidence: float = 0.0
    status: ReviewStatus = ReviewStatus.PENDING
    feedback: HumanFeedback | None = None
    added_at: datetime = Field(default_factory=datetime.now)

    model_config = {"use_enum_values": True}

    @property
    def is_reviewed(self) -> bool:
        """Check if this item has been reviewed."""
        return self.status == ReviewStatus.COMPLETED

    @property
    def is_pending(self) -> bool:
        """Check if this item is pending review."""
        return self.status == ReviewStatus.PENDING


class ReviewSessionStats(BaseModel):
    """Statistics for a review session.

    Attributes:
        session_id: Unique session identifier.
        reviewer_id: ID of the reviewer.
        started_at: When session started.
        ended_at: When session ended.
        total_reviewed: Number of analyses reviewed.
        total_agreed: Number where reviewer agreed with AI.
        total_corrected: Number with corrections.
        avg_review_time: Average review time in seconds.
        total_time_seconds: Total session time.
    """

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reviewer_id: str
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: datetime | None = None
    total_reviewed: int = 0
    total_agreed: int = 0
    total_corrected: int = 0
    avg_review_time: float = 0.0
    total_time_seconds: float = 0.0

    @property
    def agreement_rate(self) -> float:
        """Calculate the agreement rate."""
        if self.total_reviewed == 0:
            return 0.0
        return self.total_agreed / self.total_reviewed

    @property
    def correction_rate(self) -> float:
        """Calculate the correction rate."""
        if self.total_reviewed == 0:
            return 0.0
        return self.total_corrected / self.total_reviewed

    def update(self, feedback: HumanFeedback) -> None:
        """Update stats with new feedback.

        Args:
            feedback: The feedback to incorporate.
        """
        self.total_reviewed += 1
        if feedback.agrees_with_score:
            self.total_agreed += 1
        if feedback.has_corrections:
            self.total_corrected += 1

        # Update average review time
        total_time = self.avg_review_time * (self.total_reviewed - 1)
        total_time += feedback.review_time_seconds
        self.avg_review_time = total_time / self.total_reviewed

    def finalize(self) -> None:
        """Finalize the session stats."""
        self.ended_at = datetime.now()
        if self.ended_at and self.started_at:
            self.total_time_seconds = (self.ended_at - self.started_at).total_seconds()
