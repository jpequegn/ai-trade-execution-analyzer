"""Review queue management with sampling strategies.

This module provides functionality to create and manage queues of
analyses for human review, with various sampling strategies.

Example:
    >>> from src.review.queue import get_review_queue, ReviewQueue
    >>> queue = get_review_queue(results, strategy="lowest_confidence", limit=10)
    >>> for item in queue:
    ...     print(f"Review: {item.analysis_id}")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from src.review.models import ReviewItem, ReviewStatus, SamplingStrategy

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.parsers.models import AnalysisResult
    from src.review.models import HumanFeedback


@dataclass
class ReviewQueue:
    """A queue of analyses awaiting human review.

    Manages the review workflow, tracking which analyses have been
    reviewed and which are pending.

    Attributes:
        items: List of review items.
        strategy: Sampling strategy used to create queue.
        created_at: When the queue was created.

    Example:
        >>> queue = ReviewQueue(items=items, strategy=SamplingStrategy.RANDOM)
        >>> next_item = queue.get_next()
        >>> queue.mark_completed(next_item.analysis_id, feedback)
    """

    items: list[ReviewItem] = field(default_factory=list)
    strategy: SamplingStrategy = SamplingStrategy.RANDOM
    created_at: datetime = field(default_factory=datetime.now)

    def __len__(self) -> int:
        """Get total number of items in queue."""
        return len(self.items)

    def __iter__(self) -> Iterator[ReviewItem]:
        """Iterate over pending items."""
        return iter(self.get_pending())

    @property
    def pending_count(self) -> int:
        """Get count of pending reviews."""
        return sum(1 for item in self.items if item.is_pending)

    @property
    def completed_count(self) -> int:
        """Get count of completed reviews."""
        return sum(1 for item in self.items if item.is_reviewed)

    @property
    def progress(self) -> float:
        """Get completion progress as percentage."""
        if not self.items:
            return 0.0
        return self.completed_count / len(self.items) * 100

    def get_pending(self) -> list[ReviewItem]:
        """Get all pending review items.

        Returns:
            List of items with pending status.
        """
        return [item for item in self.items if item.is_pending]

    def get_completed(self) -> list[ReviewItem]:
        """Get all completed review items.

        Returns:
            List of items with completed status.
        """
        return [item for item in self.items if item.is_reviewed]

    def get_next(self) -> ReviewItem | None:
        """Get the next item to review.

        Returns:
            Next pending item, or None if queue is empty.
        """
        pending = self.get_pending()
        return pending[0] if pending else None

    def get_by_id(self, analysis_id: str) -> ReviewItem | None:
        """Get a specific item by analysis ID.

        Args:
            analysis_id: The analysis ID to find.

        Returns:
            The item if found, None otherwise.
        """
        for item in self.items:
            if item.analysis_id == analysis_id:
                return item
        return None

    def mark_in_progress(self, analysis_id: str) -> bool:
        """Mark an item as in progress.

        Args:
            analysis_id: The analysis ID to mark.

        Returns:
            True if marked, False if not found.
        """
        item = self.get_by_id(analysis_id)
        if item:
            item.status = ReviewStatus.IN_PROGRESS
            return True
        return False

    def mark_completed(
        self,
        analysis_id: str,
        feedback: HumanFeedback | None = None,
    ) -> bool:
        """Mark an item as completed with optional feedback.

        Args:
            analysis_id: The analysis ID to mark.
            feedback: Optional feedback to attach.

        Returns:
            True if marked, False if not found.
        """

        item = self.get_by_id(analysis_id)
        if item:
            item.status = ReviewStatus.COMPLETED
            if feedback:
                item.feedback = feedback
            return True
        return False

    def mark_skipped(self, analysis_id: str) -> bool:
        """Mark an item as skipped.

        Args:
            analysis_id: The analysis ID to mark.

        Returns:
            True if marked, False if not found.
        """
        item = self.get_by_id(analysis_id)
        if item:
            item.status = ReviewStatus.SKIPPED
            return True
        return False

    def add_item(self, item: ReviewItem) -> None:
        """Add an item to the queue.

        Args:
            item: The review item to add.
        """
        self.items.append(item)

    def remove_item(self, analysis_id: str) -> bool:
        """Remove an item from the queue.

        Args:
            analysis_id: The analysis ID to remove.

        Returns:
            True if removed, False if not found.
        """
        original_len = len(self.items)
        self.items = [item for item in self.items if item.analysis_id != analysis_id]
        return len(self.items) < original_len

    def reset(self) -> None:
        """Reset all items to pending status."""
        for item in self.items:
            item.status = ReviewStatus.PENDING
            item.feedback = None

    def shuffle(self) -> None:
        """Shuffle the pending items randomly."""
        pending = self.get_pending()
        completed = self.get_completed()
        skipped = [item for item in self.items if item.status == ReviewStatus.SKIPPED]

        random.shuffle(pending)
        self.items = pending + completed + skipped

    def to_summary(self) -> str:
        """Generate a summary of the queue.

        Returns:
            Human-readable summary string.
        """
        lines = [
            f"Review Queue ({self.strategy.value} strategy)",
            f"  Total items: {len(self.items)}",
            f"  Pending: {self.pending_count}",
            f"  Completed: {self.completed_count}",
            f"  Progress: {self.progress:.1f}%",
        ]

        skipped = sum(1 for item in self.items if item.status == ReviewStatus.SKIPPED)
        if skipped:
            lines.append(f"  Skipped: {skipped}")

        return "\n".join(lines)


def get_review_queue(
    results: list[AnalysisResult],
    strategy: str | SamplingStrategy = SamplingStrategy.RANDOM,
    limit: int | None = None,
    exclude_ids: set[str] | None = None,
) -> ReviewQueue:
    """Create a review queue from analysis results.

    Args:
        results: List of analysis results to queue.
        strategy: Sampling strategy to use.
        limit: Maximum number of items in queue.
        exclude_ids: Analysis IDs to exclude.

    Returns:
        ReviewQueue with selected items.

    Example:
        >>> queue = get_review_queue(results, strategy="lowest_confidence", limit=20)
        >>> print(f"Queue has {len(queue)} items")
    """
    if isinstance(strategy, str):
        strategy = SamplingStrategy(strategy)

    exclude_ids = exclude_ids or set()

    # Filter out excluded IDs and results without analysis_id
    filtered_results = [
        r for r in results if r.analysis_id is not None and r.analysis_id not in exclude_ids
    ]

    # Sort based on strategy
    sorted_results = _sort_by_strategy(filtered_results, strategy)

    # Apply limit
    if limit is not None:
        sorted_results = sorted_results[:limit]

    # Create review items (analysis_id is guaranteed non-None after filtering)
    items = [
        ReviewItem(
            analysis_id=r.analysis_id,  # type: ignore[arg-type]
            order_id=r.execution.order_id,
            symbol=r.execution.symbol,
            ai_score=r.analysis.quality_score,
            ai_confidence=r.analysis.confidence,
        )
        for r in sorted_results
    ]

    return ReviewQueue(items=items, strategy=strategy)


def _sort_by_strategy(
    results: list[AnalysisResult],
    strategy: SamplingStrategy,
) -> list[AnalysisResult]:
    """Sort results according to sampling strategy.

    Args:
        results: Results to sort.
        strategy: Strategy to use.

    Returns:
        Sorted list of results.
    """
    if strategy == SamplingStrategy.RANDOM:
        shuffled = list(results)
        random.shuffle(shuffled)
        return shuffled

    if strategy == SamplingStrategy.LOWEST_CONFIDENCE:
        return sorted(results, key=lambda r: r.analysis.confidence)

    if strategy == SamplingStrategy.HIGHEST_CONFIDENCE:
        return sorted(results, key=lambda r: r.analysis.confidence, reverse=True)

    if strategy == SamplingStrategy.NEWEST:
        return sorted(results, key=lambda r: r.analyzed_at, reverse=True)

    if strategy == SamplingStrategy.OLDEST:
        return sorted(results, key=lambda r: r.analyzed_at)

    if strategy == SamplingStrategy.LOW_SCORE:
        return sorted(results, key=lambda r: r.analysis.quality_score)

    if strategy == SamplingStrategy.HIGH_SCORE:
        return sorted(results, key=lambda r: r.analysis.quality_score, reverse=True)

    # Default to random
    shuffled = list(results)
    random.shuffle(shuffled)
    return shuffled


def create_stratified_queue(
    results: list[AnalysisResult],
    strata_key: str = "score",
    samples_per_stratum: int = 5,
) -> ReviewQueue:
    """Create a stratified sample queue.

    Samples evenly from different score/confidence ranges.

    Args:
        results: Results to sample from.
        strata_key: Key to stratify by ("score" or "confidence").
        samples_per_stratum: Number of samples per stratum.

    Returns:
        ReviewQueue with stratified samples.

    Example:
        >>> queue = create_stratified_queue(results, strata_key="score", samples_per_stratum=3)
    """
    # Filter results to only those with analysis_id
    valid_results = [r for r in results if r.analysis_id is not None]

    # Define strata based on key
    if strata_key == "score":
        # Score ranges: 1-3, 4-5, 6-7, 8-10
        strata = {
            "low": [r for r in valid_results if r.analysis.quality_score <= 3],
            "medium_low": [r for r in valid_results if 4 <= r.analysis.quality_score <= 5],
            "medium_high": [r for r in valid_results if 6 <= r.analysis.quality_score <= 7],
            "high": [r for r in valid_results if r.analysis.quality_score >= 8],
        }
    else:
        # Confidence ranges: 0-0.5, 0.5-0.7, 0.7-0.85, 0.85-1.0
        strata = {
            "low": [r for r in valid_results if r.analysis.confidence < 0.5],
            "medium_low": [r for r in valid_results if 0.5 <= r.analysis.confidence < 0.7],
            "medium_high": [r for r in valid_results if 0.7 <= r.analysis.confidence < 0.85],
            "high": [r for r in valid_results if r.analysis.confidence >= 0.85],
        }

    # Sample from each stratum
    selected: list[AnalysisResult] = []
    for _stratum_name, stratum_results in strata.items():
        if stratum_results:
            sample_size = min(samples_per_stratum, len(stratum_results))
            selected.extend(random.sample(stratum_results, sample_size))

    # Shuffle final selection
    random.shuffle(selected)

    # Create items (analysis_id is guaranteed non-None after filtering)
    items = [
        ReviewItem(
            analysis_id=r.analysis_id,  # type: ignore[arg-type]
            order_id=r.execution.order_id,
            symbol=r.execution.symbol,
            ai_score=r.analysis.quality_score,
            ai_confidence=r.analysis.confidence,
        )
        for r in selected
    ]

    return ReviewQueue(items=items, strategy=SamplingStrategy.RANDOM)
