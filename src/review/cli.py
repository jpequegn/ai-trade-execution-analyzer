"""Interactive CLI for reviewing AI trade analyses.

This module provides an interactive command-line interface for humans
to review AI-generated trade analyses and provide feedback.

Example:
    >>> from src.review.cli import ReviewSession
    >>> session = ReviewSession(reviewer_id="expert_1")
    >>> session.start(queue)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.review.models import HumanFeedback, ReviewSessionStats, ReviewStatus
from src.review.storage import FeedbackStore

if TYPE_CHECKING:
    from src.parsers.fix_parser import ExecutionReport
    from src.parsers.models import AnalysisResult, TradeAnalysis
    from src.review.models import ReviewItem
    from src.review.queue import ReviewQueue


def format_execution(execution: ExecutionReport) -> str:
    """Format execution data for display.

    Args:
        execution: The execution report to format.

    Returns:
        Formatted string for terminal display.
    """
    lines = [
        "┌─────────────────────────────────────────────┐",
        "│            EXECUTION DETAILS                │",
        "├─────────────────────────────────────────────┤",
        f"│ Order ID:  {execution.order_id:<32} │",
        f"│ Symbol:    {execution.symbol:<32} │",
        f"│ Side:      {execution.side:<32} │",
        f"│ Quantity:  {execution.quantity:<32.2f} │",
        f"│ Price:     ${execution.price:<31.2f} │",
        f"│ Venue:     {execution.venue or 'N/A':<32} │",
        f"│ Fill Type: {execution.fill_type or 'N/A':<32} │",
        f"│ Timestamp: {str(execution.timestamp)[:19]:<32} │",
        "└─────────────────────────────────────────────┘",
    ]
    return "\n".join(lines)


def format_analysis(analysis: TradeAnalysis, analysis_id: str = "") -> str:
    """Format AI analysis for display.

    Args:
        analysis: The trade analysis to format.
        analysis_id: Optional analysis ID to display.

    Returns:
        Formatted string for terminal display.
    """
    lines = [
        "┌─────────────────────────────────────────────┐",
        "│              AI ANALYSIS                    │",
        "├─────────────────────────────────────────────┤",
    ]

    if analysis_id:
        lines.append(f"│ Analysis ID: {analysis_id:<30} │")

    lines.extend(
        [
            f"│ Quality Score:  {analysis.quality_score}/10{' ' * 26}│",
            f"│ Confidence:     {analysis.confidence:.0%}{' ' * 27}│",
            "├─────────────────────────────────────────────┤",
            "│ Issues:                                     │",
        ]
    )

    if analysis.issues:
        for issue in analysis.issues[:5]:
            truncated = issue[:40] + "..." if len(issue) > 40 else issue
            lines.append(f"│   • {truncated:<39} │")
    else:
        lines.append("│   (No issues identified)                    │")

    lines.extend(
        [
            "├─────────────────────────────────────────────┤",
            "│ Observations:                               │",
        ]
    )

    if analysis.observations:
        for obs in analysis.observations[:5]:
            truncated = obs[:40] + "..." if len(obs) > 40 else obs
            lines.append(f"│   • {truncated:<39} │")
    else:
        lines.append("│   (No observations)                         │")

    lines.append("└─────────────────────────────────────────────┘")
    return "\n".join(lines)


def format_result(result: AnalysisResult) -> str:
    """Format a full analysis result for review.

    Args:
        result: The analysis result to format.

    Returns:
        Formatted string combining execution and analysis.
    """
    return (
        format_execution(result.execution)
        + "\n\n"
        + format_analysis(result.analysis, result.analysis_id or "")
    )


def clear_screen() -> None:
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def print_header(text: str) -> None:
    """Print a styled header.

    Args:
        text: Header text to display.
    """
    width = 50
    print("=" * width)
    print(f" {text.center(width - 2)} ")
    print("=" * width)


def print_progress(current: int, total: int, completed: int) -> None:
    """Print review progress bar.

    Args:
        current: Current item number.
        total: Total items.
        completed: Number of completed reviews.
    """
    pct = (current / total) * 100 if total > 0 else 0
    bar_width = 30
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n[{bar}] {current}/{total} ({pct:.0f}%) - {completed} completed")


def get_input(prompt: str, default: str = "") -> str:
    """Get user input with optional default.

    Args:
        prompt: Prompt to display.
        default: Default value if empty input.

    Returns:
        User input or default.
    """
    full_prompt = f"{prompt} [{default}]: " if default else f"{prompt}: "

    try:
        value = input(full_prompt).strip()
        return value if value else default
    except EOFError:
        return default


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user.

    Args:
        prompt: Prompt to display.
        default: Default value.

    Returns:
        Boolean response.
    """
    default_str = "Y/n" if default else "y/N"
    try:
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes", "1", "true")
    except EOFError:
        return default


def get_score(prompt: str, default: int | None = None) -> int | None:
    """Get a score (1-10) from user.

    Args:
        prompt: Prompt to display.
        default: Default score.

    Returns:
        Score value or None.
    """
    default_str = str(default) if default else ""
    while True:
        try:
            value = get_input(prompt, default_str)
            if not value:
                return None
            score = int(value)
            if 1 <= score <= 10:
                return score
            print("Score must be between 1 and 10")
        except ValueError:
            print("Please enter a valid number")
        except EOFError:
            return default


def get_list_input(prompt: str) -> list[str]:
    """Get a list of items from user (one per line).

    Args:
        prompt: Prompt to display.

    Returns:
        List of non-empty strings.
    """
    print(f"{prompt} (one per line, empty line to finish):")
    items: list[str] = []
    while True:
        try:
            line = input("  > ").strip()
            if not line:
                break
            items.append(line)
        except EOFError:
            break
    return items


class ReviewSession:
    """Interactive review session manager.

    Manages the flow of reviewing multiple analyses, collecting
    feedback, and tracking session statistics.

    Attributes:
        reviewer_id: ID of the reviewer.
        store: Feedback storage backend.
        stats: Session statistics.

    Example:
        >>> session = ReviewSession(reviewer_id="expert_1")
        >>> session.start(queue)
    """

    def __init__(
        self,
        reviewer_id: str,
        store: FeedbackStore | None = None,
        results_map: dict[str, AnalysisResult] | None = None,
    ) -> None:
        """Initialize a review session.

        Args:
            reviewer_id: ID of the reviewer.
            store: Optional feedback store (creates default if None).
            results_map: Optional map of analysis_id to AnalysisResult.
        """
        self.reviewer_id = reviewer_id
        self.store = store or FeedbackStore()
        self.results_map = results_map or {}
        self.stats = ReviewSessionStats(reviewer_id=reviewer_id)
        self._running = False

    def start(self, queue: ReviewQueue) -> ReviewSessionStats:
        """Start an interactive review session.

        Args:
            queue: Queue of items to review.

        Returns:
            Session statistics after completion.
        """
        self._running = True
        clear_screen()
        print_header("TRADE ANALYSIS REVIEW SESSION")
        print(f"\nReviewer: {self.reviewer_id}")
        print(f"Items to review: {queue.pending_count}")
        print("\nCommands: [Enter] next | [s]kip | [q]uit")
        print("-" * 50)

        input("\nPress Enter to begin...")

        item_num = 0
        pending = queue.get_pending()

        while pending and self._running:
            item = pending[0]
            item_num += 1

            clear_screen()
            print_progress(item_num, len(queue), self.stats.total_reviewed)

            # Get the full result if available
            result = self.results_map.get(item.analysis_id)

            if result:
                print("\n" + format_result(result))
            else:
                # Display what we have from the review item
                print("\n" + self._format_review_item(item))

            # Mark as in progress
            queue.mark_in_progress(item.analysis_id)

            # Collect feedback
            feedback = self._collect_feedback(item.analysis_id, item.ai_score)

            if feedback is None:
                # User quit
                queue.mark_skipped(item.analysis_id)
                break

            if feedback == "skip":
                queue.mark_skipped(item.analysis_id)
                pending = queue.get_pending()
                continue

            # At this point feedback is HumanFeedback (not None or "skip")
            assert isinstance(feedback, HumanFeedback)

            # Save feedback
            self.store.save(feedback)
            queue.mark_completed(item.analysis_id, feedback)
            self.stats.update(feedback)

            pending = queue.get_pending()

        # Finalize session
        self._running = False
        self.stats.finalize()
        self.store.save_session(self.stats)

        self._print_session_summary()
        return self.stats

    def _format_review_item(self, item: ReviewItem) -> str:
        """Format a review item for display when full result unavailable.

        Args:
            item: The review item to format.

        Returns:
            Formatted string.
        """
        lines = [
            "┌─────────────────────────────────────────────┐",
            "│            REVIEW ITEM                      │",
            "├─────────────────────────────────────────────┤",
            f"│ Analysis ID: {item.analysis_id:<30} │",
            f"│ Order ID:    {item.order_id:<30} │",
            f"│ Symbol:      {item.symbol:<30} │",
            f"│ AI Score:    {item.ai_score}/10{' ' * 26}│",
            f"│ Confidence:  {item.ai_confidence:.0%}{' ' * 27}│",
            "└─────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def _collect_feedback(
        self,
        analysis_id: str,
        ai_score: int,
    ) -> HumanFeedback | str | None:
        """Collect feedback from user for an analysis.

        Args:
            analysis_id: ID of the analysis.
            ai_score: The AI's quality score.

        Returns:
            HumanFeedback, "skip" string, or None to quit.
        """
        print("\n" + "-" * 50)
        print("YOUR REVIEW")
        print("-" * 50)

        start_time = time.time()

        try:
            # Check for skip or quit
            action = get_input("Action (Enter to review, 's' to skip, 'q' to quit)", "")
            if action.lower() == "q":
                return None
            if action.lower() == "s":
                return "skip"

            # Agreement with score
            agrees = get_yes_no(f"Do you agree with the AI score of {ai_score}/10?")

            # Human score
            human_score = None
            score_correction = None
            if not agrees:
                score_correction = get_score("Your corrected score (1-10)")
                human_score = score_correction
            else:
                human_score = get_score("Your score (1-10, Enter to skip)", ai_score)

            # Corrections
            print("\nCorrections (press Enter to skip each section):")

            missing_issues = get_list_input("Missing issues (AI should have identified)")
            incorrect_issues = get_list_input("Incorrect issues (AI wrongly identified)")
            missing_observations = get_list_input("Missing observations")
            incorrect_observations = get_list_input("Incorrect observations")

            # Notes
            notes = get_input("Additional notes (optional)", "")

            elapsed = int(time.time() - start_time)

            return HumanFeedback(
                analysis_id=analysis_id,
                reviewer_id=self.reviewer_id,
                agrees_with_score=agrees,
                human_score=human_score,
                score_correction=score_correction,
                missing_issues=missing_issues,
                incorrect_issues=incorrect_issues,
                missing_observations=missing_observations,
                incorrect_observations=incorrect_observations,
                notes=notes if notes else None,
                review_time_seconds=elapsed,
                status=ReviewStatus.COMPLETED,
            )

        except KeyboardInterrupt:
            print("\n\nReview interrupted.")
            return None

    def _print_session_summary(self) -> None:
        """Print summary of the review session."""
        clear_screen()
        print_header("SESSION COMPLETE")
        print(f"\nReviewer: {self.reviewer_id}")
        print(f"Session ID: {self.stats.session_id[:8]}...")
        print("-" * 50)
        print(f"Total reviewed:    {self.stats.total_reviewed}")
        print(f"Agreed with AI:    {self.stats.total_agreed} ({self.stats.agreement_rate:.0%})")
        print(f"Made corrections:  {self.stats.total_corrected} ({self.stats.correction_rate:.0%})")
        print(f"Avg review time:   {self.stats.avg_review_time:.1f}s")
        print(f"Total time:        {self.stats.total_time_seconds:.0f}s")
        print("-" * 50)
        print("\nFeedback saved successfully!")

    def stop(self) -> None:
        """Stop the review session."""
        self._running = False


def review_single(
    result: AnalysisResult,
    reviewer_id: str,
    store: FeedbackStore | None = None,
) -> HumanFeedback | None:
    """Review a single analysis interactively.

    Args:
        result: The analysis result to review.
        reviewer_id: ID of the reviewer.
        store: Optional feedback store.

    Returns:
        Collected feedback or None if cancelled.

    Example:
        >>> feedback = review_single(result, reviewer_id="expert_1")
    """
    store = store or FeedbackStore()

    clear_screen()
    print_header("SINGLE ANALYSIS REVIEW")
    print("\n" + format_result(result))

    session = ReviewSession(reviewer_id=reviewer_id, store=store)
    analysis_id = result.analysis_id or ""
    feedback = session._collect_feedback(analysis_id, result.analysis.quality_score)

    if isinstance(feedback, HumanFeedback):
        store.save(feedback)
        print("\nFeedback saved!")
        return feedback

    return None
