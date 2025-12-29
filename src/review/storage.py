"""Storage backends for human feedback data.

This module provides storage implementations for persisting human feedback
on AI trade analyses.

Example:
    >>> from src.review.storage import FeedbackStore
    >>> store = FeedbackStore("feedback.json")
    >>> store.save(feedback)
    >>> all_feedback = store.get_all()
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.review.models import HumanFeedback, ReviewSessionStats

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class FeedbackStore:
    """JSON file-based storage for human feedback.

    Provides persistent storage for feedback with support for
    querying, filtering, and exporting data.

    Attributes:
        path: Path to the JSON storage file.

    Example:
        >>> store = FeedbackStore("feedback.json")
        >>> store.save(feedback)
        >>> by_analysis = store.get_by_analysis("AN001")
    """

    def __init__(self, path: str | Path = "feedback.json") -> None:
        """Initialize the feedback store.

        Args:
            path: Path to JSON file for storage.
        """
        self.path = Path(path)
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create storage file if it doesn't exist."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write_data({"feedback": [], "sessions": []})
            logger.info(f"Created feedback store at {self.path}")

    def _read_data(self) -> dict[str, list[dict[str, object]]]:
        """Read data from storage file.

        Returns:
            Dictionary with feedback and sessions lists.
        """
        try:
            with self.path.open("r") as f:
                data: dict[str, list[dict[str, object]]] = json.load(f)
                return data
        except (json.JSONDecodeError, FileNotFoundError):
            return {"feedback": [], "sessions": []}

    def _write_data(self, data: dict[str, list[dict[str, object]]]) -> None:
        """Write data to storage file.

        Args:
            data: Data to write.
        """
        with self.path.open("w") as f:
            json.dump(data, f, indent=2, default=str)

    def save(self, feedback: HumanFeedback) -> None:
        """Save feedback to storage.

        Args:
            feedback: The feedback to save.
        """
        data = self._read_data()
        feedback_dict = feedback.model_dump()
        data["feedback"].append(feedback_dict)
        self._write_data(data)
        logger.debug(f"Saved feedback {feedback.feedback_id} for analysis {feedback.analysis_id}")

    def save_batch(self, feedback_list: list[HumanFeedback]) -> int:
        """Save multiple feedback items at once.

        Args:
            feedback_list: List of feedback to save.

        Returns:
            Number of items saved.
        """
        if not feedback_list:
            return 0

        data = self._read_data()
        for feedback in feedback_list:
            data["feedback"].append(feedback.model_dump())
        self._write_data(data)
        logger.info(f"Saved {len(feedback_list)} feedback items")
        return len(feedback_list)

    def get_all(self) -> list[HumanFeedback]:
        """Get all feedback from storage.

        Returns:
            List of all feedback items.
        """
        data = self._read_data()
        return [HumanFeedback.model_validate(f) for f in data.get("feedback", [])]

    def get_by_id(self, feedback_id: str) -> HumanFeedback | None:
        """Get feedback by its ID.

        Args:
            feedback_id: The feedback ID to find.

        Returns:
            The feedback if found, None otherwise.
        """
        for feedback in self.get_all():
            if feedback.feedback_id == feedback_id:
                return feedback
        return None

    def get_by_analysis(self, analysis_id: str) -> list[HumanFeedback]:
        """Get all feedback for a specific analysis.

        Args:
            analysis_id: The analysis ID to filter by.

        Returns:
            List of feedback for that analysis.
        """
        return [f for f in self.get_all() if f.analysis_id == analysis_id]

    def get_by_reviewer(self, reviewer_id: str) -> list[HumanFeedback]:
        """Get all feedback from a specific reviewer.

        Args:
            reviewer_id: The reviewer ID to filter by.

        Returns:
            List of feedback from that reviewer.
        """
        return [f for f in self.get_all() if f.reviewer_id == reviewer_id]

    def get_by_date_range(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[HumanFeedback]:
        """Get feedback within a date range.

        Args:
            start: Start of date range (inclusive).
            end: End of date range (inclusive).

        Returns:
            List of feedback within the range.
        """
        feedback_list = self.get_all()

        if start:
            feedback_list = [f for f in feedback_list if f.timestamp >= start]
        if end:
            feedback_list = [f for f in feedback_list if f.timestamp <= end]

        return feedback_list

    def get_corrections_only(self) -> list[HumanFeedback]:
        """Get only feedback that includes corrections.

        Returns:
            List of feedback with corrections.
        """
        return [f for f in self.get_all() if f.has_corrections]

    def get_disagreements(self) -> list[HumanFeedback]:
        """Get feedback where reviewer disagreed with AI.

        Returns:
            List of feedback with disagreements.
        """
        return [f for f in self.get_all() if not f.agrees_with_score]

    def count(self) -> int:
        """Get total count of feedback items.

        Returns:
            Number of feedback items stored.
        """
        data = self._read_data()
        return len(data.get("feedback", []))

    def delete(self, feedback_id: str) -> bool:
        """Delete feedback by ID.

        Args:
            feedback_id: The feedback ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        data = self._read_data()
        original_count = len(data["feedback"])
        data["feedback"] = [f for f in data["feedback"] if f.get("feedback_id") != feedback_id]

        if len(data["feedback"]) < original_count:
            self._write_data(data)
            logger.info(f"Deleted feedback {feedback_id}")
            return True
        return False

    def clear(self) -> int:
        """Clear all feedback from storage.

        Returns:
            Number of items cleared.
        """
        data = self._read_data()
        count = len(data.get("feedback", []))
        data["feedback"] = []
        self._write_data(data)
        logger.warning(f"Cleared {count} feedback items from storage")
        return count

    # Session management
    def save_session(self, session: ReviewSessionStats) -> None:
        """Save a review session.

        Args:
            session: The session stats to save.
        """
        data = self._read_data()
        if "sessions" not in data:
            data["sessions"] = []
        data["sessions"].append(session.model_dump())
        self._write_data(data)
        logger.info(f"Saved session {session.session_id}")

    def get_sessions(self) -> list[ReviewSessionStats]:
        """Get all review sessions.

        Returns:
            List of session stats.
        """
        data = self._read_data()
        return [ReviewSessionStats.model_validate(s) for s in data.get("sessions", [])]

    # Export functionality
    def export_csv(self, path: str | Path) -> int:
        """Export feedback to CSV file.

        Args:
            path: Path for the CSV file.

        Returns:
            Number of rows exported.
        """
        feedback_list = self.get_all()
        if not feedback_list:
            logger.warning("No feedback to export")
            return 0

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "feedback_id",
            "analysis_id",
            "reviewer_id",
            "timestamp",
            "agrees_with_score",
            "human_score",
            "score_correction",
            "missing_issues",
            "incorrect_issues",
            "missing_observations",
            "incorrect_observations",
            "notes",
            "review_time_seconds",
            "status",
        ]

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for feedback in feedback_list:
                row = feedback.model_dump()
                # Convert lists to semicolon-separated strings
                row["missing_issues"] = "; ".join(row["missing_issues"])
                row["incorrect_issues"] = "; ".join(row["incorrect_issues"])
                row["missing_observations"] = "; ".join(row["missing_observations"])
                row["incorrect_observations"] = "; ".join(row["incorrect_observations"])
                writer.writerow(row)

        logger.info(f"Exported {len(feedback_list)} feedback items to {path}")
        return len(feedback_list)

    def export_json(self, path: str | Path, indent: int = 2) -> int:
        """Export feedback to JSON file.

        Args:
            path: Path for the JSON file.
            indent: JSON indentation level.

        Returns:
            Number of items exported.
        """
        feedback_list = self.get_all()
        if not feedback_list:
            logger.warning("No feedback to export")
            return 0

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(
                [f.model_dump() for f in feedback_list],
                f,
                indent=indent,
                default=str,
            )

        logger.info(f"Exported {len(feedback_list)} feedback items to {path}")
        return len(feedback_list)

    def iter_feedback(self) -> Iterator[HumanFeedback]:
        """Iterate over all feedback items.

        Yields:
            HumanFeedback items one at a time.
        """
        data = self._read_data()
        for f in data.get("feedback", []):
            yield HumanFeedback.model_validate(f)

    # Statistics
    def get_statistics(self) -> dict[str, object]:
        """Calculate statistics from stored feedback.

        Returns:
            Dictionary of statistics.
        """
        feedback_list = self.get_all()

        if not feedback_list:
            return {
                "total_feedback": 0,
                "agreement_rate": 0.0,
                "correction_rate": 0.0,
                "avg_review_time": 0.0,
                "unique_reviewers": 0,
                "unique_analyses": 0,
            }

        agreed_count = sum(1 for f in feedback_list if f.agrees_with_score)
        corrected_count = sum(1 for f in feedback_list if f.has_corrections)
        total_review_time = sum(f.review_time_seconds for f in feedback_list)
        unique_reviewers = len({f.reviewer_id for f in feedback_list})
        unique_analyses = len({f.analysis_id for f in feedback_list})

        return {
            "total_feedback": len(feedback_list),
            "agreement_rate": agreed_count / len(feedback_list),
            "correction_rate": corrected_count / len(feedback_list),
            "avg_review_time": total_review_time / len(feedback_list),
            "unique_reviewers": unique_reviewers,
            "unique_analyses": unique_analyses,
            "total_corrections": sum(f.correction_count for f in feedback_list),
        }
