"""CLI entry point for review module.

Usage:
    # Start interactive review session
    python -m src.review --results results.json --reviewer expert_1

    # Review with specific strategy
    python -m src.review --results results.json --strategy lowest_confidence --limit 10

    # Export feedback
    python -m src.review --export feedback.csv

    # Show statistics
    python -m src.review --stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.parsers.models import AnalysisResult
from src.review.cli import ReviewSession, review_single
from src.review.models import SamplingStrategy
from src.review.queue import create_stratified_queue, get_review_queue
from src.review.storage import FeedbackStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive review interface for AI trade analyses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start review session with results file
  python -m src.review --results analysis_results.json --reviewer expert_1

  # Review low confidence analyses first
  python -m src.review --results results.json --strategy lowest_confidence --limit 20

  # Use stratified sampling
  python -m src.review --results results.json --stratified --samples-per-stratum 5

  # Export feedback to CSV
  python -m src.review --export feedback.csv

  # Show feedback statistics
  python -m src.review --stats

  # Clear all feedback (with confirmation)
  python -m src.review --clear
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--results",
        "-r",
        type=Path,
        help="Path to analysis results JSON file for review",
    )
    mode_group.add_argument(
        "--export",
        "-e",
        type=Path,
        help="Export feedback to file (CSV or JSON based on extension)",
    )
    mode_group.add_argument(
        "--stats",
        action="store_true",
        help="Show feedback statistics",
    )
    mode_group.add_argument(
        "--clear",
        action="store_true",
        help="Clear all stored feedback",
    )

    # Review options
    parser.add_argument(
        "--reviewer",
        type=str,
        default="reviewer",
        help="Reviewer ID (default: reviewer)",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        choices=[s.value for s in SamplingStrategy],
        default="random",
        help="Sampling strategy (default: random)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of items to review",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use stratified sampling across score ranges",
    )
    parser.add_argument(
        "--samples-per-stratum",
        type=int,
        default=5,
        help="Number of samples per stratum (default: 5)",
    )
    parser.add_argument(
        "--analysis-id",
        type=str,
        help="Review a specific analysis by ID",
    )

    # Storage options
    parser.add_argument(
        "--feedback-store",
        type=Path,
        default=Path("feedback.json"),
        help="Path to feedback storage file (default: feedback.json)",
    )

    # Output options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
    )

    return parser.parse_args()


def load_results(path: Path) -> list[AnalysisResult]:
    """Load analysis results from JSON file.

    Args:
        path: Path to results file.

    Returns:
        List of analysis results.
    """
    if not path.exists():
        logger.error(f"Results file not found: {path}")
        sys.exit(1)

    try:
        with path.open("r") as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, list):
            results_data = data
        elif isinstance(data, dict) and "results" in data:
            results_data = data["results"]
        else:
            logger.error("Invalid results file format")
            sys.exit(1)

        results = [AnalysisResult.model_validate(r) for r in results_data]
        logger.info(f"Loaded {len(results)} analysis results from {path}")
        return results

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse results file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        sys.exit(1)


def run_review_session(
    args: argparse.Namespace,
    store: FeedbackStore,
) -> int:
    """Run an interactive review session.

    Args:
        args: Parsed command line arguments.
        store: Feedback storage backend.

    Returns:
        Exit code.
    """
    # Load results
    results = load_results(args.results)

    if not results:
        logger.warning("No results to review")
        return 0

    # Build results map for session (filter out results without analysis_id)
    results_map: dict[str, AnalysisResult] = {
        r.analysis_id: r for r in results if r.analysis_id is not None
    }

    # Handle single analysis review
    if args.analysis_id:
        if args.analysis_id not in results_map:
            logger.error(f"Analysis ID not found: {args.analysis_id}")
            return 1

        result = results_map[args.analysis_id]
        feedback = review_single(result, args.reviewer, store)
        return 0 if feedback else 1

    # Get existing feedback to exclude already-reviewed
    existing = store.get_all()
    reviewed_ids = {f.analysis_id for f in existing}

    # Create queue
    if args.stratified:
        queue = create_stratified_queue(
            results,
            strata_key="score",
            samples_per_stratum=args.samples_per_stratum,
        )
    else:
        queue = get_review_queue(
            results,
            strategy=args.strategy,
            limit=args.limit,
            exclude_ids=reviewed_ids,
        )

    if not queue.pending_count:
        logger.info("No pending items to review")
        return 0

    # Run session
    session = ReviewSession(
        reviewer_id=args.reviewer,
        store=store,
        results_map=results_map,
    )

    try:
        stats = session.start(queue)
        logger.info(f"Session complete: {stats.total_reviewed} reviewed")
        return 0
    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
        return 130


def export_feedback(args: argparse.Namespace, store: FeedbackStore) -> int:
    """Export feedback to file.

    Args:
        args: Parsed command line arguments.
        store: Feedback storage backend.

    Returns:
        Exit code.
    """
    export_path = args.export

    if export_path.suffix.lower() == ".csv":
        count = store.export_csv(export_path)
    elif export_path.suffix.lower() == ".json":
        count = store.export_json(export_path)
    else:
        logger.error("Export format must be .csv or .json")
        return 1

    if count > 0:
        print(f"Exported {count} feedback items to {export_path}")
        return 0
    else:
        print("No feedback to export")
        return 0


def show_stats(store: FeedbackStore) -> int:
    """Show feedback statistics.

    Args:
        store: Feedback storage backend.

    Returns:
        Exit code.
    """
    stats = store.get_statistics()

    print("\n" + "=" * 50)
    print(" FEEDBACK STATISTICS ".center(50))
    print("=" * 50)
    print(f"\nTotal feedback items: {stats['total_feedback']}")
    print(f"Unique reviewers:     {stats['unique_reviewers']}")
    print(f"Unique analyses:      {stats['unique_analyses']}")
    print(f"\nAgreement rate:       {stats['agreement_rate']:.1%}")
    print(f"Correction rate:      {stats['correction_rate']:.1%}")
    print(f"Total corrections:    {stats.get('total_corrections', 0)}")
    print(f"\nAvg review time:      {stats['avg_review_time']:.1f}s")
    print("=" * 50)

    # Show sessions
    sessions = store.get_sessions()
    if sessions:
        print(f"\nRecent sessions ({len(sessions)} total):")
        for session in sessions[-5:]:
            print(f"  - {session.session_id[:8]}: {session.total_reviewed} reviewed")

    return 0


def clear_feedback(store: FeedbackStore) -> int:
    """Clear all feedback with confirmation.

    Args:
        store: Feedback storage backend.

    Returns:
        Exit code.
    """
    count = store.count()
    if count == 0:
        print("No feedback to clear")
        return 0

    confirm = input(f"This will delete {count} feedback items. Are you sure? [y/N]: ")
    if confirm.lower() not in ("y", "yes"):
        print("Cancelled")
        return 0

    cleared = store.clear()
    print(f"Cleared {cleared} feedback items")
    return 0


def main() -> int:
    """Main entry point for CLI."""
    args = parse_args()

    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize store
    store = FeedbackStore(args.feedback_store)

    # Handle different modes
    if args.export:
        return export_feedback(args, store)

    if args.stats:
        return show_stats(store)

    if args.clear:
        return clear_feedback(store)

    if args.results:
        return run_review_session(args, store)

    # No mode specified - show help
    print("No action specified. Use --help for usage information.")
    print("\nQuick start:")
    print("  python -m src.review --results results.json --reviewer your_name")
    print("  python -m src.review --stats")
    print("  python -m src.review --export feedback.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
