"""CLI interface for running evaluations.

Usage:
    # Run full evaluation
    python -m src.evaluation

    # Run subset of samples
    python -m src.evaluation --samples GT001,GT002,GT003

    # Generate report to file
    python -m src.evaluation --output reports/

    # Run with custom thresholds
    python -m src.evaluation --threshold overall=0.80

    # Compare with baseline
    python -m src.evaluation --compare baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.agents.analyzer import TradeAnalyzer
from src.config import get_config
from src.evaluation.reporting import (
    compare_runs,
    format_comparison_report,
    generate_markdown_report,
    save_report,
)
from src.evaluation.runner import (
    DEFAULT_THRESHOLDS,
    EvalResults,
    EvaluationRunner,
    ProgressCallback,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AI trade analysis evaluation against ground truth dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation
  python -m src.evaluation

  # Run specific samples
  python -m src.evaluation --samples GT001,GT002,GT003

  # Save report to directory
  python -m src.evaluation --output reports/

  # Use custom dataset
  python -m src.evaluation --dataset custom_ground_truth.json

  # Set custom threshold
  python -m src.evaluation --threshold overall=0.85 --threshold insight_accuracy=0.75

  # Compare with baseline run
  python -m src.evaluation --compare baseline_results.json
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=None,
        help="Path to ground truth dataset (default: built-in dataset)",
    )

    parser.add_argument(
        "--samples",
        "-s",
        type=str,
        default=None,
        help="Comma-separated list of sample IDs to evaluate (default: all)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for reports (default: print to stdout)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Report format (default: markdown)",
    )

    parser.add_argument(
        "--threshold",
        "-t",
        action="append",
        type=str,
        default=[],
        help="Custom threshold in format metric=value (can be specified multiple times)",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    parser.add_argument(
        "--compare",
        "-c",
        type=Path,
        default=None,
        help="Path to baseline results JSON for comparison",
    )

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
        help="Suppress all output except errors",
    )

    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with error code if quality thresholds not met",
    )

    return parser.parse_args()


def parse_thresholds(threshold_args: list[str]) -> dict[str, float]:
    """Parse threshold arguments into dictionary.

    Args:
        threshold_args: List of "metric=value" strings.

    Returns:
        Dictionary of metric name to threshold value.
    """
    thresholds = DEFAULT_THRESHOLDS.copy()

    for arg in threshold_args:
        if "=" not in arg:
            logger.warning(f"Invalid threshold format: {arg} (expected metric=value)")
            continue

        parts = arg.split("=", 1)
        metric_name = parts[0].strip()
        try:
            value = float(parts[1].strip())
            if 0.0 <= value <= 1.0:
                thresholds[metric_name] = value
            else:
                logger.warning(f"Threshold value must be between 0 and 1: {arg}")
        except ValueError:
            logger.warning(f"Invalid threshold value: {arg}")

    return thresholds


def progress_callback(progress: ProgressCallback) -> None:
    """Print progress updates."""
    rate = progress.samples_per_second
    eta = progress.eta_seconds
    eta_str = f"{eta:.0f}s remaining" if eta else "calculating..."

    print(
        f"\rProgress: {progress.completed}/{progress.total} "
        f"({progress.progress_pct:.0f}%) | "
        f"{progress.passed} passed, {progress.failed} failed | "
        f"{rate:.1f} samples/s | {eta_str}",
        end="",
        flush=True,
    )


def main() -> int:
    """Main entry point for CLI."""
    args = parse_args()

    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse thresholds
    thresholds = parse_thresholds(args.threshold)

    # Parse sample IDs
    sample_ids = None
    if args.samples:
        sample_ids = [s.strip() for s in args.samples.split(",")]

    try:
        # Initialize analyzer and runner
        config = get_config()
        analyzer = TradeAnalyzer(config=config)
        runner = EvaluationRunner(
            analyzer=analyzer,
            dataset_path=args.dataset,
            thresholds=thresholds,
            max_workers=args.workers,
        )

        # Run evaluation
        logger.info("Starting evaluation run...")
        results = runner.run(
            sample_ids=sample_ids,
            progress_callback=progress_callback if not args.quiet else None,
        )

        # Clear progress line
        if not args.quiet:
            print()  # New line after progress

        # Handle comparison if requested
        comparison = None
        if args.compare:
            if args.compare.exists():
                baseline_data = json.loads(args.compare.read_text())
                baseline = EvalResults.model_validate(baseline_data)
                comparison = compare_runs(results, baseline)

                if not args.quiet:
                    print("\n" + format_comparison_report(comparison))
            else:
                logger.warning(f"Baseline file not found: {args.compare}")

        # Generate and output report
        if args.output:
            # Save to files
            include_json = args.format in ("json", "both")
            include_markdown = args.format in ("markdown", "both")

            saved_files = save_report(
                results,
                args.output,
                include_json=include_json,
                include_markdown=include_markdown,
            )

            for fmt, path in saved_files.items():
                logger.info(f"Saved {fmt} report to {path}")

            # Also save comparison if generated
            if comparison:
                comparison_path = args.output / "comparison.md"
                comparison_path.write_text(format_comparison_report(comparison))
                logger.info(f"Saved comparison report to {comparison_path}")
        else:
            # Print to stdout
            if args.format == "json":
                print(results.model_dump_json(indent=2))
            else:
                print(generate_markdown_report(results))

        # Summary
        if not args.quiet:
            print(f"\n{'='*60}")
            print("Evaluation Complete")
            print(f"{'='*60}")
            print(f"Total Samples: {results.total_samples}")
            print(f"Passed: {results.passed} ({results.pass_rate:.1%})")
            print(f"Failed: {results.failed}")
            print(f"Errors: {results.error_count}")
            print(f"Duration: {results.duration_seconds:.1f}s")
            print(f"{'='*60}")

            # Show aggregate metrics
            print("\nAggregate Metrics:")
            for metric, score in results.aggregate_metrics.items():
                threshold = thresholds.get(metric, 0.0)
                status = "✅" if score >= threshold else "❌"
                print(f"  {metric}: {score:.2%} (target: {threshold:.2%}) {status}")

        # Determine exit code
        if args.fail_on_regression:
            # Check if any threshold was not met
            for metric, score in results.aggregate_metrics.items():
                if score < thresholds.get(metric, 0.0):
                    logger.error(f"Threshold not met for {metric}: {score:.2%}")
                    return 1

            # Check if comparison has regressions
            if comparison and comparison.get("has_regressions"):
                logger.error("Regressions detected compared to baseline")
                return 1

        return 0

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
