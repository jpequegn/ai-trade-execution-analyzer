"""End-to-end pipeline for trade execution analysis.

This module provides the complete pipeline from raw FIX messages
to structured analysis results, integrating all components.

Example:
    >>> from src.pipeline import analyze_fix_message
    >>> result = analyze_fix_message("8=FIX.4.4|35=8|37=ORD001|...")
    >>> print(result.analysis.quality_score)

CLI Usage:
    python -m src.pipeline --message "8=FIX.4.4|35=8|..."
    python -m src.pipeline --file executions.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import TYPE_CHECKING

from src.agents.analyzer import AnalysisError, TradeAnalyzer
from src.agents.llm_client import LLMClient
from src.agents.prompts import PromptVariant
from src.config import AppConfig, get_config
from src.observability.tracing import get_tracer, trace_context
from src.parsers.fix_parser import parse_fix_message
from src.parsers.models import AnalysisResult

if TYPE_CHECKING:
    from src.parsers.fix_parser import ExecutionReport

logger = logging.getLogger(__name__)

# Default sample FIX message for testing
SAMPLE_FIX_MESSAGE = (
    "8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|"
    "31=150.50|30=NYSE|60=20240115-10:30:00.000|39=2"
)


def analyze_fix_message(
    raw_message: str,
    variant: PromptVariant = PromptVariant.DETAILED,
    config: AppConfig | None = None,
    session_id: str | None = None,
) -> AnalysisResult:
    """Complete pipeline: FIX message string to analysis result.

    This is the main entry point for analyzing a single FIX message.
    It handles:
    1. Parsing the FIX message
    2. Creating the analyzer
    3. Running the analysis
    4. Returning the structured result

    Args:
        raw_message: Raw FIX protocol message string.
        variant: Prompt variant for analysis depth.
        config: Optional application configuration.
        session_id: Optional session ID for trace grouping.

    Returns:
        AnalysisResult containing the execution and analysis.

    Raises:
        FIXParseError: If the message cannot be parsed.
        AnalysisError: If the analysis fails.

    Example:
        >>> result = analyze_fix_message("8=FIX.4.4|35=8|37=ORD001|...")
        >>> print(f"Order {result.execution.order_id}: {result.analysis.quality_score}/10")
    """
    config = config or get_config()

    with trace_context(
        name="analyze_fix_message",
        session_id=session_id,
        tags=["pipeline", "single"],
        metadata={"variant": variant.value},
    ):
        # Parse FIX message
        logger.info("Parsing FIX message...")
        execution = parse_fix_message(raw_message)
        logger.info(f"Parsed execution: {execution.order_id} ({execution.symbol})")

        # Create analyzer and run analysis
        client = LLMClient(app_config=config)
        analyzer = TradeAnalyzer(client=client, config=config, default_variant=variant)

        logger.info("Running analysis...")
        result = analyzer.analyze(execution, session_id=session_id)

        return result


def analyze_fix_messages(
    raw_messages: list[str],
    variant: PromptVariant = PromptVariant.DETAILED,
    config: AppConfig | None = None,
    session_id: str | None = None,
    continue_on_error: bool = True,
    max_concurrent: int = 5,
) -> list[AnalysisResult | AnalysisError]:
    """Batch pipeline: Multiple FIX messages to analysis results.

    Processes multiple FIX messages concurrently for efficiency.

    Args:
        raw_messages: List of raw FIX protocol message strings.
        variant: Prompt variant for analysis depth.
        config: Optional application configuration.
        session_id: Optional session ID for trace grouping.
        continue_on_error: If True, continue on individual failures.
        max_concurrent: Maximum concurrent analyses.

    Returns:
        List of results in same order as input messages.
        Each item is either AnalysisResult or AnalysisError.

    Example:
        >>> messages = ["8=FIX.4.4|...", "8=FIX.4.4|..."]
        >>> results = analyze_fix_messages(messages)
        >>> for r in results:
        ...     if isinstance(r, AnalysisResult):
        ...         print(f"{r.execution.order_id}: {r.analysis.quality_score}")
    """
    config = config or get_config()

    with trace_context(
        name="analyze_fix_messages_batch",
        session_id=session_id,
        tags=["pipeline", "batch"],
        metadata={"count": len(raw_messages), "variant": variant.value},
    ):
        # Parse all FIX messages
        logger.info(f"Parsing {len(raw_messages)} FIX messages...")
        executions: list[ExecutionReport] = []
        parse_errors: list[tuple[int, Exception]] = []

        for i, msg in enumerate(raw_messages):
            try:
                execution = parse_fix_message(msg)
                executions.append(execution)
            except Exception as e:
                logger.warning(f"Failed to parse message {i}: {e}")
                parse_errors.append((i, e))
                if not continue_on_error:
                    raise

        logger.info(f"Parsed {len(executions)} executions ({len(parse_errors)} parse errors)")

        # Create analyzer and run batch analysis
        client = LLMClient(app_config=config)
        analyzer = TradeAnalyzer(
            client=client,
            config=config,
            default_variant=variant,
            max_concurrent=max_concurrent,
        )

        logger.info("Running batch analysis...")
        results = analyzer.analyze_batch(
            executions,
            session_id=session_id,
            continue_on_error=continue_on_error,
        )

        return results


def analyze_execution(
    execution: ExecutionReport,
    variant: PromptVariant = PromptVariant.DETAILED,
    config: AppConfig | None = None,
    session_id: str | None = None,
) -> AnalysisResult:
    """Analyze a pre-parsed ExecutionReport.

    Use this when you already have a parsed ExecutionReport object.

    Args:
        execution: Parsed execution report.
        variant: Prompt variant for analysis depth.
        config: Optional application configuration.
        session_id: Optional session ID for trace grouping.

    Returns:
        AnalysisResult containing the analysis.

    Example:
        >>> from src.parsers.fix_parser import parse_fix_message
        >>> execution = parse_fix_message("8=FIX.4.4|...")
        >>> result = analyze_execution(execution)
    """
    config = config or get_config()
    client = LLMClient(app_config=config)
    analyzer = TradeAnalyzer(client=client, config=config, default_variant=variant)
    return analyzer.analyze(execution, session_id=session_id)


def format_result_for_display(result: AnalysisResult) -> str:
    """Format an analysis result for console display.

    Args:
        result: The analysis result to format.

    Returns:
        Formatted string for display.
    """
    lines = [
        "=" * 60,
        f"Analysis Result for Order: {result.execution.order_id}",
        "=" * 60,
        "",
        "Trade Details:",
        f"  Symbol:    {result.execution.symbol}",
        f"  Side:      {result.execution.side}",
        f"  Quantity:  {result.execution.quantity:,.0f}",
        f"  Price:     ${result.execution.price:,.2f}",
        f"  Venue:     {result.execution.venue}",
        f"  Timestamp: {result.execution.timestamp}",
        "",
        "Analysis:",
        f"  Quality Score: {result.analysis.quality_score}/10",
        f"  Confidence:    {result.analysis.confidence:.0%}",
        "",
    ]

    if result.analysis.observations:
        lines.append("  Observations:")
        for obs in result.analysis.observations:
            lines.append(f"    - {obs}")
        lines.append("")

    if result.analysis.issues:
        lines.append("  Issues:")
        for issue in result.analysis.issues:
            lines.append(f"    - {issue}")
        lines.append("")

    if result.analysis.recommendations:
        lines.append("  Recommendations:")
        for rec in result.analysis.recommendations:
            lines.append(f"    - {rec}")
        lines.append("")

    lines.extend(
        [
            "Metadata:",
            f"  Model:      {result.model}",
            f"  Tokens:     {result.tokens_used}",
            f"  Latency:    {result.latency_ms:.0f}ms",
            f"  Analysis ID: {result.analysis_id}",
            "=" * 60,
        ]
    )

    return "\n".join(lines)


def format_result_as_json(result: AnalysisResult) -> str:
    """Format an analysis result as JSON.

    Args:
        result: The analysis result to format.

    Returns:
        JSON string representation.
    """
    data = {
        "execution": {
            "order_id": result.execution.order_id,
            "symbol": result.execution.symbol,
            "side": result.execution.side,
            "quantity": result.execution.quantity,
            "price": result.execution.price,
            "venue": result.execution.venue,
            "timestamp": result.execution.timestamp.isoformat(),
            "fill_type": result.execution.fill_type,
        },
        "analysis": {
            "quality_score": result.analysis.quality_score,
            "confidence": result.analysis.confidence,
            "observations": result.analysis.observations,
            "issues": result.analysis.issues,
            "recommendations": result.analysis.recommendations,
        },
        "metadata": {
            "model": result.model,
            "tokens_used": result.tokens_used,
            "latency_ms": result.latency_ms,
            "analysis_id": result.analysis_id,
            "analyzed_at": result.analyzed_at.isoformat(),
        },
    }
    return json.dumps(data, indent=2)


def main() -> int:
    """CLI entry point for the analysis pipeline.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Analyze FIX trade execution messages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single message
  python -m src.pipeline --message "8=FIX.4.4|35=8|37=ORD001|..."

  # Use the sample message
  python -m src.pipeline --sample

  # Read messages from file
  python -m src.pipeline --file executions.txt

  # Output as JSON
  python -m src.pipeline --sample --format json

  # Use quick analysis (fewer tokens)
  python -m src.pipeline --sample --variant quick
""",
    )

    parser.add_argument(
        "--message",
        "-m",
        type=str,
        help="FIX message to analyze",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="File containing FIX messages (one per line)",
    )
    parser.add_argument(
        "--sample",
        "-s",
        action="store_true",
        help="Use sample FIX message for testing",
    )
    parser.add_argument(
        "--variant",
        "-v",
        type=str,
        choices=["quick", "detailed"],
        default="detailed",
        help="Analysis variant (default: detailed)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Determine input
    if args.sample:
        messages = [SAMPLE_FIX_MESSAGE]
    elif args.message:
        messages = [args.message]
    elif args.file:
        try:
            from pathlib import Path

            with Path(args.file).open() as f:
                messages = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 1

    # Parse variant
    variant = PromptVariant(args.variant)

    # Run analysis
    try:
        if len(messages) == 1:
            result = analyze_fix_message(messages[0], variant=variant)
            if args.format == "json":
                print(format_result_as_json(result))
            else:
                print(format_result_for_display(result))
        else:
            results = analyze_fix_messages(messages, variant=variant)
            for i, batch_result in enumerate(results):
                if isinstance(batch_result, AnalysisResult):
                    if args.format == "json":
                        print(format_result_as_json(batch_result))
                    else:
                        print(format_result_for_display(batch_result))
                else:
                    print(f"Error analyzing message {i}: {batch_result}", file=sys.stderr)

        # Flush traces
        tracer = get_tracer()
        tracer.flush()

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
