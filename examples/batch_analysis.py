#!/usr/bin/env python3
"""Batch trade execution analysis example.

This example demonstrates how to:
1. Analyze multiple FIX messages concurrently
2. Handle partial failures gracefully
3. Process results efficiently

Usage:
    python examples/batch_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure the parent directory is in the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.analyzer import AnalysisError
from src.pipeline import analyze_fix_messages

if TYPE_CHECKING:
    from src.parsers.models import AnalysisResult


def main() -> int:
    """Run batch analysis example."""
    # Sample FIX messages - various scenarios
    fix_messages = [
        # Good execution - AAPL full fill
        "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD001|11=CLT001|"
        "17=EXEC001|20=0|150=F|39=2|55=AAPL|54=1|38=100|44=150.50|"
        "32=100|31=150.45|14=100|6=150.45|60=20240115-09:30:05.000|10=123|",
        # Partial fill - MSFT
        "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD002|11=CLT002|"
        "17=EXEC002|20=0|150=F|39=1|55=MSFT|54=2|38=500|44=380.00|"
        "32=200|31=379.85|14=200|6=379.85|60=20240115-14:30:00.000|10=124|",
        # Large order - GOOGL
        "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD003|11=CLT003|"
        "17=EXEC003|20=0|150=F|39=2|55=GOOGL|54=1|38=1000|44=142.00|"
        "32=1000|31=142.15|14=1000|6=142.15|60=20240115-14:32:00.000|10=125|",
        # End of day - META
        "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD004|11=CLT004|"
        "17=EXEC004|20=0|150=F|39=2|55=META|54=1|38=50|44=485.00|"
        "32=50|31=485.50|14=50|6=485.50|60=20240115-15:59:30.000|10=126|",
    ]

    print("=" * 70)
    print(" BATCH TRADE EXECUTION ANALYSIS EXAMPLE ".center(70))
    print("=" * 70)

    print(f"\nAnalyzing {len(fix_messages)} trades concurrently...")
    print("(This may take several seconds depending on LLM response times)\n")

    # Run batch analysis with concurrent processing
    try:
        results = analyze_fix_messages(
            fix_messages,
            max_concurrent=3,  # Limit concurrent LLM calls
            continue_on_error=True,  # Don't stop on individual failures
        )
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nMake sure ANTHROPIC_API_KEY is set in your environment.")
        return 1

    # Process results
    successful = 0
    failed = 0

    for i, result in enumerate(results, 1):
        print("-" * 70)
        print(f"Trade {i}:")

        if isinstance(result, AnalysisError):
            failed += 1
            print("  Status: FAILED")
            print(f"  Error: {result.message}")
            if result.execution:
                print(f"  Symbol: {result.execution.symbol}")
        else:
            successful += 1
            result: AnalysisResult
            print("  Status: SUCCESS")
            print(f"  Symbol: {result.execution.symbol}")
            print(f"  Side: {result.execution.side}")
            print(f"  Quality Score: {result.analysis.quality_score}/10")
            print(f"  Confidence: {result.analysis.confidence:.1%}")
            print(f"  Issues: {len(result.analysis.issues)}")
            print(f"  Recommendations: {len(result.analysis.recommendations)}")
            print(f"  Tokens: {result.tokens_used}")
            print(f"  Latency: {result.latency_ms:.0f}ms")

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY ".center(70))
    print("=" * 70)
    print(f"\n  Total trades: {len(fix_messages)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    if successful > 0:
        # Calculate aggregate statistics
        successful_results = [r for r in results if not isinstance(r, AnalysisError)]
        avg_score = sum(r.analysis.quality_score for r in successful_results) / len(
            successful_results
        )
        total_tokens = sum(r.tokens_used for r in successful_results)
        avg_latency = sum(r.latency_ms for r in successful_results) / len(successful_results)

        print(f"\n  Average Quality Score: {avg_score:.1f}/10")
        print(f"  Total Tokens Used: {total_tokens}")
        print(f"  Average Latency: {avg_latency:.0f}ms")

    print("\n" + "=" * 70)
    print(" EXAMPLE COMPLETE ".center(70))
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
