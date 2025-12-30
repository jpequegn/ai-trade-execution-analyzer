#!/usr/bin/env python3
"""Basic trade execution analysis example.

This example demonstrates how to:
1. Parse a FIX protocol message
2. Run AI analysis on the execution
3. Display formatted results

Usage:
    python examples/basic_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the parent directory is in the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsers import parse_fix_message
from src.pipeline import analyze_fix_message, format_result_for_display


def main() -> int:
    """Run basic analysis example."""
    # Sample FIX message - AAPL buy order, full fill
    fix_message = (
        "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD001|11=CLT001|"
        "17=EXEC001|20=0|150=F|39=2|55=AAPL|54=1|38=100|44=150.50|"
        "32=100|31=150.45|14=100|6=150.45|60=20240115-14:30:00.000|10=123|"
    )

    print("=" * 70)
    print(" BASIC TRADE EXECUTION ANALYSIS EXAMPLE ".center(70))
    print("=" * 70)

    # Step 1: Parse the FIX message (optional - pipeline does this automatically)
    print("\n[1] Parsing FIX message...")
    execution = parse_fix_message(fix_message)
    print(f"    Symbol: {execution.symbol}")
    print(f"    Side: {execution.side}")
    print(f"    Quantity: {execution.quantity}")
    print(f"    Price: ${execution.price:.2f}")
    print(f"    Fill Type: {execution.fill_type}")

    # Step 2: Run AI analysis
    print("\n[2] Running AI analysis...")
    print("    (This calls the LLM - may take a few seconds)")

    try:
        result = analyze_fix_message(fix_message)
    except Exception as e:
        print(f"\n    ERROR: {e}")
        print("\n    Make sure ANTHROPIC_API_KEY is set in your environment.")
        return 1

    # Step 3: Display results
    print("\n[3] Analysis Results:")
    print(format_result_for_display(result))

    # Step 4: Show metadata
    print("\n[4] Metadata:")
    print(f"    Analysis ID: {result.analysis_id}")
    print(f"    Model: {result.model}")
    print(f"    Tokens Used: {result.tokens_used}")
    print(f"    Latency: {result.latency_ms:.0f}ms")
    print(f"    From Cache: {result.from_cache}")

    print("\n" + "=" * 70)
    print(" EXAMPLE COMPLETE ".center(70))
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
