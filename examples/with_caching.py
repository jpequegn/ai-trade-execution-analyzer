#!/usr/bin/env python3
"""Trade analysis with caching example.

This example demonstrates how to:
1. Configure the analysis cache
2. See cache hits reduce LLM calls
3. Monitor cache statistics and savings

Usage:
    python examples/with_caching.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure the parent directory is in the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.analyzer import TradeAnalyzer
from src.agents.cache import AnalysisCache
from src.parsers import parse_fix_message


def main() -> int:
    """Run caching example."""
    print("=" * 70)
    print(" TRADE ANALYSIS WITH CACHING EXAMPLE ".center(70))
    print("=" * 70)

    # Create a temporary directory for cache storage
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "analysis_cache.db"

        # Step 1: Create cache and analyzer
        print("\n[1] Setting up cache...")
        cache = AnalysisCache(
            backend="sqlite",  # Use SQLite for persistence
            storage_path=cache_path,
            ttl_hours=24,  # Cache entries expire after 24 hours
            key_strategy="semantic",  # Use semantic matching
        )
        print("    Backend: SQLite")
        print(f"    Path: {cache_path}")
        print("    TTL: 24 hours")
        print("    Strategy: semantic")

        # Create analyzer with cache
        analyzer = TradeAnalyzer(cache=cache)

        # Step 2: First analysis - cache miss
        print("\n[2] First analysis (should be cache MISS)...")
        fix_message = (
            "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD001|11=CLT001|"
            "17=EXEC001|20=0|150=F|39=2|55=AAPL|54=1|38=100|44=150.50|"
            "32=100|31=150.45|14=100|6=150.45|60=20240115-14:30:00.000|10=123|"
        )

        execution = parse_fix_message(fix_message)
        print(f"    Symbol: {execution.symbol}")
        print(f"    Side: {execution.side}")

        try:
            result1 = analyzer.analyze(execution)
        except Exception as e:
            print(f"\n    ERROR: {e}")
            print("\n    Make sure ANTHROPIC_API_KEY is set in your environment.")
            return 1

        print(f"    From Cache: {result1.from_cache}")
        print(f"    Latency: {result1.latency_ms:.0f}ms")
        print(f"    Tokens: {result1.tokens_used}")
        print(f"    Quality Score: {result1.analysis.quality_score}/10")

        # Check cache stats after first call
        stats = cache.get_stats()
        print(f"\n    Cache Stats: {stats.hits} hits, {stats.misses} misses")

        # Step 3: Same analysis again - cache hit
        print("\n[3] Same analysis again (should be cache HIT)...")
        result2 = analyzer.analyze(execution)

        print(f"    From Cache: {result2.from_cache}")
        print(f"    Latency: {result2.latency_ms:.0f}ms")  # Much faster!
        print(f"    Quality Score: {result2.analysis.quality_score}/10")

        # Check cache stats after second call
        stats = cache.get_stats()
        print(f"\n    Cache Stats: {stats.hits} hits, {stats.misses} misses")

        # Step 4: Similar trade - semantic cache hit
        print("\n[4] Similar trade (semantic cache check)...")
        similar_message = (
            "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD002|11=CLT002|"
            "17=EXEC002|20=0|150=F|39=2|55=AAPL|54=1|38=100|44=150.55|"  # Same AAPL BUY, slightly different price
            "32=100|31=150.48|14=100|6=150.48|60=20240115-14:35:00.000|10=124|"
        )

        similar_execution = parse_fix_message(similar_message)
        print(f"    Symbol: {similar_execution.symbol}")
        print(f"    Side: {similar_execution.side}")
        print(f"    Price: ${similar_execution.price:.2f} (vs ${execution.price:.2f})")

        result3 = analyzer.analyze(similar_execution)
        print(f"    From Cache: {result3.from_cache}")
        print("    (Semantic matching may hit if prices are in same bucket)")

        # Step 5: Different trade - cache miss
        print("\n[5] Different trade (should be cache MISS)...")
        different_message = (
            "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD003|11=CLT003|"
            "17=EXEC003|20=0|150=F|39=2|55=MSFT|54=2|38=500|44=380.00|"  # MSFT SELL
            "32=500|31=379.85|14=500|6=379.85|60=20240115-14:40:00.000|10=125|"
        )

        different_execution = parse_fix_message(different_message)
        print(f"    Symbol: {different_execution.symbol}")
        print(f"    Side: {different_execution.side}")

        result4 = analyzer.analyze(different_execution)
        print(f"    From Cache: {result4.from_cache}")
        print(f"    Latency: {result4.latency_ms:.0f}ms")

        # Step 6: Final cache statistics
        print("\n[6] Final Cache Statistics:")
        final_stats = cache.get_stats()
        print(f"    Total Entries: {final_stats.size}")
        print(f"    Hits: {final_stats.hits}")
        print(f"    Misses: {final_stats.misses}")
        hit_rate = (
            final_stats.hits / (final_stats.hits + final_stats.misses)
            if (final_stats.hits + final_stats.misses) > 0
            else 0
        )
        print(f"    Hit Rate: {hit_rate:.1%}")

        # Calculate savings
        tokens_saved = result1.tokens_used  # Saved on cache hit
        print(f"\n    Tokens Saved: {tokens_saved}")
        # Rough cost estimate (assuming Claude Sonnet pricing)
        cost_saved = tokens_saved * 0.003 / 1000  # $3/1M input tokens
        print(f"    Estimated Savings: ${cost_saved:.4f}")

    print("\n" + "=" * 70)
    print(" EXAMPLE COMPLETE ".center(70))
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
