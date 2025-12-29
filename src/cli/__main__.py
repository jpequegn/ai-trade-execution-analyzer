"""CLI entry point for cache and cost management.

Usage:
    python -m src.cli cache stats
    python -m src.cli cost today
"""

from __future__ import annotations

import sys

from src.cli.cache_cli import cache_cli
from src.cli.cost_cli import cost_cli


def main() -> int:
    """Main CLI entry point.

    Routes to cache_cli or cost_cli based on first argument.

    Returns:
        Exit code.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli [cache|cost] [command] [options]")
        print("\nAvailable modules:")
        print("  cache  - Cache management (stats, clear, cleanup, export)")
        print("  cost   - Cost tracking (today, monthly, budget, alerts, export)")
        print("\nExamples:")
        print("  python -m src.cli cache stats")
        print("  python -m src.cli cost today")
        print("  python -m src.cli cost budget --daily-budget 10 --monthly-budget 200")
        return 0

    module = sys.argv[1]
    remaining_args = sys.argv[2:]

    if module == "cache":
        return cache_cli(remaining_args)
    elif module == "cost":
        return cost_cli(remaining_args)
    else:
        print(f"Unknown module: {module}")
        print("Use 'cache' or 'cost'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
