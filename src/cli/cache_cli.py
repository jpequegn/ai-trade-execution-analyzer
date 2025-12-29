"""CLI commands for cache management.

Usage:
    python -m src.cli.cache stats
    python -m src.cli.cache clear
    python -m src.cli.cache cleanup
    python -m src.cli.cache export cache_export.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.agents.cache import AnalysisCache

logger = logging.getLogger(__name__)


def show_stats(cache: AnalysisCache) -> int:
    """Display cache statistics.

    Args:
        cache: The cache instance.

    Returns:
        Exit code.
    """
    stats = cache.get_stats()

    print("\n" + "=" * 50)
    print(" CACHE STATISTICS ".center(50))
    print("=" * 50)
    print(f"\nTotal entries:     {stats.total_entries:,}")
    print(f"Cache hits:        {stats.hits:,}")
    print(f"Cache misses:      {stats.misses:,}")
    print(f"Evictions:         {stats.evictions:,}")
    print(f"Storage used:      {stats.bytes_used / 1024:.1f} KB")

    if stats.hits + stats.misses > 0:
        hit_rate = stats.hits / (stats.hits + stats.misses)
        print(f"\nHit rate:          {hit_rate:.1%}")

    print("=" * 50)
    return 0


def clear_cache(cache: AnalysisCache, force: bool = False) -> int:
    """Clear all cache entries.

    Args:
        cache: The cache instance.
        force: Skip confirmation if True.

    Returns:
        Exit code.
    """
    stats = cache.get_stats()
    count = stats.total_entries

    if count == 0:
        print("Cache is empty")
        return 0

    if not force:
        confirm = input(f"This will delete {count} cache entries. Are you sure? [y/N]: ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled")
            return 0

    cleared = cache.clear()
    print(f"Cleared {cleared} cache entries")
    return 0


def cleanup_cache(cache: AnalysisCache) -> int:
    """Remove expired cache entries.

    Args:
        cache: The cache instance.

    Returns:
        Exit code.
    """
    removed = cache.cleanup()
    print(f"Removed {removed} expired entries")
    return 0


def export_cache(cache: AnalysisCache, output_path: Path) -> int:
    """Export cache entries to JSON file.

    Args:
        cache: The cache instance.
        output_path: Path for export file.

    Returns:
        Exit code.
    """
    try:
        count = cache.export_json(output_path)
        print(f"Exported {count} entries to {output_path}")
        return 0
    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        return 1


def cache_cli(args: list[str] | None = None) -> int:
    """Cache management CLI entry point.

    Args:
        args: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Manage the analysis cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli.cache stats
  python -m src.cli.cache clear
  python -m src.cli.cache cleanup
  python -m src.cli.cache export cache_backup.json
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Stats command
    subparsers.add_parser("stats", help="Show cache statistics")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all cache entries")
    clear_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation",
    )

    # Cleanup command
    subparsers.add_parser("cleanup", help="Remove expired entries")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export cache to JSON")
    export_parser.add_argument(
        "output",
        type=Path,
        help="Output file path",
    )

    # Common options
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/analysis"),
        help="Cache directory (default: .cache/analysis)",
    )
    parser.add_argument(
        "--backend",
        choices=["file", "sqlite"],
        default="file",
        help="Cache backend (default: file)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parsed_args = parser.parse_args(args)

    # Configure logging
    if parsed_args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if not parsed_args.command:
        parser.print_help()
        return 0

    # Initialize cache
    cache = AnalysisCache(
        backend=parsed_args.backend,
        cache_dir=parsed_args.cache_dir,
    )

    # Execute command
    if parsed_args.command == "stats":
        return show_stats(cache)
    elif parsed_args.command == "clear":
        return clear_cache(cache, force=parsed_args.force)
    elif parsed_args.command == "cleanup":
        return cleanup_cache(cache)
    elif parsed_args.command == "export":
        return export_cache(cache, parsed_args.output)

    return 0


if __name__ == "__main__":
    sys.exit(cache_cli())
