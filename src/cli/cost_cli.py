"""CLI commands for cost tracking and budget management.

Usage:
    python -m src.cli.cost today
    python -m src.cli.cost monthly
    python -m src.cli.cost budget
    python -m src.cli.cost export costs.json
    python -m src.cli.cost alerts
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from src.observability.cost_tracker import CostTracker, get_cost_tracker

logger = logging.getLogger(__name__)


def show_daily_cost(tracker: CostTracker, target_date: date | None = None) -> int:
    """Display daily cost summary.

    Args:
        tracker: The cost tracker instance.
        target_date: Date to display (defaults to today).

    Returns:
        Exit code.
    """
    summary = tracker.get_daily_summary(target_date)

    print("\n" + "=" * 50)
    print(f" DAILY COST REPORT ({summary.date}) ".center(50))
    print("=" * 50)
    print(f"\nTotal cost:        ${summary.total_cost:.4f}")
    print(f"Total tokens:      {summary.total_tokens:,}")
    print(f"Analyses:          {summary.analysis_count}")
    print(f"Cache hits:        {summary.cache_hits}")
    print(f"Cache savings:     ${summary.cache_savings:.4f}")

    if summary.model_breakdown:
        print("\nCost by model:")
        for model, cost in summary.model_breakdown.items():
            print(f"  {model}: ${cost:.4f}")

    print("=" * 50)
    return 0


def show_monthly_report(tracker: CostTracker, target_month: date | None = None) -> int:
    """Display monthly cost report.

    Args:
        tracker: The cost tracker instance.
        target_month: Any date in target month (defaults to current month).

    Returns:
        Exit code.
    """
    report = tracker.get_monthly_report(target_month)

    print("\n" + "=" * 60)
    print(f" MONTHLY COST REPORT ({report.month.strftime('%B %Y')}) ".center(60))
    print("=" * 60)
    print(f"\nTotal cost:            ${report.total_cost:.2f}")
    print(f"Daily average:         ${report.daily_average:.2f}")
    print(f"Total tokens:          {report.total_tokens:,}")
    print(f"Total analyses:        {report.total_analyses}")
    print(f"Cache hit rate:        {report.cache_hit_rate:.1%}")
    print(f"Cache savings:         ${report.total_cache_savings:.2f}")

    if report.budget_remaining is not None:
        print(f"\nBudget remaining:      ${report.budget_remaining:.2f}")
        status_color = (
            "🟢"
            if report.budget_status == "ok"
            else "🟡"
            if report.budget_status == "warning"
            else "🔴"
        )
        print(f"Budget status:         {status_color} {report.budget_status}")

    if report.model_breakdown:
        print("\nCost by model:")
        for model, cost in report.model_breakdown.items():
            pct = (cost / report.total_cost * 100) if report.total_cost > 0 else 0
            print(f"  {model}: ${cost:.2f} ({pct:.1f}%)")

    print("=" * 60)
    return 0


def show_budget_status(tracker: CostTracker) -> int:
    """Display current budget status.

    Args:
        tracker: The cost tracker instance.

    Returns:
        Exit code.
    """
    status = tracker.get_budget_status()

    print("\n" + "=" * 50)
    print(" BUDGET STATUS ".center(50))
    print("=" * 50)

    # Daily budget
    daily = status["daily"]
    print("\nDaily Budget:")
    print(f"  Used:    ${daily['used']:.4f}")
    if daily["limit"]:
        print(f"  Limit:   ${daily['limit']:.2f}")
        pct = daily.get("percentage", 0)
        bar_len = int(min(pct, 100) / 100 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        status_icon = (
            "🟢" if daily["status"] == "ok" else "🟡" if daily["status"] == "warning" else "🔴"
        )
        print(f"  Status:  {status_icon} {daily['status']} [{bar}] {pct:.1f}%")
    else:
        print("  Limit:   Not set")

    # Monthly budget
    monthly = status["monthly"]
    print("\nMonthly Budget:")
    print(f"  Used:    ${monthly['used']:.2f}")
    if monthly["limit"]:
        print(f"  Limit:   ${monthly['limit']:.2f}")
        pct = monthly.get("percentage", 0)
        bar_len = int(min(pct, 100) / 100 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        status_icon = (
            "🟢" if monthly["status"] == "ok" else "🟡" if monthly["status"] == "warning" else "🔴"
        )
        print(f"  Status:  {status_icon} {monthly['status']} [{bar}] {pct:.1f}%")
    else:
        print("  Limit:   Not set")

    print("=" * 50)
    return 0


def show_alerts(tracker: CostTracker, show_all: bool = False) -> int:
    """Display budget alerts.

    Args:
        tracker: The cost tracker instance.
        show_all: Show all alerts (including acknowledged).

    Returns:
        Exit code.
    """
    alerts = tracker.get_alerts(unacknowledged_only=not show_all)

    if not alerts:
        print("No active alerts")
        return 0

    print("\n" + "=" * 60)
    print(" BUDGET ALERTS ".center(60))
    print("=" * 60)

    for alert in alerts:
        icon = "🔴" if alert.alert_type == "exceeded" else "🟡"
        print(f"\n{icon} [{alert.timestamp.strftime('%Y-%m-%d %H:%M')}]")
        print(f"   {alert.message}")

    print("\n" + "=" * 60)

    # Offer to acknowledge
    if alerts and not show_all:
        confirm = input("\nAcknowledge these alerts? [y/N]: ")
        if confirm.lower() in ("y", "yes"):
            count = tracker.acknowledge_alerts()
            print(f"Acknowledged {count} alerts")

    return 0


def export_costs(tracker: CostTracker, output_path: Path) -> int:
    """Export cost records to JSON.

    Args:
        tracker: The cost tracker instance.
        output_path: Path for export file.

    Returns:
        Exit code.
    """
    try:
        count = tracker.export_json(output_path)
        print(f"Exported {count} cost records to {output_path}")
        return 0
    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        return 1


def clear_costs(tracker: CostTracker, force: bool = False) -> int:
    """Clear all cost records.

    Args:
        tracker: The cost tracker instance.
        force: Skip confirmation if True.

    Returns:
        Exit code.
    """
    if not force:
        confirm = input("This will delete all cost records. Are you sure? [y/N]: ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled")
            return 0

    cleared = tracker.clear()
    print(f"Cleared {cleared} cost records")
    return 0


def cost_cli(args: list[str] | None = None) -> int:
    """Cost management CLI entry point.

    Args:
        args: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Manage cost tracking and budgets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli.cost today
  python -m src.cli.cost today --date 2024-01-15
  python -m src.cli.cost monthly
  python -m src.cli.cost budget
  python -m src.cli.cost alerts
  python -m src.cli.cost export costs.json
  python -m src.cli.cost clear --force
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Today command
    today_parser = subparsers.add_parser("today", help="Show today's cost summary")
    today_parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Specific date (YYYY-MM-DD)",
    )

    # Monthly command
    monthly_parser = subparsers.add_parser("monthly", help="Show monthly cost report")
    monthly_parser.add_argument(
        "--month",
        type=lambda s: datetime.strptime(s, "%Y-%m").date(),
        help="Specific month (YYYY-MM)",
    )

    # Budget command
    subparsers.add_parser("budget", help="Show budget status")

    # Alerts command
    alerts_parser = subparsers.add_parser("alerts", help="Show budget alerts")
    alerts_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all alerts including acknowledged",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export costs to JSON")
    export_parser.add_argument(
        "output",
        type=Path,
        help="Output file path",
    )

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all cost records")
    clear_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation",
    )

    # Common options
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("cost_tracking.db"),
        help="Cost tracking database path (default: cost_tracking.db)",
    )
    parser.add_argument(
        "--daily-budget",
        type=float,
        help="Daily budget limit in dollars",
    )
    parser.add_argument(
        "--monthly-budget",
        type=float,
        help="Monthly budget limit in dollars",
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

    # Initialize tracker
    tracker = get_cost_tracker(
        storage_path=parsed_args.db,
        daily_budget=parsed_args.daily_budget,
        monthly_budget=parsed_args.monthly_budget,
    )

    # Execute command
    if parsed_args.command == "today":
        return show_daily_cost(tracker, parsed_args.date)
    elif parsed_args.command == "monthly":
        return show_monthly_report(tracker, getattr(parsed_args, "month", None))
    elif parsed_args.command == "budget":
        return show_budget_status(tracker)
    elif parsed_args.command == "alerts":
        return show_alerts(tracker, show_all=parsed_args.all)
    elif parsed_args.command == "export":
        return export_costs(tracker, parsed_args.output)
    elif parsed_args.command == "clear":
        return clear_costs(tracker, force=parsed_args.force)

    return 0


if __name__ == "__main__":
    sys.exit(cost_cli())
