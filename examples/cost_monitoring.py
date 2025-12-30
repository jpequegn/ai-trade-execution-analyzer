#!/usr/bin/env python3
"""Cost monitoring example.

This example demonstrates how to:
1. Set up cost tracking with budgets
2. Record token usage
3. Monitor daily and monthly costs
4. Handle budget alerts

Usage:
    python examples/cost_monitoring.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure the parent directory is in the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability.cost_tracker import (
    BudgetAlert,
    CostTracker,
    TokenUsage,
    calculate_cost,
)


def main() -> int:
    """Run cost monitoring example."""
    print("=" * 70)
    print(" COST MONITORING EXAMPLE ".center(70))
    print("=" * 70)

    # Create a temporary directory for cost tracking database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "cost_tracking.db"

        # Step 1: Set up cost tracker with budgets
        print("\n[1] Setting up cost tracker...")
        alerts_received: list[BudgetAlert] = []

        def on_alert(alert: BudgetAlert) -> None:
            alerts_received.append(alert)
            print(f"\n    ALERT: {alert.message}")

        tracker = CostTracker(
            storage_path=db_path,
            daily_budget=1.00,  # $1 daily budget
            monthly_budget=25.00,  # $25 monthly budget
            warning_threshold=0.8,  # Warn at 80%
        )
        tracker.on_alert(on_alert)

        print(f"    Database: {db_path}")
        print("    Daily Budget: $1.00")
        print("    Monthly Budget: $25.00")
        print("    Warning Threshold: 80%")

        # Step 2: Simulate some analyses
        print("\n[2] Recording usage (simulated analyses)...")

        analyses = [
            (
                "analysis_001",
                TokenUsage(input_tokens=450, output_tokens=320),
                "claude-sonnet-4-20250514",
            ),
            (
                "analysis_002",
                TokenUsage(input_tokens=380, output_tokens=290),
                "claude-sonnet-4-20250514",
            ),
            (
                "analysis_003",
                TokenUsage(input_tokens=520, output_tokens=410),
                "claude-sonnet-4-20250514",
            ),
            (
                "analysis_004",
                TokenUsage(input_tokens=0, output_tokens=0),
                "claude-sonnet-4-20250514",
            ),  # Cache hit
            (
                "analysis_005",
                TokenUsage(input_tokens=480, output_tokens=350),
                "claude-sonnet-4-20250514",
            ),
        ]

        for analysis_id, tokens, model in analyses:
            if tokens.total == 0:
                # Record cache hit
                record = tracker.record_cache_hit(
                    analysis_id=analysis_id,
                    estimated_tokens=TokenUsage(input_tokens=450, output_tokens=320),
                    model=model,
                )
                print(f"    {analysis_id}: CACHE HIT (savings: ${record.total_cost:.4f})")
            else:
                record = tracker.record_usage(
                    analysis_id=analysis_id,
                    tokens=tokens,
                    model=model,
                    provider="anthropic",
                )
                print(f"    {analysis_id}: {tokens.total} tokens = ${record.total_cost:.4f}")

        # Step 3: Show daily summary
        print("\n[3] Daily Cost Summary:")
        summary = tracker.get_daily_summary()
        print(f"    Date: {summary.date}")
        print(f"    Total Cost: ${summary.total_cost:.4f}")
        print(f"    Total Tokens: {summary.total_tokens:,}")
        print(f"    Analyses: {summary.analysis_count}")
        print(f"    Cache Hits: {summary.cache_hits}")
        print(f"    Cache Savings: ${summary.cache_savings:.4f}")

        if summary.model_breakdown:
            print("\n    Cost by Model:")
            for model, cost in summary.model_breakdown.items():
                print(f"      {model}: ${cost:.4f}")

        # Step 4: Show budget status
        print("\n[4] Budget Status:")
        status = tracker.get_budget_status()

        daily = status["daily"]
        print("\n    Daily Budget:")
        print(f"      Used: ${daily['used']:.4f}")
        print(f"      Limit: ${daily['limit']:.2f}")
        pct = daily.get("percentage", 0)
        bar_len = int(min(pct, 100) / 100 * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"      Status: [{bar}] {pct:.1f}%")

        monthly = status["monthly"]
        print("\n    Monthly Budget:")
        print(f"      Used: ${monthly['used']:.4f}")
        print(f"      Limit: ${monthly['limit']:.2f}")
        pct = monthly.get("percentage", 0)
        bar_len = int(min(pct, 100) / 100 * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"      Status: [{bar}] {pct:.1f}%")

        # Step 5: Show monthly report
        print("\n[5] Monthly Report:")
        report = tracker.get_monthly_report()
        print(f"    Month: {report.month.strftime('%B %Y')}")
        print(f"    Total Cost: ${report.total_cost:.4f}")
        print(f"    Daily Average: ${report.daily_average:.4f}")
        print(f"    Total Tokens: {report.total_tokens:,}")
        print(f"    Total Analyses: {report.total_analyses}")
        print(f"    Cache Hit Rate: {report.cache_hit_rate:.1%}")
        print(f"    Cache Savings: ${report.total_cache_savings:.4f}")

        # Step 6: Demonstrate cost calculation
        print("\n[6] Cost Calculation Examples:")

        providers_models = [
            ("anthropic", "claude-3-sonnet-20240229"),
            ("anthropic", "claude-3-opus-20240229"),
            ("openai", "gpt-4-turbo"),
        ]

        for provider, model in providers_models:
            input_cost, output_cost = calculate_cost(
                input_tokens=1000,
                output_tokens=500,
                model=model,
                provider=provider,
            )
            print(f"\n    {model}:")
            print("      1000 input + 500 output tokens")
            print(f"      Input cost:  ${input_cost:.4f}")
            print(f"      Output cost: ${output_cost:.4f}")
            print(f"      Total cost:  ${input_cost + output_cost:.4f}")

        # Step 7: Show any alerts
        print("\n[7] Budget Alerts:")
        if alerts_received:
            for alert in alerts_received:
                icon = "(!)" if alert.alert_type == "exceeded" else "(~)"
                print(f"    {icon} {alert.message}")
        else:
            print("    No alerts triggered.")

        # Step 8: Check if budget exceeded
        print("\n[8] Budget Check:")
        if tracker.is_budget_exceeded():
            print("    WARNING: Budget exceeded!")
        else:
            print("    OK: Within budget limits.")

    print("\n" + "=" * 70)
    print(" EXAMPLE COMPLETE ".center(70))
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
