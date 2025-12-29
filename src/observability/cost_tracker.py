"""Cost tracking and budget management for LLM usage.

This module provides comprehensive cost tracking capabilities including:
- Per-analysis cost recording
- Daily/monthly cost aggregation
- Budget alerts and limits
- Cost reporting and dashboard data

Example:
    >>> from src.observability.cost_tracker import CostTracker, get_cost_tracker
    >>> tracker = get_cost_tracker()
    >>> tracker.record_usage(analysis_id="abc", tokens=TokenUsage(1000, 500), model="claude-3-sonnet")
    >>> print(tracker.get_daily_cost())
    0.0165

With budget alerts:
    >>> tracker = CostTracker(daily_budget=10.0, monthly_budget=200.0)
    >>> tracker.record_usage(analysis_id="xyz", tokens=TokenUsage(1000, 500), model="claude-3-sonnet")
    >>> if tracker.is_budget_exceeded():
    ...     print("Budget exceeded!")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Cost Constants - Pricing per 1M tokens (as of 2024)
# ============================================================================

# Anthropic Claude pricing (per 1M tokens)
ANTHROPIC_PRICING = {
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    # Default fallback
    "default": {"input": 3.00, "output": 15.00},
}

# OpenAI GPT pricing (per 1M tokens)
OPENAI_PRICING = {
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Default fallback
    "default": {"input": 2.50, "output": 10.00},
}


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class TokenUsage:
    """Token usage for a single LLM call.

    Attributes:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
    """

    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


class CostRecord(BaseModel):
    """Record of a single LLM usage cost.

    Attributes:
        analysis_id: Unique identifier for the analysis.
        timestamp: When the usage occurred.
        model: Model used for the call.
        provider: LLM provider (anthropic, openai).
        input_tokens: Input token count.
        output_tokens: Output token count.
        input_cost: Cost for input tokens.
        output_cost: Cost for output tokens.
        total_cost: Total cost for the call.
        cached: Whether this was a cache hit (zero cost).
    """

    analysis_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    model: str
    provider: str = "anthropic"
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    input_cost: float = Field(ge=0)
    output_cost: float = Field(ge=0)
    total_cost: float = Field(ge=0)
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


class DailyCostSummary(BaseModel):
    """Summary of costs for a single day.

    Attributes:
        date: The date of the summary.
        total_cost: Total cost for the day.
        total_tokens: Total tokens used.
        analysis_count: Number of analyses performed.
        cache_hits: Number of cache hits.
        cache_savings: Estimated savings from cache.
        model_breakdown: Cost breakdown by model.
    """

    date: date
    total_cost: float = 0.0
    total_tokens: int = 0
    analysis_count: int = 0
    cache_hits: int = 0
    cache_savings: float = 0.0
    model_breakdown: dict[str, float] = Field(default_factory=dict)


class MonthlyCostReport(BaseModel):
    """Monthly cost report with aggregated statistics.

    Attributes:
        month: The month (first day of month).
        total_cost: Total cost for the month.
        daily_average: Average daily cost.
        total_tokens: Total tokens used.
        total_analyses: Total analyses performed.
        cache_hit_rate: Percentage of cache hits.
        total_cache_savings: Total savings from caching.
        model_breakdown: Cost breakdown by model.
        daily_costs: Daily cost values.
        budget_remaining: Remaining monthly budget (if set).
        budget_status: Budget status description.
    """

    month: date
    total_cost: float = 0.0
    daily_average: float = 0.0
    total_tokens: int = 0
    total_analyses: int = 0
    cache_hit_rate: float = 0.0
    total_cache_savings: float = 0.0
    model_breakdown: dict[str, float] = Field(default_factory=dict)
    daily_costs: list[float] = Field(default_factory=list)
    budget_remaining: float | None = None
    budget_status: str = "ok"


class BudgetAlert(BaseModel):
    """Budget alert notification.

    Attributes:
        alert_type: Type of alert (warning, exceeded).
        budget_type: Budget type (daily, monthly).
        current_spend: Current spending amount.
        budget_limit: Budget limit.
        percentage: Percentage of budget used.
        timestamp: When the alert was triggered.
        message: Human-readable alert message.
    """

    alert_type: str  # "warning" or "exceeded"
    budget_type: str  # "daily" or "monthly"
    current_spend: float
    budget_limit: float
    percentage: float
    timestamp: datetime = Field(default_factory=datetime.now)
    message: str = ""


# ============================================================================
# Cost Calculation
# ============================================================================


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    provider: str = "anthropic",
) -> tuple[float, float]:
    """Calculate cost for token usage.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model identifier.
        provider: LLM provider.

    Returns:
        Tuple of (input_cost, output_cost).
    """
    # Get pricing table for provider
    pricing = OPENAI_PRICING if provider == "openai" else ANTHROPIC_PRICING

    # Get model pricing or default
    model_pricing = pricing.get(model, pricing["default"])

    # Calculate costs (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

    return input_cost, output_cost


# ============================================================================
# Cost Tracker
# ============================================================================


@dataclass
class CostTracker:
    """Tracks LLM usage costs with budget management.

    This class provides comprehensive cost tracking:
    - Records individual usage costs
    - Aggregates daily/monthly costs
    - Monitors budget limits
    - Generates alerts when limits approached/exceeded

    Attributes:
        storage_path: Path to SQLite database for persistence.
        daily_budget: Daily cost budget limit (None for unlimited).
        monthly_budget: Monthly cost budget limit (None for unlimited).
        warning_threshold: Percentage of budget to trigger warning (default 80%).

    Example:
        >>> tracker = CostTracker(daily_budget=10.0)
        >>> tracker.record_usage("analysis_1", TokenUsage(1000, 500), "claude-3-sonnet")
        >>> print(tracker.get_daily_cost())
    """

    storage_path: Path = field(default_factory=lambda: Path("cost_tracking.db"))
    daily_budget: float | None = None
    monthly_budget: float | None = None
    warning_threshold: float = 0.8
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _db_initialized: bool = field(default=False, init=False)
    _alert_callbacks: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize the database."""
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for cost tracking."""
        with self._lock:
            if self._db_initialized:
                return

            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            # Create cost records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cost_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    input_cost REAL NOT NULL,
                    output_cost REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    cached BOOLEAN NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for efficient querying
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost_records_timestamp
                ON cost_records(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost_records_analysis_id
                ON cost_records(analysis_id)
            """)

            # Create alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS budget_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    budget_type TEXT NOT NULL,
                    current_spend REAL NOT NULL,
                    budget_limit REAL NOT NULL,
                    percentage REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    message TEXT,
                    acknowledged BOOLEAN NOT NULL DEFAULT 0
                )
            """)

            conn.commit()
            conn.close()
            self._db_initialized = True

    def record_usage(
        self,
        analysis_id: str,
        tokens: TokenUsage,
        model: str,
        provider: str = "anthropic",
        cached: bool = False,
    ) -> CostRecord:
        """Record a usage cost.

        Args:
            analysis_id: Unique identifier for the analysis.
            tokens: Token usage details.
            model: Model used.
            provider: LLM provider.
            cached: Whether this was a cache hit.

        Returns:
            The created CostRecord.
        """
        # Calculate costs (zero if cached)
        if cached:
            input_cost, output_cost = 0.0, 0.0
        else:
            input_cost, output_cost = calculate_cost(
                tokens.input_tokens, tokens.output_tokens, model, provider
            )

        record = CostRecord(
            analysis_id=analysis_id,
            model=model,
            provider=provider,
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            cached=cached,
        )

        # Persist to database
        with self._lock:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO cost_records (
                    analysis_id, timestamp, model, provider,
                    input_tokens, output_tokens, input_cost, output_cost,
                    total_cost, cached
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.analysis_id,
                    record.timestamp.isoformat(),
                    record.model,
                    record.provider,
                    record.input_tokens,
                    record.output_tokens,
                    record.input_cost,
                    record.output_cost,
                    record.total_cost,
                    record.cached,
                ),
            )
            conn.commit()
            conn.close()

        logger.debug(
            f"Recorded cost: analysis={analysis_id}, "
            f"tokens={tokens.total}, cost=${record.total_cost:.4f}"
        )

        # Check budget limits
        self._check_budgets()

        return record

    def record_cache_hit(
        self,
        analysis_id: str,
        estimated_tokens: TokenUsage,
        model: str,
        provider: str = "anthropic",
    ) -> CostRecord:
        """Record a cache hit (zero cost but track savings).

        Args:
            analysis_id: Unique identifier for the analysis.
            estimated_tokens: Estimated tokens that would have been used.
            model: Model that would have been used.
            provider: LLM provider.

        Returns:
            The created CostRecord with cached=True.
        """
        return self.record_usage(
            analysis_id=analysis_id,
            tokens=estimated_tokens,
            model=model,
            provider=provider,
            cached=True,
        )

    def get_daily_cost(self, target_date: date | None = None) -> float:
        """Get total cost for a specific day.

        Args:
            target_date: Date to get cost for. Defaults to today.

        Returns:
            Total cost for the day.
        """
        target_date = target_date or date.today()
        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date, datetime.max.time())

        with self._lock:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COALESCE(SUM(total_cost), 0)
                FROM cost_records
                WHERE timestamp >= ? AND timestamp <= ? AND cached = 0
            """,
                (start.isoformat(), end.isoformat()),
            )
            result = cursor.fetchone()[0]
            conn.close()
            return float(result)

    def get_daily_summary(self, target_date: date | None = None) -> DailyCostSummary:
        """Get detailed cost summary for a day.

        Args:
            target_date: Date to summarize. Defaults to today.

        Returns:
            DailyCostSummary with all details.
        """
        target_date = target_date or date.today()
        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date, datetime.max.time())

        with self._lock:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            # Get aggregate stats
            cursor.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN cached = 0 THEN total_cost ELSE 0 END), 0),
                    COALESCE(SUM(input_tokens + output_tokens), 0),
                    COUNT(*),
                    COALESCE(SUM(CASE WHEN cached = 1 THEN 1 ELSE 0 END), 0)
                FROM cost_records
                WHERE timestamp >= ? AND timestamp <= ?
            """,
                (start.isoformat(), end.isoformat()),
            )
            row = cursor.fetchone()
            total_cost, total_tokens, analysis_count, cache_hits = row

            # Get model breakdown
            cursor.execute(
                """
                SELECT model, COALESCE(SUM(total_cost), 0)
                FROM cost_records
                WHERE timestamp >= ? AND timestamp <= ? AND cached = 0
                GROUP BY model
            """,
                (start.isoformat(), end.isoformat()),
            )
            model_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

            # Calculate cache savings
            cursor.execute(
                """
                SELECT model, SUM(input_tokens), SUM(output_tokens)
                FROM cost_records
                WHERE timestamp >= ? AND timestamp <= ? AND cached = 1
                GROUP BY model
            """,
                (start.isoformat(), end.isoformat()),
            )
            cache_savings = 0.0
            for row in cursor.fetchall():
                model, input_tokens, output_tokens = row
                input_cost, output_cost = calculate_cost(input_tokens, output_tokens, model)
                cache_savings += input_cost + output_cost

            conn.close()

        return DailyCostSummary(
            date=target_date,
            total_cost=float(total_cost),
            total_tokens=int(total_tokens),
            analysis_count=int(analysis_count),
            cache_hits=int(cache_hits),
            cache_savings=cache_savings,
            model_breakdown=model_breakdown,
        )

    def get_monthly_report(self, target_month: date | None = None) -> MonthlyCostReport:
        """Get monthly cost report.

        Args:
            target_month: Any date in the target month. Defaults to current month.

        Returns:
            MonthlyCostReport with all monthly statistics.
        """
        target_month = target_month or date.today()
        month_start = target_month.replace(day=1)
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1)
        month_end = month_end - timedelta(days=1)

        start = datetime.combine(month_start, datetime.min.time())
        end = datetime.combine(month_end, datetime.max.time())

        with self._lock:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            # Get aggregate stats
            cursor.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN cached = 0 THEN total_cost ELSE 0 END), 0),
                    COALESCE(SUM(input_tokens + output_tokens), 0),
                    COUNT(*),
                    COALESCE(SUM(CASE WHEN cached = 1 THEN 1 ELSE 0 END), 0)
                FROM cost_records
                WHERE timestamp >= ? AND timestamp <= ?
            """,
                (start.isoformat(), end.isoformat()),
            )
            row = cursor.fetchone()
            total_cost, total_tokens, total_analyses, cache_hits = row

            # Get model breakdown
            cursor.execute(
                """
                SELECT model, COALESCE(SUM(total_cost), 0)
                FROM cost_records
                WHERE timestamp >= ? AND timestamp <= ? AND cached = 0
                GROUP BY model
            """,
                (start.isoformat(), end.isoformat()),
            )
            model_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

            # Get daily costs for the month
            cursor.execute(
                """
                SELECT DATE(timestamp), COALESCE(SUM(total_cost), 0)
                FROM cost_records
                WHERE timestamp >= ? AND timestamp <= ? AND cached = 0
                GROUP BY DATE(timestamp)
                ORDER BY DATE(timestamp)
            """,
                (start.isoformat(), end.isoformat()),
            )
            daily_costs = [row[1] for row in cursor.fetchall()]

            # Calculate cache savings
            cursor.execute(
                """
                SELECT model, SUM(input_tokens), SUM(output_tokens)
                FROM cost_records
                WHERE timestamp >= ? AND timestamp <= ? AND cached = 1
                GROUP BY model
            """,
                (start.isoformat(), end.isoformat()),
            )
            cache_savings = 0.0
            for row in cursor.fetchall():
                model, input_tokens, output_tokens = row
                input_cost, output_cost = calculate_cost(input_tokens, output_tokens, model)
                cache_savings += input_cost + output_cost

            conn.close()

        # Calculate stats
        days_with_data = len(daily_costs) if daily_costs else 1
        daily_average = total_cost / days_with_data if days_with_data > 0 else 0.0
        cache_hit_rate = cache_hits / total_analyses if total_analyses > 0 else 0.0

        # Calculate budget status
        budget_remaining = None
        budget_status = "ok"
        if self.monthly_budget:
            budget_remaining = self.monthly_budget - total_cost
            if total_cost >= self.monthly_budget:
                budget_status = "exceeded"
            elif total_cost >= self.monthly_budget * self.warning_threshold:
                budget_status = "warning"

        return MonthlyCostReport(
            month=month_start,
            total_cost=float(total_cost),
            daily_average=daily_average,
            total_tokens=int(total_tokens),
            total_analyses=int(total_analyses),
            cache_hit_rate=cache_hit_rate,
            total_cache_savings=cache_savings,
            model_breakdown=model_breakdown,
            daily_costs=daily_costs,
            budget_remaining=budget_remaining,
            budget_status=budget_status,
        )

    def is_budget_exceeded(self) -> bool:
        """Check if any budget is exceeded.

        Returns:
            True if daily or monthly budget is exceeded.
        """
        if self.daily_budget:
            daily_cost = self.get_daily_cost()
            if daily_cost >= self.daily_budget:
                return True

        if self.monthly_budget:
            report = self.get_monthly_report()
            if report.total_cost >= self.monthly_budget:
                return True

        return False

    def get_budget_status(self) -> dict[str, Any]:
        """Get current budget status.

        Returns:
            Dict with daily and monthly budget status.
        """
        status: dict[str, Any] = {
            "daily": {"used": 0.0, "limit": self.daily_budget, "status": "ok"},
            "monthly": {"used": 0.0, "limit": self.monthly_budget, "status": "ok"},
        }

        daily_cost = self.get_daily_cost()
        status["daily"]["used"] = daily_cost
        if self.daily_budget:
            pct = daily_cost / self.daily_budget
            if pct >= 1.0:
                status["daily"]["status"] = "exceeded"
            elif pct >= self.warning_threshold:
                status["daily"]["status"] = "warning"
            status["daily"]["percentage"] = pct * 100

        report = self.get_monthly_report()
        status["monthly"]["used"] = report.total_cost
        if self.monthly_budget:
            pct = report.total_cost / self.monthly_budget
            if pct >= 1.0:
                status["monthly"]["status"] = "exceeded"
            elif pct >= self.warning_threshold:
                status["monthly"]["status"] = "warning"
            status["monthly"]["percentage"] = pct * 100

        return status

    def _check_budgets(self) -> None:
        """Check budget limits and create alerts if needed."""
        # Check daily budget
        if self.daily_budget:
            daily_cost = self.get_daily_cost()
            pct = daily_cost / self.daily_budget

            if pct >= 1.0:
                self._create_alert("exceeded", "daily", daily_cost, self.daily_budget)
            elif pct >= self.warning_threshold:
                self._create_alert("warning", "daily", daily_cost, self.daily_budget)

        # Check monthly budget
        if self.monthly_budget:
            report = self.get_monthly_report()
            pct = report.total_cost / self.monthly_budget

            if pct >= 1.0:
                self._create_alert("exceeded", "monthly", report.total_cost, self.monthly_budget)
            elif pct >= self.warning_threshold:
                self._create_alert("warning", "monthly", report.total_cost, self.monthly_budget)

    def _create_alert(
        self,
        alert_type: str,
        budget_type: str,
        current_spend: float,
        budget_limit: float,
    ) -> BudgetAlert:
        """Create and store a budget alert.

        Args:
            alert_type: Type of alert (warning, exceeded).
            budget_type: Budget type (daily, monthly).
            current_spend: Current spending amount.
            budget_limit: Budget limit.

        Returns:
            The created BudgetAlert.
        """
        percentage = (current_spend / budget_limit) * 100
        message = (
            f"{budget_type.capitalize()} budget {alert_type}: "
            f"${current_spend:.2f} / ${budget_limit:.2f} ({percentage:.1f}%)"
        )

        alert = BudgetAlert(
            alert_type=alert_type,
            budget_type=budget_type,
            current_spend=current_spend,
            budget_limit=budget_limit,
            percentage=percentage,
            message=message,
        )

        # Store alert
        with self._lock:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO budget_alerts (
                    alert_type, budget_type, current_spend, budget_limit,
                    percentage, timestamp, message
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.alert_type,
                    alert.budget_type,
                    alert.current_spend,
                    alert.budget_limit,
                    alert.percentage,
                    alert.timestamp.isoformat(),
                    alert.message,
                ),
            )
            conn.commit()
            conn.close()

        logger.warning(alert.message)

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        return alert

    def get_alerts(
        self,
        unacknowledged_only: bool = True,
        limit: int = 100,
    ) -> list[BudgetAlert]:
        """Get budget alerts.

        Args:
            unacknowledged_only: Only return unacknowledged alerts.
            limit: Maximum number of alerts to return.

        Returns:
            List of BudgetAlert objects.
        """
        with self._lock:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            query = """
                SELECT alert_type, budget_type, current_spend, budget_limit,
                       percentage, timestamp, message
                FROM budget_alerts
            """
            if unacknowledged_only:
                query += " WHERE acknowledged = 0"
            query += " ORDER BY timestamp DESC LIMIT ?"

            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            conn.close()

        return [
            BudgetAlert(
                alert_type=row[0],
                budget_type=row[1],
                current_spend=row[2],
                budget_limit=row[3],
                percentage=row[4],
                timestamp=datetime.fromisoformat(row[5]),
                message=row[6],
            )
            for row in rows
        ]

    def acknowledge_alerts(self) -> int:
        """Acknowledge all unacknowledged alerts.

        Returns:
            Number of alerts acknowledged.
        """
        with self._lock:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE budget_alerts SET acknowledged = 1 WHERE acknowledged = 0")
            count = cursor.rowcount
            conn.commit()
            conn.close()
            return count

    def on_alert(self, callback: Any) -> None:
        """Register a callback for budget alerts.

        Args:
            callback: Function to call when an alert is created.
                     Receives a BudgetAlert as argument.
        """
        self._alert_callbacks.append(callback)

    def export_json(self, output_path: Path) -> int:
        """Export all cost records to JSON.

        Args:
            output_path: Path to output file.

        Returns:
            Number of records exported.
        """
        with self._lock:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT analysis_id, timestamp, model, provider,
                       input_tokens, output_tokens, input_cost, output_cost,
                       total_cost, cached
                FROM cost_records
                ORDER BY timestamp
            """)
            rows = cursor.fetchall()
            conn.close()

        records = [
            {
                "analysis_id": row[0],
                "timestamp": row[1],
                "model": row[2],
                "provider": row[3],
                "input_tokens": row[4],
                "output_tokens": row[5],
                "input_cost": row[6],
                "output_cost": row[7],
                "total_cost": row[8],
                "cached": bool(row[9]),
            }
            for row in rows
        ]

        with output_path.open("w") as f:
            json.dump(records, f, indent=2)

        return len(records)

    def clear(self) -> int:
        """Clear all cost records.

        Returns:
            Number of records cleared.
        """
        with self._lock:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cost_records")
            count: int = cursor.fetchone()[0]
            cursor.execute("DELETE FROM cost_records")
            cursor.execute("DELETE FROM budget_alerts")
            conn.commit()
            conn.close()
            return count


# ============================================================================
# Module-level convenience functions
# ============================================================================

_default_tracker: CostTracker | None = None
_tracker_lock = threading.Lock()


def get_cost_tracker(
    storage_path: Path | None = None,
    daily_budget: float | None = None,
    monthly_budget: float | None = None,
) -> CostTracker:
    """Get or create the default cost tracker.

    Args:
        storage_path: Path to storage database.
        daily_budget: Daily budget limit.
        monthly_budget: Monthly budget limit.

    Returns:
        The CostTracker instance.
    """
    global _default_tracker

    with _tracker_lock:
        if _default_tracker is None:
            _default_tracker = CostTracker(
                storage_path=storage_path or Path("cost_tracking.db"),
                daily_budget=daily_budget,
                monthly_budget=monthly_budget,
            )
        return _default_tracker


def reset_cost_tracker() -> None:
    """Reset the default cost tracker."""
    global _default_tracker
    with _tracker_lock:
        _default_tracker = None
