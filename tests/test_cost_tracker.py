"""Tests for the cost tracking module."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from src.observability.cost_tracker import (
    BudgetAlert,
    CostRecord,
    CostTracker,
    DailyCostSummary,
    MonthlyCostReport,
    TokenUsage,
    calculate_cost,
    get_cost_tracker,
    reset_cost_tracker,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db() -> Iterator[Path]:
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_costs.db"


@pytest.fixture
def tracker(temp_db: Path) -> CostTracker:
    """Create a cost tracker for testing."""
    return CostTracker(storage_path=temp_db)


@pytest.fixture
def tracker_with_budget(temp_db: Path) -> CostTracker:
    """Create a cost tracker with budget limits."""
    return CostTracker(
        storage_path=temp_db,
        daily_budget=10.0,
        monthly_budget=100.0,
    )


@pytest.fixture(autouse=True)
def reset_tracker() -> Iterator[None]:
    """Reset the default tracker after each test."""
    yield
    reset_cost_tracker()


# ============================================================================
# TokenUsage Tests
# ============================================================================


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_total_calculation(self) -> None:
        """Test total token calculation."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        assert usage.total == 1500

    def test_zero_tokens(self) -> None:
        """Test with zero tokens."""
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        assert usage.total == 0


# ============================================================================
# Cost Calculation Tests
# ============================================================================


class TestCostCalculation:
    """Tests for cost calculation functions."""

    def test_anthropic_sonnet_pricing(self) -> None:
        """Test pricing for Claude Sonnet."""
        input_cost, output_cost = calculate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="claude-3-sonnet-20240229",
            provider="anthropic",
        )
        assert input_cost == 3.00  # $3/1M input tokens
        assert output_cost == 15.00  # $15/1M output tokens

    def test_anthropic_opus_pricing(self) -> None:
        """Test pricing for Claude Opus."""
        input_cost, output_cost = calculate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="claude-3-opus-20240229",
            provider="anthropic",
        )
        assert input_cost == 15.00
        assert output_cost == 75.00

    def test_openai_gpt4_pricing(self) -> None:
        """Test pricing for GPT-4."""
        input_cost, output_cost = calculate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="gpt-4-turbo",
            provider="openai",
        )
        assert input_cost == 10.00
        assert output_cost == 30.00

    def test_unknown_model_uses_default(self) -> None:
        """Test that unknown models use default pricing."""
        input_cost, output_cost = calculate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="unknown-model",
            provider="anthropic",
        )
        # Should use Anthropic default pricing
        assert input_cost == 3.00
        assert output_cost == 15.00

    def test_small_token_count(self) -> None:
        """Test with small token counts."""
        input_cost, output_cost = calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="claude-3-sonnet-20240229",
            provider="anthropic",
        )
        assert input_cost == pytest.approx(0.003, rel=0.01)
        assert output_cost == pytest.approx(0.0075, rel=0.01)


# ============================================================================
# CostRecord Tests
# ============================================================================


class TestCostRecord:
    """Tests for CostRecord model."""

    def test_cost_record_creation(self) -> None:
        """Test creating a cost record."""
        record = CostRecord(
            analysis_id="test_123",
            model="claude-3-sonnet-20240229",
            provider="anthropic",
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.003,
            output_cost=0.0075,
            total_cost=0.0105,
        )
        assert record.total_tokens == 1500
        assert record.cached is False

    def test_cached_cost_record(self) -> None:
        """Test creating a cached cost record."""
        record = CostRecord(
            analysis_id="test_123",
            model="claude-3-sonnet-20240229",
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            cached=True,
        )
        assert record.total_cost == 0.0
        assert record.cached is True


# ============================================================================
# CostTracker Tests
# ============================================================================


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_record_usage(self, tracker: CostTracker) -> None:
        """Test recording usage."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        record = tracker.record_usage(
            analysis_id="test_123",
            tokens=usage,
            model="claude-3-sonnet-20240229",
        )

        assert record.analysis_id == "test_123"
        assert record.total_tokens == 1500
        assert record.total_cost > 0

    def test_record_cache_hit(self, tracker: CostTracker) -> None:
        """Test recording a cache hit."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        record = tracker.record_cache_hit(
            analysis_id="test_123",
            estimated_tokens=usage,
            model="claude-3-sonnet-20240229",
        )

        assert record.cached is True
        assert record.total_cost == 0.0

    def test_get_daily_cost(self, tracker: CostTracker) -> None:
        """Test getting daily cost."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker.record_usage(
            analysis_id="test_1",
            tokens=usage,
            model="claude-3-sonnet-20240229",
        )
        tracker.record_usage(
            analysis_id="test_2",
            tokens=usage,
            model="claude-3-sonnet-20240229",
        )

        daily_cost = tracker.get_daily_cost()
        assert daily_cost > 0

    def test_get_daily_summary(self, tracker: CostTracker) -> None:
        """Test getting daily summary."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker.record_usage(
            analysis_id="test_1",
            tokens=usage,
            model="claude-3-sonnet-20240229",
        )
        tracker.record_cache_hit(
            analysis_id="test_2",
            estimated_tokens=usage,
            model="claude-3-sonnet-20240229",
        )

        summary = tracker.get_daily_summary()
        assert isinstance(summary, DailyCostSummary)
        assert summary.analysis_count == 2
        assert summary.cache_hits == 1
        assert summary.total_cost > 0

    def test_get_monthly_report(self, tracker: CostTracker) -> None:
        """Test getting monthly report."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker.record_usage(
            analysis_id="test_1",
            tokens=usage,
            model="claude-3-sonnet-20240229",
        )

        report = tracker.get_monthly_report()
        assert isinstance(report, MonthlyCostReport)
        assert report.total_analyses == 1
        assert report.total_cost > 0

    def test_export_json(self, tracker: CostTracker, temp_db: Path) -> None:
        """Test exporting costs to JSON."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker.record_usage(
            analysis_id="test_1",
            tokens=usage,
            model="claude-3-sonnet-20240229",
        )

        export_path = temp_db.parent / "export.json"
        count = tracker.export_json(export_path)

        assert count == 1
        assert export_path.exists()

    def test_clear(self, tracker: CostTracker) -> None:
        """Test clearing cost records."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker.record_usage(
            analysis_id="test_1",
            tokens=usage,
            model="claude-3-sonnet-20240229",
        )
        tracker.record_usage(
            analysis_id="test_2",
            tokens=usage,
            model="claude-3-sonnet-20240229",
        )

        cleared = tracker.clear()
        assert cleared == 2
        assert tracker.get_daily_cost() == 0


# ============================================================================
# Budget Tests
# ============================================================================


class TestBudgetManagement:
    """Tests for budget management."""

    def test_budget_not_exceeded(self, tracker_with_budget: CostTracker) -> None:
        """Test budget check when not exceeded."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        tracker_with_budget.record_usage(
            analysis_id="test_1",
            tokens=usage,
            model="claude-3-sonnet-20240229",
        )

        assert not tracker_with_budget.is_budget_exceeded()

    def test_get_budget_status(self, tracker_with_budget: CostTracker) -> None:
        """Test getting budget status."""
        status = tracker_with_budget.get_budget_status()

        assert "daily" in status
        assert "monthly" in status
        assert status["daily"]["limit"] == 10.0
        assert status["monthly"]["limit"] == 100.0

    def test_budget_warning_alert(self, temp_db: Path) -> None:
        """Test budget warning alert is created."""
        alerts_received: list[BudgetAlert] = []

        def capture_alert(alert: BudgetAlert) -> None:
            alerts_received.append(alert)

        # Create tracker with low budget
        tracker = CostTracker(
            storage_path=temp_db,
            daily_budget=0.01,  # Very low budget
            warning_threshold=0.5,
        )
        tracker.on_alert(capture_alert)

        # Record usage that exceeds warning threshold
        usage = TokenUsage(input_tokens=10000, output_tokens=5000)
        tracker.record_usage(
            analysis_id="test_1",
            tokens=usage,
            model="claude-3-sonnet-20240229",
        )

        # Should have triggered a warning or exceeded alert
        assert len(alerts_received) > 0

    def test_acknowledge_alerts(self, tracker_with_budget: CostTracker) -> None:
        """Test acknowledging alerts."""
        # Record that may trigger alerts
        count = tracker_with_budget.acknowledge_alerts()
        # Should work even with no alerts
        assert count >= 0


# ============================================================================
# Module-level Function Tests
# ============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_cost_tracker_singleton(self, temp_db: Path) -> None:
        """Test that get_cost_tracker returns the same instance."""
        reset_cost_tracker()
        tracker1 = get_cost_tracker(storage_path=temp_db)
        tracker2 = get_cost_tracker()
        assert tracker1 is tracker2

    def test_reset_cost_tracker(self, temp_db: Path) -> None:
        """Test resetting the cost tracker."""
        tracker1 = get_cost_tracker(storage_path=temp_db)
        reset_cost_tracker()
        tracker2 = get_cost_tracker(storage_path=temp_db)
        # After reset, should be a new instance
        assert tracker1 is not tracker2


# ============================================================================
# DailyCostSummary Tests
# ============================================================================


class TestDailyCostSummary:
    """Tests for DailyCostSummary model."""

    def test_summary_creation(self) -> None:
        """Test creating a daily summary."""
        summary = DailyCostSummary(
            date=date.today(),
            total_cost=10.50,
            total_tokens=100000,
            analysis_count=50,
            cache_hits=10,
            cache_savings=2.50,
            model_breakdown={"claude-3-sonnet": 8.00, "claude-3-haiku": 2.50},
        )
        assert summary.total_cost == 10.50
        assert summary.analysis_count == 50


# ============================================================================
# MonthlyCostReport Tests
# ============================================================================


class TestMonthlyCostReport:
    """Tests for MonthlyCostReport model."""

    def test_report_creation(self) -> None:
        """Test creating a monthly report."""
        report = MonthlyCostReport(
            month=date(2024, 1, 1),
            total_cost=100.00,
            daily_average=3.33,
            total_tokens=1000000,
            total_analyses=500,
            cache_hit_rate=0.2,
            total_cache_savings=25.00,
        )
        assert report.total_cost == 100.00
        assert report.budget_status == "ok"

    def test_report_with_budget_exceeded(self) -> None:
        """Test report with exceeded budget."""
        report = MonthlyCostReport(
            month=date(2024, 1, 1),
            total_cost=150.00,
            total_analyses=500,
            budget_remaining=-50.00,
            budget_status="exceeded",
        )
        assert report.budget_status == "exceeded"
        assert report.budget_remaining == -50.00
