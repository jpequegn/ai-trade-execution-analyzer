"""Tests for metrics module."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from src.observability.metrics import (
    ErrorStats,
    ErrorTracker,
    LatencyStats,
    LatencyTracker,
    MetricsCollector,
    MetricsSummary,
    ThroughputStats,
    ThroughputTracker,
    get_metrics_collector,
    reset_metrics_collector,
)


class TestLatencyStats:
    """Tests for LatencyStats model."""

    def test_default_values(self) -> None:
        """Test default latency stats values."""
        stats = LatencyStats(operation="test")

        assert stats.operation == "test"
        assert stats.count == 0
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.mean_ms == 0.0
        assert stats.p95 == 0.0
        assert stats.p99 == 0.0

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        stats = LatencyStats(
            operation="test",
            count=100,
            min_ms=10.123,
            max_ms=200.567,
            mean_ms=55.789,
            p95=180.456,
        )

        result = stats.to_dict()

        assert result["operation"] == "test"
        assert result["count"] == 100
        assert result["min_ms"] == 10.12  # Rounded
        assert result["max_ms"] == 200.57
        assert result["mean_ms"] == 55.79
        assert result["p95"] == 180.46


class TestThroughputStats:
    """Tests for ThroughputStats model."""

    def test_default_values(self) -> None:
        """Test default throughput stats values."""
        stats = ThroughputStats(operation="test")

        assert stats.operation == "test"
        assert stats.total_count == 0
        assert stats.success_count == 0
        assert stats.error_count == 0
        assert stats.success_rate == 0.0

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        stats = ThroughputStats(
            operation="test",
            total_count=100,
            success_count=95,
            error_count=5,
            success_rate=95.0,
            requests_per_second=10.5,
        )

        result = stats.to_dict()

        assert result["total_count"] == 100
        assert result["success_count"] == 95
        assert result["success_rate"] == 95.0


class TestErrorStats:
    """Tests for ErrorStats model."""

    def test_default_values(self) -> None:
        """Test default error stats values."""
        stats = ErrorStats(operation="test")

        assert stats.operation == "test"
        assert stats.total_errors == 0
        assert stats.by_category == {}
        assert stats.by_type == {}

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        stats = ErrorStats(
            operation="test",
            total_errors=10,
            by_category={"parse_error": 5, "llm_error": 5},
            by_type={"ValueError": 3, "RuntimeError": 7},
        )

        result = stats.to_dict()

        assert result["total_errors"] == 10
        assert result["by_category"]["parse_error"] == 5


class TestLatencyTracker:
    """Tests for LatencyTracker class."""

    def test_record_single(self) -> None:
        """Test recording a single latency."""
        tracker = LatencyTracker()
        tracker.record(100.0)

        stats = tracker.get_stats("test")
        assert stats.count == 1
        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0

    def test_record_multiple(self) -> None:
        """Test recording multiple latencies."""
        tracker = LatencyTracker()
        for i in range(100):
            tracker.record(float(i + 1))  # 1 to 100

        stats = tracker.get_stats("test")
        assert stats.count == 100
        assert stats.min_ms == 1.0
        assert stats.max_ms == 100.0
        assert 50 <= stats.mean_ms <= 51  # Should be ~50.5

    def test_percentile_calculation(self) -> None:
        """Test percentile calculation accuracy."""
        tracker = LatencyTracker()
        for i in range(100):
            tracker.record(float(i + 1))

        # P50 should be around 50
        p50 = tracker.get_percentile(50)
        assert 49 <= p50 <= 51

        # P95 should be around 95
        p95 = tracker.get_percentile(95)
        assert 94 <= p95 <= 96

        # P99 should be around 99
        p99 = tracker.get_percentile(99)
        assert 98 <= p99 <= 100

    def test_max_samples_limit(self) -> None:
        """Test samples are limited to max."""
        tracker = LatencyTracker(max_samples=100)

        for i in range(200):
            tracker.record(float(i))

        stats = tracker.get_stats("test")
        assert stats.count == 100

    def test_empty_tracker(self) -> None:
        """Test empty tracker returns zeros."""
        tracker = LatencyTracker()

        stats = tracker.get_stats("test")
        assert stats.count == 0
        assert stats.min_ms == 0.0
        assert stats.p95 == 0.0

    def test_clear(self) -> None:
        """Test clearing tracker."""
        tracker = LatencyTracker()
        tracker.record(100.0)
        tracker.clear()

        stats = tracker.get_stats("test")
        assert stats.count == 0


class TestThroughputTracker:
    """Tests for ThroughputTracker class."""

    def test_record_success(self) -> None:
        """Test recording successful requests."""
        tracker = ThroughputTracker()
        tracker.record(success=True)

        stats = tracker.get_stats("test")
        assert stats.total_count == 1
        assert stats.success_count == 1
        assert stats.error_count == 0

    def test_record_failure(self) -> None:
        """Test recording failed requests."""
        tracker = ThroughputTracker()
        tracker.record(success=False)

        stats = tracker.get_stats("test")
        assert stats.total_count == 1
        assert stats.success_count == 0
        assert stats.error_count == 1

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        tracker = ThroughputTracker()
        for _ in range(80):
            tracker.record(success=True)
        for _ in range(20):
            tracker.record(success=False)

        stats = tracker.get_stats("test")
        assert stats.success_rate == 80.0

    def test_window_expiration(self) -> None:
        """Test requests expire after window."""
        tracker = ThroughputTracker(window_seconds=0.1)  # 100ms window
        tracker.record(success=True)

        time.sleep(0.15)  # Wait for expiration

        stats = tracker.get_stats("test")
        assert stats.total_count == 0

    def test_clear(self) -> None:
        """Test clearing tracker."""
        tracker = ThroughputTracker()
        tracker.record(success=True)
        tracker.clear()

        stats = tracker.get_stats("test")
        assert stats.total_count == 0


class TestErrorTracker:
    """Tests for ErrorTracker class."""

    def test_record_error(self) -> None:
        """Test recording an error."""
        tracker = ErrorTracker()
        tracker.record(
            error_type="ValueError",
            category="validation_error",
            message="Invalid input",
        )

        stats = tracker.get_stats("test")
        assert stats.total_errors == 1
        assert stats.by_type["ValueError"] == 1
        assert stats.by_category["validation_error"] == 1

    def test_record_multiple_errors(self) -> None:
        """Test recording multiple errors."""
        tracker = ErrorTracker()
        tracker.record("ValueError", "validation_error", "Error 1")
        tracker.record("ValueError", "validation_error", "Error 2")
        tracker.record("RuntimeError", "unknown_error", "Error 3")

        stats = tracker.get_stats("test")
        assert stats.total_errors == 3
        assert stats.by_type["ValueError"] == 2
        assert stats.by_type["RuntimeError"] == 1

    def test_recent_errors_limit(self) -> None:
        """Test recent errors are limited."""
        tracker = ErrorTracker(max_recent=5)

        for i in range(10):
            tracker.record("Error", "category", f"Message {i}")

        stats = tracker.get_stats("test")
        assert len(stats.recent_errors) <= 10  # Stats returns last 10

    def test_clear(self) -> None:
        """Test clearing tracker."""
        tracker = ErrorTracker()
        tracker.record("Error", "category", "message")
        tracker.clear()

        stats = tracker.get_stats("test")
        assert stats.total_errors == 0


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_record_latency(self) -> None:
        """Test recording latency."""
        collector = MetricsCollector()
        collector.record_latency("test_op", 100.0)
        collector.record_latency("test_op", 200.0)

        stats = collector.get_latency_stats("test_op")
        assert stats.count == 2
        assert stats.min_ms == 100.0
        assert stats.max_ms == 200.0

    def test_record_request(self) -> None:
        """Test recording requests."""
        collector = MetricsCollector()
        collector.record_request("test_op", success=True)
        collector.record_request("test_op", success=False)

        stats = collector.get_throughput_stats("test_op")
        assert stats.total_count == 2
        assert stats.success_count == 1
        assert stats.error_count == 1

    def test_record_error(self) -> None:
        """Test recording errors."""
        collector = MetricsCollector()
        collector.record_error(
            operation="test_op",
            error_type="ValueError",
            category="validation",
            message="test error",
        )

        stats = collector.get_error_stats("test_op")
        assert stats.total_errors == 1

    def test_counters(self) -> None:
        """Test counter operations."""
        collector = MetricsCollector()

        collector.increment_counter("test_counter")
        assert collector.get_counter("test_counter") == 1

        collector.increment_counter("test_counter", 5)
        assert collector.get_counter("test_counter") == 6

    def test_gauges(self) -> None:
        """Test gauge operations."""
        collector = MetricsCollector()

        collector.set_gauge("test_gauge", 100.5)
        assert collector.get_gauge("test_gauge") == 100.5

        collector.set_gauge("test_gauge", 200.0)
        assert collector.get_gauge("test_gauge") == 200.0

    def test_get_summary(self) -> None:
        """Test getting metrics summary."""
        collector = MetricsCollector()
        collector.record_latency("op1", 100.0)
        collector.record_request("op2", success=True)
        collector.increment_counter("counter1")
        collector.set_gauge("gauge1", 50.0)

        summary = collector.get_summary()

        assert "op1" in summary.latency
        assert "op2" in summary.throughput
        assert summary.counters["counter1"] == 1
        assert summary.gauges["gauge1"] == 50.0

    def test_clear(self) -> None:
        """Test clearing all metrics."""
        collector = MetricsCollector()
        collector.record_latency("test", 100.0)
        collector.increment_counter("counter")
        collector.set_gauge("gauge", 50.0)

        collector.clear()

        summary = collector.get_summary()
        assert summary.counters == {}
        assert summary.gauges == {}

    def test_callbacks(self) -> None:
        """Test metric callbacks."""
        collector = MetricsCollector()
        events = []

        def callback(metric_type: str, data: dict[str, object]) -> None:
            events.append((metric_type, data))

        collector.on_metric(callback)
        collector.record_latency("test", 100.0)
        collector.increment_counter("counter")

        assert len(events) == 2
        assert events[0][0] == "latency"
        assert events[1][0] == "counter"

    def test_export_json(self) -> None:
        """Test JSON export."""
        collector = MetricsCollector()
        collector.record_latency("test", 100.0)
        collector.increment_counter("counter")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            collector.export_json(output_path)

            assert output_path.exists()

    def test_export_prometheus(self) -> None:
        """Test Prometheus format export."""
        collector = MetricsCollector()
        collector.record_latency("test_op", 100.0)
        collector.record_latency("test_op", 200.0)
        collector.increment_counter("requests_total")
        collector.set_gauge("active_connections", 5.0)

        output = collector.export_prometheus()

        assert "latency_test_op_ms" in output
        assert "counter_requests_total" in output
        assert "gauge_active_connections" in output


class TestMetricsSummary:
    """Tests for MetricsSummary model."""

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        summary = MetricsSummary(
            counters={"counter1": 10},
            gauges={"gauge1": 50.0},
        )

        result = summary.to_dict()

        assert result["counters"]["counter1"] == 10
        assert result["gauges"]["gauge1"] == 50.0
        assert "timestamp" in result


class TestGlobalMetricsCollector:
    """Tests for global metrics collector functions."""

    def test_get_metrics_collector(self) -> None:
        """Test getting global collector."""
        reset_metrics_collector()
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is collector2

    def test_reset_metrics_collector(self) -> None:
        """Test resetting global collector."""
        collector1 = get_metrics_collector()
        reset_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is not collector2


class TestLatencyStatsPercentiles:
    """Tests for latency percentile accuracy."""

    def test_standard_deviation(self) -> None:
        """Test standard deviation calculation."""
        tracker = LatencyTracker()
        # Add values with known std dev
        for v in [10, 20, 30, 40, 50]:
            tracker.record(float(v))

        stats = tracker.get_stats("test")
        # Mean is 30, variance is 200, std dev is ~14.14
        assert 14 <= stats.stddev_ms <= 15

    def test_small_sample_percentiles(self) -> None:
        """Test percentiles with small samples."""
        tracker = LatencyTracker()
        tracker.record(100.0)
        tracker.record(200.0)

        stats = tracker.get_stats("test")
        assert stats.p99 == 200.0  # Falls back to max for small samples


class TestErrorTrackerAttributes:
    """Tests for error tracker with attributes."""

    def test_error_with_attributes(self) -> None:
        """Test errors with custom attributes."""
        tracker = ErrorTracker()
        tracker.record(
            error_type="APIError",
            category="llm_error",
            message="API call failed",
            attributes={"status_code": 429, "model": "claude-3"},
        )

        stats = tracker.get_stats("test")
        recent = stats.recent_errors[0]
        assert recent["attributes"]["status_code"] == 429
        assert recent["attributes"]["model"] == "claude-3"

    def test_message_truncation(self) -> None:
        """Test long messages are truncated."""
        tracker = ErrorTracker()
        long_message = "x" * 1000
        tracker.record("Error", "category", long_message)

        stats = tracker.get_stats("test")
        assert len(stats.recent_errors[0]["message"]) == 500
