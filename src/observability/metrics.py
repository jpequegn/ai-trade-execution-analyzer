"""Metrics collection and export for observability.

This module provides latency tracking with percentile calculations,
throughput metrics, and export functionality for external monitoring systems.

Example:
    >>> from src.observability.metrics import MetricsCollector, get_metrics_collector
    >>> collector = get_metrics_collector()
    >>> collector.record_latency("llm_call", 150.5)
    >>> collector.record_latency("llm_call", 200.3)
    >>> stats = collector.get_latency_stats("llm_call")
    >>> print(f"P95: {stats.p95}ms")
"""

from __future__ import annotations

import bisect
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class LatencyStats(BaseModel):
    """Statistics for latency measurements.

    Attributes:
        operation: Name of the operation.
        count: Number of measurements.
        min_ms: Minimum latency in milliseconds.
        max_ms: Maximum latency in milliseconds.
        mean_ms: Mean latency in milliseconds.
        median_ms: Median (P50) latency.
        p75: 75th percentile latency.
        p90: 90th percentile latency.
        p95: 95th percentile latency.
        p99: 99th percentile latency.
        stddev_ms: Standard deviation.
        last_updated: When stats were last updated.
    """

    operation: str
    count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    stddev_ms: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "count": self.count,
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p75": round(self.p75, 2),
            "p90": round(self.p90, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
            "stddev_ms": round(self.stddev_ms, 2),
            "last_updated": self.last_updated.isoformat(),
        }


class ThroughputStats(BaseModel):
    """Statistics for throughput measurements.

    Attributes:
        operation: Name of the operation.
        total_count: Total operations recorded.
        success_count: Successful operations.
        error_count: Failed operations.
        success_rate: Success rate as percentage.
        requests_per_second: Average throughput.
        window_seconds: Measurement window duration.
        last_updated: When stats were last updated.
    """

    operation: str
    total_count: int = 0
    success_count: int = 0
    error_count: int = 0
    success_rate: float = 0.0
    requests_per_second: float = 0.0
    window_seconds: float = 60.0
    last_updated: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "total_count": self.total_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": round(self.success_rate, 2),
            "requests_per_second": round(self.requests_per_second, 4),
            "window_seconds": self.window_seconds,
            "last_updated": self.last_updated.isoformat(),
        }


class ErrorStats(BaseModel):
    """Statistics for error tracking.

    Attributes:
        operation: Name of the operation.
        total_errors: Total error count.
        by_category: Errors grouped by category.
        by_type: Errors grouped by exception type.
        recent_errors: Most recent error messages.
        last_error_time: When the last error occurred.
    """

    operation: str
    total_errors: int = 0
    by_category: dict[str, int] = Field(default_factory=dict)
    by_type: dict[str, int] = Field(default_factory=dict)
    recent_errors: list[dict[str, Any]] = Field(default_factory=list)
    last_error_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "total_errors": self.total_errors,
            "by_category": self.by_category,
            "by_type": self.by_type,
            "recent_errors": self.recent_errors[-10:],  # Last 10
            "last_error_time": (self.last_error_time.isoformat() if self.last_error_time else None),
        }


class MetricsSummary(BaseModel):
    """Complete metrics summary.

    Attributes:
        timestamp: When the summary was generated.
        latency: Latency statistics by operation.
        throughput: Throughput statistics by operation.
        errors: Error statistics by operation.
        counters: Custom counter values.
        gauges: Custom gauge values.
    """

    timestamp: datetime = Field(default_factory=datetime.now)
    latency: dict[str, LatencyStats] = Field(default_factory=dict)
    throughput: dict[str, ThroughputStats] = Field(default_factory=dict)
    errors: dict[str, ErrorStats] = Field(default_factory=dict)
    counters: dict[str, int] = Field(default_factory=dict)
    gauges: dict[str, float] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "latency": {k: v.to_dict() for k, v in self.latency.items()},
            "throughput": {k: v.to_dict() for k, v in self.throughput.items()},
            "errors": {k: v.to_dict() for k, v in self.errors.items()},
            "counters": self.counters,
            "gauges": self.gauges,
        }


@dataclass
class LatencyTracker:
    """Tracks latency measurements with percentile calculation.

    Uses a sorted list approach for accurate percentile calculation.
    Maintains a configurable window of recent measurements.

    Attributes:
        max_samples: Maximum number of samples to retain.
    """

    max_samples: int = 10000
    _samples: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            latency_ms: Latency in milliseconds.
        """
        with self._lock:
            # Insert in sorted order
            bisect.insort(self._samples, latency_ms)

            # Trim if over max
            if len(self._samples) > self.max_samples:
                # Remove oldest samples (we keep most recent by value distribution)
                self._samples = self._samples[-self.max_samples :]

    def get_percentile(self, percentile: float) -> float:
        """Get a specific percentile value.

        Args:
            percentile: Percentile to calculate (0-100).

        Returns:
            The percentile value, or 0.0 if no samples.
        """
        with self._lock:
            if not self._samples:
                return 0.0

            idx = int((percentile / 100.0) * len(self._samples))
            idx = min(idx, len(self._samples) - 1)
            return self._samples[idx]

    def get_stats(self, operation: str) -> LatencyStats:
        """Get complete statistics.

        Args:
            operation: Name of the operation.

        Returns:
            LatencyStats with all calculated values.
        """
        with self._lock:
            if not self._samples:
                return LatencyStats(operation=operation)

            n = len(self._samples)
            total = sum(self._samples)
            mean = total / n

            # Calculate standard deviation
            variance = sum((x - mean) ** 2 for x in self._samples) / n
            stddev = variance**0.5

            return LatencyStats(
                operation=operation,
                count=n,
                min_ms=self._samples[0],
                max_ms=self._samples[-1],
                mean_ms=mean,
                median_ms=self._samples[n // 2],
                p75=self._samples[int(n * 0.75)],
                p90=self._samples[int(n * 0.90)],
                p95=self._samples[int(n * 0.95)],
                p99=self._samples[int(n * 0.99)] if n >= 100 else self._samples[-1],
                stddev_ms=stddev,
            )

    def clear(self) -> None:
        """Clear all samples."""
        with self._lock:
            self._samples.clear()


@dataclass
class ThroughputTracker:
    """Tracks request throughput with time-based windowing.

    Attributes:
        window_seconds: Size of the measurement window.
    """

    window_seconds: float = 60.0
    _timestamps: list[tuple[float, bool]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, success: bool = True) -> None:
        """Record a request.

        Args:
            success: Whether the request was successful.
        """
        with self._lock:
            now = time.time()
            self._timestamps.append((now, success))
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove timestamps outside the window."""
        cutoff = time.time() - self.window_seconds
        self._timestamps = [(t, s) for t, s in self._timestamps if t >= cutoff]

    def get_stats(self, operation: str) -> ThroughputStats:
        """Get throughput statistics.

        Args:
            operation: Name of the operation.

        Returns:
            ThroughputStats for the operation.
        """
        with self._lock:
            self._cleanup()

            if not self._timestamps:
                return ThroughputStats(
                    operation=operation,
                    window_seconds=self.window_seconds,
                )

            success_count = sum(1 for _, s in self._timestamps if s)
            error_count = len(self._timestamps) - success_count
            total = len(self._timestamps)

            # Calculate RPS based on actual time span
            if len(self._timestamps) > 1:
                time_span = self._timestamps[-1][0] - self._timestamps[0][0]
                rps = total / time_span if time_span > 0 else 0.0
            else:
                rps = 0.0

            return ThroughputStats(
                operation=operation,
                total_count=total,
                success_count=success_count,
                error_count=error_count,
                success_rate=(success_count / total * 100) if total > 0 else 0.0,
                requests_per_second=rps,
                window_seconds=self.window_seconds,
            )

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._timestamps.clear()


@dataclass
class ErrorTracker:
    """Tracks errors with categorization.

    Attributes:
        max_recent: Maximum recent errors to retain.
    """

    max_recent: int = 100
    _errors: list[dict[str, Any]] = field(default_factory=list)
    _by_category: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self,
        error_type: str,
        category: str,
        message: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record an error.

        Args:
            error_type: Type/class of the error.
            category: Error category.
            message: Error message.
            attributes: Additional attributes.
        """
        with self._lock:
            self._by_category[category] += 1
            self._by_type[error_type] += 1

            error_record = {
                "type": error_type,
                "category": category,
                "message": message[:500],  # Truncate long messages
                "timestamp": datetime.now().isoformat(),
                "attributes": attributes or {},
            }
            self._errors.append(error_record)

            # Trim to max
            if len(self._errors) > self.max_recent:
                self._errors = self._errors[-self.max_recent :]

    def get_stats(self, operation: str) -> ErrorStats:
        """Get error statistics.

        Args:
            operation: Name of the operation.

        Returns:
            ErrorStats for the operation.
        """
        with self._lock:
            total = sum(self._by_category.values())
            last_error = self._errors[-1] if self._errors else None

            return ErrorStats(
                operation=operation,
                total_errors=total,
                by_category=dict(self._by_category),
                by_type=dict(self._by_type),
                recent_errors=self._errors[-10:],
                last_error_time=(
                    datetime.fromisoformat(last_error["timestamp"]) if last_error else None
                ),
            )

    def clear(self) -> None:
        """Clear all error records."""
        with self._lock:
            self._errors.clear()
            self._by_category.clear()
            self._by_type.clear()


@dataclass
class MetricsCollector:
    """Central collector for all metrics.

    Provides methods for recording latency, throughput, errors,
    and custom counters/gauges. Supports export to various formats.

    Attributes:
        latency_max_samples: Max samples for latency tracking.
        throughput_window: Window for throughput calculation.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_latency("api_call", 150.0)
        >>> collector.increment_counter("requests_total")
        >>> summary = collector.get_summary()
    """

    latency_max_samples: int = 10000
    throughput_window: float = 60.0

    _latency_trackers: dict[str, LatencyTracker] = field(default_factory=dict)
    _throughput_trackers: dict[str, ThroughputTracker] = field(default_factory=dict)
    _error_trackers: dict[str, ErrorTracker] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _gauges: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _callbacks: list[Callable[[str, Any], None]] = field(default_factory=list)

    def _get_latency_tracker(self, operation: str) -> LatencyTracker:
        """Get or create a latency tracker for an operation."""
        with self._lock:
            if operation not in self._latency_trackers:
                self._latency_trackers[operation] = LatencyTracker(
                    max_samples=self.latency_max_samples
                )
            return self._latency_trackers[operation]

    def _get_throughput_tracker(self, operation: str) -> ThroughputTracker:
        """Get or create a throughput tracker for an operation."""
        with self._lock:
            if operation not in self._throughput_trackers:
                self._throughput_trackers[operation] = ThroughputTracker(
                    window_seconds=self.throughput_window
                )
            return self._throughput_trackers[operation]

    def _get_error_tracker(self, operation: str) -> ErrorTracker:
        """Get or create an error tracker for an operation."""
        with self._lock:
            if operation not in self._error_trackers:
                self._error_trackers[operation] = ErrorTracker()
            return self._error_trackers[operation]

    def record_latency(self, operation: str, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            operation: Name of the operation.
            latency_ms: Latency in milliseconds.
        """
        tracker = self._get_latency_tracker(operation)
        tracker.record(latency_ms)
        self._notify("latency", {"operation": operation, "latency_ms": latency_ms})

    def record_request(self, operation: str, success: bool = True) -> None:
        """Record a request for throughput tracking.

        Args:
            operation: Name of the operation.
            success: Whether the request was successful.
        """
        tracker = self._get_throughput_tracker(operation)
        tracker.record(success)
        self._notify("request", {"operation": operation, "success": success})

    def record_error(
        self,
        operation: str,
        error_type: str,
        category: str,
        message: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record an error.

        Args:
            operation: Name of the operation.
            error_type: Type/class of the error.
            category: Error category.
            message: Error message.
            attributes: Additional attributes.
        """
        tracker = self._get_error_tracker(operation)
        tracker.record(error_type, category, message, attributes)
        # Also record as failed request
        self.record_request(operation, success=False)
        self._notify(
            "error",
            {
                "operation": operation,
                "type": error_type,
                "category": category,
            },
        )

    def increment_counter(self, name: str, delta: int = 1) -> None:
        """Increment a counter.

        Args:
            name: Counter name.
            delta: Amount to increment by.
        """
        with self._lock:
            self._counters[name] += delta
        self._notify("counter", {"name": name, "value": self._counters[name]})

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value.

        Args:
            name: Gauge name.
            value: Gauge value.
        """
        with self._lock:
            self._gauges[name] = value
        self._notify("gauge", {"name": name, "value": value})

    def get_counter(self, name: str) -> int:
        """Get a counter value.

        Args:
            name: Counter name.

        Returns:
            Current counter value.
        """
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float | None:
        """Get a gauge value.

        Args:
            name: Gauge name.

        Returns:
            Current gauge value or None.
        """
        with self._lock:
            return self._gauges.get(name)

    def get_latency_stats(self, operation: str) -> LatencyStats:
        """Get latency statistics for an operation.

        Args:
            operation: Name of the operation.

        Returns:
            LatencyStats for the operation.
        """
        tracker = self._get_latency_tracker(operation)
        return tracker.get_stats(operation)

    def get_throughput_stats(self, operation: str) -> ThroughputStats:
        """Get throughput statistics for an operation.

        Args:
            operation: Name of the operation.

        Returns:
            ThroughputStats for the operation.
        """
        tracker = self._get_throughput_tracker(operation)
        return tracker.get_stats(operation)

    def get_error_stats(self, operation: str) -> ErrorStats:
        """Get error statistics for an operation.

        Args:
            operation: Name of the operation.

        Returns:
            ErrorStats for the operation.
        """
        tracker = self._get_error_tracker(operation)
        return tracker.get_stats(operation)

    def get_summary(self) -> MetricsSummary:
        """Get a complete metrics summary.

        Returns:
            MetricsSummary with all metrics.
        """
        with self._lock:
            latency = {op: tracker.get_stats(op) for op, tracker in self._latency_trackers.items()}
            throughput = {
                op: tracker.get_stats(op) for op, tracker in self._throughput_trackers.items()
            }
            errors = {op: tracker.get_stats(op) for op, tracker in self._error_trackers.items()}

            return MetricsSummary(
                latency=latency,
                throughput=throughput,
                errors=errors,
                counters=dict(self._counters),
                gauges=dict(self._gauges),
            )

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            for latency_tracker in self._latency_trackers.values():
                latency_tracker.clear()
            for throughput_tracker in self._throughput_trackers.values():
                throughput_tracker.clear()
            for error_tracker in self._error_trackers.values():
                error_tracker.clear()
            self._counters.clear()
            self._gauges.clear()

    def on_metric(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback for metric updates.

        Args:
            callback: Function to call on metric updates.
                     Receives (metric_type, data) arguments.
        """
        self._callbacks.append(callback)

    def _notify(self, metric_type: str, data: Any) -> None:
        """Notify callbacks of a metric update."""
        for callback in self._callbacks:
            try:
                callback(metric_type, data)
            except Exception as e:
                logger.error(f"Metric callback failed: {e}")

    def export_json(self, output_path: Path) -> None:
        """Export all metrics to JSON.

        Args:
            output_path: Path to output file.
        """
        summary = self.get_summary()
        with output_path.open("w") as f:
            json.dump(summary.to_dict(), f, indent=2)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string.
        """
        lines = []
        summary = self.get_summary()

        # Export latency metrics
        for op, stats in summary.latency.items():
            safe_op = op.replace("-", "_").replace(".", "_")
            lines.append(f"# HELP latency_{safe_op}_ms Latency for {op}")
            lines.append(f"# TYPE latency_{safe_op}_ms gauge")
            lines.append(f'latency_{safe_op}_ms{{quantile="0.5"}} {stats.median_ms}')
            lines.append(f'latency_{safe_op}_ms{{quantile="0.95"}} {stats.p95}')
            lines.append(f'latency_{safe_op}_ms{{quantile="0.99"}} {stats.p99}')
            lines.append(f"latency_{safe_op}_count {stats.count}")

        # Export throughput metrics
        for t_op, t_stats in summary.throughput.items():
            safe_t_op = t_op.replace("-", "_").replace(".", "_")
            lines.append(f"# HELP throughput_{safe_t_op} Throughput for {t_op}")
            lines.append(f"# TYPE throughput_{safe_t_op} counter")
            lines.append(f"throughput_{safe_t_op}_total {t_stats.total_count}")
            lines.append(f"throughput_{safe_t_op}_success {t_stats.success_count}")
            lines.append(f"throughput_{safe_t_op}_errors {t_stats.error_count}")

        # Export counters
        for c_name, c_value in summary.counters.items():
            safe_c_name = c_name.replace("-", "_").replace(".", "_")
            lines.append(f"# HELP counter_{safe_c_name} Counter {c_name}")
            lines.append(f"# TYPE counter_{safe_c_name} counter")
            lines.append(f"counter_{safe_c_name} {c_value}")

        # Export gauges
        for g_name, g_value in summary.gauges.items():
            safe_g_name = g_name.replace("-", "_").replace(".", "_")
            lines.append(f"# HELP gauge_{safe_g_name} Gauge {g_name}")
            lines.append(f"# TYPE gauge_{safe_g_name} gauge")
            lines.append(f"gauge_{safe_g_name} {g_value}")

        return "\n".join(lines)


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics_collector(
    latency_max_samples: int = 10000,
    throughput_window: float = 60.0,
) -> MetricsCollector:
    """Get or create the global metrics collector.

    Args:
        latency_max_samples: Max samples for latency tracking.
        throughput_window: Window for throughput calculation.

    Returns:
        The MetricsCollector instance.
    """
    global _metrics_collector

    with _collector_lock:
        if _metrics_collector is None:
            _metrics_collector = MetricsCollector(
                latency_max_samples=latency_max_samples,
                throughput_window=throughput_window,
            )
        return _metrics_collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector."""
    global _metrics_collector
    with _collector_lock:
        _metrics_collector = None
