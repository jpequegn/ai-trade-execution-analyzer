"""Observability and tracing utilities.

This module provides comprehensive observability capabilities including:
- Trace context management with hierarchical spans
- Langfuse integration for LLM observability
- Cost tracking and budget management
- Latency metrics with percentile tracking
- Error tracking and categorization
- Metrics export for external monitoring

Example:
    >>> from src.observability import TraceContext, get_metrics_collector
    >>> collector = get_metrics_collector()
    >>> with TraceContext("analysis_pipeline") as ctx:
    ...     with ctx.span("parse") as span:
    ...         span.set_attribute("message_type", "ExecutionReport")
    ...         # do work
    ...     collector.record_latency("analysis", ctx.data.duration_ms or 0)
"""

from src.observability.context import (
    ErrorCategory,
    Span,
    SpanData,
    SpanStatus,
    TraceContext,
    TraceData,
    get_current_context,
    set_current_context,
    trace,
)
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
from src.observability.tracing import (
    Tracer,
    generate_trace_id,
    get_tracer,
    reset_tracer,
    trace_context,
    traced,
)

__all__ = [
    "BudgetAlert",
    "CostRecord",
    "CostTracker",
    "DailyCostSummary",
    "ErrorCategory",
    "ErrorStats",
    "ErrorTracker",
    "LatencyStats",
    "LatencyTracker",
    "MetricsCollector",
    "MetricsSummary",
    "MonthlyCostReport",
    "Span",
    "SpanData",
    "SpanStatus",
    "ThroughputStats",
    "ThroughputTracker",
    "TokenUsage",
    "TraceContext",
    "TraceData",
    "Tracer",
    "calculate_cost",
    "generate_trace_id",
    "get_cost_tracker",
    "get_current_context",
    "get_metrics_collector",
    "get_tracer",
    "reset_cost_tracker",
    "reset_metrics_collector",
    "reset_tracer",
    "set_current_context",
    "trace",
    "trace_context",
    "traced",
]
