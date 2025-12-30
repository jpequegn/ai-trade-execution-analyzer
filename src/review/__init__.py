"""Human review interface for AI trade analysis.

This module provides tools for humans to review AI-generated trade analyses,
provide corrections, and submit feedback for continuous improvement.

Example:
    >>> from src.review.cli import ReviewSession
    >>> from src.review.storage import FeedbackStore
    >>> store = FeedbackStore()
    >>> session = ReviewSession(reviewer_id="expert_1", store=store)
    >>> session.start(queue)

Agreement tracking example:
    >>> from src.review import calculate_agreement_rate, AlertManager
    >>> store = FeedbackStore()
    >>> metrics = calculate_agreement_rate(store.get_all())
    >>> print(f"Agreement: {metrics.overall_agreement:.1%}")
"""

from src.review.agreement import (
    AgreementCalculator,
    AgreementMetrics,
    AgreementTracker,
    calculate_agreement_rate,
)
from src.review.alerts import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertThresholds,
    ConsoleAlertHandler,
    FileAlertHandler,
    LogAlertHandler,
    create_default_alert_manager,
)
from src.review.disagreements import (
    DisagreementPattern,
    DisagreementReport,
    DisagreementTracker,
    analyze_disagreements,
)
from src.review.models import (
    HumanFeedback,
    ReviewItem,
    ReviewSessionStats,
    ReviewStatus,
    SamplingStrategy,
)
from src.review.queue import (
    ReviewQueue,
    create_stratified_queue,
    get_review_queue,
)
from src.review.storage import FeedbackStore

__all__ = [
    "AgreementCalculator",
    "AgreementMetrics",
    "AgreementTracker",
    "Alert",
    "AlertManager",
    "AlertSeverity",
    "AlertThresholds",
    "ConsoleAlertHandler",
    "DisagreementPattern",
    "DisagreementReport",
    "DisagreementTracker",
    "FeedbackStore",
    "FileAlertHandler",
    "HumanFeedback",
    "LogAlertHandler",
    "ReviewItem",
    "ReviewQueue",
    "ReviewSessionStats",
    "ReviewStatus",
    "SamplingStrategy",
    "analyze_disagreements",
    "calculate_agreement_rate",
    "create_default_alert_manager",
    "create_stratified_queue",
    "get_review_queue",
]
