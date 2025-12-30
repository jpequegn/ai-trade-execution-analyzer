"""Alerting system for AI-human agreement monitoring.

This module provides alerting capabilities when agreement metrics
fall below thresholds or new disagreement patterns emerge.

Example:
    >>> from src.review.alerts import AlertManager, AlertThresholds
    >>> from src.review.storage import FeedbackStore
    >>> manager = AlertManager(thresholds=AlertThresholds())
    >>> store = FeedbackStore()
    >>> alerts = manager.check_alerts(store.get_all())
    >>> for alert in alerts:
    ...     print(f"[{alert.severity}] {alert.message}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.review.models import HumanFeedback

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Severity level for alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert(BaseModel):
    """An alert for agreement monitoring.

    Attributes:
        alert_id: Unique identifier for this alert.
        severity: Severity level of the alert.
        alert_type: Type/category of alert.
        message: Human-readable alert message.
        details: Additional alert details.
        timestamp: When the alert was generated.
        acknowledged: Whether the alert has been acknowledged.
    """

    alert_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    severity: AlertSeverity = AlertSeverity.WARNING
    alert_type: str = "general"
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the alert.
        """
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "alert_type": self.alert_type,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class AlertThresholds:
    """Configuration for alert thresholds.

    Attributes:
        score_deviation: Maximum acceptable score difference.
        agreement_rate_warning: Warn if agreement drops below this.
        agreement_rate_critical: Critical if agreement drops below this.
        issue_agreement_threshold: Threshold for issue agreement alerts.
        new_disagreement_pattern: Alert on new patterns.
        consecutive_disagreements: Alert after N consecutive disagreements.
    """

    score_deviation: int = 2
    agreement_rate_warning: float = 0.85
    agreement_rate_critical: float = 0.75
    issue_agreement_threshold: float = 0.80
    new_disagreement_pattern: bool = True
    consecutive_disagreements: int = 5


class AlertHandler(Protocol):
    """Protocol for alert handlers."""

    def handle(self, alert: Alert) -> None:
        """Handle an alert.

        Args:
            alert: The alert to handle.
        """
        ...


@dataclass
class ConsoleAlertHandler:
    """Handler that outputs alerts to console."""

    def handle(self, alert: Alert) -> None:
        """Print alert to console.

        Args:
            alert: The alert to handle.
        """
        severity_icons = {
            AlertSeverity.INFO: "[i] ",
            AlertSeverity.WARNING: "[!] ",
            AlertSeverity.CRITICAL: "[X]",
        }
        icon = severity_icons.get(alert.severity, "")
        print(f"{icon} [{alert.severity.value.upper()}] {alert.message}")


@dataclass
class LogAlertHandler:
    """Handler that logs alerts."""

    logger_name: str = "alerts"

    def __post_init__(self) -> None:
        """Initialize logger."""
        self._logger = logging.getLogger(self.logger_name)

    def handle(self, alert: Alert) -> None:
        """Log the alert.

        Args:
            alert: The alert to handle.
        """
        log_levels = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }
        level = log_levels.get(alert.severity, logging.WARNING)
        self._logger.log(level, f"[{alert.alert_type}] {alert.message}", extra=alert.details)


@dataclass
class FileAlertHandler:
    """Handler that writes alerts to a JSON file."""

    file_path: Path = field(default_factory=lambda: Path("alerts.json"))

    def handle(self, alert: Alert) -> None:
        """Append alert to JSON file.

        Args:
            alert: The alert to handle.
        """
        alerts = []
        if self.file_path.exists():
            try:
                with self.file_path.open() as f:
                    alerts = json.load(f)
            except (json.JSONDecodeError, OSError):
                alerts = []

        alerts.append(alert.to_dict())

        with self.file_path.open("w") as f:
            json.dump(alerts, f, indent=2)


@dataclass
class WebhookAlertHandler:
    """Handler that sends alerts to a webhook (Slack, etc.).

    Note: This is a placeholder implementation. In production,
    you would implement actual HTTP calls to your webhook.
    """

    webhook_url: str = ""
    enabled: bool = False

    def handle(self, alert: Alert) -> None:
        """Send alert to webhook.

        Args:
            alert: The alert to handle.
        """
        if not self.enabled or not self.webhook_url:
            logger.debug("Webhook handler disabled or no URL configured")
            return

        # Placeholder - in production, implement actual HTTP POST
        logger.info(f"Would send to webhook {self.webhook_url}: {alert.message}")


@dataclass
class AlertManager:
    """Manager for checking and dispatching alerts.

    Attributes:
        thresholds: Alert threshold configuration.
        handlers: List of alert handlers.
        history: Historical alerts.
    """

    thresholds: AlertThresholds = field(default_factory=AlertThresholds)
    handlers: list[AlertHandler] = field(default_factory=list)
    history: list[Alert] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Add default console handler if none provided."""
        if not self.handlers:
            self.handlers = [ConsoleAlertHandler()]

    def check_alerts(
        self,
        feedbacks: Sequence[HumanFeedback],
    ) -> list[Alert]:
        """Check for alert conditions and dispatch any alerts.

        Args:
            feedbacks: List of feedback to analyze.

        Returns:
            List of generated alerts.
        """
        from src.review.agreement import calculate_agreement_rate
        from src.review.disagreements import analyze_disagreements

        if not feedbacks:
            return []

        alerts = []

        # Calculate metrics
        agreement = calculate_agreement_rate(feedbacks)
        disagreement = analyze_disagreements(feedbacks)

        # Check agreement rate thresholds
        if agreement.overall_agreement < self.thresholds.agreement_rate_critical:
            alerts.append(
                Alert(
                    severity=AlertSeverity.CRITICAL,
                    alert_type="agreement_critical",
                    message=f"Agreement rate critically low at {agreement.overall_agreement:.1%}",
                    details={
                        "current_rate": agreement.overall_agreement,
                        "threshold": self.thresholds.agreement_rate_critical,
                        "score_agreement": agreement.score_agreement_rate,
                        "issue_agreement": agreement.issue_agreement_rate,
                    },
                )
            )
        elif agreement.overall_agreement < self.thresholds.agreement_rate_warning:
            alerts.append(
                Alert(
                    severity=AlertSeverity.WARNING,
                    alert_type="agreement_warning",
                    message=f"Agreement rate dropped to {agreement.overall_agreement:.1%}",
                    details={
                        "current_rate": agreement.overall_agreement,
                        "threshold": self.thresholds.agreement_rate_warning,
                    },
                )
            )

        # Check issue agreement
        if agreement.issue_agreement_rate < self.thresholds.issue_agreement_threshold:
            alerts.append(
                Alert(
                    severity=AlertSeverity.WARNING,
                    alert_type="issue_agreement",
                    message=f"Issue agreement below threshold at {agreement.issue_agreement_rate:.1%}",
                    details={
                        "current_rate": agreement.issue_agreement_rate,
                        "threshold": self.thresholds.issue_agreement_threshold,
                    },
                )
            )

        # Check for high-severity patterns
        if self.thresholds.new_disagreement_pattern:
            high_severity_patterns = [p for p in disagreement.patterns if p.severity == "high"]
            for pattern in high_severity_patterns:
                alerts.append(
                    Alert(
                        severity=AlertSeverity.WARNING,
                        alert_type="pattern_detected",
                        message=f"Disagreement pattern detected: {pattern.description}",
                        details={
                            "pattern_type": pattern.pattern_type,
                            "frequency": pattern.frequency,
                            "examples": pattern.examples[:3],
                        },
                    )
                )

        # Check for score bias
        if abs(disagreement.score_bias) > self.thresholds.score_deviation:
            alerts.append(
                Alert(
                    severity=AlertSeverity.WARNING,
                    alert_type="score_bias",
                    message=f"AI scoring bias detected: {disagreement.score_bias_description}",
                    details={
                        "bias": disagreement.score_bias,
                        "description": disagreement.score_bias_description,
                    },
                )
            )

        # Check consecutive disagreements
        recent_feedbacks = sorted(feedbacks, key=lambda f: f.timestamp, reverse=True)
        consecutive = 0
        for feedback in recent_feedbacks:
            if not feedback.agrees_with_score:
                consecutive += 1
            else:
                break

        if consecutive >= self.thresholds.consecutive_disagreements:
            alerts.append(
                Alert(
                    severity=AlertSeverity.WARNING,
                    alert_type="consecutive_disagreements",
                    message=f"{consecutive} consecutive disagreements detected",
                    details={"count": consecutive},
                )
            )

        # Check declining trend
        if agreement.trend == "declining":
            alerts.append(
                Alert(
                    severity=AlertSeverity.INFO,
                    alert_type="trend_declining",
                    message="Agreement trend is declining",
                    details={"trend": agreement.trend},
                )
            )

        # Dispatch alerts
        for alert in alerts:
            self._dispatch(alert)

        # Record history
        self.history.extend(alerts)

        return alerts

    def _dispatch(self, alert: Alert) -> None:
        """Dispatch alert to all handlers.

        Args:
            alert: The alert to dispatch.
        """
        for handler in self.handlers:
            try:
                handler.handle(alert)
            except Exception as e:
                logger.exception(f"Alert handler failed: {e}")

    def add_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler.

        Args:
            handler: The handler to add.
        """
        self.handlers.append(handler)

    def remove_handler(self, handler: AlertHandler) -> None:
        """Remove an alert handler.

        Args:
            handler: The handler to remove.
        """
        if handler in self.handlers:
            self.handlers.remove(handler)

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: ID of the alert to acknowledge.

        Returns:
            True if alert was found and acknowledged.
        """
        for alert in self.history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_unacknowledged(self) -> list[Alert]:
        """Get all unacknowledged alerts.

        Returns:
            List of unacknowledged alerts.
        """
        return [a for a in self.history if not a.acknowledged]

    def get_by_severity(self, severity: AlertSeverity) -> list[Alert]:
        """Get alerts by severity.

        Args:
            severity: The severity to filter by.

        Returns:
            List of alerts with given severity.
        """
        return [a for a in self.history if a.severity == severity]

    def get_summary(self) -> str:
        """Generate a summary of alert history.

        Returns:
            Summary string.
        """
        if not self.history:
            return "No alerts recorded."

        by_severity: dict[AlertSeverity, list[Alert]] = {
            AlertSeverity.CRITICAL: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.INFO: [],
        }

        for alert in self.history:
            by_severity[alert.severity].append(alert)

        lines = [
            "Alert Summary",
            "=" * 40,
            f"Total Alerts: {len(self.history)}",
            f"  Critical: {len(by_severity[AlertSeverity.CRITICAL])}",
            f"  Warning: {len(by_severity[AlertSeverity.WARNING])}",
            f"  Info: {len(by_severity[AlertSeverity.INFO])}",
            f"Unacknowledged: {len(self.get_unacknowledged())}",
        ]

        return "\n".join(lines)


def create_default_alert_manager(
    log_to_file: bool = False,
    file_path: Path | None = None,
) -> AlertManager:
    """Create an alert manager with default configuration.

    Args:
        log_to_file: Whether to log alerts to file.
        file_path: Path for file logging.

    Returns:
        Configured AlertManager.
    """
    handlers: list[AlertHandler] = [
        ConsoleAlertHandler(),
        LogAlertHandler(),
    ]

    if log_to_file:
        handlers.append(FileAlertHandler(file_path=file_path or Path("alerts.json")))

    return AlertManager(
        thresholds=AlertThresholds(),
        handlers=handlers,
    )
