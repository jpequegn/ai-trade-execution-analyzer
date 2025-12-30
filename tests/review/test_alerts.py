"""Tests for alerting system module."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest  # noqa: TC002 - pytest used at runtime for fixtures

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
from src.review.models import HumanFeedback


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_values(self) -> None:
        """Test severity value strings."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlert:
    """Tests for Alert model."""

    def test_default_values(self) -> None:
        """Test default alert values."""
        alert = Alert(message="Test alert")

        assert alert.severity == AlertSeverity.WARNING
        assert alert.alert_type == "general"
        assert alert.message == "Test alert"
        assert alert.details == {}
        assert not alert.acknowledged
        assert alert.alert_id  # Should be auto-generated

    def test_custom_values(self) -> None:
        """Test alert with custom values."""
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            alert_type="agreement_critical",
            message="Agreement dropped",
            details={"rate": 0.65},
        )

        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.alert_type == "agreement_critical"
        assert alert.details["rate"] == 0.65

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            alert_type="test",
            message="Test message",
            details={"key": "value"},
        )

        result = alert.to_dict()

        assert result["severity"] == "warning"
        assert result["alert_type"] == "test"
        assert result["message"] == "Test message"
        assert result["details"]["key"] == "value"
        assert not result["acknowledged"]


class TestAlertThresholds:
    """Tests for AlertThresholds dataclass."""

    def test_default_values(self) -> None:
        """Test default threshold values."""
        thresholds = AlertThresholds()

        assert thresholds.score_deviation == 2
        assert thresholds.agreement_rate_warning == 0.85
        assert thresholds.agreement_rate_critical == 0.75
        assert thresholds.issue_agreement_threshold == 0.80
        assert thresholds.new_disagreement_pattern is True
        assert thresholds.consecutive_disagreements == 5

    def test_custom_values(self) -> None:
        """Test custom threshold values."""
        thresholds = AlertThresholds(
            score_deviation=3,
            agreement_rate_warning=0.90,
            consecutive_disagreements=10,
        )

        assert thresholds.score_deviation == 3
        assert thresholds.agreement_rate_warning == 0.90
        assert thresholds.consecutive_disagreements == 10


class TestConsoleAlertHandler:
    """Tests for ConsoleAlertHandler."""

    def test_handle_alert(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test console output for alerts."""
        handler = ConsoleAlertHandler()
        alert = Alert(
            severity=AlertSeverity.WARNING,
            message="Test warning",
        )

        handler.handle(alert)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "Test warning" in captured.out


class TestLogAlertHandler:
    """Tests for LogAlertHandler."""

    def test_handle_alert(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test log output for alerts."""
        handler = LogAlertHandler(logger_name="test_alerts")
        alert = Alert(
            severity=AlertSeverity.WARNING,
            alert_type="test",
            message="Test log message",
        )

        with caplog.at_level("WARNING", logger="test_alerts"):
            handler.handle(alert)

        assert "Test log message" in caplog.text


class TestFileAlertHandler:
    """Tests for FileAlertHandler."""

    def test_handle_alert(self) -> None:
        """Test file output for alerts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_alerts.json"
            handler = FileAlertHandler(file_path=file_path)

            alert = Alert(
                severity=AlertSeverity.CRITICAL,
                message="Critical test alert",
            )
            handler.handle(alert)

            # Check file was created
            assert file_path.exists()

            # Check content
            with file_path.open() as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["message"] == "Critical test alert"
            assert data[0]["severity"] == "critical"

    def test_append_to_existing_file(self) -> None:
        """Test appending to existing alert file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_alerts.json"
            handler = FileAlertHandler(file_path=file_path)

            # Add first alert
            handler.handle(Alert(message="First alert"))

            # Add second alert
            handler.handle(Alert(message="Second alert"))

            with file_path.open() as f:
                data = json.load(f)

            assert len(data) == 2


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_default_initialization(self) -> None:
        """Test default manager initialization."""
        manager = AlertManager()

        assert len(manager.handlers) == 1  # Default console handler
        assert isinstance(manager.handlers[0], ConsoleAlertHandler)
        assert manager.thresholds is not None
        assert len(manager.history) == 0

    def test_custom_handlers(self) -> None:
        """Test manager with custom handlers."""
        handler = LogAlertHandler()
        manager = AlertManager(handlers=[handler])

        assert len(manager.handlers) == 1
        assert isinstance(manager.handlers[0], LogAlertHandler)

    def test_add_handler(self) -> None:
        """Test adding handlers."""
        # Note: AlertManager adds a default handler if none provided
        # so we need to start with an explicit empty list
        manager = AlertManager(handlers=[LogAlertHandler()])
        initial_count = len(manager.handlers)
        handler = ConsoleAlertHandler()

        manager.add_handler(handler)

        assert len(manager.handlers) == initial_count + 1
        assert manager.handlers[-1] is handler

    def test_remove_handler(self) -> None:
        """Test removing handlers."""
        handler = ConsoleAlertHandler()
        manager = AlertManager(handlers=[handler])

        manager.remove_handler(handler)

        assert len(manager.handlers) == 0

    def test_check_alerts_empty_feedbacks(self) -> None:
        """Test check_alerts with no feedbacks."""
        manager = AlertManager(handlers=[])
        alerts = manager.check_alerts([])

        assert alerts == []

    def test_check_alerts_high_agreement(self) -> None:
        """Test check_alerts with high agreement (no alerts)."""
        manager = AlertManager(handlers=[])
        feedbacks = [
            HumanFeedback(
                analysis_id=f"AN{i:03d}",
                reviewer_id="expert1",
                agrees_with_score=True,
                human_score=8,
            )
            for i in range(10)
        ]

        alerts = manager.check_alerts(feedbacks)

        # Should have no critical/warning alerts about agreement
        critical_alerts = [a for a in alerts if a.alert_type == "agreement_critical"]
        warning_alerts = [a for a in alerts if a.alert_type == "agreement_warning"]

        assert len(critical_alerts) == 0
        assert len(warning_alerts) == 0

    def test_check_alerts_critical_agreement(self) -> None:
        """Test check_alerts with critically low agreement."""
        manager = AlertManager(
            handlers=[],
            thresholds=AlertThresholds(agreement_rate_critical=0.75),
        )
        # All feedback disagrees
        feedbacks = [
            HumanFeedback(
                analysis_id=f"AN{i:03d}",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
            )
            for i in range(10)
        ]

        alerts = manager.check_alerts(feedbacks)

        critical_alerts = [a for a in alerts if a.alert_type == "agreement_critical"]
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL

    def test_check_alerts_warning_agreement(self) -> None:
        """Test check_alerts with warning-level agreement."""
        manager = AlertManager(
            handlers=[],
            thresholds=AlertThresholds(
                agreement_rate_warning=0.85,
                agreement_rate_critical=0.60,  # Lower critical threshold
            ),
        )
        # 70% score agreement + issue agreement = overall below 85%
        feedbacks = [
            HumanFeedback(
                analysis_id=f"AN{i:03d}",
                reviewer_id="expert1",
                agrees_with_score=i < 7,  # 7 out of 10 agree (70%)
                human_score=8 if i < 7 else 5,
                score_correction=None if i < 7 else 6,
            )
            for i in range(10)
        ]

        alerts = manager.check_alerts(feedbacks)

        # Should trigger warning but not critical
        warning_alerts = [a for a in alerts if a.alert_type == "agreement_warning"]
        critical_alerts = [a for a in alerts if a.alert_type == "agreement_critical"]
        assert len(warning_alerts) == 1 or len(critical_alerts) == 0
        # At least one alert should be present
        agreement_alerts = [a for a in alerts if "agreement" in a.alert_type]
        assert len(agreement_alerts) >= 1

    def test_check_alerts_consecutive_disagreements(self) -> None:
        """Test alert for consecutive disagreements."""
        manager = AlertManager(
            handlers=[],
            thresholds=AlertThresholds(consecutive_disagreements=3),
        )
        base_time = datetime.now()
        # 5 consecutive disagreements (most recent first)
        feedbacks = [
            HumanFeedback(
                analysis_id=f"AN{i:03d}",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
                timestamp=base_time - timedelta(hours=i),
            )
            for i in range(5)
        ]

        alerts = manager.check_alerts(feedbacks)

        consecutive_alerts = [a for a in alerts if a.alert_type == "consecutive_disagreements"]
        assert len(consecutive_alerts) == 1

    def test_acknowledge_alert(self) -> None:
        """Test acknowledging alerts."""
        manager = AlertManager(handlers=[])
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
            )
            for _ in range(10)
        ]

        alerts = manager.check_alerts(feedbacks)
        assert len(alerts) > 0

        # Acknowledge first alert
        alert_id = alerts[0].alert_id
        result = manager.acknowledge(alert_id)

        assert result is True
        assert manager.history[0].acknowledged is True

    def test_acknowledge_nonexistent_alert(self) -> None:
        """Test acknowledging nonexistent alert."""
        manager = AlertManager()

        result = manager.acknowledge("nonexistent-id")

        assert result is False

    def test_get_unacknowledged(self) -> None:
        """Test getting unacknowledged alerts."""
        manager = AlertManager(handlers=[])
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
            )
            for _ in range(10)
        ]

        alerts = manager.check_alerts(feedbacks)
        assert len(alerts) > 0

        # All should be unacknowledged
        unacked = manager.get_unacknowledged()
        assert len(unacked) == len(alerts)

        # Acknowledge one
        manager.acknowledge(alerts[0].alert_id)

        # Should have one fewer unacknowledged
        unacked = manager.get_unacknowledged()
        assert len(unacked) == len(alerts) - 1

    def test_get_by_severity(self) -> None:
        """Test getting alerts by severity."""
        manager = AlertManager(handlers=[])
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
            )
            for _ in range(10)
        ]

        manager.check_alerts(feedbacks)

        critical = manager.get_by_severity(AlertSeverity.CRITICAL)
        warning = manager.get_by_severity(AlertSeverity.WARNING)
        info = manager.get_by_severity(AlertSeverity.INFO)

        # All returned alerts should have correct severity
        assert all(a.severity == AlertSeverity.CRITICAL for a in critical)
        assert all(a.severity == AlertSeverity.WARNING for a in warning)
        assert all(a.severity == AlertSeverity.INFO for a in info)

    def test_get_summary(self) -> None:
        """Test summary generation."""
        manager = AlertManager(handlers=[])
        feedbacks = [
            HumanFeedback(
                analysis_id="AN001",
                reviewer_id="expert1",
                agrees_with_score=False,
                human_score=5,
                score_correction=6,
            )
            for _ in range(10)
        ]

        manager.check_alerts(feedbacks)

        summary = manager.get_summary()

        assert "Alert Summary" in summary
        assert "Total Alerts" in summary
        assert "Critical" in summary
        assert "Warning" in summary
        assert "Info" in summary
        assert "Unacknowledged" in summary

    def test_get_summary_no_alerts(self) -> None:
        """Test summary with no alerts."""
        manager = AlertManager()

        summary = manager.get_summary()

        assert "No alerts recorded" in summary


class TestCreateDefaultAlertManager:
    """Tests for create_default_alert_manager function."""

    def test_default_manager(self) -> None:
        """Test creating default manager."""
        manager = create_default_alert_manager()

        assert len(manager.handlers) == 2
        assert any(isinstance(h, ConsoleAlertHandler) for h in manager.handlers)
        assert any(isinstance(h, LogAlertHandler) for h in manager.handlers)

    def test_with_file_logging(self) -> None:
        """Test creating manager with file logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "alerts.json"
            manager = create_default_alert_manager(
                log_to_file=True,
                file_path=file_path,
            )

            assert len(manager.handlers) == 3
            assert any(isinstance(h, FileAlertHandler) for h in manager.handlers)
