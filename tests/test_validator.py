"""Tests for the ground truth dataset validator."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.evaluation.validator import (
    ValidationError,
    ValidationResult,
    validate_dataset,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_initial_state(self) -> None:
        """Test initial validation result state."""
        result = ValidationResult()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.stats == {}

    def test_add_error(self) -> None:
        """Test adding an error invalidates result."""
        result = ValidationResult()
        result.add_error("GT001", "field", "Error message")

        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].sample_id == "GT001"
        assert result.errors[0].field == "field"
        assert result.errors[0].message == "Error message"

    def test_add_warning(self) -> None:
        """Test adding a warning doesn't invalidate result."""
        result = ValidationResult()
        result.add_warning("GT001", "field", "Warning message")

        assert result.valid is True
        assert len(result.warnings) == 1

    def test_summary_output(self) -> None:
        """Test summary output format."""
        result = ValidationResult()
        result.stats["total"] = 10
        result.add_error("GT001", "id", "Duplicate ID")
        result.add_warning("GT002", "score", "Low score without issues")

        summary = result.summary()

        assert "FAILED" in summary
        assert "Total Samples: 10" in summary
        assert "Errors: 1" in summary
        assert "Warnings: 1" in summary
        assert "[GT001] id: Duplicate ID" in summary
        assert "[GT002] score:" in summary


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_validate_nonexistent_file(self) -> None:
        """Test validation of nonexistent file."""
        result = validate_dataset(Path("/nonexistent/path.json"))
        assert result.valid is False
        assert any("not found" in e.message for e in result.errors)

    def test_validate_invalid_json(self) -> None:
        """Test validation of invalid JSON."""
        with NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json")
            temp_path = Path(f.name)

        try:
            result = validate_dataset(temp_path)
            assert result.valid is False
            assert any("parse" in e.message.lower() for e in result.errors)
        finally:
            temp_path.unlink()

    def test_validate_empty_dataset(self) -> None:
        """Test validation of empty dataset."""
        with NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"version": "1.0.0", "samples": []}, f)
            temp_path = Path(f.name)

        try:
            result = validate_dataset(temp_path)
            assert result.valid is False
            assert any("empty" in e.message.lower() for e in result.errors)
        finally:
            temp_path.unlink()

    def test_validate_duplicate_ids(self) -> None:
        """Test detection of duplicate IDs."""
        dataset = {
            "version": "1.0.0",
            "samples": [
                {
                    "id": "GT001",
                    "fix_message": "8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|31=185.50|30=NYSE|60=20240115-10:30:00.000|39=2",
                    "execution": {
                        "order_id": "ORD001",
                        "symbol": "AAPL",
                        "side": "BUY",
                        "quantity": 100.0,
                        "price": 185.50,
                        "venue": "NYSE",
                        "timestamp": "2024-01-15T10:30:00",
                        "fill_type": "FULL",
                    },
                    "expert_analysis": {
                        "quality_score": 9,
                        "key_issues": [],
                        "expected_observations": [],
                        "severity": "none",
                        "category": "none",
                    },
                    "metadata": {
                        "annotator": "expert",
                        "annotation_date": "2024-01-15",
                        "confidence": "high",
                    },
                },
                {
                    "id": "GT001",  # Duplicate ID
                    "fix_message": "8=FIX.4.4|35=8|37=ORD002|55=MSFT|54=1|32=100|31=400.00|30=NASDAQ|60=20240115-10:30:00.000|39=2",
                    "execution": {
                        "order_id": "ORD002",
                        "symbol": "MSFT",
                        "side": "BUY",
                        "quantity": 100.0,
                        "price": 400.00,
                        "venue": "NASDAQ",
                        "timestamp": "2024-01-15T10:30:00",
                        "fill_type": "FULL",
                    },
                    "expert_analysis": {
                        "quality_score": 8,
                        "key_issues": [],
                        "expected_observations": [],
                        "severity": "none",
                        "category": "none",
                    },
                    "metadata": {
                        "annotator": "expert",
                        "annotation_date": "2024-01-15",
                        "confidence": "high",
                    },
                },
            ],
        }

        with NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(dataset, f)
            temp_path = Path(f.name)

        try:
            result = validate_dataset(temp_path)
            assert result.valid is False
            assert any("Duplicate ID" in e.message for e in result.errors)
        finally:
            temp_path.unlink()

    def test_validate_mismatched_execution(self) -> None:
        """Test detection of mismatched execution data."""
        dataset = {
            "version": "1.0.0",
            "samples": [
                {
                    "id": "GT001",
                    "fix_message": "8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|31=185.50|30=NYSE|60=20240115-10:30:00.000|39=2",
                    "execution": {
                        "order_id": "ORD001",
                        "symbol": "WRONG",  # Mismatch - FIX message has AAPL
                        "side": "BUY",
                        "quantity": 100.0,
                        "price": 185.50,
                        "venue": "NYSE",
                        "timestamp": "2024-01-15T10:30:00",
                        "fill_type": "FULL",
                    },
                    "expert_analysis": {
                        "quality_score": 9,
                        "key_issues": [],
                        "expected_observations": [],
                        "severity": "none",
                        "category": "none",
                    },
                    "metadata": {
                        "annotator": "expert",
                        "annotation_date": "2024-01-15",
                        "confidence": "high",
                    },
                },
            ],
        }

        with NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(dataset, f)
            temp_path = Path(f.name)

        try:
            result = validate_dataset(temp_path)
            assert result.valid is False
            assert any("Mismatch" in e.message and "symbol" in e.field for e in result.errors)
        finally:
            temp_path.unlink()

    def test_validate_builtin_dataset(self) -> None:
        """Test that the built-in dataset passes validation."""
        result = validate_dataset()

        assert result.valid is True, f"Built-in dataset validation failed:\n{result.summary()}"
        assert result.stats["total"] >= 50
        assert result.stats["unique_symbols"] > 10

    def test_statistics_calculated(self) -> None:
        """Test that statistics are properly calculated."""
        result = validate_dataset()

        assert "score_distribution" in result.stats
        assert "average_score" in result.stats
        assert "categories" in result.stats
        assert "severities" in result.stats


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_error_attributes(self) -> None:
        """Test error has expected attributes."""
        error = ValidationError(
            sample_id="GT001",
            field="symbol",
            message="Mismatch detected",
            severity="error",
        )
        assert error.sample_id == "GT001"
        assert error.field == "symbol"
        assert error.message == "Mismatch detected"
        assert error.severity == "error"

    def test_warning_severity(self) -> None:
        """Test warning severity."""
        warning = ValidationError(
            sample_id="GT002",
            field="score",
            message="Inconsistent score",
            severity="warning",
        )
        assert warning.severity == "warning"
