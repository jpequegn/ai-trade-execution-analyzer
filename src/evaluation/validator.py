"""Validator for ground truth dataset integrity.

This module provides validation utilities to ensure the ground truth
dataset is complete, consistent, and parseable.

Example:
    >>> from src.evaluation.validator import validate_dataset
    >>> result = validate_dataset()
    >>> print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003

from src.evaluation.ground_truth import (
    DATASET_PATH,
    IssueCategory,
    IssueSeverity,
    load_ground_truth,
)
from src.parsers.fix_parser import parse_fix_message

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """A single validation error."""

    sample_id: str
    field: str
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    def add_error(self, sample_id: str, field_name: str, message: str) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(sample_id, field_name, message, "error"))
        self.valid = False

    def add_warning(self, sample_id: str, field_name: str, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationError(sample_id, field_name, message, "warning"))

    def summary(self) -> str:
        """Generate a summary of validation results."""
        lines = [
            "=" * 60,
            "Ground Truth Dataset Validation Report",
            "=" * 60,
            "",
            f"Status: {'PASSED' if self.valid else 'FAILED'}",
            f"Total Samples: {self.stats.get('total', 0)}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
            "",
        ]

        if self.stats:
            lines.append("Statistics:")
            for key, value in self.stats.items():
                if key != "total":
                    lines.append(f"  {key}: {value}")
            lines.append("")

        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  [{err.sample_id}] {err.field}: {err.message}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  [{warn.sample_id}] {warn.field}: {warn.message}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


def validate_dataset(path: Path | None = None) -> ValidationResult:
    """Validate the ground truth dataset.

    Performs the following checks:
    1. File exists and is valid JSON
    2. All samples have unique IDs
    3. All FIX messages can be parsed
    4. Parsed executions match stored data
    5. All required fields are present
    6. Score consistency (low scores have issues)
    7. Coverage statistics

    Args:
        path: Optional path to the dataset. Defaults to built-in location.

    Returns:
        ValidationResult with errors, warnings, and statistics.
    """
    result = ValidationResult()
    dataset_path = path or DATASET_PATH

    # Check file exists
    if not dataset_path.exists():
        result.add_error("N/A", "file", f"Dataset file not found: {dataset_path}")
        return result

    # Load and parse dataset
    try:
        dataset = load_ground_truth(dataset_path)
    except Exception as e:
        result.add_error("N/A", "parse", f"Failed to parse dataset: {e}")
        return result

    samples = dataset.samples
    result.stats["total"] = len(samples)

    if not samples:
        result.add_error("N/A", "samples", "Dataset is empty")
        return result

    # Check for unique IDs
    seen_ids: set[str] = set()
    for sample in samples:
        if sample.id in seen_ids:
            result.add_error(sample.id, "id", f"Duplicate ID: {sample.id}")
        seen_ids.add(sample.id)

    # Validate each sample
    for sample in samples:
        # Validate FIX message parses
        try:
            parsed = parse_fix_message(sample.fix_message)

            # Check parsed values match stored execution
            if parsed.order_id != sample.execution.order_id:
                result.add_error(
                    sample.id,
                    "order_id",
                    f"Mismatch: parsed={parsed.order_id}, stored={sample.execution.order_id}",
                )
            if parsed.symbol != sample.execution.symbol:
                result.add_error(
                    sample.id,
                    "symbol",
                    f"Mismatch: parsed={parsed.symbol}, stored={sample.execution.symbol}",
                )
            if abs(parsed.quantity - sample.execution.quantity) > 0.01:
                result.add_error(
                    sample.id,
                    "quantity",
                    f"Mismatch: parsed={parsed.quantity}, stored={sample.execution.quantity}",
                )
            if abs(parsed.price - sample.execution.price) > 0.01:
                result.add_error(
                    sample.id,
                    "price",
                    f"Mismatch: parsed={parsed.price}, stored={sample.execution.price}",
                )

        except Exception as e:
            result.add_error(sample.id, "fix_message", f"Failed to parse: {e}")

        # Validate score consistency
        analysis = sample.expert_analysis
        if analysis.quality_score <= 3 and not analysis.key_issues:
            result.add_warning(
                sample.id,
                "quality_score",
                f"Low score ({analysis.quality_score}) but no key_issues",
            )
        if analysis.quality_score >= 8 and not analysis.expected_observations:
            result.add_warning(
                sample.id,
                "quality_score",
                f"High score ({analysis.quality_score}) but no expected_observations",
            )

        # Validate severity matches score
        if analysis.quality_score <= 2 and analysis.severity not in [
            IssueSeverity.HIGH,
            IssueSeverity.CRITICAL,
        ]:
            result.add_warning(
                sample.id,
                "severity",
                f"Very low score ({analysis.quality_score}) but severity is {analysis.severity.value}",
            )
        if analysis.quality_score >= 9 and analysis.severity != IssueSeverity.NONE:
            result.add_warning(
                sample.id,
                "severity",
                f"Very high score ({analysis.quality_score}) but has severity {analysis.severity.value}",
            )

        # Validate category matches issues
        if analysis.key_issues and analysis.category == IssueCategory.NONE:
            result.add_warning(
                sample.id,
                "category",
                "Has key_issues but category is 'none'",
            )

    # Calculate coverage statistics
    scores = [s.expert_analysis.quality_score for s in samples]
    result.stats["score_distribution"] = {
        "good (8-10)": len([s for s in scores if 8 <= s <= 10]),
        "average (5-7)": len([s for s in scores if 5 <= s <= 7]),
        "poor (1-4)": len([s for s in scores if 1 <= s <= 4]),
    }
    result.stats["average_score"] = round(sum(scores) / len(scores), 2)

    # Category coverage
    categories: dict[str, int] = {}
    for sample in samples:
        cat = sample.expert_analysis.category.value
        categories[cat] = categories.get(cat, 0) + 1
    result.stats["categories"] = categories

    # Severity coverage
    severities: dict[str, int] = {}
    for sample in samples:
        sev = sample.expert_analysis.severity.value
        severities[sev] = severities.get(sev, 0) + 1
    result.stats["severities"] = severities

    # Symbol coverage
    symbols: set[str] = set()
    for sample in samples:
        symbols.add(sample.execution.symbol)
    result.stats["unique_symbols"] = len(symbols)

    # Venue coverage
    venues: set[str] = set()
    for sample in samples:
        venues.add(sample.execution.venue)
    result.stats["unique_venues"] = len(venues)

    return result


def main() -> int:
    """CLI entry point for validation."""

    result = validate_dataset()
    print(result.summary())
    return 0 if result.valid else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
