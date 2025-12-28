"""Ground truth dataset models for evaluation.

This module provides Pydantic models for the expert-annotated ground truth
dataset used to evaluate AI analysis quality.

Example:
    >>> from src.evaluation.ground_truth import GroundTruthSample, load_ground_truth
    >>> samples = load_ground_truth()
    >>> for sample in samples:
    ...     print(f"{sample.id}: score={sample.expert_analysis.quality_score}")
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from src.parsers.fix_parser import ExecutionReport, parse_fix_message

logger = logging.getLogger(__name__)

# Path to the ground truth dataset
DATASET_PATH = Path(__file__).parent / "datasets" / "ground_truth.json"


class IssueSeverity(str, Enum):
    """Severity level of execution issues."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueCategory(str, Enum):
    """Category of execution issues."""

    NONE = "none"
    VENUE_SELECTION = "venue_selection"
    TIMING = "timing"
    FILL_QUALITY = "fill_quality"
    PRICE_SLIPPAGE = "price_slippage"
    ORDER_HANDLING = "order_handling"
    MARKET_CONDITIONS = "market_conditions"


class AnnotatorConfidence(str, Enum):
    """Confidence level of the human annotator."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExpertAnalysis(BaseModel):
    """Expert-provided ground truth analysis.

    This represents the human expert's assessment of the trade execution,
    serving as the benchmark for evaluating AI analysis quality.

    Attributes:
        quality_score: Expert-assigned quality score (1-10).
        key_issues: List of key issues identified by the expert.
        expected_observations: List of observations the AI should identify.
        severity: Overall severity of issues in the execution.
        category: Primary category of issues.
    """

    quality_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Expert-assigned quality score (1=poor, 10=excellent)",
    )
    key_issues: list[str] = Field(
        default_factory=list,
        description="Key issues identified by the expert",
    )
    expected_observations: list[str] = Field(
        default_factory=list,
        description="Observations the AI should identify",
    )
    severity: IssueSeverity = Field(
        default=IssueSeverity.NONE,
        description="Overall severity of issues",
    )
    category: IssueCategory = Field(
        default=IssueCategory.NONE,
        description="Primary category of issues",
    )

    @field_validator("key_issues", "expected_observations", mode="before")
    @classmethod
    def ensure_list(cls, v: list[str] | None) -> list[str]:
        """Ensure list fields are never None."""
        if v is None:
            return []
        return v


class AnnotationMetadata(BaseModel):
    """Metadata about the ground truth annotation.

    Attributes:
        annotator: Identifier for who created the annotation.
        annotation_date: Date when the annotation was created.
        confidence: Annotator's confidence in their assessment.
        notes: Optional notes about the annotation.
    """

    annotator: str = Field(
        default="expert",
        description="Identifier for the annotator",
    )
    annotation_date: dt.date = Field(
        default_factory=dt.date.today,
        description="Date of annotation",
    )
    confidence: AnnotatorConfidence = Field(
        default=AnnotatorConfidence.HIGH,
        description="Annotator confidence level",
    )
    notes: str | None = Field(
        default=None,
        description="Optional notes about the annotation",
    )


class GroundTruthSample(BaseModel):
    """A single ground truth sample for evaluation.

    Combines a FIX message, its parsed execution, expert analysis,
    and annotation metadata.

    Attributes:
        id: Unique identifier for this sample (e.g., "GT001").
        fix_message: Raw FIX protocol message string.
        execution: Parsed ExecutionReport from the FIX message.
        expert_analysis: Human expert's analysis and scoring.
        metadata: Information about the annotation process.
    """

    id: str = Field(
        ...,
        pattern=r"^GT\d{3}$",
        description="Unique sample ID (format: GT001)",
    )
    fix_message: str = Field(
        ...,
        min_length=10,
        description="Raw FIX protocol message",
    )
    execution: ExecutionReport = Field(
        ...,
        description="Parsed execution report",
    )
    expert_analysis: ExpertAnalysis = Field(
        ...,
        description="Expert-provided ground truth analysis",
    )
    metadata: AnnotationMetadata = Field(
        default_factory=AnnotationMetadata,
        description="Annotation metadata",
    )

    model_config = {"arbitrary_types_allowed": True}


class GroundTruthDataset(BaseModel):
    """Collection of ground truth samples.

    Attributes:
        version: Dataset version string.
        description: Description of the dataset.
        samples: List of ground truth samples.
    """

    version: str = Field(
        default="1.0.0",
        description="Dataset version",
    )
    description: str = Field(
        default="Expert-annotated ground truth dataset for trade execution analysis",
        description="Dataset description",
    )
    samples: list[GroundTruthSample] = Field(
        default_factory=list,
        description="List of ground truth samples",
    )

    def get_by_id(self, sample_id: str) -> GroundTruthSample | None:
        """Get a sample by its ID."""
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        return None

    def get_by_score_range(
        self, min_score: int = 1, max_score: int = 10
    ) -> list[GroundTruthSample]:
        """Get samples within a score range."""
        return [
            s for s in self.samples if min_score <= s.expert_analysis.quality_score <= max_score
        ]

    def get_by_category(self, category: IssueCategory) -> list[GroundTruthSample]:
        """Get samples by issue category."""
        return [s for s in self.samples if s.expert_analysis.category == category]

    def get_by_severity(self, severity: IssueSeverity) -> list[GroundTruthSample]:
        """Get samples by issue severity."""
        return [s for s in self.samples if s.expert_analysis.severity == severity]

    @property
    def good_executions(self) -> list[GroundTruthSample]:
        """Get samples with score 8-10."""
        return self.get_by_score_range(8, 10)

    @property
    def average_executions(self) -> list[GroundTruthSample]:
        """Get samples with score 5-7."""
        return self.get_by_score_range(5, 7)

    @property
    def poor_executions(self) -> list[GroundTruthSample]:
        """Get samples with score 1-4."""
        return self.get_by_score_range(1, 4)

    def statistics(self) -> dict[str, object]:
        """Get dataset statistics."""
        if not self.samples:
            return {"total": 0}

        scores = [s.expert_analysis.quality_score for s in self.samples]
        categories: dict[str, int] = {}
        severities: dict[str, int] = {}

        for sample in self.samples:
            cat = sample.expert_analysis.category.value
            sev = sample.expert_analysis.severity.value
            categories[cat] = categories.get(cat, 0) + 1
            severities[sev] = severities.get(sev, 0) + 1

        return {
            "total": len(self.samples),
            "score_distribution": {
                "good (8-10)": len(self.good_executions),
                "average (5-7)": len(self.average_executions),
                "poor (1-4)": len(self.poor_executions),
            },
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "categories": categories,
            "severities": severities,
        }


def load_ground_truth(path: Path | None = None) -> GroundTruthDataset:
    """Load the ground truth dataset from JSON file.

    Args:
        path: Optional path to the dataset file. Defaults to the
            built-in dataset location.

    Returns:
        GroundTruthDataset containing all samples.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
        ValueError: If the dataset fails validation.
    """
    dataset_path = path or DATASET_PATH

    if not dataset_path.exists():
        logger.warning(f"Ground truth dataset not found at {dataset_path}")
        return GroundTruthDataset()

    with dataset_path.open() as f:
        data = json.load(f)

    return GroundTruthDataset.model_validate(data)


def save_ground_truth(dataset: GroundTruthDataset, path: Path | None = None) -> None:
    """Save the ground truth dataset to JSON file.

    Args:
        dataset: The dataset to save.
        path: Optional path for the output file. Defaults to the
            built-in dataset location.
    """
    dataset_path = path or DATASET_PATH
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("w") as f:
        json.dump(dataset.model_dump(mode="json"), f, indent=2, default=str)

    logger.info(f"Saved {len(dataset.samples)} samples to {dataset_path}")


def create_sample(
    sample_id: str,
    fix_message: str,
    quality_score: int,
    key_issues: list[str] | None = None,
    expected_observations: list[str] | None = None,
    severity: IssueSeverity = IssueSeverity.NONE,
    category: IssueCategory = IssueCategory.NONE,
    annotator: str = "expert",
    confidence: AnnotatorConfidence = AnnotatorConfidence.HIGH,
    notes: str | None = None,
) -> GroundTruthSample:
    """Helper function to create a ground truth sample.

    Args:
        sample_id: Unique identifier (format: GT001).
        fix_message: Raw FIX protocol message.
        quality_score: Expert-assigned quality score (1-10).
        key_issues: List of key issues.
        expected_observations: List of expected observations.
        severity: Issue severity level.
        category: Issue category.
        annotator: Annotator identifier.
        confidence: Annotator confidence.
        notes: Optional annotation notes.

    Returns:
        GroundTruthSample ready to add to dataset.

    Raises:
        FIXParseError: If the FIX message cannot be parsed.
    """
    execution = parse_fix_message(fix_message)

    return GroundTruthSample(
        id=sample_id,
        fix_message=fix_message,
        execution=execution,
        expert_analysis=ExpertAnalysis(
            quality_score=quality_score,
            key_issues=key_issues or [],
            expected_observations=expected_observations or [],
            severity=severity,
            category=category,
        ),
        metadata=AnnotationMetadata(
            annotator=annotator,
            confidence=confidence,
            notes=notes,
        ),
    )
