"""Tests for ground truth dataset module."""

from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from src.evaluation.ground_truth import (
    AnnotationMetadata,
    AnnotatorConfidence,
    ExpertAnalysis,
    GroundTruthDataset,
    GroundTruthSample,
    IssueCategory,
    IssueSeverity,
    create_sample,
    load_ground_truth,
    save_ground_truth,
)
from src.parsers.fix_parser import ExecutionReport


class TestExpertAnalysis:
    """Tests for ExpertAnalysis model."""

    def test_valid_analysis(self) -> None:
        """Test creating a valid expert analysis."""
        analysis = ExpertAnalysis(
            quality_score=8,
            key_issues=["Minor timing issue"],
            expected_observations=["Full fill achieved"],
            severity=IssueSeverity.LOW,
            category=IssueCategory.TIMING,
        )
        assert analysis.quality_score == 8
        assert len(analysis.key_issues) == 1
        assert analysis.severity == IssueSeverity.LOW

    def test_score_validation(self) -> None:
        """Test quality score validation."""
        with pytest.raises(ValueError):
            ExpertAnalysis(quality_score=0)  # Below minimum

        with pytest.raises(ValueError):
            ExpertAnalysis(quality_score=11)  # Above maximum

    def test_empty_lists_default(self) -> None:
        """Test that list fields default to empty lists."""
        analysis = ExpertAnalysis(quality_score=5)
        assert analysis.key_issues == []
        assert analysis.expected_observations == []

    def test_none_lists_converted(self) -> None:
        """Test that None values are converted to empty lists."""
        analysis = ExpertAnalysis(
            quality_score=5,
            key_issues=None,
            expected_observations=None,
        )
        assert analysis.key_issues == []
        assert analysis.expected_observations == []


class TestAnnotationMetadata:
    """Tests for AnnotationMetadata model."""

    def test_default_values(self) -> None:
        """Test default metadata values."""
        metadata = AnnotationMetadata()
        assert metadata.annotator == "expert"
        assert metadata.confidence == AnnotatorConfidence.HIGH
        assert metadata.notes is None
        assert metadata.annotation_date == date.today()

    def test_custom_values(self) -> None:
        """Test custom metadata values."""
        metadata = AnnotationMetadata(
            annotator="john_doe",
            annotation_date=date(2024, 1, 15),
            confidence=AnnotatorConfidence.MEDIUM,
            notes="Edge case scenario",
        )
        assert metadata.annotator == "john_doe"
        assert metadata.annotation_date == date(2024, 1, 15)
        assert metadata.notes == "Edge case scenario"


class TestGroundTruthSample:
    """Tests for GroundTruthSample model."""

    @pytest.fixture
    def sample_execution(self) -> ExecutionReport:
        """Create a sample execution."""
        return ExecutionReport(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            price=185.50,
            venue="NYSE",
            timestamp="2024-01-15T10:30:00",
            fill_type="FULL",
        )

    def test_valid_sample(self, sample_execution: ExecutionReport) -> None:
        """Test creating a valid sample."""
        sample = GroundTruthSample(
            id="GT001",
            fix_message="8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|31=185.50|30=NYSE|60=20240115-10:30:00.000|39=2",
            execution=sample_execution,
            expert_analysis=ExpertAnalysis(quality_score=9),
        )
        assert sample.id == "GT001"
        assert sample.execution.symbol == "AAPL"

    def test_invalid_id_format(self, sample_execution: ExecutionReport) -> None:
        """Test that invalid ID format raises error."""
        with pytest.raises(ValueError):
            GroundTruthSample(
                id="INVALID",  # Must be GT followed by 3 digits
                fix_message="8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|31=185.50|30=NYSE|60=20240115-10:30:00.000|39=2",
                execution=sample_execution,
                expert_analysis=ExpertAnalysis(quality_score=9),
            )


class TestGroundTruthDataset:
    """Tests for GroundTruthDataset model."""

    @pytest.fixture
    def sample_dataset(self) -> GroundTruthDataset:
        """Create a sample dataset with multiple samples."""
        samples = []
        for i, (score, cat, sev) in enumerate(
            [
                (9, IssueCategory.NONE, IssueSeverity.NONE),
                (7, IssueCategory.TIMING, IssueSeverity.LOW),
                (5, IssueCategory.VENUE_SELECTION, IssueSeverity.MEDIUM),
                (3, IssueCategory.FILL_QUALITY, IssueSeverity.HIGH),
                (8, IssueCategory.NONE, IssueSeverity.NONE),
            ]
        ):
            execution = ExecutionReport(
                order_id=f"ORD00{i + 1}",
                symbol=["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"][i],
                side="BUY",
                quantity=100.0,
                price=100.0 + i * 10,
                venue="NYSE",
                timestamp="2024-01-15T10:30:00",
                fill_type="FULL",
            )
            samples.append(
                GroundTruthSample(
                    id=f"GT00{i + 1}",
                    fix_message=f"8=FIX.4.4|35=8|37=ORD00{i + 1}|55={execution.symbol}|54=1|32=100|31={execution.price}|30=NYSE|60=20240115-10:30:00.000|39=2",
                    execution=execution,
                    expert_analysis=ExpertAnalysis(
                        quality_score=score,
                        severity=sev,
                        category=cat,
                    ),
                )
            )
        return GroundTruthDataset(samples=samples)

    def test_get_by_id(self, sample_dataset: GroundTruthDataset) -> None:
        """Test getting sample by ID."""
        sample = sample_dataset.get_by_id("GT001")
        assert sample is not None
        assert sample.execution.symbol == "AAPL"

        assert sample_dataset.get_by_id("GT999") is None

    def test_get_by_score_range(self, sample_dataset: GroundTruthDataset) -> None:
        """Test filtering by score range."""
        good = sample_dataset.get_by_score_range(8, 10)
        assert len(good) == 2  # Scores 9 and 8

        poor = sample_dataset.get_by_score_range(1, 4)
        assert len(poor) == 1  # Score 3

    def test_get_by_category(self, sample_dataset: GroundTruthDataset) -> None:
        """Test filtering by category."""
        timing = sample_dataset.get_by_category(IssueCategory.TIMING)
        assert len(timing) == 1

        none = sample_dataset.get_by_category(IssueCategory.NONE)
        assert len(none) == 2

    def test_get_by_severity(self, sample_dataset: GroundTruthDataset) -> None:
        """Test filtering by severity."""
        high = sample_dataset.get_by_severity(IssueSeverity.HIGH)
        assert len(high) == 1

    def test_execution_properties(self, sample_dataset: GroundTruthDataset) -> None:
        """Test convenience properties."""
        assert len(sample_dataset.good_executions) == 2
        assert len(sample_dataset.average_executions) == 2
        assert len(sample_dataset.poor_executions) == 1

    def test_statistics(self, sample_dataset: GroundTruthDataset) -> None:
        """Test statistics calculation."""
        stats = sample_dataset.statistics()
        assert stats["total"] == 5
        assert stats["average_score"] == 6.4
        assert stats["min_score"] == 3
        assert stats["max_score"] == 9
        assert stats["score_distribution"]["good (8-10)"] == 2

    def test_empty_dataset_stats(self) -> None:
        """Test statistics for empty dataset."""
        dataset = GroundTruthDataset()
        stats = dataset.statistics()
        assert stats["total"] == 0


class TestCreateSample:
    """Tests for create_sample helper function."""

    def test_create_sample_basic(self) -> None:
        """Test creating a sample from FIX message."""
        sample = create_sample(
            sample_id="GT001",
            fix_message="8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|31=185.50|30=NYSE|60=20240115-10:30:00.000|39=2",
            quality_score=9,
        )
        assert sample.id == "GT001"
        assert sample.execution.symbol == "AAPL"
        assert sample.execution.order_id == "ORD001"
        assert sample.expert_analysis.quality_score == 9

    def test_create_sample_with_analysis(self) -> None:
        """Test creating a sample with full analysis details."""
        sample = create_sample(
            sample_id="GT002",
            fix_message="8=FIX.4.4|35=8|37=ORD002|55=MSFT|54=2|32=50|31=400.00|30=NASDAQ|60=20240115-11:00:00.000|39=2",
            quality_score=6,
            key_issues=["Timing during lunch hour"],
            expected_observations=["NASDAQ execution", "Sell order"],
            severity=IssueSeverity.MEDIUM,
            category=IssueCategory.TIMING,
            annotator="test_user",
            confidence=AnnotatorConfidence.MEDIUM,
            notes="Test annotation",
        )
        assert sample.expert_analysis.quality_score == 6
        assert len(sample.expert_analysis.key_issues) == 1
        assert sample.expert_analysis.severity == IssueSeverity.MEDIUM
        assert sample.metadata.annotator == "test_user"


class TestLoadSaveGroundTruth:
    """Tests for load/save functions."""

    def test_load_nonexistent_file(self) -> None:
        """Test loading from nonexistent path returns empty dataset."""
        dataset = load_ground_truth(Path("/nonexistent/path.json"))
        assert len(dataset.samples) == 0

    def test_save_and_load_roundtrip(self) -> None:
        """Test saving and loading a dataset."""
        # Create a dataset
        sample = create_sample(
            sample_id="GT001",
            fix_message="8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|31=185.50|30=NYSE|60=20240115-10:30:00.000|39=2",
            quality_score=9,
            key_issues=[],
            expected_observations=["Full fill"],
        )
        dataset = GroundTruthDataset(samples=[sample])

        # Save to temp file
        with NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_ground_truth(dataset, temp_path)

            # Load and verify
            loaded = load_ground_truth(temp_path)
            assert len(loaded.samples) == 1
            assert loaded.samples[0].id == "GT001"
            assert loaded.samples[0].execution.symbol == "AAPL"
            assert loaded.samples[0].expert_analysis.quality_score == 9
        finally:
            temp_path.unlink()


class TestLoadBuiltinDataset:
    """Tests for loading the built-in ground truth dataset."""

    def test_load_builtin_dataset(self) -> None:
        """Test loading the built-in dataset."""
        dataset = load_ground_truth()
        assert len(dataset.samples) >= 50, "Expected at least 50 samples"

    def test_dataset_has_variety(self) -> None:
        """Test that dataset has variety of scores."""
        dataset = load_ground_truth()
        stats = dataset.statistics()

        assert stats["score_distribution"]["good (8-10)"] >= 10
        assert stats["score_distribution"]["average (5-7)"] >= 10
        assert stats["score_distribution"]["poor (1-4)"] >= 10

    def test_all_samples_have_required_fields(self) -> None:
        """Test that all samples have required fields."""
        dataset = load_ground_truth()

        for sample in dataset.samples:
            assert sample.id is not None
            assert sample.fix_message is not None
            assert sample.execution is not None
            assert sample.expert_analysis is not None
            assert 1 <= sample.expert_analysis.quality_score <= 10
