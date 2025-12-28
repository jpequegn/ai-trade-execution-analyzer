"""Tests for evaluation metrics."""

from datetime import datetime

import pytest

from src.evaluation.ground_truth import ExpertAnalysis, IssueCategory, IssueSeverity
from src.evaluation.metrics import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_WEIGHTS,
    EvaluationResult,
    MetricResult,
    completeness,
    completeness_detailed,
    evaluate,
    evaluate_batch,
    extract_factual_claims,
    factual_correctness,
    factual_correctness_detailed,
    insight_accuracy,
    insight_accuracy_detailed,
    overall_quality,
    score_accuracy,
    score_accuracy_detailed,
)
from src.parsers.fix_parser import ExecutionReport
from src.parsers.models import TradeAnalysis

# --- Fixtures ---


@pytest.fixture
def sample_execution() -> ExecutionReport:
    """Sample execution report for testing."""
    return ExecutionReport(
        order_id="ORD001",
        symbol="AAPL",
        side="BUY",
        quantity=100.0,
        price=185.50,
        venue="NYSE",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        fill_type="FULL",
    )


@pytest.fixture
def sample_expert_analysis() -> ExpertAnalysis:
    """Sample expert analysis with issues and observations."""
    return ExpertAnalysis(
        quality_score=7,
        key_issues=["suboptimal venue selection", "timing at market open"],
        expected_observations=[
            "Full fill achieved",
            "Executed on NYSE",
            "Buy order for AAPL",
        ],
        severity=IssueSeverity.MEDIUM,
        category=IssueCategory.VENUE_SELECTION,
    )


@pytest.fixture
def good_ai_analysis() -> TradeAnalysis:
    """AI analysis that closely matches expert analysis."""
    return TradeAnalysis(
        quality_score=7,
        issues=["poor venue choice", "execution at market open risky"],
        observations=[
            "Complete fill on order",
            "Trade executed on NYSE exchange",
            "Bought AAPL shares",
        ],
        recommendations=["Consider dark pool for large orders"],
        confidence=0.9,
    )


@pytest.fixture
def poor_ai_analysis() -> TradeAnalysis:
    """AI analysis that misses key points."""
    return TradeAnalysis(
        quality_score=4,
        issues=["general market volatility"],
        observations=["Order was placed"],
        recommendations=[],
        confidence=0.6,
    )


@pytest.fixture
def empty_ai_analysis() -> TradeAnalysis:
    """AI analysis with no issues or observations."""
    return TradeAnalysis(
        quality_score=8,
        issues=[],
        observations=[],
        recommendations=[],
        confidence=0.5,
    )


# --- Insight Accuracy Tests ---


class TestInsightAccuracy:
    """Tests for insight_accuracy metric."""

    def test_perfect_match(
        self, good_ai_analysis: TradeAnalysis, sample_expert_analysis: ExpertAnalysis
    ) -> None:
        """Test good AI analysis gets high insight score."""
        score = insight_accuracy(good_ai_analysis, sample_expert_analysis)
        assert score >= 0.5  # Should match at least one key issue

    def test_no_matching_issues(
        self, poor_ai_analysis: TradeAnalysis, sample_expert_analysis: ExpertAnalysis
    ) -> None:
        """Test AI missing key issues gets low score."""
        score = insight_accuracy(poor_ai_analysis, sample_expert_analysis)
        assert score < 0.5  # "general market volatility" doesn't match venue/timing

    def test_empty_ai_issues(
        self, empty_ai_analysis: TradeAnalysis, sample_expert_analysis: ExpertAnalysis
    ) -> None:
        """Test AI with no issues gets zero score."""
        score = insight_accuracy(empty_ai_analysis, sample_expert_analysis)
        assert score == 0.0

    def test_no_expected_issues(self, good_ai_analysis: TradeAnalysis) -> None:
        """Test perfect score when no issues expected."""
        expert = ExpertAnalysis(
            quality_score=9,
            key_issues=[],
            expected_observations=[],
        )
        score = insight_accuracy(good_ai_analysis, expert)
        assert score == 1.0

    def test_threshold_effect(
        self, good_ai_analysis: TradeAnalysis, sample_expert_analysis: ExpertAnalysis
    ) -> None:
        """Test higher threshold reduces matches."""
        low_threshold = insight_accuracy(good_ai_analysis, sample_expert_analysis, threshold=0.3)
        high_threshold = insight_accuracy(good_ai_analysis, sample_expert_analysis, threshold=0.9)
        assert low_threshold >= high_threshold


class TestInsightAccuracyDetailed:
    """Tests for detailed insight accuracy calculation."""

    def test_detailed_result_structure(
        self, good_ai_analysis: TradeAnalysis, sample_expert_analysis: ExpertAnalysis
    ) -> None:
        """Test detailed result has expected structure."""
        result = insight_accuracy_detailed(good_ai_analysis, sample_expert_analysis)
        assert isinstance(result, MetricResult)
        assert result.name == "insight_accuracy"
        assert 0.0 <= result.score <= 1.0
        assert "expected" in result.details
        assert "found" in result.details

    def test_no_expected_issues_message(self, good_ai_analysis: TradeAnalysis) -> None:
        """Test message when no issues expected."""
        expert = ExpertAnalysis(quality_score=9, key_issues=[])
        result = insight_accuracy_detailed(good_ai_analysis, expert)
        assert result.score == 1.0
        assert "message" in result.details

    def test_unmatched_issues_tracked(
        self, poor_ai_analysis: TradeAnalysis, sample_expert_analysis: ExpertAnalysis
    ) -> None:
        """Test unmatched issues are tracked in details."""
        result = insight_accuracy_detailed(poor_ai_analysis, sample_expert_analysis)
        assert "unmatched" in result.details
        assert len(result.details["unmatched"]) > 0


# --- Completeness Tests ---


class TestCompleteness:
    """Tests for completeness metric."""

    def test_good_coverage(
        self, good_ai_analysis: TradeAnalysis, sample_expert_analysis: ExpertAnalysis
    ) -> None:
        """Test good observation coverage."""
        # Use lower threshold for fuzzy matching to capture semantic similarity
        score = completeness(good_ai_analysis, sample_expert_analysis, threshold=0.35)
        assert score >= 0.5

    def test_poor_coverage(
        self, poor_ai_analysis: TradeAnalysis, sample_expert_analysis: ExpertAnalysis
    ) -> None:
        """Test poor observation coverage."""
        score = completeness(poor_ai_analysis, sample_expert_analysis)
        assert score < 0.5

    def test_no_observations(
        self, empty_ai_analysis: TradeAnalysis, sample_expert_analysis: ExpertAnalysis
    ) -> None:
        """Test AI with no observations gets zero."""
        score = completeness(empty_ai_analysis, sample_expert_analysis)
        assert score == 0.0

    def test_no_expected_observations(self, good_ai_analysis: TradeAnalysis) -> None:
        """Test perfect score when nothing expected."""
        expert = ExpertAnalysis(quality_score=9, expected_observations=[])
        score = completeness(good_ai_analysis, expert)
        assert score == 1.0


class TestCompletenessDetailed:
    """Tests for detailed completeness calculation."""

    def test_detailed_result_structure(
        self, good_ai_analysis: TradeAnalysis, sample_expert_analysis: ExpertAnalysis
    ) -> None:
        """Test detailed result has expected structure."""
        result = completeness_detailed(good_ai_analysis, sample_expert_analysis)
        assert isinstance(result, MetricResult)
        assert result.name == "completeness"
        assert "expected" in result.details
        assert "found" in result.details


# --- Score Accuracy Tests ---


class TestScoreAccuracy:
    """Tests for score_accuracy metric."""

    def test_exact_match(self) -> None:
        """Test exact score match returns 1.0."""
        assert score_accuracy(7, 7) == 1.0

    def test_within_tolerance(self) -> None:
        """Test within tolerance returns 1.0."""
        assert score_accuracy(7, 8, tolerance=1) == 1.0
        assert score_accuracy(7, 6, tolerance=1) == 1.0

    def test_beyond_tolerance(self) -> None:
        """Test penalty for exceeding tolerance."""
        score = score_accuracy(5, 8, tolerance=1)
        assert 0.0 < score < 1.0

    def test_maximum_difference(self) -> None:
        """Test maximum difference approaches 0."""
        score = score_accuracy(1, 10, tolerance=1)
        assert score == 0.0

    def test_custom_tolerance(self) -> None:
        """Test custom tolerance value."""
        # With tolerance=2, diff of 2 should still be perfect
        assert score_accuracy(5, 7, tolerance=2) == 1.0
        assert score_accuracy(5, 8, tolerance=2) < 1.0


class TestScoreAccuracyDetailed:
    """Tests for detailed score accuracy calculation."""

    def test_detailed_result_structure(self) -> None:
        """Test detailed result has expected structure."""
        result = score_accuracy_detailed(7, 8)
        assert isinstance(result, MetricResult)
        assert result.name == "score_accuracy"
        assert result.details["ai_score"] == 7
        assert result.details["expert_score"] == 8
        assert result.details["difference"] == 1
        assert result.details["within_tolerance"] is True


# --- Factual Correctness Tests ---


class TestFactualCorrectness:
    """Tests for factual_correctness metric."""

    def test_correct_symbol_mention(self, sample_execution: ExecutionReport) -> None:
        """Test correct symbol mention is verified."""
        ai = TradeAnalysis(
            quality_score=8,
            observations=["Traded AAPL on NYSE"],
        )
        score = factual_correctness(ai, sample_execution)
        assert score == 1.0

    def test_incorrect_symbol_mention(self, sample_execution: ExecutionReport) -> None:
        """Test incorrect claims reduce score."""
        ai = TradeAnalysis(
            quality_score=8,
            observations=["Sold shares on NYSE"],  # Wrong side (actually BUY)
        )
        score = factual_correctness(ai, sample_execution)
        # Should have lower score due to incorrect side claim
        assert score < 1.0

    def test_no_observations(self, sample_execution: ExecutionReport) -> None:
        """Test no observations returns perfect score."""
        ai = TradeAnalysis(quality_score=8, observations=[])
        score = factual_correctness(ai, sample_execution)
        assert score == 1.0


class TestFactualCorrectnessDetailed:
    """Tests for detailed factual correctness calculation."""

    def test_detailed_result_structure(self, sample_execution: ExecutionReport) -> None:
        """Test detailed result has expected structure."""
        ai = TradeAnalysis(
            quality_score=8,
            observations=["Buy order for AAPL executed at $185.50"],
        )
        result = factual_correctness_detailed(ai, sample_execution)
        assert isinstance(result, MetricResult)
        assert result.name == "factual_correctness"


class TestExtractFactualClaims:
    """Tests for factual claim extraction."""

    def test_symbol_mention(self, sample_execution: ExecutionReport) -> None:
        """Test symbol mention detection."""
        claims = extract_factual_claims("Traded AAPL shares", sample_execution)
        assert claims.get("symbol_mention") is True

    def test_venue_mention(self, sample_execution: ExecutionReport) -> None:
        """Test venue mention detection."""
        claims = extract_factual_claims("Executed on NYSE", sample_execution)
        assert claims.get("venue_mention") is True

    def test_side_mention_buy(self, sample_execution: ExecutionReport) -> None:
        """Test buy side detection."""
        claims = extract_factual_claims("Bought shares", sample_execution)
        assert claims.get("side_mention") is True

    def test_side_mention_sell_incorrect(self, sample_execution: ExecutionReport) -> None:
        """Test incorrect sell claim."""
        claims = extract_factual_claims("Sold shares", sample_execution)
        assert claims.get("side_mention") is False  # Execution is BUY

    def test_full_fill_mention(self, sample_execution: ExecutionReport) -> None:
        """Test full fill detection."""
        claims = extract_factual_claims("Complete fill achieved", sample_execution)
        assert claims.get("fill_type_mention") is True

    def test_quantity_mention(self, sample_execution: ExecutionReport) -> None:
        """Test quantity detection."""
        claims = extract_factual_claims("Order for 100 shares", sample_execution)
        assert claims.get("quantity_mention") is True

    def test_price_mention(self, sample_execution: ExecutionReport) -> None:
        """Test price detection."""
        claims = extract_factual_claims("Executed at $185.50", sample_execution)
        assert claims.get("price_mention") is True


# --- Overall Quality Tests ---


class TestOverallQuality:
    """Tests for overall_quality aggregation."""

    def test_good_analysis_high_score(
        self,
        good_ai_analysis: TradeAnalysis,
        sample_expert_analysis: ExpertAnalysis,
        sample_execution: ExecutionReport,
    ) -> None:
        """Test good analysis gets high overall score."""
        score = overall_quality(good_ai_analysis, sample_expert_analysis, sample_execution)
        assert score >= 0.5

    def test_poor_analysis_low_score(
        self,
        poor_ai_analysis: TradeAnalysis,
        sample_expert_analysis: ExpertAnalysis,
        sample_execution: ExecutionReport,
    ) -> None:
        """Test poor analysis gets lower score."""
        score = overall_quality(poor_ai_analysis, sample_expert_analysis, sample_execution)
        # Poor analysis should score lower than good
        good_score = overall_quality(
            TradeAnalysis(
                quality_score=7,
                issues=["poor venue choice"],
                observations=["Full fill on NYSE"],
            ),
            sample_expert_analysis,
            sample_execution,
        )
        assert score <= good_score

    def test_custom_weights(
        self,
        good_ai_analysis: TradeAnalysis,
        sample_expert_analysis: ExpertAnalysis,
        sample_execution: ExecutionReport,
    ) -> None:
        """Test custom weight configuration."""
        custom_weights = {
            "insight_accuracy": 0.5,
            "factual_correctness": 0.2,
            "completeness": 0.2,
            "score_accuracy": 0.1,
        }
        score_custom = overall_quality(
            good_ai_analysis,
            sample_expert_analysis,
            sample_execution,
            weights=custom_weights,
        )
        score_default = overall_quality(good_ai_analysis, sample_expert_analysis, sample_execution)
        # Different weights should produce different scores
        # (unless all individual scores are exactly equal)
        assert isinstance(score_custom, float)
        assert isinstance(score_default, float)


# --- Evaluate Tests ---


class TestEvaluate:
    """Tests for complete evaluation."""

    def test_evaluation_result_structure(
        self,
        good_ai_analysis: TradeAnalysis,
        sample_expert_analysis: ExpertAnalysis,
        sample_execution: ExecutionReport,
    ) -> None:
        """Test evaluation result has all expected fields."""
        result = evaluate(good_ai_analysis, sample_expert_analysis, sample_execution)
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.insight_accuracy <= 1.0
        assert 0.0 <= result.factual_correctness <= 1.0
        assert 0.0 <= result.completeness <= 1.0
        assert 0.0 <= result.score_accuracy <= 1.0
        assert 0.0 <= result.overall_score <= 1.0
        assert "insight_accuracy" in result.metrics
        assert "factual_correctness" in result.metrics
        assert "completeness" in result.metrics
        assert "score_accuracy" in result.metrics

    def test_metrics_contain_details(
        self,
        good_ai_analysis: TradeAnalysis,
        sample_expert_analysis: ExpertAnalysis,
        sample_execution: ExecutionReport,
    ) -> None:
        """Test metric results contain details."""
        result = evaluate(good_ai_analysis, sample_expert_analysis, sample_execution)
        for metric_name, metric_result in result.metrics.items():
            assert isinstance(metric_result, MetricResult)
            assert metric_result.name == metric_name


# --- Evaluate Batch Tests ---


class TestEvaluateBatch:
    """Tests for batch evaluation."""

    def test_empty_batch(self) -> None:
        """Test empty batch returns minimal result."""
        result = evaluate_batch([])
        assert result["count"] == 0

    def test_single_sample_batch(
        self,
        good_ai_analysis: TradeAnalysis,
        sample_expert_analysis: ExpertAnalysis,
        sample_execution: ExecutionReport,
    ) -> None:
        """Test batch with single sample."""
        samples = [(good_ai_analysis, sample_expert_analysis, sample_execution)]
        result = evaluate_batch(samples)
        assert result["count"] == 1
        assert "insight_accuracy" in result
        assert "overall" in result

    def test_multiple_samples_batch(
        self,
        good_ai_analysis: TradeAnalysis,
        poor_ai_analysis: TradeAnalysis,
        sample_expert_analysis: ExpertAnalysis,
        sample_execution: ExecutionReport,
    ) -> None:
        """Test batch with multiple samples."""
        samples = [
            (good_ai_analysis, sample_expert_analysis, sample_execution),
            (poor_ai_analysis, sample_expert_analysis, sample_execution),
        ]
        result = evaluate_batch(samples)
        assert result["count"] == 2
        # Mean should be between min and max
        insight = result["insight_accuracy"]
        assert isinstance(insight, dict)
        assert insight["min"] <= insight["mean"] <= insight["max"]

    def test_aggregate_statistics(
        self,
        good_ai_analysis: TradeAnalysis,
        sample_expert_analysis: ExpertAnalysis,
        sample_execution: ExecutionReport,
    ) -> None:
        """Test aggregate statistics are calculated."""
        samples = [(good_ai_analysis, sample_expert_analysis, sample_execution)]
        result = evaluate_batch(samples)
        for metric_name in [
            "insight_accuracy",
            "factual_correctness",
            "completeness",
            "score_accuracy",
            "overall",
        ]:
            metric_stats = result[metric_name]
            assert isinstance(metric_stats, dict)
            assert "mean" in metric_stats
            assert "min" in metric_stats
            assert "max" in metric_stats


# --- Constants Tests ---


class TestConstants:
    """Tests for module constants."""

    def test_default_threshold(self) -> None:
        """Test default match threshold value."""
        assert DEFAULT_MATCH_THRESHOLD == 0.5

    def test_default_weights_sum(self) -> None:
        """Test default weights sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_default_weights_keys(self) -> None:
        """Test default weights have expected keys."""
        expected_keys = {
            "insight_accuracy",
            "factual_correctness",
            "completeness",
            "score_accuracy",
        }
        assert set(DEFAULT_WEIGHTS.keys()) == expected_keys


# --- Edge Cases ---


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_strings_in_lists(self) -> None:
        """Test handling of empty strings in lists."""
        ai = TradeAnalysis(quality_score=5, issues=["", "real issue"])
        expert = ExpertAnalysis(quality_score=5, key_issues=["real issue"])
        score = insight_accuracy(ai, expert)
        assert score > 0.0

    def test_very_long_observation(self, sample_execution: ExecutionReport) -> None:
        """Test handling of very long observation text."""
        long_obs = "This is a very long observation. " * 100 + "AAPL"
        ai = TradeAnalysis(quality_score=8, observations=[long_obs])
        score = factual_correctness(ai, sample_execution)
        assert score >= 0.0

    def test_score_at_boundaries(self) -> None:
        """Test score accuracy at score boundaries."""
        assert score_accuracy(1, 1) == 1.0
        assert score_accuracy(10, 10) == 1.0
        assert score_accuracy(1, 10) == 0.0
        assert score_accuracy(10, 1) == 0.0
