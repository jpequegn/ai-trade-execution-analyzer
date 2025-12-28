"""Evaluation metrics for measuring AI analysis quality.

This module provides quantitative metrics to compare AI-generated trade
analysis against expert ground truth annotations.

Example:
    >>> from src.evaluation.metrics import insight_accuracy, overall_quality
    >>> from src.parsers.models import TradeAnalysis
    >>> from src.evaluation import ExpertAnalysis
    >>>
    >>> ai_analysis = TradeAnalysis(
    ...     quality_score=7,
    ...     issues=["poor venue selection"],
    ...     observations=["Full fill on NYSE"]
    ... )
    >>> ground_truth = ExpertAnalysis(
    ...     quality_score=7,
    ...     key_issues=["suboptimal venue choice"],
    ...     expected_observations=["Full fill achieved"]
    ... )
    >>> insight_accuracy(ai_analysis, ground_truth)
    1.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.evaluation.matching import count_matches

if TYPE_CHECKING:
    from src.evaluation.ground_truth import ExpertAnalysis
    from src.parsers.fix_parser import ExecutionReport
    from src.parsers.models import TradeAnalysis

# Default match threshold for fuzzy comparisons
DEFAULT_MATCH_THRESHOLD = 0.5

# Default weights for overall quality calculation
DEFAULT_WEIGHTS = {
    "insight_accuracy": 0.35,
    "factual_correctness": 0.30,
    "completeness": 0.20,
    "score_accuracy": 0.15,
}


@dataclass
class MetricResult:
    """Result of a single metric calculation.

    Attributes:
        name: Name of the metric.
        score: Calculated score (0.0-1.0).
        details: Additional details about the calculation.
    """

    name: str
    score: float
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result with all metrics.

    Attributes:
        insight_accuracy: Score for identifying key issues.
        factual_correctness: Score for factual accuracy.
        completeness: Score for observation coverage.
        score_accuracy: Score for quality score alignment.
        overall_score: Weighted overall quality score.
        metrics: Individual metric results with details.
    """

    insight_accuracy: float
    factual_correctness: float
    completeness: float
    score_accuracy: float
    overall_score: float
    metrics: dict[str, MetricResult] = field(default_factory=dict)


def insight_accuracy(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> float:
    """Measure if AI identified the key issues from ground truth.

    This metric evaluates whether the AI correctly identified the important
    issues that an expert would flag in a trade execution.

    Args:
        ai_analysis: The AI-generated trade analysis.
        ground_truth: The expert-annotated ground truth.
        threshold: Minimum fuzzy match score for a match.

    Returns:
        Score between 0.0 and 1.0.
        - 1.0 = All key issues identified
        - 0.0 = No key issues identified

    Example:
        >>> insight_accuracy(ai_analysis, ground_truth)
        0.8  # 4 of 5 key issues identified
    """
    key_issues = ground_truth.key_issues

    # If no key issues expected, perfect score (nothing to miss)
    if not key_issues:
        return 1.0

    ai_issues = ai_analysis.issues

    # If no AI issues found but issues expected, score is 0
    if not ai_issues:
        return 0.0

    matched, _ = count_matches(key_issues, ai_issues, threshold)
    return matched / len(key_issues)


def insight_accuracy_detailed(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> MetricResult:
    """Detailed insight accuracy calculation with match information.

    Args:
        ai_analysis: The AI-generated trade analysis.
        ground_truth: The expert-annotated ground truth.
        threshold: Minimum fuzzy match score for a match.

    Returns:
        MetricResult with score and detailed match information.
    """
    key_issues = ground_truth.key_issues

    if not key_issues:
        return MetricResult(
            name="insight_accuracy",
            score=1.0,
            details={"message": "No key issues in ground truth"},
        )

    ai_issues = ai_analysis.issues

    if not ai_issues:
        return MetricResult(
            name="insight_accuracy",
            score=0.0,
            details={
                "expected": len(key_issues),
                "found": 0,
                "unmatched": key_issues,
            },
        )

    matched, match_details = count_matches(key_issues, ai_issues, threshold)
    score = matched / len(key_issues)

    return MetricResult(
        name="insight_accuracy",
        score=score,
        details={
            "expected": len(key_issues),
            "found": matched,
            "matches": [
                {"expected": exp, "matched": match, "score": sc} for exp, match, sc in match_details
            ],
            "unmatched": [exp for exp, match, _ in match_details if match is None],
        },
    )


def completeness(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> float:
    """Measure if AI found all expected observations.

    This metric evaluates the coverage of the AI's observations against
    what an expert would expect to see mentioned.

    Args:
        ai_analysis: The AI-generated trade analysis.
        ground_truth: The expert-annotated ground truth.
        threshold: Minimum fuzzy match score for a match.

    Returns:
        Score between 0.0 and 1.0.
        - 1.0 = All expected observations found
        - 0.0 = No expected observations found

    Example:
        >>> completeness(ai_analysis, ground_truth)
        0.75  # 3 of 4 expected observations found
    """
    expected_obs = ground_truth.expected_observations

    # If no observations expected, perfect score
    if not expected_obs:
        return 1.0

    ai_observations = ai_analysis.observations

    # If no AI observations but some expected, score is 0
    if not ai_observations:
        return 0.0

    matched, _ = count_matches(expected_obs, ai_observations, threshold)
    return matched / len(expected_obs)


def completeness_detailed(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> MetricResult:
    """Detailed completeness calculation with match information.

    Args:
        ai_analysis: The AI-generated trade analysis.
        ground_truth: The expert-annotated ground truth.
        threshold: Minimum fuzzy match score for a match.

    Returns:
        MetricResult with score and detailed match information.
    """
    expected_obs = ground_truth.expected_observations

    if not expected_obs:
        return MetricResult(
            name="completeness",
            score=1.0,
            details={"message": "No observations expected in ground truth"},
        )

    ai_observations = ai_analysis.observations

    if not ai_observations:
        return MetricResult(
            name="completeness",
            score=0.0,
            details={
                "expected": len(expected_obs),
                "found": 0,
                "unmatched": expected_obs,
            },
        )

    matched, match_details = count_matches(expected_obs, ai_observations, threshold)
    score = matched / len(expected_obs)

    return MetricResult(
        name="completeness",
        score=score,
        details={
            "expected": len(expected_obs),
            "found": matched,
            "matches": [
                {"expected": exp, "matched": match, "score": sc} for exp, match, sc in match_details
            ],
            "unmatched": [exp for exp, match, _ in match_details if match is None],
        },
    )


def score_accuracy(
    ai_score: int,
    expert_score: int,
    tolerance: int = 1,
    max_diff: int = 9,
) -> float:
    """Measure if AI score matches expert score within tolerance.

    This metric evaluates how close the AI's quality score is to the
    expert's assessment.

    Args:
        ai_score: The AI-assigned quality score (1-10).
        expert_score: The expert-assigned quality score (1-10).
        tolerance: Score difference allowed for perfect score (default 1).
        max_diff: Maximum possible difference for scaling (default 9).

    Returns:
        Score between 0.0 and 1.0.
        - 1.0 = Within tolerance
        - Scaled penalty for larger differences

    Example:
        >>> score_accuracy(7, 8)  # Within tolerance
        1.0
        >>> score_accuracy(5, 8)  # 3 points off
        0.6
    """
    diff = abs(ai_score - expert_score)

    if diff <= tolerance:
        return 1.0

    # Scale the penalty based on how far beyond tolerance
    excess = diff - tolerance
    penalty_per_point = 1.0 / (max_diff - tolerance)
    return max(0.0, 1.0 - excess * penalty_per_point)


def score_accuracy_detailed(
    ai_score: int,
    expert_score: int,
    tolerance: int = 1,
) -> MetricResult:
    """Detailed score accuracy calculation.

    Args:
        ai_score: The AI-assigned quality score (1-10).
        expert_score: The expert-assigned quality score (1-10).
        tolerance: Score difference allowed for perfect score.

    Returns:
        MetricResult with score and calculation details.
    """
    diff = abs(ai_score - expert_score)
    score = score_accuracy(ai_score, expert_score, tolerance)

    return MetricResult(
        name="score_accuracy",
        score=score,
        details={
            "ai_score": ai_score,
            "expert_score": expert_score,
            "difference": diff,
            "within_tolerance": diff <= tolerance,
            "tolerance": tolerance,
        },
    )


def factual_correctness(
    ai_analysis: TradeAnalysis,
    execution: ExecutionReport,
) -> float:
    """Measure if AI observations match actual execution data.

    This metric verifies that the AI's factual claims about the trade
    execution are accurate and not hallucinated.

    Args:
        ai_analysis: The AI-generated trade analysis.
        execution: The actual execution report data.

    Returns:
        Score between 0.0 and 1.0.
        - 1.0 = All factual claims verified
        - Penalty for each incorrect claim

    Example:
        >>> factual_correctness(ai_analysis, execution)
        0.9  # 9 of 10 claims correct
    """
    observations = ai_analysis.observations

    if not observations:
        return 1.0  # No claims to verify

    correct = 0
    total = 0

    for obs in observations:
        claims = extract_factual_claims(obs, execution)
        for claim_correct in claims.values():
            total += 1
            if claim_correct:
                correct += 1

    if total == 0:
        return 1.0  # No verifiable claims

    return correct / total


def factual_correctness_detailed(
    ai_analysis: TradeAnalysis,
    execution: ExecutionReport,
) -> MetricResult:
    """Detailed factual correctness calculation.

    Args:
        ai_analysis: The AI-generated trade analysis.
        execution: The actual execution report data.

    Returns:
        MetricResult with score and claim verification details.
    """
    observations = ai_analysis.observations

    if not observations:
        return MetricResult(
            name="factual_correctness",
            score=1.0,
            details={"message": "No observations to verify"},
        )

    all_claims: list[dict[str, object]] = []
    correct = 0
    total = 0

    for obs in observations:
        claims = extract_factual_claims(obs, execution)
        for claim_type, is_correct in claims.items():
            total += 1
            if is_correct:
                correct += 1
            all_claims.append(
                {
                    "observation": obs,
                    "claim_type": claim_type,
                    "correct": is_correct,
                }
            )

    if total == 0:
        return MetricResult(
            name="factual_correctness",
            score=1.0,
            details={"message": "No verifiable factual claims"},
        )

    score = correct / total

    return MetricResult(
        name="factual_correctness",
        score=score,
        details={
            "total_claims": total,
            "correct_claims": correct,
            "claims": all_claims,
        },
    )


def extract_factual_claims(
    observation: str,
    execution: ExecutionReport,
) -> dict[str, bool]:
    """Extract and verify factual claims from an observation.

    Args:
        observation: The observation text to analyze.
        execution: The execution data to verify against.

    Returns:
        Dictionary of claim_type -> is_correct mappings.
    """
    claims: dict[str, bool] = {}
    obs_lower = observation.lower()

    # Check symbol mentions
    if execution.symbol.lower() in obs_lower:
        claims["symbol_mention"] = True

    # Check venue mentions
    if execution.venue.lower() in obs_lower:
        claims["venue_mention"] = True

    # Check side mentions
    side_terms = {
        "BUY": ["buy", "bought", "purchase", "long"],
        "SELL": ["sell", "sold", "sale", "short"],
    }
    for side, terms in side_terms.items():
        if any(term in obs_lower for term in terms):
            claims["side_mention"] = execution.side == side

    # Check fill type mentions
    if "full" in obs_lower or "complete" in obs_lower:
        claims["fill_type_mention"] = execution.fill_type == "FULL"
    elif "partial" in obs_lower:
        claims["fill_type_mention"] = execution.fill_type == "PARTIAL"

    # Check quantity mentions
    qty_patterns = [
        rf"\b{int(execution.quantity)}\b",
        rf"\b{int(execution.quantity):,}\b",
    ]
    for pattern in qty_patterns:
        if re.search(pattern, observation):
            claims["quantity_mention"] = True
            break

    # Check price mentions
    price_patterns = [
        rf"\${execution.price:.2f}",
        rf"\b{execution.price:.2f}\b",
        rf"\${execution.price:,.2f}",
    ]
    for pattern in price_patterns:
        if re.search(pattern, observation):
            claims["price_mention"] = True
            break

    return claims


def overall_quality(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
    execution: ExecutionReport,
    weights: dict[str, float] | None = None,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> float:
    """Calculate overall quality score from all metrics.

    Args:
        ai_analysis: The AI-generated trade analysis.
        ground_truth: The expert-annotated ground truth.
        execution: The actual execution report data.
        weights: Optional custom weights for each metric.
        threshold: Match threshold for fuzzy comparisons.

    Returns:
        Weighted overall quality score between 0.0 and 1.0.

    Example:
        >>> overall_quality(ai_analysis, ground_truth, execution)
        0.82
    """
    weights = weights or DEFAULT_WEIGHTS

    insight = insight_accuracy(ai_analysis, ground_truth, threshold)
    factual = factual_correctness(ai_analysis, execution)
    complete = completeness(ai_analysis, ground_truth, threshold)
    score_acc = score_accuracy(ai_analysis.quality_score, ground_truth.quality_score)

    total = (
        weights.get("insight_accuracy", 0.35) * insight
        + weights.get("factual_correctness", 0.30) * factual
        + weights.get("completeness", 0.20) * complete
        + weights.get("score_accuracy", 0.15) * score_acc
    )

    return total


def evaluate(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
    execution: ExecutionReport,
    weights: dict[str, float] | None = None,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> EvaluationResult:
    """Perform complete evaluation with all metrics and details.

    Args:
        ai_analysis: The AI-generated trade analysis.
        ground_truth: The expert-annotated ground truth.
        execution: The actual execution report data.
        weights: Optional custom weights for each metric.
        threshold: Match threshold for fuzzy comparisons.

    Returns:
        Complete EvaluationResult with all metrics and details.

    Example:
        >>> result = evaluate(ai_analysis, ground_truth, execution)
        >>> print(f"Overall: {result.overall_score:.2%}")
        >>> print(f"Insight accuracy: {result.insight_accuracy:.2%}")
    """
    weights = weights or DEFAULT_WEIGHTS

    # Calculate detailed metrics
    insight_result = insight_accuracy_detailed(ai_analysis, ground_truth, threshold)
    factual_result = factual_correctness_detailed(ai_analysis, execution)
    complete_result = completeness_detailed(ai_analysis, ground_truth, threshold)
    score_result = score_accuracy_detailed(ai_analysis.quality_score, ground_truth.quality_score)

    # Calculate overall score
    overall = (
        weights.get("insight_accuracy", 0.35) * insight_result.score
        + weights.get("factual_correctness", 0.30) * factual_result.score
        + weights.get("completeness", 0.20) * complete_result.score
        + weights.get("score_accuracy", 0.15) * score_result.score
    )

    return EvaluationResult(
        insight_accuracy=insight_result.score,
        factual_correctness=factual_result.score,
        completeness=complete_result.score,
        score_accuracy=score_result.score,
        overall_score=overall,
        metrics={
            "insight_accuracy": insight_result,
            "factual_correctness": factual_result,
            "completeness": complete_result,
            "score_accuracy": score_result,
        },
    )


def evaluate_batch(
    results: list[tuple[TradeAnalysis, ExpertAnalysis, ExecutionReport]],
    weights: dict[str, float] | None = None,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict[str, object]:
    """Evaluate multiple samples and aggregate statistics.

    Args:
        results: List of (ai_analysis, ground_truth, execution) tuples.
        weights: Optional custom weights for each metric.
        threshold: Match threshold for fuzzy comparisons.

    Returns:
        Aggregated statistics across all samples.
    """
    if not results:
        return {"count": 0}

    scores: dict[str, list[float]] = {
        "insight_accuracy": [],
        "factual_correctness": [],
        "completeness": [],
        "score_accuracy": [],
        "overall": [],
    }

    for ai_analysis, ground_truth, execution in results:
        eval_result = evaluate(ai_analysis, ground_truth, execution, weights, threshold)
        scores["insight_accuracy"].append(eval_result.insight_accuracy)
        scores["factual_correctness"].append(eval_result.factual_correctness)
        scores["completeness"].append(eval_result.completeness)
        scores["score_accuracy"].append(eval_result.score_accuracy)
        scores["overall"].append(eval_result.overall_score)

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "count": len(results),
        "insight_accuracy": {
            "mean": avg(scores["insight_accuracy"]),
            "min": min(scores["insight_accuracy"]),
            "max": max(scores["insight_accuracy"]),
        },
        "factual_correctness": {
            "mean": avg(scores["factual_correctness"]),
            "min": min(scores["factual_correctness"]),
            "max": max(scores["factual_correctness"]),
        },
        "completeness": {
            "mean": avg(scores["completeness"]),
            "min": min(scores["completeness"]),
            "max": max(scores["completeness"]),
        },
        "score_accuracy": {
            "mean": avg(scores["score_accuracy"]),
            "min": min(scores["score_accuracy"]),
            "max": max(scores["score_accuracy"]),
        },
        "overall": {
            "mean": avg(scores["overall"]),
            "min": min(scores["overall"]),
            "max": max(scores["overall"]),
        },
    }
