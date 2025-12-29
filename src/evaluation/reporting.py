"""Reporting utilities for evaluation results.

This module provides functions to generate markdown and JSON reports
from evaluation results, with support for run comparisons.

Example:
    >>> from src.evaluation.reporting import generate_markdown_report
    >>> from src.evaluation.runner import EvalResults
    >>> report = generate_markdown_report(results)
    >>> print(report)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.evaluation.runner import EvalResults, SingleEvalResult


def generate_markdown_report(
    results: EvalResults,
    include_failed_details: bool = True,
    include_all_samples: bool = False,
    max_failed_samples: int = 20,
) -> str:
    """Generate a markdown report from evaluation results.

    Args:
        results: The evaluation results to report.
        include_failed_details: Whether to include details of failed samples.
        include_all_samples: Whether to include all sample results (not just failed).
        max_failed_samples: Maximum number of failed samples to show in detail.

    Returns:
        Markdown-formatted report string.

    Example:
        >>> report = generate_markdown_report(results)
        >>> Path("report.md").write_text(report)
    """
    lines: list[str] = []

    # Header
    timestamp_str = results.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# Evaluation Report - {timestamp_str}")
    lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Run ID**: `{results.run_id}`")
    lines.append(f"- **Dataset**: {results.dataset_path or 'Default'}")
    lines.append(f"- **Total Samples**: {results.total_samples}")
    lines.append(f"- **Passed**: {results.passed} ({results.pass_rate:.1%})")
    lines.append(f"- **Failed**: {results.failed} ({1 - results.pass_rate:.1%})")
    if results.error_count > 0:
        lines.append(f"- **Errors**: {results.error_count}")
    lines.append(f"- **Duration**: {results.duration_seconds:.1f}s")
    lines.append("")

    # Aggregate metrics table
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Score | Target | Status |")
    lines.append("|--------|-------|--------|--------|")

    metric_display_names = {
        "insight_accuracy": "Insight Accuracy",
        "factual_correctness": "Factual Correctness",
        "completeness": "Completeness",
        "score_accuracy": "Score Accuracy",
        "overall": "Overall",
    }

    for metric_key, display_name in metric_display_names.items():
        score = results.aggregate_metrics.get(metric_key, 0.0)
        threshold = results.thresholds.get(metric_key, 0.0)
        status = "✅" if score >= threshold else "❌"
        lines.append(f"| {display_name} | {score:.2%} | {threshold:.2%} | {status} |")

    lines.append("")

    # Failed samples section
    if include_failed_details:
        failed_samples = results.get_failed_samples()
        if failed_samples:
            lines.append("## Failed Samples")
            lines.append("")

            for sample in failed_samples[:max_failed_samples]:
                lines.append(_format_failed_sample(sample, results.thresholds))
                lines.append("")

            if len(failed_samples) > max_failed_samples:
                remaining = len(failed_samples) - max_failed_samples
                lines.append(f"*... and {remaining} more failed samples*")
                lines.append("")

    # All samples section (optional)
    if include_all_samples:
        lines.append("## All Sample Results")
        lines.append("")
        lines.append("| Sample ID | Overall | Insight | Factual | Complete | Score | Status |")
        lines.append("|-----------|---------|---------|---------|----------|-------|--------|")

        for sample in sorted(results.results, key=lambda s: s.sample_id):
            status = "✅" if sample.passed else "❌"
            if sample.errors:
                status = "⚠️"
            insight = sample.metrics.get("insight_accuracy", 0.0)
            factual = sample.metrics.get("factual_correctness", 0.0)
            complete = sample.metrics.get("completeness", 0.0)
            score_acc = sample.metrics.get("score_accuracy", 0.0)
            lines.append(
                f"| {sample.sample_id} | {sample.overall_score:.2%} | "
                f"{insight:.2%} | {factual:.2%} | {complete:.2%} | "
                f"{score_acc:.2%} | {status} |"
            )

        lines.append("")

    # Thresholds section
    lines.append("## Quality Thresholds")
    lines.append("")
    lines.append("| Metric | Threshold |")
    lines.append("|--------|-----------|")
    for metric_key, display_name in metric_display_names.items():
        threshold = results.thresholds.get(metric_key, 0.0)
        lines.append(f"| {display_name} | {threshold:.2%} |")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(lines)


def _format_failed_sample(sample: SingleEvalResult, thresholds: dict[str, float]) -> str:
    """Format a single failed sample for the report.

    Args:
        sample: The failed sample result.
        thresholds: Quality thresholds for context.

    Returns:
        Markdown-formatted sample details.
    """
    lines: list[str] = []

    # Sample header with primary failure reason
    failure_reasons = _get_failure_reasons(sample, thresholds)
    reason_str = ", ".join(failure_reasons[:2])  # Show top 2 reasons
    lines.append(f"### {sample.sample_id}: {reason_str}")

    # Metrics breakdown
    lines.append("")
    lines.append("**Metrics:**")
    for metric_name, score in sample.metrics.items():
        threshold = thresholds.get(metric_name, 0.0)
        status = "✅" if score >= threshold else "❌"
        lines.append(f"- {metric_name}: {score:.2%} (target: {threshold:.2%}) {status}")

    lines.append(f"- overall: {sample.overall_score:.2%}")

    # Score comparison
    ai_score = sample.ai_analysis.quality_score
    expert_score = sample.ground_truth.get("quality_score", 0)
    lines.append("")
    lines.append(f"**Score Comparison:** AI={ai_score}, Expert={expert_score}")

    # Show errors if any
    if sample.errors:
        lines.append("")
        lines.append("**Errors:**")
        for error in sample.errors:
            lines.append(f"- {error}")

    return "\n".join(lines)


def _get_failure_reasons(sample: SingleEvalResult, thresholds: dict[str, float]) -> list[str]:
    """Get list of reasons why a sample failed.

    Args:
        sample: The failed sample result.
        thresholds: Quality thresholds.

    Returns:
        List of failure reason descriptions.
    """
    reasons: list[str] = []

    # Check each metric
    metric_names = {
        "insight_accuracy": "Low insight accuracy",
        "factual_correctness": "Factual errors",
        "completeness": "Incomplete observations",
        "score_accuracy": "Score deviation",
    }

    for metric_key, description in metric_names.items():
        score = sample.metrics.get(metric_key, 0.0)
        threshold = thresholds.get(metric_key, 1.0)
        if score < threshold:
            reasons.append(f"{description} ({score:.0%})")

    # Check overall
    overall_threshold = thresholds.get("overall", 0.75)
    if sample.overall_score < overall_threshold:
        reasons.append(f"Low overall ({sample.overall_score:.0%})")

    # Check errors
    if sample.errors:
        reasons.append("Analysis errors")

    return reasons or ["Unknown"]


def generate_json_export(results: EvalResults) -> str:
    """Generate JSON export of evaluation results.

    Args:
        results: The evaluation results to export.

    Returns:
        JSON-formatted string.
    """
    return results.model_dump_json(indent=2)


def save_report(
    results: EvalResults,
    output_dir: Path | str,
    basename: str | None = None,
    include_json: bool = True,
    include_markdown: bool = True,
) -> dict[str, Path]:
    """Save evaluation report to files.

    Args:
        results: The evaluation results to save.
        output_dir: Directory to save reports.
        basename: Base filename (default: uses run_id).
        include_json: Whether to save JSON export.
        include_markdown: Whether to save markdown report.

    Returns:
        Dictionary mapping format to saved file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = basename or f"eval_{results.run_id[:8]}"
    saved_files: dict[str, Path] = {}

    if include_json:
        json_path = output_dir / f"{basename}.json"
        json_path.write_text(generate_json_export(results))
        saved_files["json"] = json_path

    if include_markdown:
        md_path = output_dir / f"{basename}.md"
        md_path.write_text(generate_markdown_report(results))
        saved_files["markdown"] = md_path

    return saved_files


def compare_runs(
    current: EvalResults,
    baseline: EvalResults,
) -> dict[str, object]:
    """Compare two evaluation runs for regression detection.

    Args:
        current: Current evaluation results.
        baseline: Baseline results to compare against.

    Returns:
        Comparison report with deltas and regressions.
    """
    comparison: dict[str, object] = {
        "current_run_id": current.run_id,
        "baseline_run_id": baseline.run_id,
        "current_timestamp": current.timestamp.isoformat(),
        "baseline_timestamp": baseline.timestamp.isoformat(),
        "metric_deltas": {},
        "regressions": [],
        "improvements": [],
        "sample_changes": {},
    }

    # Compare aggregate metrics
    metric_deltas: dict[str, dict[str, float]] = {}
    regressions: list[str] = []
    improvements: list[str] = []

    for metric_name in current.aggregate_metrics:
        current_score = current.aggregate_metrics.get(metric_name, 0.0)
        baseline_score = baseline.aggregate_metrics.get(metric_name, 0.0)
        delta = current_score - baseline_score
        delta_pct = (delta / baseline_score * 100) if baseline_score > 0 else 0.0

        metric_deltas[metric_name] = {
            "current": current_score,
            "baseline": baseline_score,
            "delta": delta,
            "delta_pct": delta_pct,
        }

        # Flag significant changes (> 5% change)
        if delta < -0.05:
            regressions.append(
                f"{metric_name}: {baseline_score:.2%} → {current_score:.2%} " f"({delta_pct:+.1f}%)"
            )
        elif delta > 0.05:
            improvements.append(
                f"{metric_name}: {baseline_score:.2%} → {current_score:.2%} " f"({delta_pct:+.1f}%)"
            )

    comparison["metric_deltas"] = metric_deltas
    comparison["regressions"] = regressions
    comparison["improvements"] = improvements

    # Compare individual samples
    current_samples = {r.sample_id: r for r in current.results}
    baseline_samples = {r.sample_id: r for r in baseline.results}

    sample_changes: dict[str, dict[str, object]] = {}

    # Find status changes
    for sample_id in current_samples:
        if sample_id in baseline_samples:
            current_passed = current_samples[sample_id].passed
            baseline_passed = baseline_samples[sample_id].passed
            if current_passed != baseline_passed:
                sample_changes[sample_id] = {
                    "status_change": "improved" if current_passed else "regressed",
                    "current_overall": current_samples[sample_id].overall_score,
                    "baseline_overall": baseline_samples[sample_id].overall_score,
                }

    comparison["sample_changes"] = sample_changes
    comparison["has_regressions"] = len(regressions) > 0

    return comparison


def format_comparison_report(comparison: dict[str, object]) -> str:
    """Format a comparison report as markdown.

    Args:
        comparison: Comparison data from compare_runs().

    Returns:
        Markdown-formatted comparison report.
    """
    lines: list[str] = []

    lines.append("# Evaluation Run Comparison")
    lines.append("")
    lines.append(f"- **Current Run**: `{comparison['current_run_id']}`")
    lines.append(f"- **Baseline Run**: `{comparison['baseline_run_id']}`")
    lines.append("")

    # Overall status
    has_regressions = comparison.get("has_regressions", False)
    status = "⚠️ REGRESSIONS DETECTED" if has_regressions else "✅ No Regressions"
    lines.append(f"## Status: {status}")
    lines.append("")

    # Metric changes
    lines.append("## Metric Changes")
    lines.append("")
    lines.append("| Metric | Baseline | Current | Delta | Change |")
    lines.append("|--------|----------|---------|-------|--------|")

    metric_deltas_obj = comparison.get("metric_deltas", {})
    if isinstance(metric_deltas_obj, dict):
        for metric_name, delta_info in metric_deltas_obj.items():
            if isinstance(delta_info, dict):
                baseline = float(delta_info.get("baseline", 0.0))
                current = float(delta_info.get("current", 0.0))
                delta = float(delta_info.get("delta", 0.0))
                delta_pct = float(delta_info.get("delta_pct", 0.0))
                change_icon = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
                lines.append(
                    f"| {metric_name} | {baseline:.2%} | {current:.2%} | "
                    f"{delta:+.2%} | {change_icon} {delta_pct:+.1f}% |"
                )

    lines.append("")

    # Regressions
    regressions_obj = comparison.get("regressions", [])
    if isinstance(regressions_obj, list) and regressions_obj:
        lines.append("## Regressions")
        lines.append("")
        for reg in regressions_obj:
            lines.append(f"- ❌ {reg}")
        lines.append("")

    # Improvements
    improvements_obj = comparison.get("improvements", [])
    if isinstance(improvements_obj, list) and improvements_obj:
        lines.append("## Improvements")
        lines.append("")
        for imp in improvements_obj:
            lines.append(f"- ✅ {imp}")
        lines.append("")

    # Sample status changes
    sample_changes_obj = comparison.get("sample_changes", {})
    if isinstance(sample_changes_obj, dict) and sample_changes_obj:
        lines.append("## Sample Status Changes")
        lines.append("")
        lines.append("| Sample | Change | Baseline | Current |")
        lines.append("|--------|--------|----------|---------|")
        for sample_id, change in sample_changes_obj.items():
            if isinstance(change, dict):
                status_change = str(change.get("status_change", ""))
                baseline_overall = float(change.get("baseline_overall", 0.0))
                current_overall = float(change.get("current_overall", 0.0))
                icon = "📈" if status_change == "improved" else "📉"
                lines.append(
                    f"| {sample_id} | {icon} {status_change} | "
                    f"{baseline_overall:.2%} | {current_overall:.2%} |"
                )
        lines.append("")

    return "\n".join(lines)
