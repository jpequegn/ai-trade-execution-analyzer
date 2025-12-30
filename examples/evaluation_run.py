#!/usr/bin/env python3
"""Evaluation pipeline example.

This example demonstrates how to:
1. Create a ground truth dataset
2. Run AI analysis on the dataset
3. Evaluate AI performance against expert annotations
4. Generate evaluation reports

Usage:
    python examples/evaluation_run.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Ensure the parent directory is in the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ground_truth import (
    ExpertAnalysis,
    GroundTruthDataset,
    GroundTruthSample,
)
from src.evaluation.metrics import evaluate
from src.parsers import parse_fix_message
from src.pipeline import analyze_fix_message


def create_sample_dataset() -> GroundTruthDataset:
    """Create a small ground truth dataset for demonstration."""
    samples = []

    # Sample 1: Good execution - expect high score
    sample1_msg = (
        "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD001|11=CLT001|"
        "17=EXEC001|20=0|150=F|39=2|55=AAPL|54=1|38=100|44=150.50|"
        "32=100|31=150.45|14=100|6=150.45|60=20240115-09:30:05.000|10=123|"
    )
    samples.append(
        GroundTruthSample(
            execution=parse_fix_message(sample1_msg),
            expert_analysis=ExpertAnalysis(
                quality_score=9,
                key_issues=[],  # No issues for good execution
                expected_observations=[
                    "Full fill achieved",
                    "Price improvement of $0.05",
                    "Market open execution timing",
                ],
                expected_recommendations=[
                    "Continue current execution strategy",
                ],
            ),
            sample_id="SAMPLE001",
            source="manual",
            annotator="expert1",
            created_at=datetime.now(),
        )
    )

    # Sample 2: Partial fill - expect medium score
    sample2_msg = (
        "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD002|11=CLT002|"
        "17=EXEC002|20=0|150=F|39=1|55=MSFT|54=2|38=500|44=380.00|"
        "32=200|31=379.85|14=200|6=379.85|60=20240115-12:30:00.000|10=124|"
    )
    samples.append(
        GroundTruthSample(
            execution=parse_fix_message(sample2_msg),
            expert_analysis=ExpertAnalysis(
                quality_score=6,
                key_issues=[
                    "Partial fill only 40% of order",
                    "Lunch hour execution may impact liquidity",
                ],
                expected_observations=[
                    "Partial fill of 200 shares out of 500 ordered",
                    "Price improvement on filled portion",
                    "Mid-day execution timing",
                ],
                expected_recommendations=[
                    "Consider TWAP for larger orders",
                    "Avoid lunch hour for large orders",
                ],
            ),
            sample_id="SAMPLE002",
            source="manual",
            annotator="expert1",
            created_at=datetime.now(),
        )
    )

    # Sample 3: Poor execution - expect low score
    sample3_msg = (
        "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD003|11=CLT003|"
        "17=EXEC003|20=0|150=F|39=2|55=GME|54=1|38=1000|44=25.00|"
        "32=1000|31=26.50|14=1000|6=26.50|60=20240115-15:55:00.000|10=125|"
    )
    samples.append(
        GroundTruthSample(
            execution=parse_fix_message(sample3_msg),
            expert_analysis=ExpertAnalysis(
                quality_score=3,
                key_issues=[
                    "Significant slippage of $1.50 per share",
                    "Near market close execution timing",
                    "High volatility stock",
                ],
                expected_observations=[
                    "Full fill achieved but with 6% slippage",
                    "Executed 5 minutes before market close",
                    "Volatile meme stock",
                ],
                expected_recommendations=[
                    "Use limit orders for volatile stocks",
                    "Avoid trading volatile stocks near close",
                    "Consider breaking order into smaller pieces",
                ],
            ),
            sample_id="SAMPLE003",
            source="manual",
            annotator="expert1",
            created_at=datetime.now(),
        )
    )

    return GroundTruthDataset(samples=samples)


def main() -> int:
    """Run evaluation example."""
    print("=" * 70)
    print(" EVALUATION PIPELINE EXAMPLE ".center(70))
    print("=" * 70)

    # Step 1: Create ground truth dataset
    print("\n[1] Creating ground truth dataset...")
    dataset = create_sample_dataset()
    print(f"    Samples: {len(dataset)}")

    # Step 2: Run AI analysis on each sample
    print("\n[2] Running AI analysis on samples...")
    print("    (This will make LLM calls for each sample)")

    results = []
    for i, sample in enumerate(dataset.samples, 1):
        print(f"\n    Sample {i}/{len(dataset)}:")
        print(f"    Symbol: {sample.execution.symbol}")

        try:
            # Analyze using the pipeline
            fix_msg = (
                f"8={sample.execution.fix_version}|35=8|"
                f"37={sample.execution.order_id}|55={sample.execution.symbol}|"
                f"54={'1' if sample.execution.side == 'BUY' else '2'}|"
                f"38={int(sample.execution.quantity)}|44={sample.execution.price}|"
                f"32={int(sample.execution.quantity)}|31={sample.execution.price}|"
                f"39={'2' if sample.execution.fill_type == 'FULL' else '1'}|"
                f"60={sample.execution.timestamp.strftime('%Y%m%d-%H:%M:%S.000')}|"
                "10=123|"
            )
            result = analyze_fix_message(fix_msg)
            results.append((sample, result))

            print(f"    AI Score: {result.analysis.quality_score}/10")
            print(f"    Expert Score: {sample.expert_analysis.quality_score}/10")

        except Exception as e:
            print(f"    ERROR: {e}")
            print("    (Skipping this sample)")

    if not results:
        print("\n    No samples analyzed successfully.")
        print("    Make sure ANTHROPIC_API_KEY is set in your environment.")
        return 1

    # Step 3: Evaluate AI performance
    print("\n[3] Evaluating AI performance...")
    evaluations = []

    for sample, result in results:
        eval_result = evaluate(result.analysis, sample.expert_analysis)
        evaluations.append((sample, result, eval_result))

        print(f"\n    {sample.execution.symbol}:")
        print(f"    - Insight Accuracy:    {eval_result.insight_accuracy:.1%}")
        print(f"    - Factual Correctness: {eval_result.factual_correctness:.1%}")
        print(f"    - Completeness:        {eval_result.completeness:.1%}")
        print(f"    - Score Accuracy:      {eval_result.score_accuracy:.1%}")
        print(f"    - Overall:             {eval_result.overall_score:.1%}")

    # Step 4: Generate summary
    print("\n[4] Summary:")
    avg_overall = sum(e[2].overall_score for e in evaluations) / len(evaluations)
    avg_insight = sum(e[2].insight_accuracy for e in evaluations) / len(evaluations)
    avg_factual = sum(e[2].factual_correctness for e in evaluations) / len(evaluations)
    avg_complete = sum(e[2].completeness for e in evaluations) / len(evaluations)
    avg_score = sum(e[2].score_accuracy for e in evaluations) / len(evaluations)

    print(f"\n    Samples Evaluated: {len(evaluations)}")
    print("\n    Average Metrics:")
    print(f"    - Insight Accuracy:    {avg_insight:.1%}")
    print(f"    - Factual Correctness: {avg_factual:.1%}")
    print(f"    - Completeness:        {avg_complete:.1%}")
    print(f"    - Score Accuracy:      {avg_score:.1%}")
    print(f"    - Overall:             {avg_overall:.1%}")

    # Step 5: Check against thresholds
    print("\n[5] Success Criteria Check:")
    thresholds = {
        "Insight Accuracy": (avg_insight, 0.80),
        "Factual Correctness": (avg_factual, 0.95),
        "Completeness": (avg_complete, 0.70),
        "Score Accuracy": (avg_score, 0.85),
    }

    all_passed = True
    for metric, (value, threshold) in thresholds.items():
        status = "PASS" if value >= threshold else "FAIL"
        symbol = "[x]" if value >= threshold else "[ ]"
        print(f"    {symbol} {metric}: {value:.1%} (threshold: {threshold:.0%}) - {status}")
        if value < threshold:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print(" ALL CRITERIA MET ".center(70))
    else:
        print(" SOME CRITERIA NOT MET ".center(70))
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
