"""Prompt templates for trade execution analysis.

This module provides carefully crafted prompts for LLM-based trade analysis,
including system prompts, analysis templates, and structured output schemas.

The prompts are designed to:
- Produce consistent, structured JSON output
- Avoid hallucination by emphasizing data-driven analysis
- Support different analysis depths (quick, detailed, batch)

Example:
    >>> from src.agents.prompts import build_analysis_prompt, SYSTEM_PROMPT
    >>> from src.parsers.fix_parser import ExecutionReport
    >>> prompt = build_analysis_prompt(execution, variant="detailed")
"""

from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.parsers.fix_parser import ExecutionReport

# Output JSON schema for reference in prompts
OUTPUT_SCHEMA = {
    "quality_score": "integer (1-10)",
    "observations": ["string - factual observations about the execution"],
    "issues": ["string - identified problems or concerns"],
    "recommendations": ["string - actionable suggestions for improvement"],
    "confidence": "float (0.0-1.0) - your confidence in this analysis",
}

OUTPUT_SCHEMA_JSON = json.dumps(OUTPUT_SCHEMA, indent=2)

# System prompt establishing the AI's role and constraints
SYSTEM_PROMPT = """You are an expert trade execution analyst with deep knowledge of:
- Market microstructure and venue selection
- Order execution quality metrics
- Trading best practices and regulatory requirements
- Common execution issues and their causes

Your role is to analyze trade executions for quality issues, timing problems,
and venue selection optimization.

CRITICAL RULES:
1. ONLY analyze the data provided. Never make assumptions about data not given.
2. Base observations ONLY on facts present in the execution data.
3. Be specific and quantitative when possible.
4. If information is insufficient for analysis, state that clearly.
5. Quality scores must be justified by the data provided.

SCORING GUIDELINES (1-10 scale):
- 9-10: Exceptional execution, optimal price, ideal venue, perfect timing
- 7-8: Good execution with minor areas for improvement
- 5-6: Average execution, notable issues but acceptable
- 3-4: Poor execution, significant problems identified
- 1-2: Very poor execution, major issues present

Always respond with valid JSON matching the specified schema."""


class PromptVariant(str, Enum):
    """Variants for analysis prompts with different depth/cost tradeoffs.

    Attributes:
        QUICK: Cost-optimized, brief analysis for high-volume screening.
        DETAILED: Comprehensive analysis with full context.
        BATCH: Optimized for analyzing multiple trades together.
    """

    QUICK = "quick"
    DETAILED = "detailed"
    BATCH = "batch"


# Template for quick analysis (cost-optimized)
QUICK_ANALYSIS_TEMPLATE = """Analyze this trade execution briefly:

Order: {order_id} | {symbol} | {side} | {quantity} @ ${price} | {venue} | {timestamp}

Provide a JSON response with:
- quality_score (1-10)
- observations (1-2 key points)
- issues (if any critical issues)
- recommendations (if any immediate actions needed)
- confidence (0.0-1.0)

JSON Response:"""


# Template for detailed analysis (comprehensive)
DETAILED_ANALYSIS_TEMPLATE = """Analyze this trade execution in detail:

## Trade Details
- **Order ID**: {order_id}
- **Symbol**: {symbol}
- **Side**: {side}
- **Quantity**: {quantity:,.0f} shares
- **Price**: ${price:,.2f}
- **Venue**: {venue}
- **Timestamp**: {timestamp}
- **Fill Type**: {fill_type}
{extra_fields}

## Analysis Required
Provide a comprehensive analysis including:

1. **Quality Score** (1-10): Rate the overall execution quality
   - Consider price, timing, venue selection, and fill completeness

2. **Observations**: List factual observations about this execution
   - What stands out about this trade?
   - Any notable characteristics?

3. **Issues**: Identify any problems or concerns
   - Suboptimal venue selection?
   - Timing concerns?
   - Price slippage indicators?

4. **Recommendations**: Suggest specific improvements
   - Alternative venues to consider?
   - Timing optimizations?
   - Order type suggestions?

5. **Confidence**: Your confidence in this analysis (0.0-1.0)

## Output Format
Respond with valid JSON matching this schema:
{output_schema}

JSON Response:"""


# Template for batch analysis (multiple trades)
BATCH_ANALYSIS_TEMPLATE = """Analyze these {count} trade executions:

## Executions
{executions_list}

## Analysis Required
For EACH execution, provide:
- quality_score (1-10)
- observations (key points)
- issues (problems identified)
- recommendations (improvements)
- confidence (0.0-1.0)

## Output Format
Respond with a JSON array of analyses, one per execution, in order:
[
  {{"order_id": "...", "quality_score": N, "observations": [...], "issues": [...], "recommendations": [...], "confidence": N.N}},
  ...
]

JSON Response:"""


def format_timestamp(timestamp: object) -> str:
    """Format a timestamp for prompt display.

    Args:
        timestamp: datetime object or string representation.

    Returns:
        Formatted timestamp string.
    """
    from datetime import datetime

    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return str(timestamp)


def format_execution_for_prompt(execution: ExecutionReport) -> dict[str, str | float]:
    """Extract and format execution fields for prompt templates.

    Args:
        execution: Parsed execution report.

    Returns:
        Dictionary of formatted field values for template substitution.
    """
    extra_fields_parts = []
    if execution.exec_type:
        extra_fields_parts.append(f"- **Exec Type**: {execution.exec_type}")
    if execution.cum_qty is not None:
        extra_fields_parts.append(f"- **Cumulative Qty**: {execution.cum_qty:,.0f}")
    if execution.avg_px is not None:
        extra_fields_parts.append(f"- **Average Price**: ${execution.avg_px:,.2f}")

    return {
        "order_id": execution.order_id,
        "symbol": execution.symbol,
        "side": execution.side,
        "quantity": execution.quantity,
        "price": execution.price,
        "venue": execution.venue,
        "timestamp": format_timestamp(execution.timestamp),
        "fill_type": execution.fill_type,
        "extra_fields": "\n".join(extra_fields_parts) if extra_fields_parts else "",
    }


def build_analysis_prompt(
    execution: ExecutionReport,
    variant: PromptVariant | str = PromptVariant.DETAILED,
) -> str:
    """Build an analysis prompt for a single execution.

    Args:
        execution: The execution report to analyze.
        variant: The prompt variant to use (quick, detailed, or batch).

    Returns:
        Formatted prompt string ready for LLM input.

    Example:
        >>> from src.parsers.fix_parser import parse_fix_message
        >>> execution = parse_fix_message("8=FIX.4.4|35=8|37=ORD001|...")
        >>> prompt = build_analysis_prompt(execution, variant="quick")
    """
    if isinstance(variant, str):
        variant = PromptVariant(variant)

    fields = format_execution_for_prompt(execution)

    if variant == PromptVariant.QUICK:
        return QUICK_ANALYSIS_TEMPLATE.format(**fields)
    elif variant == PromptVariant.DETAILED:
        return DETAILED_ANALYSIS_TEMPLATE.format(
            **fields,
            output_schema=OUTPUT_SCHEMA_JSON,
        )
    else:
        # For single execution, use detailed template even if batch requested
        return DETAILED_ANALYSIS_TEMPLATE.format(
            **fields,
            output_schema=OUTPUT_SCHEMA_JSON,
        )


def build_batch_prompt(
    executions: list[ExecutionReport],
) -> str:
    """Build a batch analysis prompt for multiple executions.

    Args:
        executions: List of execution reports to analyze.

    Returns:
        Formatted batch prompt string.

    Raises:
        ValueError: If executions list is empty.
    """
    if not executions:
        raise ValueError("Cannot build batch prompt with empty executions list")

    # Format each execution as a compact line
    execution_lines = []
    for i, exec_ in enumerate(executions, 1):
        line = (
            f"{i}. Order: {exec_.order_id} | {exec_.symbol} | {exec_.side} | "
            f"{exec_.quantity:,.0f} @ ${exec_.price:,.2f} | {exec_.venue} | "
            f"{format_timestamp(exec_.timestamp)}"
        )
        execution_lines.append(line)

    return BATCH_ANALYSIS_TEMPLATE.format(
        count=len(executions),
        executions_list="\n".join(execution_lines),
    )


def get_system_prompt() -> str:
    """Get the system prompt for trade analysis.

    Returns:
        The system prompt string.
    """
    return SYSTEM_PROMPT


def estimate_prompt_tokens(prompt: str) -> int:
    """Estimate token count for a prompt.

    Uses a simple heuristic of ~4 characters per token for English text.
    This is approximate and varies by model.

    Args:
        prompt: The prompt text.

    Returns:
        Estimated token count.
    """
    # Rough estimate: ~4 characters per token for English text
    return len(prompt) // 4
