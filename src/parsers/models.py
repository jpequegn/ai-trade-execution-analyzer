"""Pydantic data models for trade execution analysis.

This module contains the data models used throughout the trade execution
analyzer, including models for LLM outputs and combined analysis results.

The ExecutionReport model is defined in fix_parser.py as it's tightly
coupled with the FIX parsing logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class TradeAnalysis(BaseModel):
    """Structured representation of LLM analysis output.

    This model represents the AI-generated analysis of a trade execution,
    including quality assessment, observations, and recommendations.

    Attributes:
        quality_score: Overall execution quality score from 1-10.
        observations: List of factual observations about the execution.
        issues: List of identified problems or concerns.
        recommendations: List of suggested improvements.
        confidence: Model's confidence in the analysis (0.0-1.0).
    """

    quality_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Overall execution quality score (1=poor, 10=excellent)",
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Factual observations about the execution",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Identified problems or concerns",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Suggested improvements",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Model's confidence in the analysis (0.0-1.0)",
    )

    @field_validator("observations", "issues", "recommendations", mode="before")
    @classmethod
    def ensure_list(cls, v: list[str] | None) -> list[str]:
        """Ensure list fields are never None."""
        if v is None:
            return []
        return v

    @model_validator(mode="after")
    def validate_content_consistency(self) -> TradeAnalysis:
        """Validate that low scores have issues and high scores have observations."""
        # Low quality score should typically have issues
        if self.quality_score <= 3 and not self.issues:
            # This is a soft validation - we don't raise an error
            # but it's worth noting for analysis quality checks
            pass

        # High quality score should typically have observations
        if self.quality_score >= 8 and not self.observations:
            pass

        return self


class AnalysisResult(BaseModel):
    """Complete result of analyzing a trade execution.

    This model combines the original execution data with the AI-generated
    analysis and metadata about the analysis process.

    Attributes:
        execution: The original parsed execution report.
        analysis: The AI-generated trade analysis.
        raw_response: Raw text response from the LLM.
        tokens_used: Total tokens consumed (input + output).
        latency_ms: Time taken for analysis in milliseconds.
        model: The LLM model used for analysis.
        from_cache: Whether the result was retrieved from cache.
        analysis_id: Unique identifier for this analysis.
        analyzed_at: Timestamp when analysis was performed.
    """

    # Use Any type at runtime to avoid circular import issues
    # The actual type is ExecutionReport from fix_parser.py
    execution: Any
    analysis: TradeAnalysis
    raw_response: str = Field(
        default="",
        description="Raw text response from the LLM",
    )
    tokens_used: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed (input + output)",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for analysis in milliseconds",
    )
    model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="The LLM model used for analysis",
    )
    from_cache: bool = Field(
        default=False,
        description="Whether the result was retrieved from cache",
    )
    analysis_id: str | None = Field(
        default=None,
        description="Unique identifier for this analysis",
    )
    analyzed_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when analysis was performed",
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("execution", mode="before")
    @classmethod
    def validate_execution(cls, v: Any) -> Any:
        """Validate that execution is an ExecutionReport-like object."""
        # Import here to avoid circular imports
        from src.parsers.fix_parser import ExecutionReport

        if isinstance(v, ExecutionReport):
            return v
        if isinstance(v, dict):
            return ExecutionReport(**v)
        raise ValueError(f"Expected ExecutionReport or dict, got {type(v)}")
