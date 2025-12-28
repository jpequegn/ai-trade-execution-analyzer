"""Tests for prompt templates module."""

from datetime import datetime

import pytest

from src.agents.prompts import (
    SYSTEM_PROMPT,
    PromptVariant,
    build_analysis_prompt,
    build_batch_prompt,
    estimate_prompt_tokens,
    format_execution_for_prompt,
    format_timestamp,
    get_system_prompt,
)
from src.parsers.fix_parser import ExecutionReport


@pytest.fixture
def sample_execution() -> ExecutionReport:
    """Create a sample execution report for testing."""
    return ExecutionReport(
        order_id="ORD001",
        symbol="AAPL",
        side="BUY",
        quantity=100.0,
        price=150.50,
        venue="NYSE",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        fill_type="FULL",
    )


@pytest.fixture
def execution_with_extras() -> ExecutionReport:
    """Create an execution report with optional fields."""
    return ExecutionReport(
        order_id="ORD002",
        symbol="GOOGL",
        side="SELL",
        quantity=500.0,
        price=175.25,
        venue="NASDAQ",
        timestamp=datetime(2024, 1, 15, 14, 45, 30),
        fill_type="PARTIAL",
        exec_type="F",
        cum_qty=250.0,
        avg_px=175.10,
    )


class TestSystemPrompt:
    """Tests for system prompt."""

    def test_system_prompt_exists(self) -> None:
        """Test that system prompt is defined."""
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_key_elements(self) -> None:
        """Test that system prompt contains essential instructions."""
        assert "trade execution analyst" in SYSTEM_PROMPT.lower()
        assert "JSON" in SYSTEM_PROMPT
        assert "quality" in SYSTEM_PROMPT.lower()
        assert "1-10" in SYSTEM_PROMPT

    def test_get_system_prompt_returns_same(self) -> None:
        """Test that get_system_prompt returns the same content."""
        assert get_system_prompt() == SYSTEM_PROMPT


class TestPromptVariant:
    """Tests for PromptVariant enum."""

    def test_variant_values(self) -> None:
        """Test that all variants have expected values."""
        assert PromptVariant.QUICK.value == "quick"
        assert PromptVariant.DETAILED.value == "detailed"
        assert PromptVariant.BATCH.value == "batch"

    def test_variant_from_string(self) -> None:
        """Test creating variant from string."""
        assert PromptVariant("quick") == PromptVariant.QUICK
        assert PromptVariant("detailed") == PromptVariant.DETAILED
        assert PromptVariant("batch") == PromptVariant.BATCH


class TestFormatTimestamp:
    """Tests for timestamp formatting."""

    def test_format_datetime(self) -> None:
        """Test formatting a datetime object."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = format_timestamp(dt)
        assert result == "2024-01-15 10:30:45"

    def test_format_string(self) -> None:
        """Test formatting a string passes through."""
        result = format_timestamp("2024-01-15")
        assert result == "2024-01-15"


class TestFormatExecutionForPrompt:
    """Tests for execution formatting."""

    def test_basic_execution(self, sample_execution: ExecutionReport) -> None:
        """Test formatting a basic execution."""
        fields = format_execution_for_prompt(sample_execution)

        assert fields["order_id"] == "ORD001"
        assert fields["symbol"] == "AAPL"
        assert fields["side"] == "BUY"
        assert fields["quantity"] == 100.0
        assert fields["price"] == 150.50
        assert fields["venue"] == "NYSE"
        assert fields["fill_type"] == "FULL"
        assert fields["extra_fields"] == ""

    def test_execution_with_extras(self, execution_with_extras: ExecutionReport) -> None:
        """Test formatting an execution with optional fields."""
        fields = format_execution_for_prompt(execution_with_extras)

        assert fields["order_id"] == "ORD002"
        assert "Exec Type" in fields["extra_fields"]
        assert "Cumulative Qty" in fields["extra_fields"]
        assert "Average Price" in fields["extra_fields"]


class TestBuildAnalysisPrompt:
    """Tests for build_analysis_prompt function."""

    def test_quick_variant(self, sample_execution: ExecutionReport) -> None:
        """Test building quick analysis prompt."""
        prompt = build_analysis_prompt(sample_execution, PromptVariant.QUICK)

        assert "ORD001" in prompt
        assert "AAPL" in prompt
        assert "BUY" in prompt
        assert "150.5" in prompt or "150.50" in prompt
        assert "NYSE" in prompt
        # Quick prompts are shorter
        assert len(prompt) < 500

    def test_detailed_variant(self, sample_execution: ExecutionReport) -> None:
        """Test building detailed analysis prompt."""
        prompt = build_analysis_prompt(sample_execution, PromptVariant.DETAILED)

        assert "ORD001" in prompt
        assert "AAPL" in prompt
        assert "Trade Details" in prompt
        assert "Analysis Required" in prompt
        assert "Quality Score" in prompt
        assert "Observations" in prompt
        assert "JSON" in prompt

    def test_string_variant(self, sample_execution: ExecutionReport) -> None:
        """Test that string variant is accepted."""
        prompt = build_analysis_prompt(sample_execution, "quick")
        assert "ORD001" in prompt

    def test_batch_variant_single_execution(self, sample_execution: ExecutionReport) -> None:
        """Test batch variant with single execution falls back to detailed."""
        prompt = build_analysis_prompt(sample_execution, PromptVariant.BATCH)
        # Should use detailed template for single execution
        assert "Trade Details" in prompt

    def test_prompt_includes_all_fields(self, execution_with_extras: ExecutionReport) -> None:
        """Test that detailed prompt includes extra fields."""
        prompt = build_analysis_prompt(execution_with_extras, PromptVariant.DETAILED)

        assert "Exec Type" in prompt
        assert "Cumulative Qty" in prompt
        assert "Average Price" in prompt


class TestBuildBatchPrompt:
    """Tests for build_batch_prompt function."""

    def test_batch_prompt_multiple_executions(
        self, sample_execution: ExecutionReport, execution_with_extras: ExecutionReport
    ) -> None:
        """Test building batch prompt with multiple executions."""
        executions = [sample_execution, execution_with_extras]
        prompt = build_batch_prompt(executions)

        assert "2 trade executions" in prompt
        assert "ORD001" in prompt
        assert "ORD002" in prompt
        assert "AAPL" in prompt
        assert "GOOGL" in prompt
        assert "JSON array" in prompt

    def test_batch_prompt_empty_list_raises(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            build_batch_prompt([])

    def test_batch_prompt_single_execution(self, sample_execution: ExecutionReport) -> None:
        """Test batch prompt with single execution."""
        prompt = build_batch_prompt([sample_execution])

        assert "1 trade executions" in prompt
        assert "ORD001" in prompt


class TestEstimatePromptTokens:
    """Tests for token estimation."""

    def test_estimate_short_prompt(self) -> None:
        """Test token estimation for short text."""
        tokens = estimate_prompt_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10

    def test_estimate_longer_prompt(self, sample_execution: ExecutionReport) -> None:
        """Test token estimation for actual prompt."""
        prompt = build_analysis_prompt(sample_execution, PromptVariant.DETAILED)
        tokens = estimate_prompt_tokens(prompt)

        # Detailed prompt should be substantial but under 500 tokens
        assert tokens > 50
        assert tokens < 500

    def test_estimate_empty_string(self) -> None:
        """Test token estimation for empty string."""
        assert estimate_prompt_tokens("") == 0


class TestPromptQuality:
    """Tests for prompt quality and consistency."""

    def test_system_prompt_token_count(self) -> None:
        """Test that system prompt is reasonably sized."""
        tokens = estimate_prompt_tokens(SYSTEM_PROMPT)
        # System prompt should be substantial but not excessive
        assert 50 < tokens < 300

    def test_quick_prompt_under_limit(self, sample_execution: ExecutionReport) -> None:
        """Test that quick prompts stay under token budget."""
        prompt = build_analysis_prompt(sample_execution, PromptVariant.QUICK)
        tokens = estimate_prompt_tokens(prompt)
        # Quick prompts should be concise
        assert tokens < 200

    def test_detailed_prompt_under_limit(self, sample_execution: ExecutionReport) -> None:
        """Test that detailed prompts stay under token budget."""
        prompt = build_analysis_prompt(sample_execution, PromptVariant.DETAILED)
        tokens = estimate_prompt_tokens(prompt)
        # Detailed prompts should be under 500 tokens per acceptance criteria
        assert tokens < 500

    def test_prompt_includes_json_schema(self, sample_execution: ExecutionReport) -> None:
        """Test that detailed prompt includes output schema."""
        prompt = build_analysis_prompt(sample_execution, PromptVariant.DETAILED)
        assert "quality_score" in prompt
        assert "observations" in prompt
        assert "issues" in prompt
        assert "recommendations" in prompt
        assert "confidence" in prompt
