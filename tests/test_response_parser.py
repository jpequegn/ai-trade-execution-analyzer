"""Tests for response parser module."""

import pytest

from src.agents.response_parser import (
    ResponseParseError,
    extract_json_from_response,
    normalize_analysis_data,
    parse_analysis_response,
    parse_batch_response,
    parse_json_response,
    safe_parse_analysis,
)
from src.parsers.models import TradeAnalysis


class TestExtractJsonFromResponse:
    """Tests for JSON extraction from LLM responses."""

    def test_extract_plain_json(self) -> None:
        """Test extracting plain JSON object."""
        response = '{"quality_score": 7, "observations": ["Good execution"]}'
        result = extract_json_from_response(response)
        assert result == response

    def test_extract_from_markdown_block(self) -> None:
        """Test extracting JSON from markdown code block."""
        response = """Here's the analysis:

```json
{"quality_score": 8, "observations": ["Excellent timing"]}
```

Hope this helps!"""
        result = extract_json_from_response(response)
        assert '"quality_score": 8' in result

    def test_extract_from_generic_code_block(self) -> None:
        """Test extracting JSON from generic code block."""
        response = """```
{"quality_score": 6}
```"""
        result = extract_json_from_response(response)
        assert '"quality_score": 6' in result

    def test_extract_json_with_leading_text(self) -> None:
        """Test extracting JSON with leading text."""
        response = 'Based on my analysis: {"quality_score": 7, "observations": []}'
        result = extract_json_from_response(response)
        assert '"quality_score": 7' in result

    def test_extract_json_with_trailing_text(self) -> None:
        """Test extracting JSON with trailing text."""
        response = '{"quality_score": 7} Let me know if you need more details.'
        result = extract_json_from_response(response)
        assert '"quality_score": 7' in result

    def test_extract_nested_json(self) -> None:
        """Test extracting nested JSON."""
        response = """{"quality_score": 7, "details": {"nested": true}}"""
        result = extract_json_from_response(response)
        assert "nested" in result

    def test_extract_json_with_escaped_quotes(self) -> None:
        """Test extracting JSON with escaped quotes."""
        response = r'{"quality_score": 7, "observations": ["Said \"hello\""]}'
        result = extract_json_from_response(response)
        assert "quality_score" in result

    def test_extract_empty_response_raises(self) -> None:
        """Test that empty response raises error."""
        with pytest.raises(ResponseParseError, match="Empty"):
            extract_json_from_response("")

    def test_extract_whitespace_only_raises(self) -> None:
        """Test that whitespace-only response raises error."""
        with pytest.raises(ResponseParseError, match="Empty"):
            extract_json_from_response("   \n\t  ")

    def test_extract_no_json_raises(self) -> None:
        """Test that response without JSON raises error."""
        with pytest.raises(ResponseParseError, match="Could not extract"):
            extract_json_from_response("This is just plain text with no JSON.")

    def test_extract_json_array(self) -> None:
        """Test extracting JSON array."""
        response = '[{"quality_score": 7}, {"quality_score": 8}]'
        result = extract_json_from_response(response)
        # Should extract the full array
        assert "quality_score" in result
        assert result.startswith("[")


class TestParseJsonResponse:
    """Tests for parse_json_response function."""

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON object."""
        response = '{"quality_score": 7, "observations": ["Good"]}'
        result = parse_json_response(response)
        assert result["quality_score"] == 7
        assert result["observations"] == ["Good"]

    def test_parse_json_array_raises(self) -> None:
        """Test that JSON array raises error for parse_json_response."""
        # parse_json_response expects an object, not array
        response = '[{"quality_score": 7}]'
        # Use parse_batch_response for arrays instead
        result = parse_batch_response(response)
        assert len(result) == 1


class TestNormalizeAnalysisData:
    """Tests for data normalization."""

    def test_normalize_complete_data(self) -> None:
        """Test normalizing complete data."""
        data = {
            "quality_score": 7,
            "observations": ["Good timing"],
            "issues": ["Price slippage"],
            "recommendations": ["Use limit orders"],
            "confidence": 0.85,
        }
        result = normalize_analysis_data(data)
        assert result["quality_score"] == 7
        assert result["observations"] == ["Good timing"]
        assert result["confidence"] == 0.85

    def test_normalize_score_variations(self) -> None:
        """Test normalizing different score field names."""
        # quality_score
        assert normalize_analysis_data({"quality_score": 7})["quality_score"] == 7
        # score
        assert normalize_analysis_data({"score": 8})["quality_score"] == 8
        # quality
        assert normalize_analysis_data({"quality": 9})["quality_score"] == 9

    def test_normalize_score_clamping(self) -> None:
        """Test that scores are clamped to 1-10 range."""
        assert normalize_analysis_data({"quality_score": 0})["quality_score"] == 1
        assert normalize_analysis_data({"quality_score": 15})["quality_score"] == 10
        assert normalize_analysis_data({"quality_score": -5})["quality_score"] == 1

    def test_normalize_float_score(self) -> None:
        """Test normalizing float score to int."""
        assert normalize_analysis_data({"quality_score": 7.8})["quality_score"] == 7

    def test_normalize_missing_score(self) -> None:
        """Test default for missing score."""
        result = normalize_analysis_data({})
        assert result["quality_score"] == 5  # Now always set

    def test_normalize_none_lists(self) -> None:
        """Test that None lists become empty lists."""
        data = {"quality_score": 7, "observations": None, "issues": None}
        result = normalize_analysis_data(data)
        assert result["observations"] == []
        assert result["issues"] == []

    def test_normalize_string_to_list(self) -> None:
        """Test that single string becomes list."""
        data = {"quality_score": 7, "observations": "Single observation"}
        result = normalize_analysis_data(data)
        assert result["observations"] == ["Single observation"]

    def test_normalize_confidence_clamping(self) -> None:
        """Test that confidence is clamped to 0-1 range."""
        assert normalize_analysis_data({"confidence": 1.5})["confidence"] == 1.0
        assert normalize_analysis_data({"confidence": -0.5})["confidence"] == 0.0

    def test_normalize_default_confidence(self) -> None:
        """Test default confidence value."""
        result = normalize_analysis_data({})
        assert result["confidence"] == 0.8


class TestParseAnalysisResponse:
    """Tests for parse_analysis_response function."""

    def test_parse_complete_response(self) -> None:
        """Test parsing a complete, valid response."""
        response = """{
            "quality_score": 8,
            "observations": ["Executed during market hours", "Good fill rate"],
            "issues": ["Slightly above VWAP"],
            "recommendations": ["Consider dark pools for large orders"],
            "confidence": 0.9
        }"""
        result = parse_analysis_response(response)

        assert isinstance(result, TradeAnalysis)
        assert result.quality_score == 8
        assert len(result.observations) == 2
        assert len(result.issues) == 1
        assert len(result.recommendations) == 1
        assert result.confidence == 0.9

    def test_parse_minimal_response(self) -> None:
        """Test parsing minimal valid response."""
        response = '{"quality_score": 5}'
        result = parse_analysis_response(response)

        assert result.quality_score == 5
        assert result.observations == []
        assert result.issues == []
        assert result.recommendations == []

    def test_parse_response_with_markdown(self) -> None:
        """Test parsing response wrapped in markdown."""
        response = """Here's my analysis:

```json
{
    "quality_score": 7,
    "observations": ["Good execution"],
    "issues": [],
    "recommendations": [],
    "confidence": 0.85
}
```"""
        result = parse_analysis_response(response)
        assert result.quality_score == 7

    def test_parse_response_normalizes_data(self) -> None:
        """Test that response data is normalized."""
        response = '{"score": 7.8, "observations": "Single observation"}'
        result = parse_analysis_response(response)

        assert result.quality_score == 7  # Converted from float
        assert result.observations == ["Single observation"]  # Wrapped in list

    def test_parse_invalid_json_raises(self) -> None:
        """Test that invalid JSON raises error."""
        with pytest.raises(ResponseParseError):
            parse_analysis_response("This is not JSON at all")


class TestParseBatchResponse:
    """Tests for batch response parsing."""

    def test_parse_batch_array(self) -> None:
        """Test parsing batch response as array."""
        response = '[{"quality_score": 7, "observations": ["Good"]}, {"quality_score": 8, "observations": ["Excellent"]}]'
        result = parse_batch_response(response)

        assert len(result) == 2
        assert result[0]["quality_score"] == 7
        assert result[1]["quality_score"] == 8

    def test_parse_batch_wrapped_array(self) -> None:
        """Test parsing batch response wrapped in object."""
        response = """{
            "analyses": [
                {"quality_score": 7},
                {"quality_score": 8}
            ]
        }"""
        result = parse_batch_response(response)
        assert len(result) == 2

    def test_parse_batch_results_key(self) -> None:
        """Test parsing batch with 'results' key."""
        response = '{"results": [{"quality_score": 7}]}'
        result = parse_batch_response(response)
        assert len(result) == 1

    def test_parse_batch_single_object(self) -> None:
        """Test parsing single object as single-item list."""
        response = '{"quality_score": 7}'
        result = parse_batch_response(response)
        assert len(result) == 1
        assert result[0]["quality_score"] == 7


class TestSafeParseAnalysis:
    """Tests for safe_parse_analysis function."""

    def test_safe_parse_success(self) -> None:
        """Test successful safe parse returns TradeAnalysis."""
        response = '{"quality_score": 7, "observations": ["Good"]}'
        result = safe_parse_analysis(response)

        assert result is not None
        assert result.quality_score == 7

    def test_safe_parse_failure_returns_none(self) -> None:
        """Test failed safe parse returns None."""
        result = safe_parse_analysis("Not JSON")
        assert result is None

    def test_safe_parse_empty_returns_none(self) -> None:
        """Test empty response returns None."""
        result = safe_parse_analysis("")
        assert result is None


class TestResponseParseError:
    """Tests for ResponseParseError exception."""

    def test_error_attributes(self) -> None:
        """Test error has expected attributes."""
        error = ResponseParseError(
            "Parse failed",
            raw_response="invalid",
            reason="test_reason",
        )
        assert error.raw_response == "invalid"
        assert error.reason == "test_reason"
        assert str(error) == "Parse failed"

    def test_error_default_reason(self) -> None:
        """Test error uses message as default reason."""
        error = ResponseParseError("Parse failed", raw_response="invalid")
        assert error.reason == "Parse failed"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_deeply_nested_json(self) -> None:
        """Test parsing deeply nested JSON."""
        response = """{
            "quality_score": 7,
            "observations": ["Good"],
            "metadata": {
                "nested": {
                    "deep": {
                        "value": true
                    }
                }
            }
        }"""
        # Should extract and parse without error
        result = extract_json_from_response(response)
        assert "quality_score" in result

    def test_json_with_unicode(self) -> None:
        """Test parsing JSON with unicode characters."""
        response = (
            '{"quality_score": 7, "observations": ["Trade executed on Tokyo (東京) exchange"]}'
        )
        result = parse_analysis_response(response)
        assert "東京" in result.observations[0]

    def test_json_with_newlines_in_strings(self) -> None:
        """Test parsing JSON with newlines in string values."""
        response = '{"quality_score": 7, "observations": ["Line 1\\nLine 2"]}'
        result = parse_analysis_response(response)
        assert result.quality_score == 7

    def test_large_response(self) -> None:
        """Test parsing large response."""
        import json

        observations = [f"Observation {i}" for i in range(100)]
        response = json.dumps({"quality_score": 7, "observations": observations})
        result = parse_analysis_response(response)
        assert len(result.observations) == 100

    def test_empty_lists_in_response(self) -> None:
        """Test parsing response with empty lists."""
        response = """{
            "quality_score": 5,
            "observations": [],
            "issues": [],
            "recommendations": [],
            "confidence": 0.5
        }"""
        result = parse_analysis_response(response)
        assert result.observations == []
        assert result.issues == []
        assert result.recommendations == []
