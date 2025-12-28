"""Response parsing utilities for LLM outputs.

This module provides robust parsing of LLM responses into structured data,
handling common issues like malformed JSON, markdown code blocks, and
missing fields.

Example:
    >>> from src.agents.response_parser import parse_analysis_response
    >>> from src.parsers.models import TradeAnalysis
    >>> result = parse_analysis_response(llm_output)
    >>> print(result.quality_score)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from src.parsers.models import TradeAnalysis

logger = logging.getLogger(__name__)


class ResponseParseError(Exception):
    """Raised when LLM response cannot be parsed.

    Attributes:
        raw_response: The original unparseable response.
        reason: Human-readable explanation of the parse failure.
    """

    def __init__(self, message: str, raw_response: str, reason: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.reason = reason or message


def extract_json_from_response(response: str) -> str:
    """Extract JSON content from LLM response.

    Handles common response formats:
    - Plain JSON
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON with leading/trailing text
    - Multiple JSON objects (extracts first valid one)

    Args:
        response: Raw LLM response text.

    Returns:
        Extracted JSON string.

    Raises:
        ResponseParseError: If no valid JSON can be extracted.
    """
    if not response or not response.strip():
        raise ResponseParseError(
            "Empty response from LLM",
            raw_response=response,
            reason="empty_response",
        )

    text = response.strip()

    # Strategy 1: Try to extract from markdown code block
    code_block_patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]*?)\s*```",  # ``` ... ``` (any language)
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            if _is_valid_json(json_str):
                return json_str

    # Strategy 2: Try the whole response as JSON first
    if _is_valid_json(text):
        return text

    # Strategy 3: Find JSON array by matching brackets (try first for arrays)
    if text.lstrip().startswith("["):
        json_str = _extract_json_array(text)
        if json_str:
            return json_str

    # Strategy 4: Find JSON object by matching braces
    json_str = _extract_json_object(text)
    if json_str:
        return json_str

    # Strategy 5: Find JSON array anywhere in text
    json_str = _extract_json_array(text)
    if json_str:
        return json_str

    raise ResponseParseError(
        "Could not extract valid JSON from response",
        raw_response=response,
        reason="no_json_found",
    )


def _is_valid_json(text: str) -> bool:
    """Check if a string is valid JSON."""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _extract_json_object(text: str) -> str | None:
    """Extract a JSON object by matching braces."""
    # Find the first { and try to match closing }
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                json_str = text[start : i + 1]
                if _is_valid_json(json_str):
                    return json_str
                break

    return None


def _extract_json_array(text: str) -> str | None:
    """Extract a JSON array by matching brackets."""
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                json_str = text[start : i + 1]
                if _is_valid_json(json_str):
                    return json_str
                break

    return None


def parse_json_response(response: str) -> dict[str, Any]:
    """Parse JSON from LLM response.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed JSON as dictionary.

    Raises:
        ResponseParseError: If JSON cannot be extracted or parsed.
    """
    json_str = extract_json_from_response(response)

    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            raise ResponseParseError(
                "Expected JSON object, got array or primitive",
                raw_response=response,
                reason="wrong_json_type",
            )
        return parsed
    except json.JSONDecodeError as e:
        raise ResponseParseError(
            f"Invalid JSON: {e}",
            raw_response=response,
            reason="json_decode_error",
        ) from e


def normalize_analysis_data(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize and fill in missing fields for analysis data.

    Handles common variations in field names and provides defaults
    for missing optional fields.

    Args:
        data: Raw parsed JSON data.

    Returns:
        Normalized dictionary ready for TradeAnalysis validation.
    """
    normalized: dict[str, Any] = {}

    # Handle quality_score variations - check each explicitly (0 is valid but falsy)
    score = None
    for key in ["quality_score", "score", "quality"]:
        if key in data and data[key] is not None:
            score = data[key]
            break

    if score is not None:
        # Ensure it's an integer in valid range
        try:
            score_int = int(float(score))
            normalized["quality_score"] = max(1, min(10, score_int))
        except (ValueError, TypeError):
            normalized["quality_score"] = 5  # Default to middle score
    else:
        normalized["quality_score"] = 5  # Default score

    # Handle list fields - ensure they're lists of strings
    for field in ["observations", "issues", "recommendations"]:
        value = data.get(field, [])
        if value is None:
            normalized[field] = []
        elif isinstance(value, list):
            normalized[field] = [str(item) for item in value if item]
        elif isinstance(value, str):
            # If a single string was provided, wrap in list
            normalized[field] = [value] if value else []
        else:
            normalized[field] = []

    # Handle confidence
    confidence = data.get("confidence")
    if confidence is not None:
        try:
            conf_float = float(confidence)
            normalized["confidence"] = max(0.0, min(1.0, conf_float))
        except (ValueError, TypeError):
            normalized["confidence"] = 0.8  # Default confidence
    else:
        normalized["confidence"] = 0.8

    return normalized


def parse_analysis_response(response: str) -> TradeAnalysis:
    """Parse LLM response into a TradeAnalysis model.

    This is the main entry point for parsing analysis responses.
    It handles JSON extraction, normalization, and validation.

    Args:
        response: Raw LLM response text.

    Returns:
        Validated TradeAnalysis model.

    Raises:
        ResponseParseError: If response cannot be parsed.
    """
    # Extract and parse JSON
    data = parse_json_response(response)

    # Normalize the data
    normalized = normalize_analysis_data(data)

    # Validate with Pydantic
    try:
        return TradeAnalysis(**normalized)
    except ValidationError as e:
        logger.warning(f"Validation error, attempting recovery: {e}")
        # Try with more aggressive defaults
        fallback_data = {
            "quality_score": normalized.get("quality_score", 5),
            "observations": normalized.get("observations", []),
            "issues": normalized.get("issues", []),
            "recommendations": normalized.get("recommendations", []),
            "confidence": normalized.get("confidence", 0.5),
        }
        try:
            return TradeAnalysis(**fallback_data)
        except ValidationError as e2:
            raise ResponseParseError(
                f"Failed to validate analysis: {e2}",
                raw_response=response,
                reason="validation_error",
            ) from e2


def parse_batch_response(response: str) -> list[dict[str, Any]]:
    """Parse a batch analysis response containing multiple results.

    Args:
        response: Raw LLM response with array of analyses.

    Returns:
        List of raw analysis dictionaries (not yet validated).

    Raises:
        ResponseParseError: If response cannot be parsed as array.
    """
    json_str = extract_json_from_response(response)

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return list(parsed)
        if isinstance(parsed, dict):
            # Some LLMs wrap array in object like {"analyses": [...]}
            for key in ["analyses", "results", "executions", "data"]:
                if key in parsed and isinstance(parsed[key], list):
                    return list(parsed[key])
            # Return single item as list
            return [parsed]

        raise ResponseParseError(
            "Expected JSON array for batch response",
            raw_response=response,
            reason="expected_array",
        )
    except json.JSONDecodeError as e:
        raise ResponseParseError(
            f"Invalid JSON in batch response: {e}",
            raw_response=response,
            reason="json_decode_error",
        ) from e


def safe_parse_analysis(response: str) -> TradeAnalysis | None:
    """Safely parse analysis, returning None on failure.

    This is a convenience wrapper that doesn't raise exceptions,
    useful for batch processing where partial failures are acceptable.

    Args:
        response: Raw LLM response text.

    Returns:
        TradeAnalysis if parsing succeeds, None otherwise.
    """
    try:
        return parse_analysis_response(response)
    except ResponseParseError as e:
        logger.warning(f"Failed to parse analysis: {e.reason}")
        return None
