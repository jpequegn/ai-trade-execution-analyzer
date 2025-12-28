"""Fuzzy matching utilities for comparing AI analysis with ground truth.

This module provides text similarity functions for comparing issues,
observations, and other text-based analysis components.

Example:
    >>> from src.evaluation.matching import fuzzy_match, best_match
    >>> fuzzy_match("suboptimal venue selection", "poor venue choice")
    0.72
    >>> best_match("timing issue", ["bad execution time", "venue problem"])
    ("bad execution time", 0.65)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

# Synonym mappings for common trade execution terms
SYNONYMS: dict[str, list[str]] = {
    # Venue-related
    "venue": ["exchange", "market", "trading venue", "execution venue"],
    "suboptimal": ["poor", "bad", "wrong", "inappropriate", "inefficient"],
    "selection": ["choice", "pick", "decision"],
    "dark pool": ["dark venue", "off-exchange", "alternative venue"],
    "primary": ["main", "principal", "primary listing"],
    # Timing-related
    "timing": ["time", "execution time", "temporal"],
    "open": ["market open", "opening", "at open"],
    "close": ["market close", "closing", "at close", "near close"],
    "after hours": ["extended hours", "post-market", "pre-market"],
    "lunch": ["midday", "noon", "lunch hour"],
    "volatility": ["volatile", "price swings", "unstable"],
    # Fill-related
    "partial": ["partial fill", "incomplete", "unfilled"],
    "full": ["complete", "full fill", "fully filled"],
    "fill": ["execution", "filled", "fill quality"],
    # Price-related
    "slippage": ["price impact", "adverse selection", "execution cost"],
    "spread": ["bid-ask", "bid ask spread", "quote spread"],
    "vwap": ["volume weighted", "average price"],
    # General
    "issue": ["problem", "concern", "error", "failure"],
    "good": ["excellent", "optimal", "great", "strong"],
    "bad": ["poor", "weak", "suboptimal", "negative"],
}

# Common stop words to filter out
STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "and",
    "but",
    "or",
    "nor",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "not",
    "only",
    "same",
    "than",
    "too",
    "very",
    "just",
    "also",
}


@dataclass
class MatchResult:
    """Result of a fuzzy match operation.

    Attributes:
        candidate: The matched candidate string.
        score: Similarity score (0.0-1.0).
        match_type: Type of match (exact, sequence, keyword, synonym).
    """

    candidate: str
    score: float
    match_type: str


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Args:
        text: Input text string.

    Returns:
        Lowercased text with extra whitespace removed.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text.

    Args:
        text: Input text string.

    Returns:
        Set of keywords with stop words removed.
    """
    words = re.findall(r"\b[a-z]+\b", normalize_text(text))
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def expand_synonyms(keywords: set[str]) -> set[str]:
    """Expand keywords with their synonyms.

    Args:
        keywords: Set of keywords to expand.

    Returns:
        Expanded set including synonyms.
    """
    expanded = set(keywords)
    for keyword in keywords:
        # Check if keyword has synonyms
        if keyword in SYNONYMS:
            expanded.update(SYNONYMS[keyword])
        # Also check if keyword is a synonym of something
        for base, syns in SYNONYMS.items():
            if keyword in syns:
                expanded.add(base)
                expanded.update(syns)
    return expanded


def sequence_similarity(text1: str, text2: str) -> float:
    """Calculate sequence-based similarity using difflib.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    return SequenceMatcher(None, t1, t2).ratio()


def keyword_overlap(text1: str, text2: str, use_synonyms: bool = True) -> float:
    """Calculate keyword overlap between two texts.

    Args:
        text1: First text string.
        text2: Second text string.
        use_synonyms: Whether to expand with synonyms.

    Returns:
        Jaccard similarity of keywords (0.0-1.0).
    """
    kw1 = extract_keywords(text1)
    kw2 = extract_keywords(text2)

    if use_synonyms:
        kw1 = expand_synonyms(kw1)
        kw2 = expand_synonyms(kw2)

    if not kw1 or not kw2:
        return 0.0

    intersection = kw1 & kw2
    union = kw1 | kw2

    return len(intersection) / len(union)


def fuzzy_match(
    text1: str,
    text2: str,
    sequence_weight: float = 0.4,
    keyword_weight: float = 0.6,
) -> float:
    """Calculate fuzzy similarity between two text strings.

    Combines sequence similarity and keyword overlap for robust matching.

    Args:
        text1: First text string.
        text2: Second text string.
        sequence_weight: Weight for sequence similarity (default 0.4).
        keyword_weight: Weight for keyword overlap (default 0.6).

    Returns:
        Combined similarity score between 0.0 and 1.0.

    Example:
        >>> fuzzy_match("suboptimal venue selection", "poor venue choice")
        0.72
    """
    if not text1 or not text2:
        return 0.0

    # Check for exact match first
    if normalize_text(text1) == normalize_text(text2):
        return 1.0

    seq_sim = sequence_similarity(text1, text2)
    kw_sim = keyword_overlap(text1, text2, use_synonyms=True)

    return sequence_weight * seq_sim + keyword_weight * kw_sim


def best_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.5,
) -> MatchResult | None:
    """Find the best matching candidate for a query.

    Args:
        query: The text to match.
        candidates: List of candidate strings to compare against.
        threshold: Minimum score required for a match (default 0.5).

    Returns:
        MatchResult if a match is found above threshold, None otherwise.

    Example:
        >>> best_match("timing issue", ["bad execution time", "venue problem"])
        MatchResult(candidate="bad execution time", score=0.65, match_type="fuzzy")
    """
    if not query or not candidates:
        return None

    best: MatchResult | None = None
    best_score = 0.0

    query_norm = normalize_text(query)

    for candidate in candidates:
        cand_norm = normalize_text(candidate)

        # Check exact match
        if query_norm == cand_norm:
            return MatchResult(candidate=candidate, score=1.0, match_type="exact")

        # Calculate fuzzy score
        score = fuzzy_match(query, candidate)

        if score > best_score:
            best_score = score
            best = MatchResult(candidate=candidate, score=score, match_type="fuzzy")

    if best and best.score >= threshold:
        return best

    return None


def find_all_matches(
    query: str,
    candidates: list[str],
    threshold: float = 0.5,
) -> list[MatchResult]:
    """Find all matching candidates above a threshold.

    Args:
        query: The text to match.
        candidates: List of candidate strings to compare against.
        threshold: Minimum score required for a match.

    Returns:
        List of MatchResults sorted by score descending.
    """
    if not query or not candidates:
        return []

    matches = []
    query_norm = normalize_text(query)

    for candidate in candidates:
        cand_norm = normalize_text(candidate)

        # Check exact match
        if query_norm == cand_norm:
            matches.append(MatchResult(candidate=candidate, score=1.0, match_type="exact"))
            continue

        # Calculate fuzzy score
        score = fuzzy_match(query, candidate)
        if score >= threshold:
            matches.append(MatchResult(candidate=candidate, score=score, match_type="fuzzy"))

    return sorted(matches, key=lambda m: m.score, reverse=True)


def count_matches(
    expected: list[str],
    actual: list[str],
    threshold: float = 0.5,
) -> tuple[int, list[tuple[str, str | None, float]]]:
    """Count how many expected items are matched in actual.

    Args:
        expected: List of expected items to find.
        actual: List of actual items to search in.
        threshold: Minimum similarity for a match.

    Returns:
        Tuple of (match_count, match_details).
        match_details is a list of (expected_item, matched_item, score).
    """
    matched = 0
    details: list[tuple[str, str | None, float]] = []
    used_candidates: set[int] = set()

    for exp in expected:
        best_idx = -1
        best_score = 0.0
        best_candidate: str | None = None

        for i, act in enumerate(actual):
            if i in used_candidates:
                continue

            score = fuzzy_match(exp, act)
            if score > best_score:
                best_score = score
                best_idx = i
                best_candidate = act

        if best_score >= threshold and best_idx >= 0:
            matched += 1
            used_candidates.add(best_idx)
            details.append((exp, best_candidate, best_score))
        else:
            details.append((exp, None, best_score))

    return matched, details
