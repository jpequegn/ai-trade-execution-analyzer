"""Tests for fuzzy matching utilities."""

from src.evaluation.matching import (
    MatchResult,
    best_match,
    count_matches,
    expand_synonyms,
    extract_keywords,
    find_all_matches,
    fuzzy_match,
    keyword_overlap,
    normalize_text,
    sequence_similarity,
)


class TestNormalizeText:
    """Tests for text normalization."""

    def test_lowercase(self) -> None:
        """Test lowercasing."""
        assert normalize_text("HELLO") == "hello"

    def test_strip_whitespace(self) -> None:
        """Test whitespace stripping."""
        assert normalize_text("  hello  ") == "hello"

    def test_collapse_spaces(self) -> None:
        """Test collapsing multiple spaces."""
        assert normalize_text("hello   world") == "hello world"

    def test_mixed_normalization(self) -> None:
        """Test combined normalization."""
        assert normalize_text("  HELLO   World  ") == "hello world"


class TestExtractKeywords:
    """Tests for keyword extraction."""

    def test_basic_extraction(self) -> None:
        """Test basic keyword extraction."""
        keywords = extract_keywords("Poor venue selection")
        assert "poor" in keywords
        assert "venue" in keywords
        assert "selection" in keywords

    def test_stop_words_removed(self) -> None:
        """Test stop words are filtered out."""
        keywords = extract_keywords("The execution was on the NYSE")
        assert "the" not in keywords
        assert "was" not in keywords
        assert "on" not in keywords
        assert "execution" in keywords
        assert "nyse" in keywords

    def test_short_words_filtered(self) -> None:
        """Test short words (<=2 chars) are filtered."""
        keywords = extract_keywords("Go to NY")
        assert "go" not in keywords
        assert "to" not in keywords
        assert "ny" not in keywords  # Only 2 chars


class TestExpandSynonyms:
    """Tests for synonym expansion."""

    def test_direct_synonym(self) -> None:
        """Test direct synonym expansion."""
        expanded = expand_synonyms({"venue"})
        assert "exchange" in expanded
        assert "market" in expanded
        assert "trading venue" in expanded

    def test_reverse_synonym(self) -> None:
        """Test reverse synonym lookup."""
        expanded = expand_synonyms({"exchange"})
        assert "venue" in expanded
        assert "market" in expanded

    def test_no_synonyms(self) -> None:
        """Test terms without synonyms."""
        expanded = expand_synonyms({"xyz123"})
        assert "xyz123" in expanded
        assert len(expanded) == 1


class TestSequenceSimilarity:
    """Tests for sequence-based similarity."""

    def test_identical_strings(self) -> None:
        """Test identical strings return 1.0."""
        assert sequence_similarity("hello", "hello") == 1.0

    def test_different_case(self) -> None:
        """Test case-insensitive comparison."""
        assert sequence_similarity("HELLO", "hello") == 1.0

    def test_completely_different(self) -> None:
        """Test completely different strings."""
        score = sequence_similarity("abc", "xyz")
        assert score < 0.5

    def test_partial_match(self) -> None:
        """Test partially matching strings."""
        score = sequence_similarity("venue selection", "venue choice")
        assert 0.5 <= score <= 0.9


class TestKeywordOverlap:
    """Tests for keyword overlap calculation."""

    def test_identical_text(self) -> None:
        """Test identical text returns 1.0."""
        assert keyword_overlap("venue selection", "venue selection") == 1.0

    def test_synonym_match(self) -> None:
        """Test synonym-enhanced matching."""
        # "suboptimal" and "poor" are synonyms
        score = keyword_overlap("suboptimal venue", "poor venue")
        assert score > 0.5

    def test_no_overlap(self) -> None:
        """Test non-overlapping keywords."""
        score = keyword_overlap("venue selection", "timing execution")
        assert score < 0.3

    def test_without_synonyms(self) -> None:
        """Test matching without synonym expansion."""
        score_with = keyword_overlap("suboptimal venue", "poor venue", use_synonyms=True)
        score_without = keyword_overlap("suboptimal venue", "poor venue", use_synonyms=False)
        assert score_with > score_without


class TestFuzzyMatch:
    """Tests for combined fuzzy matching."""

    def test_exact_match(self) -> None:
        """Test exact match returns 1.0."""
        assert fuzzy_match("venue selection", "venue selection") == 1.0

    def test_case_insensitive(self) -> None:
        """Test case-insensitive exact match."""
        assert fuzzy_match("VENUE SELECTION", "venue selection") == 1.0

    def test_empty_string(self) -> None:
        """Test empty strings return 0.0."""
        assert fuzzy_match("", "venue") == 0.0
        assert fuzzy_match("venue", "") == 0.0

    def test_semantic_match(self) -> None:
        """Test semantically similar phrases match well."""
        score = fuzzy_match("suboptimal venue selection", "poor venue choice")
        assert score > 0.5

    def test_custom_weights(self) -> None:
        """Test custom weighting."""
        # More weight on sequence = more sensitive to character-level similarity
        score_seq = fuzzy_match("abc", "abd", sequence_weight=0.8, keyword_weight=0.2)
        score_kw = fuzzy_match("abc", "abd", sequence_weight=0.2, keyword_weight=0.8)
        # Character difference matters more with higher sequence weight
        assert score_seq > score_kw


class TestBestMatch:
    """Tests for best match finding."""

    def test_exact_match_returns_immediately(self) -> None:
        """Test exact match is returned immediately."""
        result = best_match("venue issue", ["venue issue", "timing problem"])
        assert result is not None
        assert result.candidate == "venue issue"
        assert result.score == 1.0
        assert result.match_type == "exact"

    def test_fuzzy_match(self) -> None:
        """Test fuzzy match selection."""
        result = best_match("poor venue", ["bad venue choice", "timing delay"])
        assert result is not None
        assert result.candidate == "bad venue choice"
        assert result.match_type == "fuzzy"
        assert result.score >= 0.5

    def test_threshold_filtering(self) -> None:
        """Test threshold filters out low matches."""
        result = best_match("venue issue", ["unrelated topic"], threshold=0.8)
        assert result is None

    def test_empty_query(self) -> None:
        """Test empty query returns None."""
        assert best_match("", ["candidate"]) is None

    def test_empty_candidates(self) -> None:
        """Test empty candidates returns None."""
        assert best_match("query", []) is None


class TestFindAllMatches:
    """Tests for finding all matches."""

    def test_multiple_matches(self) -> None:
        """Test finding multiple matches."""
        matches = find_all_matches(
            "venue",
            ["venue selection", "venue issue", "timing problem"],
            threshold=0.3,
        )
        assert len(matches) >= 2

    def test_sorted_by_score(self) -> None:
        """Test results are sorted by score descending."""
        matches = find_all_matches(
            "venue issue",
            ["venue problem", "venue selection issue", "timing delay"],
            threshold=0.3,
        )
        if len(matches) >= 2:
            assert matches[0].score >= matches[1].score

    def test_exact_match_included(self) -> None:
        """Test exact matches are included."""
        matches = find_all_matches("exact", ["exact", "similar"])
        assert any(m.match_type == "exact" for m in matches)


class TestCountMatches:
    """Tests for counting matches between expected and actual lists."""

    def test_perfect_matches(self) -> None:
        """Test all expected items match."""
        expected = ["venue issue", "timing problem"]
        actual = ["venue issue", "timing problem"]
        matched, details = count_matches(expected, actual)
        assert matched == 2
        assert all(d[1] is not None for d in details)

    def test_partial_matches(self) -> None:
        """Test some expected items match."""
        expected = ["venue issue", "timing problem", "unmatched"]
        actual = ["venue problem", "timing delay"]
        matched, details = count_matches(expected, actual, threshold=0.4)
        assert matched == 2
        assert any(d[1] is None for d in details)

    def test_no_matches(self) -> None:
        """Test no matches found."""
        expected = ["abc", "def"]
        actual = ["xyz", "uvw"]
        matched, details = count_matches(expected, actual)
        assert matched == 0
        assert all(d[1] is None for d in details)

    def test_candidate_used_once(self) -> None:
        """Test each candidate can only match once."""
        expected = ["venue", "venue issue"]
        actual = ["venue"]  # Only one candidate
        matched, _details = count_matches(expected, actual)
        # Should only match one, not both
        assert matched == 1

    def test_detail_structure(self) -> None:
        """Test detail tuple structure."""
        expected = ["venue issue"]
        actual = ["venue problem"]
        _, details = count_matches(expected, actual, threshold=0.3)
        assert len(details) == 1
        exp, _match, score = details[0]
        assert exp == "venue issue"
        assert isinstance(score, float)


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_attributes(self) -> None:
        """Test MatchResult has expected attributes."""
        result = MatchResult(candidate="test", score=0.8, match_type="fuzzy")
        assert result.candidate == "test"
        assert result.score == 0.8
        assert result.match_type == "fuzzy"
