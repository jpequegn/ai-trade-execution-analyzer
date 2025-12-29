"""Tests for the caching module."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.agents.cache import (
    AnalysisCache,
    CacheEntry,
    CacheStats,
    FileCacheBackend,
    SQLiteCacheBackend,
    generate_cache_key,
    is_similar_execution,
)
from src.parsers.fix_parser import ExecutionReport
from src.parsers.models import TradeAnalysis

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_execution() -> ExecutionReport:
    """Create a sample execution report for testing."""
    return ExecutionReport(
        order_id="ORD001",
        symbol="AAPL",
        side="BUY",
        quantity=100,
        price=150.50,
        venue="NYSE",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        fill_type="FULL",
    )


@pytest.fixture
def sample_execution2() -> ExecutionReport:
    """Create a second sample execution for testing."""
    return ExecutionReport(
        order_id="ORD002",
        symbol="GOOGL",
        side="SELL",
        quantity=50,
        price=140.25,
        venue="NASDAQ",
        timestamp=datetime(2024, 1, 15, 11, 0, 0),
        fill_type="PARTIAL",
    )


@pytest.fixture
def sample_analysis() -> TradeAnalysis:
    """Create a sample trade analysis for testing."""
    return TradeAnalysis(
        quality_score=8,
        confidence=0.85,
        observations=["Good execution", "Reasonable price"],
        issues=[],
        recommendations=["Consider larger order sizes"],
    )


@pytest.fixture
def temp_cache_dir() -> Iterator[Path]:
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def file_cache(temp_cache_dir: Path) -> FileCacheBackend:
    """Create a file cache backend for testing."""
    return FileCacheBackend(cache_dir=temp_cache_dir)


@pytest.fixture
def sqlite_cache(temp_cache_dir: Path) -> SQLiteCacheBackend:
    """Create a SQLite cache backend for testing."""
    return SQLiteCacheBackend(db_path=temp_cache_dir / "test.db")


# ============================================================================
# Cache Key Generation Tests
# ============================================================================


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_exact_key_generation(self, sample_execution: ExecutionReport) -> None:
        """Test exact cache key generation."""
        key = generate_cache_key(sample_execution, strategy="exact")
        assert key.startswith("exact:")
        assert len(key) > 10  # Should have hash component

    def test_exact_key_deterministic(self, sample_execution: ExecutionReport) -> None:
        """Test that exact keys are deterministic."""
        key1 = generate_cache_key(sample_execution, strategy="exact")
        key2 = generate_cache_key(sample_execution, strategy="exact")
        assert key1 == key2

    def test_exact_key_different_for_different_executions(
        self,
        sample_execution: ExecutionReport,
        sample_execution2: ExecutionReport,
    ) -> None:
        """Test that different executions get different exact keys."""
        key1 = generate_cache_key(sample_execution, strategy="exact")
        key2 = generate_cache_key(sample_execution2, strategy="exact")
        assert key1 != key2

    def test_semantic_key_generation(self, sample_execution: ExecutionReport) -> None:
        """Test semantic cache key generation."""
        key = generate_cache_key(sample_execution, strategy="semantic")
        assert key.startswith("semantic:")
        assert "AAPL" in key
        assert "BUY" in key

    def test_semantic_key_buckets_similar_quantities(self) -> None:
        """Test that similar quantities get same semantic key."""
        exec1 = ExecutionReport(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.00,
            venue="NYSE",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            fill_type="FULL",
        )
        exec2 = ExecutionReport(
            order_id="ORD002",
            symbol="AAPL",
            side="BUY",
            quantity=105,  # Similar quantity
            price=150.50,  # Similar price
            venue="NYSE",
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
            fill_type="FULL",
        )
        key1 = generate_cache_key(exec1, strategy="semantic")
        key2 = generate_cache_key(exec2, strategy="semantic")
        # Keys should be the same due to bucketing
        assert key1 == key2

    def test_invalid_strategy_raises_error(self, sample_execution: ExecutionReport) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cache key strategy"):
            generate_cache_key(sample_execution, strategy="invalid")


# ============================================================================
# Semantic Similarity Tests
# ============================================================================


class TestSemanticSimilarity:
    """Tests for semantic similarity detection."""

    def test_identical_executions_similar(self, sample_execution: ExecutionReport) -> None:
        """Test that identical executions are similar."""
        assert is_similar_execution(sample_execution, sample_execution)

    def test_different_symbols_not_similar(
        self,
        sample_execution: ExecutionReport,
        sample_execution2: ExecutionReport,
    ) -> None:
        """Test that different symbols are not similar."""
        assert not is_similar_execution(sample_execution, sample_execution2)

    def test_similar_quantities(self, sample_execution: ExecutionReport) -> None:
        """Test similar quantities within tolerance."""
        similar_exec = ExecutionReport(
            order_id="ORD002",
            symbol="AAPL",
            side="BUY",
            quantity=110,  # 10% difference
            price=150.50,
            venue="NYSE",
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
            fill_type="FULL",
        )
        assert is_similar_execution(sample_execution, similar_exec, quantity_tolerance=0.2)

    def test_different_quantities_not_similar(self, sample_execution: ExecutionReport) -> None:
        """Test different quantities outside tolerance."""
        different_exec = ExecutionReport(
            order_id="ORD002",
            symbol="AAPL",
            side="BUY",
            quantity=200,  # 100% difference
            price=150.50,
            venue="NYSE",
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
            fill_type="FULL",
        )
        assert not is_similar_execution(sample_execution, different_exec, quantity_tolerance=0.2)

    def test_different_sides_not_similar(self, sample_execution: ExecutionReport) -> None:
        """Test different sides are not similar."""
        sell_exec = ExecutionReport(
            order_id="ORD002",
            symbol="AAPL",
            side="SELL",  # Different side
            quantity=100,
            price=150.50,
            venue="NYSE",
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
            fill_type="FULL",
        )
        assert not is_similar_execution(sample_execution, sell_exec)


# ============================================================================
# Cache Entry Tests
# ============================================================================


class TestCacheEntry:
    """Tests for CacheEntry model."""

    def test_cache_entry_creation(self, sample_analysis: TradeAnalysis) -> None:
        """Test creating a cache entry."""
        entry = CacheEntry(
            key="test_key",
            analysis=sample_analysis,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
        )
        assert entry.key == "test_key"
        assert entry.analysis == sample_analysis
        assert entry.hit_count == 0

    def test_cache_entry_not_expired(self, sample_analysis: TradeAnalysis) -> None:
        """Test that fresh entry is not expired."""
        entry = CacheEntry(
            key="test_key",
            analysis=sample_analysis,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
        )
        assert not entry.is_expired

    def test_cache_entry_expired(self, sample_analysis: TradeAnalysis) -> None:
        """Test that old entry is expired."""
        entry = CacheEntry(
            key="test_key",
            analysis=sample_analysis,
            created_at=datetime.now() - timedelta(hours=48),
            expires_at=datetime.now() - timedelta(hours=24),
        )
        assert entry.is_expired


# ============================================================================
# File Cache Backend Tests
# ============================================================================


class TestFileCacheBackend:
    """Tests for file-based cache backend."""

    def test_set_and_get(
        self,
        file_cache: FileCacheBackend,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test setting and getting cache entries."""
        entry = CacheEntry(
            key="test_key",
            analysis=sample_analysis,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
        )
        file_cache.set(entry)

        retrieved = file_cache.get("test_key")
        assert retrieved is not None
        assert retrieved.key == "test_key"
        assert retrieved.analysis.quality_score == sample_analysis.quality_score

    def test_get_nonexistent(self, file_cache: FileCacheBackend) -> None:
        """Test getting a nonexistent key returns None."""
        result = file_cache.get("nonexistent_key")
        assert result is None

    def test_delete(
        self,
        file_cache: FileCacheBackend,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test deleting cache entries."""
        entry = CacheEntry(
            key="test_key",
            analysis=sample_analysis,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
        )
        file_cache.set(entry)

        assert file_cache.delete("test_key")
        assert file_cache.get("test_key") is None

    def test_clear(
        self,
        file_cache: FileCacheBackend,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test clearing all cache entries."""
        for i in range(5):
            entry = CacheEntry(
                key=f"test_key_{i}",
                analysis=sample_analysis,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=24),
            )
            file_cache.set(entry)

        cleared = file_cache.clear()
        assert cleared == 5
        assert file_cache.get("test_key_0") is None

    def test_get_stats(
        self,
        file_cache: FileCacheBackend,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test getting cache statistics."""
        entry = CacheEntry(
            key="test_key",
            analysis=sample_analysis,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
        )
        file_cache.set(entry)
        file_cache.get("test_key")
        file_cache.get("nonexistent")

        stats = file_cache.get_stats()
        assert stats.total_entries == 1
        assert stats.hits >= 1
        assert stats.misses >= 1

    def test_cleanup_expired(
        self,
        file_cache: FileCacheBackend,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test cleaning up expired entries."""
        # Add expired entry
        expired_entry = CacheEntry(
            key="expired_key",
            analysis=sample_analysis,
            created_at=datetime.now() - timedelta(hours=48),
            expires_at=datetime.now() - timedelta(hours=24),
        )
        file_cache.set(expired_entry)

        # Add fresh entry
        fresh_entry = CacheEntry(
            key="fresh_key",
            analysis=sample_analysis,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
        )
        file_cache.set(fresh_entry)

        removed = file_cache.cleanup_expired()
        assert removed == 1
        assert file_cache.get("expired_key") is None
        assert file_cache.get("fresh_key") is not None


# ============================================================================
# SQLite Cache Backend Tests
# ============================================================================


class TestSQLiteCacheBackend:
    """Tests for SQLite-based cache backend."""

    def test_set_and_get(
        self,
        sqlite_cache: SQLiteCacheBackend,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test setting and getting cache entries."""
        entry = CacheEntry(
            key="test_key",
            analysis=sample_analysis,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
        )
        sqlite_cache.set(entry)

        retrieved = sqlite_cache.get("test_key")
        assert retrieved is not None
        assert retrieved.key == "test_key"
        assert retrieved.analysis.quality_score == sample_analysis.quality_score

    def test_get_nonexistent(self, sqlite_cache: SQLiteCacheBackend) -> None:
        """Test getting a nonexistent key returns None."""
        result = sqlite_cache.get("nonexistent_key")
        assert result is None

    def test_delete(
        self,
        sqlite_cache: SQLiteCacheBackend,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test deleting cache entries."""
        entry = CacheEntry(
            key="test_key",
            analysis=sample_analysis,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
        )
        sqlite_cache.set(entry)

        assert sqlite_cache.delete("test_key")
        assert sqlite_cache.get("test_key") is None

    def test_clear(
        self,
        sqlite_cache: SQLiteCacheBackend,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test clearing all cache entries."""
        for i in range(5):
            entry = CacheEntry(
                key=f"test_key_{i}",
                analysis=sample_analysis,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=24),
            )
            sqlite_cache.set(entry)

        cleared = sqlite_cache.clear()
        assert cleared == 5
        assert sqlite_cache.get("test_key_0") is None

    def test_get_stats(
        self,
        sqlite_cache: SQLiteCacheBackend,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test getting cache statistics."""
        entry = CacheEntry(
            key="test_key",
            analysis=sample_analysis,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
        )
        sqlite_cache.set(entry)
        sqlite_cache.get("test_key")
        sqlite_cache.get("nonexistent")

        stats = sqlite_cache.get_stats()
        assert stats.total_entries == 1
        assert stats.hits >= 1
        assert stats.misses >= 1


# ============================================================================
# Analysis Cache Tests
# ============================================================================


class TestAnalysisCache:
    """Tests for the main AnalysisCache class."""

    def test_init_file_backend(self, temp_cache_dir: Path) -> None:
        """Test initialization with file backend."""
        cache = AnalysisCache(backend="file", cache_dir=temp_cache_dir)
        assert isinstance(cache.backend, FileCacheBackend)

    def test_init_sqlite_backend(self, temp_cache_dir: Path) -> None:
        """Test initialization with SQLite backend."""
        cache = AnalysisCache(backend="sqlite", cache_dir=temp_cache_dir / "test.db")
        assert isinstance(cache.backend, SQLiteCacheBackend)

    def test_init_invalid_backend(self) -> None:
        """Test initialization with invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend type"):
            AnalysisCache(backend="invalid")

    def test_get_and_set(
        self,
        temp_cache_dir: Path,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test getting and setting with AnalysisCache."""
        cache = AnalysisCache(backend="file", cache_dir=temp_cache_dir)
        cache.set("test_key", sample_analysis)

        retrieved = cache.get("test_key")
        assert retrieved is not None
        assert retrieved.quality_score == sample_analysis.quality_score

    def test_get_for_execution(
        self,
        temp_cache_dir: Path,
        sample_execution: ExecutionReport,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test getting analysis for execution."""
        cache = AnalysisCache(backend="file", cache_dir=temp_cache_dir)
        cache.set_for_execution(sample_execution, sample_analysis)

        retrieved = cache.get_for_execution(sample_execution)
        assert retrieved is not None
        assert retrieved.quality_score == sample_analysis.quality_score

    def test_disabled_cache(
        self,
        temp_cache_dir: Path,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test that disabled cache returns None and doesn't store."""
        cache = AnalysisCache(backend="file", cache_dir=temp_cache_dir, enabled=False)
        cache.set("test_key", sample_analysis)
        assert cache.get("test_key") is None

    def test_invalidate(
        self,
        temp_cache_dir: Path,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test invalidating cache entries."""
        cache = AnalysisCache(backend="file", cache_dir=temp_cache_dir)
        cache.set("test_key", sample_analysis)

        assert cache.invalidate("test_key")
        assert cache.get("test_key") is None

    def test_export_json(
        self,
        temp_cache_dir: Path,
        sample_analysis: TradeAnalysis,
    ) -> None:
        """Test exporting cache to JSON."""
        cache = AnalysisCache(backend="file", cache_dir=temp_cache_dir)
        cache.set("test_key_1", sample_analysis)
        cache.set("test_key_2", sample_analysis)

        export_path = temp_cache_dir / "export.json"
        count = cache.export_json(export_path)

        assert count == 2
        assert export_path.exists()

        with export_path.open() as f:
            data = json.load(f)
        assert len(data) == 2


# ============================================================================
# CacheStats Tests
# ============================================================================


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8

    def test_hit_rate_no_requests(self) -> None:
        """Test hit rate with no requests."""
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        stats = CacheStats(
            total_entries=100,
            hits=80,
            misses=20,
            evictions=5,
            bytes_used=1024,
        )
        data = stats.to_dict()
        assert data["total_entries"] == 100
        assert data["hits"] == 80
        assert data["hit_rate"] == "80.0%"
