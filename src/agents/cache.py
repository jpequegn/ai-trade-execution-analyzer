"""Caching layer for trade execution analysis.

This module provides caching functionality to reduce redundant LLM calls
by storing and reusing analysis results for similar executions.

Example:
    >>> from src.agents.cache import AnalysisCache, generate_cache_key
    >>> cache = AnalysisCache(backend="file")
    >>> key = generate_cache_key(execution)
    >>> cached = cache.get(key)
    >>> if not cached:
    ...     result = analyzer.analyze(execution)
    ...     cache.set(key, result.analysis)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.parsers.models import TradeAnalysis

if TYPE_CHECKING:
    from src.parsers.fix_parser import ExecutionReport

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Key Generation
# ============================================================================


def generate_cache_key(
    execution: ExecutionReport,
    strategy: str = "exact",
) -> str:
    """Generate a cache key for an execution report.

    Args:
        execution: The execution report to generate a key for.
        strategy: Key generation strategy:
            - "exact": Hash of full execution data (strict matching)
            - "semantic": Key based on execution characteristics (fuzzy matching)

    Returns:
        A unique cache key string.

    Example:
        >>> key = generate_cache_key(execution, strategy="exact")
        >>> print(key)
        'exact:a1b2c3d4e5...'
    """
    if strategy == "exact":
        return _generate_exact_key(execution)
    elif strategy == "semantic":
        return _generate_semantic_key(execution)
    else:
        raise ValueError(f"Unknown cache key strategy: {strategy}")


def _generate_exact_key(execution: ExecutionReport) -> str:
    """Generate an exact match cache key from execution data.

    Uses a SHA256 hash of the serialized execution to ensure
    only identical executions match.
    """
    # Create a deterministic representation
    data = {
        "order_id": execution.order_id,
        "symbol": execution.symbol,
        "side": execution.side,
        "quantity": execution.quantity,
        "price": execution.price,
        "venue": execution.venue,
        "fill_type": execution.fill_type,
        "timestamp": execution.timestamp.isoformat(),
    }
    serialized = json.dumps(data, sort_keys=True)
    hash_value = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    return f"exact:{hash_value}"


def _generate_semantic_key(execution: ExecutionReport) -> str:
    """Generate a semantic cache key based on execution characteristics.

    Groups similar executions together by bucketing numeric values
    and ignoring order-specific details like order_id and timestamp.
    """
    # Bucket quantity by order of magnitude
    qty_bucket = _bucket_value(execution.quantity, [100, 1000, 10000, 100000])

    # Bucket price by ranges
    price_bucket = _bucket_value(execution.price, [10, 50, 100, 500, 1000])

    # Create semantic key
    key_parts = [
        execution.symbol.upper(),
        execution.side,
        execution.venue or "UNKNOWN",
        execution.fill_type,
        f"qty:{qty_bucket}",
        f"px:{price_bucket}",
    ]
    return f"semantic:{':'.join(key_parts)}"


def _bucket_value(value: float, thresholds: list[float]) -> str:
    """Bucket a numeric value into named ranges.

    Args:
        value: The value to bucket.
        thresholds: List of threshold values defining bucket boundaries.

    Returns:
        Bucket identifier string.
    """
    for threshold in thresholds:
        if value < threshold:
            return f"lt{threshold}"
    return f"gte{thresholds[-1]}"


# ============================================================================
# Semantic Similarity Detection
# ============================================================================


def is_similar_execution(
    exec1: ExecutionReport,
    exec2: ExecutionReport,
    quantity_tolerance: float = 0.2,
    price_tolerance: float = 0.05,
) -> bool:
    """Check if two executions are similar enough to reuse analysis.

    Two executions are considered similar if they have:
    - Same symbol and side
    - Same venue
    - Same fill type
    - Quantity within tolerance (default 20%)
    - Price within tolerance (default 5%)

    Args:
        exec1: First execution report.
        exec2: Second execution report.
        quantity_tolerance: Maximum relative difference in quantity.
        price_tolerance: Maximum relative difference in price.

    Returns:
        True if executions are similar, False otherwise.

    Example:
        >>> similar = is_similar_execution(exec1, exec2)
        >>> if similar:
        ...     # Can reuse analysis from exec1 for exec2
    """
    # Must match exactly
    if exec1.symbol != exec2.symbol:
        return False
    if exec1.side != exec2.side:
        return False
    if exec1.venue != exec2.venue:
        return False
    if exec1.fill_type != exec2.fill_type:
        return False

    # Check quantity within tolerance
    if exec1.quantity > 0:
        qty_diff = abs(exec1.quantity - exec2.quantity) / exec1.quantity
        if qty_diff > quantity_tolerance:
            return False

    # Check price within tolerance
    if exec1.price > 0:
        price_diff = abs(exec1.price - exec2.price) / exec1.price
        if price_diff > price_tolerance:
            return False

    return True


# ============================================================================
# Cache Entry Model
# ============================================================================


class CacheEntry(BaseModel):
    """A cached analysis entry with metadata.

    Attributes:
        key: The cache key.
        analysis: The cached trade analysis.
        created_at: When the entry was cached.
        expires_at: When the entry expires.
        hit_count: Number of times this entry has been retrieved.
        execution_summary: Brief summary of the cached execution.
    """

    key: str
    analysis: TradeAnalysis
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    execution_summary: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return datetime.now() > self.expires_at


# ============================================================================
# Cache Statistics
# ============================================================================


@dataclass
class CacheStats:
    """Statistics about cache performance.

    Attributes:
        total_entries: Total number of entries in cache.
        hits: Number of cache hits.
        misses: Number of cache misses.
        evictions: Number of entries evicted.
        bytes_used: Approximate storage used in bytes.
    """

    total_entries: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_used: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        """Convert stats to dictionary."""
        return {
            "total_entries": self.total_entries,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": f"{self.hit_rate:.1%}",
            "bytes_used": self.bytes_used,
        }


# ============================================================================
# Cache Backend Interface
# ============================================================================


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        """Get a cache entry by key."""

    @abstractmethod
    def set(self, entry: CacheEntry) -> None:
        """Store a cache entry."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""

    @abstractmethod
    def clear(self) -> int:
        """Clear all cache entries. Returns count of entries cleared."""

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of entries removed."""

    @abstractmethod
    def get_all_entries(self) -> list[CacheEntry]:
        """Get all cache entries (for export). Returns list of entries."""


# ============================================================================
# File-based Cache Backend
# ============================================================================


class FileCacheBackend(CacheBackend):
    """File-based JSON cache backend.

    Stores each cache entry as a separate JSON file in a directory.
    Simple and portable, suitable for single-instance deployments.

    Attributes:
        cache_dir: Directory for storing cache files.
        stats: Cache statistics.
    """

    def __init__(self, cache_dir: str | Path = ".cache/analysis") -> None:
        """Initialize file cache backend.

        Args:
            cache_dir: Directory path for cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stats = CacheStats()
        self._load_stats()
        logger.info(f"Initialized file cache at {self.cache_dir}")

    def _get_file_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Sanitize key for filesystem
        safe_key = key.replace(":", "_").replace("/", "_")
        return self.cache_dir / f"{safe_key}.json"

    def _load_stats(self) -> None:
        """Load stats and count entries."""
        entries = list(self.cache_dir.glob("*.json"))
        self._stats.total_entries = len([f for f in entries if f.name != "_stats.json"])

    def get(self, key: str) -> CacheEntry | None:
        """Get a cache entry by key."""
        file_path = self._get_file_path(key)

        if not file_path.exists():
            self._stats.misses += 1
            return None

        try:
            with file_path.open("r") as f:
                data = json.load(f)
            entry = CacheEntry.model_validate(data)

            if entry.is_expired:
                self.delete(key)
                self._stats.misses += 1
                return None

            # Update hit count
            entry.hit_count += 1
            self.set(entry)  # Save updated hit count
            self._stats.hits += 1
            return entry

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to load cache entry {key}: {e}")
            self._stats.misses += 1
            return None

    def set(self, entry: CacheEntry) -> None:
        """Store a cache entry."""
        file_path = self._get_file_path(entry.key)

        try:
            with file_path.open("w") as f:
                json.dump(entry.model_dump(mode="json"), f, indent=2, default=str)
            self._stats.total_entries = len(
                [f for f in self.cache_dir.glob("*.json") if f.name != "_stats.json"]
            )
        except OSError as e:
            logger.error(f"Failed to write cache entry {entry.key}: {e}")

    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        file_path = self._get_file_path(key)

        if file_path.exists():
            try:
                file_path.unlink()
                self._stats.evictions += 1
                self._stats.total_entries = max(0, self._stats.total_entries - 1)
                return True
            except OSError as e:
                logger.error(f"Failed to delete cache entry {key}: {e}")
        return False

    def clear(self) -> int:
        """Clear all cache entries."""
        count = 0
        for file_path in self.cache_dir.glob("*.json"):
            if file_path.name != "_stats.json":
                try:
                    file_path.unlink()
                    count += 1
                except OSError:
                    pass

        self._stats.total_entries = 0
        self._stats.evictions += count
        logger.info(f"Cleared {count} cache entries")
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        # Calculate bytes used
        total_bytes = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.json") if f.name != "_stats.json"
        )
        self._stats.bytes_used = total_bytes
        return self._stats

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        count = 0
        for file_path in self.cache_dir.glob("*.json"):
            if file_path.name == "_stats.json":
                continue

            try:
                with file_path.open("r") as f:
                    data = json.load(f)
                entry = CacheEntry.model_validate(data)

                if entry.is_expired:
                    file_path.unlink()
                    count += 1
            except (json.JSONDecodeError, ValueError, OSError):
                # Remove corrupted files
                try:
                    file_path.unlink()
                    count += 1
                except OSError:
                    pass

        self._stats.evictions += count
        self._stats.total_entries = len(
            [f for f in self.cache_dir.glob("*.json") if f.name != "_stats.json"]
        )
        logger.info(f"Cleaned up {count} expired cache entries")
        return count

    def get_all_entries(self) -> list[CacheEntry]:
        """Get all cache entries."""
        entries: list[CacheEntry] = []
        for file_path in self.cache_dir.glob("*.json"):
            if file_path.name == "_stats.json":
                continue
            try:
                data = json.loads(file_path.read_text())
                entry = CacheEntry(
                    key=data["key"],
                    analysis=TradeAnalysis.model_validate(data["analysis"]),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    expires_at=datetime.fromisoformat(data["expires_at"]),
                    hit_count=data.get("hit_count", 0),
                    execution_summary=data.get("execution_summary", ""),
                )
                entries.append(entry)
            except (json.JSONDecodeError, ValueError, KeyError, OSError):
                pass
        return entries


# ============================================================================
# SQLite Cache Backend
# ============================================================================


class SQLiteCacheBackend(CacheBackend):
    """SQLite-based cache backend.

    Uses a SQLite database for efficient storage and querying.
    Better for larger datasets and supports concurrent access.

    Attributes:
        db_path: Path to the SQLite database file.
        stats: Cache statistics.
    """

    def __init__(self, db_path: str | Path = ".cache/analysis.db") -> None:
        """Initialize SQLite cache backend.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._stats = CacheStats()
        self._init_db()
        logger.info(f"Initialized SQLite cache at {self.db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    analysis_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    execution_summary TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON cache_entries(expires_at)
            """)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)

    def get(self, key: str) -> CacheEntry | None:
        """Get a cache entry by key."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT key, analysis_json, created_at, expires_at, hit_count, execution_summary "
                "FROM cache_entries WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()

            if not row:
                self._stats.misses += 1
                return None

            try:
                analysis = TradeAnalysis.model_validate_json(row[1])
                entry = CacheEntry(
                    key=row[0],
                    analysis=analysis,
                    created_at=datetime.fromisoformat(row[2]),
                    expires_at=datetime.fromisoformat(row[3]),
                    hit_count=row[4],
                    execution_summary=row[5],
                )

                if entry.is_expired:
                    self.delete(key)
                    self._stats.misses += 1
                    return None

                # Update hit count
                conn.execute(
                    "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE key = ?",
                    (key,),
                )
                conn.commit()
                entry.hit_count += 1
                self._stats.hits += 1
                return entry

            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
                self._stats.misses += 1
                return None

    def set(self, entry: CacheEntry) -> None:
        """Store a cache entry."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                (key, analysis_json, created_at, expires_at, hit_count, execution_summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.key,
                    entry.analysis.model_dump_json(),
                    entry.created_at.isoformat(),
                    entry.expires_at.isoformat(),
                    entry.hit_count,
                    entry.execution_summary,
                ),
            )
            conn.commit()

    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            conn.commit()
            if cursor.rowcount > 0:
                self._stats.evictions += 1
                return True
            return False

    def clear(self) -> int:
        """Clear all cache entries."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM cache_entries")
            count = cursor.rowcount
            conn.commit()
            self._stats.evictions += count
            logger.info(f"Cleared {count} cache entries")
            return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
            self._stats.total_entries = cursor.fetchone()[0]

            # Get database file size
            self._stats.bytes_used = self.db_path.stat().st_size if self.db_path.exists() else 0

        return self._stats

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM cache_entries WHERE expires_at < ?",
                (now,),
            )
            count = cursor.rowcount
            conn.commit()

        self._stats.evictions += count
        logger.info(f"Cleaned up {count} expired cache entries")
        return count

    def get_all_entries(self) -> list[CacheEntry]:
        """Get all cache entries."""
        entries: list[CacheEntry] = []
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT key, analysis_json, created_at, expires_at, hit_count, execution_summary "
                "FROM cache_entries"
            )
            for row in cursor.fetchall():
                try:
                    entry = CacheEntry(
                        key=row[0],
                        analysis=TradeAnalysis.model_validate_json(row[1]),
                        created_at=datetime.fromisoformat(row[2]),
                        expires_at=datetime.fromisoformat(row[3]),
                        hit_count=row[4],
                        execution_summary=row[5] or "",
                    )
                    entries.append(entry)
                except (ValueError, KeyError):
                    pass
        return entries


# ============================================================================
# Main Cache Class
# ============================================================================


@dataclass
class AnalysisCache:
    """Main cache interface for trade analysis caching.

    Provides a unified interface for caching analysis results with
    support for multiple backends and cache key strategies.

    Attributes:
        backend: Cache storage backend.
        ttl_hours: Time-to-live for cache entries in hours.
        key_strategy: Cache key generation strategy.
        enabled: Whether caching is enabled.

    Example:
        >>> cache = AnalysisCache(backend="file", ttl_hours=24)
        >>> key = generate_cache_key(execution)
        >>> cached = cache.get(key)
        >>> if cached is None:
        ...     result = analyzer.analyze(execution)
        ...     cache.set(key, result.analysis, execution)
    """

    backend: CacheBackend = field(default_factory=lambda: FileCacheBackend())
    ttl_hours: int = 24
    key_strategy: str = "exact"
    enabled: bool = True

    def __init__(
        self,
        backend: str | CacheBackend = "file",
        ttl_hours: int = 24,
        key_strategy: str = "exact",
        enabled: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize the analysis cache.

        Args:
            backend: Backend type ("file", "sqlite") or a CacheBackend instance.
            ttl_hours: Time-to-live for cache entries in hours.
            key_strategy: Cache key generation strategy ("exact" or "semantic").
            enabled: Whether caching is enabled.
            cache_dir: Custom directory/path for cache storage.
        """
        self.ttl_hours = ttl_hours
        self.key_strategy = key_strategy
        self.enabled = enabled

        if isinstance(backend, CacheBackend):
            self.backend = backend
        elif backend == "file":
            path = cache_dir or ".cache/analysis"
            self.backend = FileCacheBackend(cache_dir=path)
        elif backend == "sqlite":
            path = cache_dir or ".cache/analysis.db"
            self.backend = SQLiteCacheBackend(db_path=path)
        else:
            raise ValueError(f"Unknown backend type: {backend}")

    def get(self, key: str) -> TradeAnalysis | None:
        """Get a cached analysis by key.

        Args:
            key: The cache key.

        Returns:
            The cached TradeAnalysis if found and not expired, None otherwise.
        """
        if not self.enabled:
            return None

        entry = self.backend.get(key)
        if entry:
            logger.debug(f"Cache hit for key {key}")
            return entry.analysis
        return None

    def get_for_execution(self, execution: ExecutionReport) -> TradeAnalysis | None:
        """Get a cached analysis for an execution report.

        Generates the cache key using the configured strategy
        and retrieves the cached analysis.

        Args:
            execution: The execution report to look up.

        Returns:
            The cached TradeAnalysis if found, None otherwise.
        """
        key = generate_cache_key(execution, strategy=self.key_strategy)
        return self.get(key)

    def set(
        self,
        key: str,
        analysis: TradeAnalysis,
        execution: ExecutionReport | None = None,
    ) -> None:
        """Store an analysis in the cache.

        Args:
            key: The cache key.
            analysis: The analysis to cache.
            execution: Optional execution for summary metadata.
        """
        if not self.enabled:
            return

        summary = ""
        if execution:
            summary = f"{execution.symbol} {execution.side} {execution.quantity}@{execution.price}"

        entry = CacheEntry(
            key=key,
            analysis=analysis,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=self.ttl_hours),
            hit_count=0,
            execution_summary=summary,
        )
        self.backend.set(entry)
        logger.debug(f"Cached analysis with key {key}")

    def set_for_execution(
        self,
        execution: ExecutionReport,
        analysis: TradeAnalysis,
    ) -> str:
        """Store an analysis for an execution report.

        Generates the cache key using the configured strategy
        and stores the analysis.

        Args:
            execution: The execution report.
            analysis: The analysis to cache.

        Returns:
            The generated cache key.
        """
        key = generate_cache_key(execution, strategy=self.key_strategy)
        self.set(key, analysis, execution)
        return key

    def invalidate(self, key: str) -> bool:
        """Invalidate (delete) a cache entry.

        Args:
            key: The cache key to invalidate.

        Returns:
            True if entry was deleted, False if not found.
        """
        if not self.enabled:
            return False
        return self.backend.delete(key)

    def invalidate_for_execution(self, execution: ExecutionReport) -> bool:
        """Invalidate the cache entry for an execution.

        Args:
            execution: The execution report.

        Returns:
            True if entry was deleted, False if not found.
        """
        key = generate_cache_key(execution, strategy=self.key_strategy)
        return self.invalidate(key)

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        return self.backend.clear()

    def cleanup(self) -> int:
        """Clean up expired cache entries.

        Returns:
            Number of entries removed.
        """
        return self.backend.cleanup_expired()

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with current cache metrics.
        """
        return self.backend.get_stats()

    def to_summary(self) -> str:
        """Generate a summary of cache status.

        Returns:
            Human-readable summary string.
        """
        stats = self.get_stats()
        lines = [
            "Analysis Cache Status",
            f"  Enabled: {self.enabled}",
            f"  Backend: {type(self.backend).__name__}",
            f"  TTL: {self.ttl_hours} hours",
            f"  Key Strategy: {self.key_strategy}",
            f"  Total Entries: {stats.total_entries}",
            f"  Hit Rate: {stats.hit_rate:.1%}",
            f"  Hits: {stats.hits}",
            f"  Misses: {stats.misses}",
            f"  Storage Used: {stats.bytes_used:,} bytes",
        ]
        return "\n".join(lines)

    def export_json(self, output_path: Path) -> int:
        """Export cache entries to JSON file.

        Args:
            output_path: Path for the export file.

        Returns:
            Number of entries exported.
        """
        entries = self.backend.get_all_entries()

        data = [
            {
                "key": entry.key,
                "analysis": entry.analysis.model_dump(),
                "created_at": entry.created_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "hit_count": entry.hit_count,
                "execution_summary": entry.execution_summary,
            }
            for entry in entries
        ]

        with output_path.open("w") as f:
            json.dump(data, f, indent=2)

        return len(data)
