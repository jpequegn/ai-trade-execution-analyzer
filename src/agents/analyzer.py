"""Trade execution analyzer using LLM for quality analysis.

This module provides the main TradeAnalyzer class that orchestrates
FIX parsing, LLM analysis, and result formatting.

Example:
    >>> from src.agents.analyzer import TradeAnalyzer
    >>> from src.agents import LLMClient
    >>> analyzer = TradeAnalyzer(LLMClient())
    >>> result = analyzer.analyze(execution)
    >>> print(result.analysis.quality_score)

With caching enabled:
    >>> from src.agents.cache import AnalysisCache
    >>> cache = AnalysisCache(backend="file", ttl_hours=24)
    >>> analyzer = TradeAnalyzer(LLMClient(), cache=cache)
    >>> result = analyzer.analyze(execution)
    >>> print(f"From cache: {result.from_cache}")
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING

from src.agents.cache import AnalysisCache, generate_cache_key
from src.agents.llm_client import LLMClient, LLMError
from src.agents.prompts import (
    PromptVariant,
    build_analysis_prompt,
    get_system_prompt,
)
from src.agents.response_parser import (
    ResponseParseError,
    parse_analysis_response,
)
from src.config import AppConfig, get_config
from src.observability.tracing import trace_context, traced
from src.parsers.models import AnalysisResult, TradeAnalysis

if TYPE_CHECKING:
    from src.parsers.fix_parser import ExecutionReport

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Base exception for analysis errors.

    Attributes:
        message: Error description.
        execution: The execution that failed analysis (if available).
        cause: The underlying exception that caused this error.
    """

    def __init__(
        self,
        message: str,
        execution: ExecutionReport | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.execution = execution
        self.cause = cause


class TradeAnalyzer:
    """Analyzer for trade execution quality using LLM.

    This class orchestrates the analysis pipeline:
    1. Checks cache for existing analysis (if caching enabled)
    2. Validates input ExecutionReport
    3. Formats prompt with execution data
    4. Calls LLM with tracing
    5. Parses structured response
    6. Caches result (if caching enabled)
    7. Returns AnalysisResult with metadata

    Attributes:
        client: LLM client for making API calls.
        config: Application configuration.
        cache: Optional cache for storing analysis results.
        default_variant: Default prompt variant to use.
        max_concurrent: Maximum concurrent analyses for batch processing.

    Example:
        >>> from src.agents.analyzer import TradeAnalyzer
        >>> from src.agents import LLMClient
        >>> analyzer = TradeAnalyzer(LLMClient())
        >>> result = analyzer.analyze(execution)
        >>> print(f"Quality: {result.analysis.quality_score}/10")

    With caching:
        >>> from src.agents.cache import AnalysisCache
        >>> cache = AnalysisCache(backend="file", ttl_hours=24)
        >>> analyzer = TradeAnalyzer(LLMClient(), cache=cache)
        >>> result = analyzer.analyze(execution)
        >>> # Second call returns cached result
        >>> result2 = analyzer.analyze(execution)
        >>> print(f"From cache: {result2.from_cache}")  # True
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        config: AppConfig | None = None,
        cache: AnalysisCache | None = None,
        default_variant: PromptVariant = PromptVariant.DETAILED,
        max_concurrent: int = 5,
    ) -> None:
        """Initialize the trade analyzer.

        Args:
            client: LLM client for API calls. Creates new client if not provided.
            config: Application configuration. Loads from environment if not provided.
            cache: Optional cache for storing analysis results.
            default_variant: Default prompt variant for analysis.
            max_concurrent: Maximum concurrent LLM calls for batch processing.
        """
        self.config = config or get_config()
        self.client = client or LLMClient(app_config=self.config)
        self.cache = cache
        self.default_variant = default_variant
        self.max_concurrent = max_concurrent

    @traced(name="analyze_execution", tags=["analysis", "single"])
    def analyze(
        self,
        execution: ExecutionReport,
        variant: PromptVariant | None = None,
        session_id: str | None = None,
        skip_cache: bool = False,
    ) -> AnalysisResult:
        """Analyze a single trade execution.

        Args:
            execution: The parsed execution report to analyze.
            variant: Prompt variant to use. Defaults to instance default.
            session_id: Optional session ID for tracing grouping.
            skip_cache: If True, bypass cache lookup and storage.

        Returns:
            AnalysisResult containing the analysis and metadata.

        Raises:
            AnalysisError: If analysis fails after retries.

        Example:
            >>> result = analyzer.analyze(execution)
            >>> print(result.analysis.observations)
        """
        analysis_id = str(uuid.uuid4())
        variant = variant or self.default_variant

        # Check cache first (if caching enabled and not skipped)
        cache_key = None
        if self.cache and not skip_cache:
            cache_key = generate_cache_key(execution, strategy=self.cache.key_strategy)
            cached_analysis = self.cache.get(cache_key)
            if cached_analysis:
                logger.info(f"Cache hit for order {execution.order_id} (key={cache_key})")
                return AnalysisResult(
                    execution=execution,
                    analysis=cached_analysis,
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model="cached",
                    from_cache=True,
                    analysis_id=analysis_id,
                    analyzed_at=datetime.now(),
                )

        logger.info(
            f"Starting analysis for order {execution.order_id} "
            f"(variant={variant.value}, id={analysis_id})"
        )

        try:
            # Build the prompt
            prompt = build_analysis_prompt(execution, variant)
            system_prompt = get_system_prompt()

            # Call LLM
            messages = [{"role": "user", "content": prompt}]
            response = self.client.complete(messages, system=system_prompt)

            # Parse response
            try:
                analysis = parse_analysis_response(response.content)
            except ResponseParseError as e:
                logger.warning(f"Failed to parse response, using fallback: {e}")
                # Create a fallback analysis
                analysis = TradeAnalysis(
                    quality_score=5,
                    observations=["Analysis parsing failed"],
                    issues=["Could not parse LLM response"],
                    recommendations=["Retry analysis"],
                    confidence=0.1,
                )

            # Build result
            result = AnalysisResult(
                execution=execution,
                analysis=analysis,
                raw_response=response.content,
                tokens_used=response.total_tokens,
                latency_ms=response.latency_ms,
                model=response.model,
                from_cache=False,
                analysis_id=analysis_id,
                analyzed_at=datetime.now(),
            )

            # Cache the result (if caching enabled and not skipped)
            if self.cache and not skip_cache and cache_key:
                self.cache.set(cache_key, analysis, execution)
                logger.debug(f"Cached analysis for order {execution.order_id}")

            logger.info(
                f"Completed analysis for order {execution.order_id}: "
                f"score={analysis.quality_score}, "
                f"tokens={response.total_tokens}, "
                f"latency={response.latency_ms:.0f}ms"
            )

            return result

        except LLMError as e:
            logger.error(f"LLM error analyzing order {execution.order_id}: {e}")
            raise AnalysisError(
                f"LLM error: {e}",
                execution=execution,
                cause=e,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error analyzing order {execution.order_id}: {e}")
            raise AnalysisError(
                f"Analysis failed: {e}",
                execution=execution,
                cause=e,
            ) from e

    def get_cache_stats(self) -> dict[str, object] | None:
        """Get cache statistics if caching is enabled.

        Returns:
            Cache statistics dictionary or None if caching is disabled.
        """
        if self.cache:
            return self.cache.get_stats().to_dict()
        return None

    @traced(name="analyze_batch", tags=["analysis", "batch"])
    def analyze_batch(
        self,
        executions: list[ExecutionReport],
        variant: PromptVariant | None = None,
        session_id: str | None = None,
        continue_on_error: bool = True,
    ) -> list[AnalysisResult | AnalysisError]:
        """Analyze multiple trade executions concurrently.

        Uses a thread pool to process executions in parallel,
        up to the configured max_concurrent limit.

        Args:
            executions: List of execution reports to analyze.
            variant: Prompt variant to use for all analyses.
            session_id: Optional session ID for tracing grouping.
            continue_on_error: If True, continue processing on errors.
                Failed analyses return AnalysisError instead of raising.

        Returns:
            List of results in the same order as input executions.
            Each item is either AnalysisResult or AnalysisError.

        Example:
            >>> results = analyzer.analyze_batch(executions)
            >>> for result in results:
            ...     if isinstance(result, AnalysisResult):
            ...         print(f"{result.execution.order_id}: {result.analysis.quality_score}")
        """
        if not executions:
            return []

        variant = variant or self.default_variant
        batch_id = str(uuid.uuid4())

        logger.info(
            f"Starting batch analysis of {len(executions)} executions "
            f"(batch_id={batch_id}, max_concurrent={self.max_concurrent})"
        )

        results: list[AnalysisResult | AnalysisError | None] = [None] * len(executions)

        with (
            trace_context(
                name="batch_analysis",
                session_id=session_id,
                tags=["batch"],
                metadata={"batch_id": batch_id, "count": len(executions)},
            ),
            ThreadPoolExecutor(max_workers=self.max_concurrent) as executor,
        ):
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self._analyze_single,
                    execution,
                    variant,
                    session_id,
                ): i
                for i, execution in enumerate(executions)
            }

            # Collect results as they complete
            completed = 0
            errors = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                completed += 1  # noqa: SIM113

                try:
                    result = future.result()
                    results[index] = result
                except AnalysisError as e:
                    errors += 1
                    if continue_on_error:
                        results[index] = e
                    else:
                        raise

                if completed % 10 == 0 or completed == len(executions):
                    logger.info(
                        f"Batch progress: {completed}/{len(executions)} " f"({errors} errors)"
                    )

        # Filter out None values and cast
        final_results: list[AnalysisResult | AnalysisError] = [r for r in results if r is not None]

        logger.info(
            f"Completed batch analysis: {len(executions)} executions, "
            f"{len([r for r in final_results if isinstance(r, AnalysisResult)])} successful, "
            f"{len([r for r in final_results if isinstance(r, AnalysisError)])} failed"
        )

        return final_results

    def _analyze_single(
        self,
        execution: ExecutionReport,
        variant: PromptVariant,
        session_id: str | None,
    ) -> AnalysisResult:
        """Internal method for single analysis (used by batch processing).

        This method is separate to allow it to be called from thread pool
        without the @traced decorator creating nested traces.
        """
        return self.analyze(execution, variant=variant, session_id=session_id)

    async def analyze_async(
        self,
        execution: ExecutionReport,
        variant: PromptVariant | None = None,
        session_id: str | None = None,
    ) -> AnalysisResult:
        """Async version of analyze for use in async contexts.

        Args:
            execution: The parsed execution report to analyze.
            variant: Prompt variant to use. Defaults to instance default.
            session_id: Optional session ID for tracing grouping.

        Returns:
            AnalysisResult containing the analysis and metadata.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.analyze(execution, variant=variant, session_id=session_id),
        )

    async def analyze_batch_async(
        self,
        executions: list[ExecutionReport],
        variant: PromptVariant | None = None,
        session_id: str | None = None,
        continue_on_error: bool = True,
    ) -> list[AnalysisResult | AnalysisError]:
        """Async version of analyze_batch.

        Args:
            executions: List of execution reports to analyze.
            variant: Prompt variant to use for all analyses.
            session_id: Optional session ID for tracing grouping.
            continue_on_error: If True, continue processing on errors.

        Returns:
            List of results in the same order as input executions.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.analyze_batch(
                executions,
                variant=variant,
                session_id=session_id,
                continue_on_error=continue_on_error,
            ),
        )
