# API Reference

This document provides complete API documentation for the AI Trade Execution Analyzer.

## Table of Contents

- [Pipeline](#pipeline)
- [FIX Parser](#fix-parser)
- [Data Models](#data-models)
- [Trade Analyzer](#trade-analyzer)
- [LLM Client](#llm-client)
- [Prompts](#prompts)
- [Response Parser](#response-parser)
- [Caching](#caching)
- [Observability](#observability)
- [Evaluation](#evaluation)
- [Human Review](#human-review)
- [CLI](#cli)
- [Configuration](#configuration)
- [Exceptions](#exceptions)

---

## Pipeline

The main entry point for trade execution analysis.

**Module:** `src.pipeline`

### `analyze_fix_message`

Parse and analyze a FIX protocol message.

```python
def analyze_fix_message(
    raw_message: str,
    variant: PromptVariant = PromptVariant.DETAILED,
    config: AppConfig | None = None,
    session_id: str | None = None,
) -> AnalysisResult
```

**Parameters:**
- `raw_message` (str): Raw FIX message string with `|` delimiters
- `variant` (PromptVariant): Analysis depth - `QUICK`, `DETAILED`, or `BATCH`
- `config` (AppConfig | None): Configuration override
- `session_id` (str | None): Session ID for tracing

**Returns:**
- `AnalysisResult`: Complete analysis with execution data and AI insights

**Raises:**
- `FIXParseError`: If message is malformed
- `FIXMissingFieldError`: If required field missing
- `AnalysisError`: If LLM analysis fails

**Example:**
```python
from src.pipeline import analyze_fix_message

result = analyze_fix_message(
    "8=FIX.4.4|35=8|49=BROKER|56=CLIENT|37=ORD001|"
    "11=CLT001|17=EXEC001|20=0|150=F|39=2|55=AAPL|"
    "54=1|38=100|44=150.50|32=100|31=150.45|14=100|"
    "6=150.45|60=20240115-14:30:00.000|10=123|"
)

print(f"Quality Score: {result.analysis.quality_score}/10")
print(f"Issues: {result.analysis.issues}")
```

---

### `analyze_fix_messages`

Batch analyze multiple FIX messages concurrently.

```python
def analyze_fix_messages(
    raw_messages: list[str],
    variant: PromptVariant = PromptVariant.DETAILED,
    config: AppConfig | None = None,
    session_id: str | None = None,
    continue_on_error: bool = True,
    max_concurrent: int = 5,
) -> list[AnalysisResult | AnalysisError]
```

**Parameters:**
- `raw_messages` (list[str]): List of raw FIX messages
- `variant` (PromptVariant): Analysis depth
- `config` (AppConfig | None): Configuration override
- `session_id` (str | None): Session ID for tracing
- `continue_on_error` (bool): Continue if individual analyses fail
- `max_concurrent` (int): Maximum concurrent LLM calls

**Returns:**
- `list[AnalysisResult | AnalysisError]`: Results or errors for each message

**Example:**
```python
from src.pipeline import analyze_fix_messages

messages = [
    "8=FIX.4.4|35=8|...",
    "8=FIX.4.4|35=8|...",
]

results = analyze_fix_messages(messages, max_concurrent=3)

for result in results:
    if isinstance(result, AnalysisError):
        print(f"Error: {result.message}")
    else:
        print(f"Score: {result.analysis.quality_score}")
```

---

### `analyze_execution`

Analyze a pre-parsed ExecutionReport.

```python
def analyze_execution(
    execution: ExecutionReport,
    variant: PromptVariant = PromptVariant.DETAILED,
    config: AppConfig | None = None,
    session_id: str | None = None,
) -> AnalysisResult
```

**Parameters:**
- `execution` (ExecutionReport): Parsed execution data
- `variant` (PromptVariant): Analysis depth
- `config` (AppConfig | None): Configuration override
- `session_id` (str | None): Session ID for tracing

**Returns:**
- `AnalysisResult`: Complete analysis result

---

### `format_result_for_display`

Format analysis result for console output.

```python
def format_result_for_display(result: AnalysisResult) -> str
```

**Example:**
```python
from src.pipeline import analyze_fix_message, format_result_for_display

result = analyze_fix_message("8=FIX.4.4|35=8|...")
print(format_result_for_display(result))
```

---

### `format_result_as_json`

Format analysis result as JSON string.

```python
def format_result_as_json(result: AnalysisResult) -> str
```

---

## FIX Parser

Parse FIX protocol messages into structured data.

**Module:** `src.parsers.fix_parser`

### `parse_fix_message`

Parse a FIX message string into an ExecutionReport.

```python
def parse_fix_message(raw: str) -> ExecutionReport
```

**Parameters:**
- `raw` (str): Raw FIX message with `|` as field delimiter

**Returns:**
- `ExecutionReport`: Structured execution data

**Raises:**
- `FIXParseError`: If message format is invalid
- `FIXMissingFieldError`: If required field is missing
- `FIXValidationError`: If field value is invalid

**Example:**
```python
from src.parsers import parse_fix_message

execution = parse_fix_message(
    "8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|"
    "38=100|32=100|31=150.50|60=20240115-14:30:00.000|10=123|"
)

print(f"Symbol: {execution.symbol}")
print(f"Side: {execution.side}")  # "BUY" or "SELL"
print(f"Price: ${execution.price}")
print(f"Quantity: {execution.quantity}")
```

---

### `tokenize_fix_message`

Tokenize a FIX message into tag-value pairs.

```python
def tokenize_fix_message(raw: str) -> dict[int, str]
```

**Parameters:**
- `raw` (str): Raw FIX message

**Returns:**
- `dict[int, str]`: Mapping of tag numbers to values

**Example:**
```python
from src.parsers.fix_parser import tokenize_fix_message

tokens = tokenize_fix_message("8=FIX.4.4|35=8|55=AAPL|")
print(tokens[55])  # "AAPL"
```

---

### FIX Tag Constants

```python
# Common FIX tags
TAG_SYMBOL = 55          # Trading symbol
TAG_ORDER_ID = 37        # Order ID
TAG_EXEC_ID = 17         # Execution ID
TAG_SIDE = 54            # Side (1=Buy, 2=Sell)
TAG_ORDER_QTY = 38       # Order quantity
TAG_LAST_QTY = 32        # Last fill quantity
TAG_LAST_PX = 31         # Last fill price
TAG_CUM_QTY = 14         # Cumulative quantity
TAG_AVG_PX = 6           # Average price
TAG_TRANSACT_TIME = 60   # Transaction time
TAG_EXEC_TYPE = 150      # Execution type
TAG_ORD_STATUS = 39      # Order status

# Side values
SIDE_BUY = "1"
SIDE_SELL = "2"

# Order status values
ORD_STATUS_FILLED = "2"
ORD_STATUS_PARTIAL = "1"
```

---

## Data Models

Pydantic models for structured data.

**Module:** `src.parsers.models`

### `ExecutionReport`

Represents a parsed FIX execution report.

```python
class ExecutionReport(BaseModel):
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float
    price: float
    venue: str = "UNKNOWN"
    timestamp: datetime
    fill_type: Literal["FULL", "PARTIAL"] = "FULL"
    exec_type: str | None = None
    cum_qty: float | None = None
    avg_px: float | None = None
    fix_version: str = "FIX.4.4"
```

**Fields:**
- `order_id`: Unique order identifier
- `symbol`: Trading symbol (e.g., "AAPL", "MSFT")
- `side`: Trade direction - "BUY" or "SELL"
- `quantity`: Fill quantity
- `price`: Execution price
- `venue`: Execution venue (default: "UNKNOWN")
- `timestamp`: Execution timestamp
- `fill_type`: "FULL" or "PARTIAL"
- `exec_type`: FIX execution type code
- `cum_qty`: Cumulative filled quantity
- `avg_px`: Average execution price
- `fix_version`: FIX protocol version

---

### `TradeAnalysis`

AI-generated analysis of a trade execution.

```python
class TradeAnalysis(BaseModel):
    quality_score: int  # 1-10
    observations: list[str]
    issues: list[str]
    recommendations: list[str]
    confidence: float  # 0.0-1.0
```

**Fields:**
- `quality_score`: Overall execution quality (1-10)
- `observations`: Factual observations about the execution
- `issues`: Identified problems or concerns
- `recommendations`: Suggested improvements
- `confidence`: Model's confidence in the analysis

**Validators:**
- `quality_score` must be between 1 and 10
- `confidence` must be between 0.0 and 1.0
- Lists default to empty if not provided

---

### `AnalysisResult`

Complete analysis result combining execution and analysis.

```python
class AnalysisResult(BaseModel):
    execution: ExecutionReport
    analysis: TradeAnalysis
    raw_response: str
    tokens_used: int
    latency_ms: float
    model: str
    from_cache: bool = False
    analysis_id: str | None = None
    analyzed_at: datetime
```

**Fields:**
- `execution`: Original execution report
- `analysis`: AI-generated analysis
- `raw_response`: Raw LLM response text
- `tokens_used`: Total tokens consumed
- `latency_ms`: Analysis latency in milliseconds
- `model`: Model identifier used
- `from_cache`: Whether result was from cache
- `analysis_id`: Unique analysis identifier
- `analyzed_at`: Timestamp of analysis

---

## Trade Analyzer

Orchestrates trade execution analysis.

**Module:** `src.agents.analyzer`

### `TradeAnalyzer`

Main analyzer class that coordinates LLM calls and caching.

```python
class TradeAnalyzer:
    def __init__(
        self,
        client: LLMClient | None = None,
        config: AppConfig | None = None,
        cache: AnalysisCache | None = None,
        default_variant: PromptVariant = PromptVariant.DETAILED,
        max_concurrent: int = 5,
    )
```

**Parameters:**
- `client`: LLM client instance (created if not provided)
- `config`: Application configuration
- `cache`: Optional analysis cache
- `default_variant`: Default prompt variant
- `max_concurrent`: Maximum concurrent analyses

**Methods:**

#### `analyze`

```python
def analyze(
    self,
    execution: ExecutionReport,
    variant: PromptVariant | None = None,
    skip_cache: bool = False,
) -> AnalysisResult
```

Analyze a single execution.

#### `analyze_batch`

```python
async def analyze_batch(
    self,
    executions: list[ExecutionReport],
    variant: PromptVariant | None = None,
) -> list[AnalysisResult | AnalysisError]
```

Analyze multiple executions concurrently.

#### `get_cache_stats`

```python
def get_cache_stats(self) -> CacheStats | None
```

Get cache statistics if cache is enabled.

**Example:**
```python
from src.agents.analyzer import TradeAnalyzer
from src.agents.cache import AnalysisCache
from src.parsers import parse_fix_message

# Create analyzer with cache
cache = AnalysisCache()
analyzer = TradeAnalyzer(cache=cache)

# Analyze execution
execution = parse_fix_message("8=FIX.4.4|35=8|...")
result = analyzer.analyze(execution)

# Check cache stats
stats = analyzer.get_cache_stats()
print(f"Cache hits: {stats.hits}, misses: {stats.misses}")
```

---

### `AnalysisError`

Exception raised when analysis fails.

```python
class AnalysisError(Exception):
    def __init__(
        self,
        message: str,
        execution: ExecutionReport | None = None,
        cause: Exception | None = None,
    )
```

**Attributes:**
- `message`: Error description
- `execution`: The execution that failed
- `cause`: Underlying exception

---

## LLM Client

Unified client for LLM providers (Anthropic, OpenAI).

**Module:** `src.agents.llm_client`

### `LLMClient`

```python
class LLMClient:
    def __init__(self, config: LLMConfig | None = None)
```

**Parameters:**
- `config`: LLM configuration (uses default config if not provided)

**Methods:**

#### `complete`

```python
def complete(self, messages: list[dict]) -> LLMResponse
```

Call the LLM API with messages.

**Parameters:**
- `messages`: List of message dicts with `role` and `content`

**Returns:**
- `LLMResponse`: Response with content and usage info

**Example:**
```python
from src.agents.llm_client import LLMClient

client = LLMClient()
response = client.complete([
    {"role": "system", "content": "You are a trading expert."},
    {"role": "user", "content": "Analyze this trade..."},
])

print(response.content)
print(f"Tokens: {response.total_tokens}")
```

---

### `LLMResponse`

Response from an LLM call.

```python
class LLMResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    provider: str

    @property
    def total_tokens(self) -> int
```

---

### LLM Exceptions

```python
class LLMError(Exception):
    """Base LLM error"""

class LLMRateLimitError(LLMError):
    """Rate limit exceeded"""
    retry_after: float | None

class LLMTimeoutError(LLMError):
    """Request timed out"""

class LLMProviderError(LLMError):
    """Provider returned an error"""
    status_code: int | None
```

---

## Prompts

Prompt templates for trade analysis.

**Module:** `src.agents.prompts`

### `PromptVariant`

```python
class PromptVariant(Enum):
    QUICK = "quick"       # Cost-optimized, brief analysis
    DETAILED = "detailed" # Comprehensive analysis
    BATCH = "batch"       # Multi-trade batch analysis
```

---

### `build_analysis_prompt`

Build a prompt for single trade analysis.

```python
def build_analysis_prompt(
    execution: ExecutionReport,
    variant: PromptVariant = PromptVariant.DETAILED,
) -> str
```

---

### `build_batch_prompt`

Build a prompt for batch trade analysis.

```python
def build_batch_prompt(
    executions: list[ExecutionReport],
    variant: PromptVariant = PromptVariant.BATCH,
) -> str
```

---

### `get_system_prompt`

Get the system prompt for a variant.

```python
def get_system_prompt(variant: PromptVariant) -> str
```

---

### `estimate_prompt_tokens`

Estimate token count for a prompt.

```python
def estimate_prompt_tokens(prompt: str) -> int
```

---

## Response Parser

Parse LLM responses into structured data.

**Module:** `src.agents.response_parser`

### `parse_analysis_response`

Parse LLM response into TradeAnalysis.

```python
def parse_analysis_response(response: str) -> TradeAnalysis
```

**Raises:**
- `ResponseParseError`: If parsing fails

---

### `parse_batch_response`

Parse batch response into list of analyses.

```python
def parse_batch_response(response: str) -> list[TradeAnalysis]
```

---

### `safe_parse_analysis`

Safely parse with fallback to defaults.

```python
def safe_parse_analysis(response: str) -> TradeAnalysis | None
```

---

### `extract_json_from_response`

Extract JSON from LLM response text.

```python
def extract_json_from_response(response: str) -> str
```

---

### `ResponseParseError`

```python
class ResponseParseError(Exception):
    def __init__(
        self,
        message: str,
        raw_response: str,
        reason: str | None = None,
    )
```

---

## Caching

Cache analysis results to reduce LLM costs.

**Module:** `src.agents.cache`

### `AnalysisCache`

Main cache interface.

```python
class AnalysisCache:
    def __init__(
        self,
        backend: str = "sqlite",
        storage_path: Path | None = None,
        ttl_hours: int = 168,  # 7 days
        key_strategy: str = "semantic",
    )
```

**Parameters:**
- `backend`: "sqlite" or "file"
- `storage_path`: Path for cache storage
- `ttl_hours`: Time-to-live in hours
- `key_strategy`: "exact" or "semantic"

**Methods:**

#### `get`

```python
def get(
    self,
    execution: ExecutionReport,
) -> CacheEntry | None
```

Get cached analysis for an execution.

#### `set`

```python
def set(
    self,
    execution: ExecutionReport,
    result: AnalysisResult,
) -> None
```

Store analysis result in cache.

#### `get_stats`

```python
def get_stats(self) -> CacheStats
```

Get cache statistics.

#### `clear`

```python
def clear(self) -> int
```

Clear all cache entries. Returns count cleared.

#### `cleanup_expired`

```python
def cleanup_expired(self) -> int
```

Remove expired entries. Returns count removed.

---

### `CacheEntry`

```python
class CacheEntry(BaseModel):
    key: str
    result: AnalysisResult
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
```

---

### `CacheStats`

```python
class CacheStats(BaseModel):
    hits: int
    misses: int
    size: int
    oldest_entry: datetime | None
    hit_rate: float  # Computed property
```

---

### `generate_cache_key`

Generate cache key for an execution.

```python
def generate_cache_key(
    execution: ExecutionReport,
    strategy: str = "semantic",
) -> str
```

**Strategies:**
- `exact`: SHA256 hash of all fields
- `semantic`: Hash of key characteristics (symbol, side, price bucket)

---

### `is_similar_execution`

Check if two executions are semantically similar.

```python
def is_similar_execution(
    exec1: ExecutionReport,
    exec2: ExecutionReport,
    threshold: float = 0.9,
) -> bool
```

---

## Observability

Tracing and cost tracking.

### Tracing

**Module:** `src.observability.tracing`

#### `get_tracer`

Get the global tracer instance.

```python
def get_tracer() -> Tracer
```

#### `trace_context`

Context manager for tracing operations.

```python
@contextmanager
def trace_context(
    name: str,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
)
```

**Example:**
```python
from src.observability.tracing import trace_context

with trace_context("analyze_trade", session_id="sess-123"):
    result = analyze_fix_message("8=FIX.4.4|...")
```

#### `@traced`

Decorator for function tracing.

```python
@traced("operation_name")
def my_function():
    pass
```

---

### Cost Tracking

**Module:** `src.observability.cost_tracker`

#### `CostTracker`

Track LLM costs and manage budgets.

```python
class CostTracker:
    def __init__(
        self,
        storage_path: Path | None = None,
        daily_budget: float | None = None,
        monthly_budget: float | None = None,
        warning_threshold: float = 0.8,
    )
```

**Methods:**

##### `record_usage`

```python
def record_usage(
    self,
    analysis_id: str,
    tokens: TokenUsage,
    model: str,
    provider: str = "anthropic",
) -> CostRecord
```

Record token usage for an analysis.

##### `record_cache_hit`

```python
def record_cache_hit(
    self,
    analysis_id: str,
    estimated_tokens: TokenUsage,
    model: str,
) -> CostRecord
```

Record a cache hit (zero cost).

##### `get_daily_cost`

```python
def get_daily_cost(self, target_date: date | None = None) -> float
```

##### `get_daily_summary`

```python
def get_daily_summary(
    self,
    target_date: date | None = None,
) -> DailyCostSummary
```

##### `get_monthly_report`

```python
def get_monthly_report(
    self,
    target_month: date | None = None,
) -> MonthlyCostReport
```

##### `is_budget_exceeded`

```python
def is_budget_exceeded(self) -> bool
```

##### `get_budget_status`

```python
def get_budget_status(self) -> dict
```

##### `get_alerts`

```python
def get_alerts(
    self,
    unacknowledged_only: bool = True,
) -> list[BudgetAlert]
```

---

#### `calculate_cost`

Calculate cost for token usage.

```python
def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    provider: str = "anthropic",
) -> tuple[float, float]
```

**Returns:**
- Tuple of (input_cost, output_cost)

---

#### `TokenUsage`

```python
@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int
```

---

#### `DailyCostSummary`

```python
class DailyCostSummary(BaseModel):
    date: date
    total_cost: float
    total_tokens: int
    analysis_count: int
    cache_hits: int
    cache_savings: float
    model_breakdown: dict[str, float]
```

---

#### `MonthlyCostReport`

```python
class MonthlyCostReport(BaseModel):
    month: date
    total_cost: float
    daily_average: float
    total_tokens: int
    total_analyses: int
    cache_hit_rate: float
    total_cache_savings: float
    budget_remaining: float | None
    budget_status: str  # "ok", "warning", "exceeded"
    model_breakdown: dict[str, float] | None
```

---

## Evaluation

Evaluate AI analysis quality against ground truth.

**Module:** `src.evaluation`

### Metrics

**Module:** `src.evaluation.metrics`

#### `evaluate`

Evaluate AI analysis against ground truth.

```python
def evaluate(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
) -> EvaluationResult
```

#### `EvaluationResult`

```python
class EvaluationResult(BaseModel):
    insight_accuracy: float      # 0.0-1.0
    factual_correctness: float   # 0.0-1.0
    completeness: float          # 0.0-1.0
    score_accuracy: float        # 0.0-1.0
    overall_score: float         # Weighted average
    metrics: dict[str, MetricResult]
```

#### Individual Metrics

```python
def insight_accuracy(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
) -> float

def factual_correctness(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
) -> float

def completeness(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
) -> float

def score_accuracy(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
) -> float
```

---

### Ground Truth

**Module:** `src.evaluation.ground_truth`

#### `ExpertAnalysis`

Expert-annotated ground truth.

```python
class ExpertAnalysis(BaseModel):
    quality_score: int  # 1-10
    key_issues: list[str]
    expected_observations: list[str]
    expected_recommendations: list[str] | None
```

#### `GroundTruthSample`

```python
class GroundTruthSample(BaseModel):
    execution: ExecutionReport
    expert_analysis: ExpertAnalysis
    sample_id: str
    source: str
    annotator: str
    created_at: datetime
```

#### `GroundTruthDataset`

```python
class GroundTruthDataset(BaseModel):
    samples: list[GroundTruthSample]

    def __len__(self) -> int
    def __getitem__(self, idx: int) -> GroundTruthSample
    def to_json(self) -> str
```

#### `load_ground_truth`

```python
def load_ground_truth(path: str) -> GroundTruthDataset
```

#### `save_ground_truth`

```python
def save_ground_truth(
    dataset: GroundTruthDataset,
    path: str,
) -> None
```

---

### Runner

**Module:** `src.evaluation.runner`

#### `EvaluationRunner`

```python
class EvaluationRunner:
    def __init__(
        self,
        analyzer: TradeAnalyzer | None = None,
        config: AppConfig | None = None,
    )

    def run(
        self,
        dataset: GroundTruthDataset,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EvalResults
```

#### `EvalResults`

```python
class EvalResults(BaseModel):
    samples_evaluated: int
    mean_overall_score: float
    mean_insight_accuracy: float
    mean_factual_correctness: float
    mean_completeness: float
    mean_score_accuracy: float
    individual_results: list[SingleEvalResult]
```

---

### Reporting

**Module:** `src.evaluation.reporting`

```python
def generate_markdown_report(results: EvalResults) -> str
def generate_json_export(results: EvalResults) -> str
def save_report(
    results: EvalResults,
    path: str,
    format: str = "markdown",
) -> None
def compare_runs(
    results1: EvalResults,
    results2: EvalResults,
) -> dict
```

---

## Human Review

Human-in-the-loop review interface.

**Module:** `src.review`

### Models

**Module:** `src.review.models`

#### `HumanFeedback`

```python
class HumanFeedback(BaseModel):
    feedback_id: str
    analysis_id: str
    reviewer_id: str
    timestamp: datetime
    agrees_with_score: bool
    human_score: int  # 1-10
    score_correction: int | None
    missing_issues: list[str]
    incorrect_issues: list[str]
    missing_observations: list[str]
    incorrect_observations: list[str]
    notes: str
    review_time_seconds: float
    status: ReviewStatus
```

#### `SamplingStrategy`

```python
class SamplingStrategy(Enum):
    RANDOM = "random"
    LOWEST_CONFIDENCE = "lowest_confidence"
    HIGHEST_CONFIDENCE = "highest_confidence"
    NEWEST = "newest"
    OLDEST = "oldest"
    LOW_SCORE = "low_score"
    HIGH_SCORE = "high_score"
```

---

### Queue

**Module:** `src.review.queue`

#### `ReviewQueue`

```python
class ReviewQueue:
    def add(self, analysis_result: AnalysisResult) -> None
    def next(self) -> ReviewItem | None
    def get_pending_count(self) -> int
    def get_by_id(self, item_id: str) -> ReviewItem | None
```

#### `create_stratified_queue`

```python
def create_stratified_queue(
    results: list[AnalysisResult],
    strategy: SamplingStrategy,
) -> ReviewQueue
```

---

### Storage

**Module:** `src.review.storage`

#### `FeedbackStore`

```python
class FeedbackStore:
    def __init__(self, storage_path: Path | None = None)

    def save_feedback(self, feedback: HumanFeedback) -> None
    def load_feedback(self, feedback_id: str) -> HumanFeedback | None
    def load_all_feedback(self) -> list[HumanFeedback]
    def get_feedback_by_analysis(
        self,
        analysis_id: str,
    ) -> list[HumanFeedback]
    def get_agreement_rate(self) -> float
```

---

## CLI

Command-line interfaces for cache and cost management.

### Cache CLI

```bash
python -m src.cli cache stats           # Show cache statistics
python -m src.cli cache clear           # Clear all cache entries
python -m src.cli cache cleanup         # Remove expired entries
python -m src.cli cache export FILE     # Export cache to JSON
```

### Cost CLI

```bash
python -m src.cli cost today            # Show today's cost
python -m src.cli cost today --date 2024-01-15
python -m src.cli cost monthly          # Show monthly report
python -m src.cli cost budget           # Show budget status
python -m src.cli cost alerts           # Show budget alerts
python -m src.cli cost export FILE      # Export costs to JSON
python -m src.cli cost clear            # Clear all cost records
```

**Options:**
```bash
--db PATH              # Database path
--daily-budget FLOAT   # Daily budget limit
--monthly-budget FLOAT # Monthly budget limit
--verbose, -v          # Verbose output
```

---

## Configuration

**Module:** `src.config`

### `AppConfig`

```python
@dataclass
class AppConfig:
    llm: LLMConfig
    langfuse: LangfuseConfig
    log_level: str = "INFO"
    debug: bool = False

    @classmethod
    def from_env(cls) -> AppConfig

    def validate(self) -> None
```

### `LLMConfig`

```python
@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: int = 60
    max_retries: int = 3

    def validate(self) -> None
```

### `LangfuseConfig`

```python
@dataclass
class LangfuseConfig:
    enabled: bool = True
    public_key: str | None = None
    secret_key: str | None = None
    host: str = "https://cloud.langfuse.com"
    release: str | None = None
    debug: bool = False

    def validate(self) -> None
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider | `anthropic` |
| `LLM_MODEL` | Model identifier | `claude-sonnet-4-20250514` |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required if OpenAI |
| `LLM_MAX_TOKENS` | Max response tokens | `4096` |
| `LLM_TEMPERATURE` | Sampling temperature | `0.0` |
| `LLM_TIMEOUT` | Request timeout (s) | `60` |
| `LLM_MAX_RETRIES` | Retry attempts | `3` |
| `LANGFUSE_ENABLED` | Enable tracing | `true` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | - |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | - |
| `LANGFUSE_HOST` | Langfuse endpoint | `https://cloud.langfuse.com` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DEBUG` | Debug mode | `false` |

---

## Exceptions

### FIX Parsing Exceptions

**Module:** `src.parsers.exceptions`

```python
class FIXParseError(Exception):
    """Base FIX parsing error"""
    message: str
    raw_message: str | None

class FIXValidationError(FIXParseError):
    """Field validation failed"""
    field_tag: int
    field_value: str

class FIXMissingFieldError(FIXParseError):
    """Required field missing"""
    field_tag: int
    field_name: str
```

### Analysis Exceptions

```python
class AnalysisError(Exception):
    """Analysis failed"""
    message: str
    execution: ExecutionReport | None
    cause: Exception | None
```

### LLM Exceptions

```python
class LLMError(Exception):
    """Base LLM error"""

class LLMRateLimitError(LLMError):
    """Rate limit exceeded"""
    retry_after: float | None

class LLMTimeoutError(LLMError):
    """Request timed out"""

class LLMProviderError(LLMError):
    """Provider error"""
    status_code: int | None
```

### Response Exceptions

```python
class ResponseParseError(Exception):
    """Failed to parse LLM response"""
    message: str
    raw_response: str
    reason: str | None
```

### Configuration Exceptions

```python
class ConfigurationError(Exception):
    """Configuration error"""
```
