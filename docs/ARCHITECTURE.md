# Architecture Documentation

This document describes the system architecture, design decisions, and data flow of the AI Trade Execution Analyzer.

## Table of Contents

- [System Overview](#system-overview)
- [Architecture Diagram](#architecture-diagram)
- [Component Design](#component-design)
- [Data Flow](#data-flow)
- [Design Decisions](#design-decisions)
- [Technology Choices](#technology-choices)
- [Extension Points](#extension-points)

---

## System Overview

The AI Trade Execution Analyzer is an AI-native system for analyzing trade execution quality. It parses FIX protocol messages, uses LLMs to generate insights, and provides tools for evaluation and continuous improvement.

### Core Capabilities

1. **FIX Protocol Parsing**: Transform raw FIX messages into structured execution data
2. **AI Analysis**: Generate quality assessments, observations, and recommendations
3. **Caching**: Reduce costs through semantic result caching
4. **Evaluation**: Measure AI accuracy against expert ground truth
5. **Observability**: Full tracing and cost tracking
6. **Human Review**: Continuous improvement through feedback

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  CLI Interface     │    Python API    │    Human Review UI                  │
│  (pipeline.py)     │  (analyze_*)     │    (review/cli.py)                  │
└────────┬───────────┴────────┬─────────┴────────┬────────────────────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Core Pipeline                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  FIX Parser  │───▶│    Cache     │───▶│  Analyzer    │                   │
│  │              │    │   (Check)    │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ExecutionReport    CacheEntry | None     AnalysisResult                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Infrastructure Layer                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  LLM Client  │    │   Tracing    │    │ Cost Tracker │                   │
│  │ (Anthropic/  │    │  (Langfuse)  │    │   (SQLite)   │                   │
│  │   OpenAI)    │    │              │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          External Services                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│   Anthropic API     │     OpenAI API     │      Langfuse Cloud              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. FIX Parser (`src/parsers/`)

Transforms raw FIX protocol messages into structured `ExecutionReport` objects.

```
Raw FIX String → Tokenizer → Field Extractor → Validator → ExecutionReport
```

**Key Components:**
- `tokenize_fix_message()`: Split message into tag-value pairs
- `parse_fix_message()`: Parse and validate into ExecutionReport
- Custom exceptions for specific error types

**Design Principles:**
- Fail-fast with descriptive errors
- Support multiple FIX versions (4.2, 4.4, 5.0)
- Preserve raw message for debugging

---

### 2. Analysis Cache (`src/agents/cache.py`)

Reduces LLM costs through intelligent result caching.

```
                    ┌─────────────────┐
                    │  Cache Request  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Generate Key    │
                    │ (exact/semantic)│
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌────────────────┐            ┌────────────────┐
     │  Cache Hit     │            │  Cache Miss    │
     │  Return Entry  │            │  Call LLM      │
     └────────────────┘            │  Store Result  │
                                   └────────────────┘
```

**Key Strategies:**
- **Exact**: SHA256 hash of all execution fields
- **Semantic**: Hash of key characteristics (symbol, side, price bucket)

**Backend Options:**
- `SQLiteCacheBackend`: Persistent, queryable
- `FileCacheBackend`: Simple JSON files

---

### 3. Trade Analyzer (`src/agents/analyzer.py`)

Orchestrates the analysis workflow.

```
ExecutionReport
      │
      ▼
┌─────────────────┐     Cache Hit?
│  Check Cache    │────────────────▶ Return Cached Result
└────────┬────────┘
         │ Miss
         ▼
┌─────────────────┐
│  Build Prompt   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Client    │◀──── Retry Logic
│  (with tracing) │      Rate Limit Handling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Parse Response │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Store in Cache │
└────────┬────────┘
         │
         ▼
   AnalysisResult
```

**Features:**
- Automatic prompt variant selection
- Concurrent batch processing
- Graceful error handling

---

### 4. LLM Client (`src/agents/llm_client.py`)

Unified interface for multiple LLM providers.

```
┌─────────────────────────────────────────┐
│              LLMClient                   │
├─────────────────────────────────────────┤
│  - Provider abstraction                  │
│  - Retry with exponential backoff        │
│  - Rate limit handling                   │
│  - Langfuse integration                  │
└────────────────┬────────────────────────┘
                 │
     ┌───────────┴───────────┐
     ▼                       ▼
┌──────────┐           ┌──────────┐
│Anthropic │           │  OpenAI  │
│  Client  │           │  Client  │
└──────────┘           └──────────┘
```

**Retry Strategy:**
- Exponential backoff: 1s, 2s, 4s, 8s...
- Max 3 retries by default
- Rate limit errors: respect `retry_after` header

---

### 5. Observability (`src/observability/`)

Full visibility into system behavior and costs.

#### Tracing (Langfuse)

```
┌─────────────────────────────────────────────────────────────────┐
│                          Trace                                   │
│  Session: sess-123                                               │
│  ├── Span: parse_fix_message                                     │
│  │   └── Duration: 2ms                                           │
│  ├── Span: check_cache                                           │
│  │   └── Result: miss                                            │
│  ├── Generation: llm_call                                        │
│  │   ├── Model: claude-sonnet-4-20250514                                     │
│  │   ├── Input tokens: 450                                       │
│  │   ├── Output tokens: 320                                      │
│  │   └── Duration: 1250ms                                        │
│  └── Span: parse_response                                        │
│      └── Duration: 5ms                                           │
└─────────────────────────────────────────────────────────────────┘
```

#### Cost Tracking (SQLite)

```
┌─────────────────────────────────────────┐
│            Cost Tracker                  │
├─────────────────────────────────────────┤
│  Daily Budget: $10.00                    │
│  Monthly Budget: $200.00                 │
│  Warning Threshold: 80%                  │
├─────────────────────────────────────────┤
│  Today's Usage:                          │
│  - Analyses: 45                          │
│  - Tokens: 125,000                       │
│  - Cost: $2.35                           │
│  - Cache savings: $1.20                  │
└─────────────────────────────────────────┘
```

---

### 6. Evaluation Framework (`src/evaluation/`)

Measures AI quality against expert ground truth.

```
┌──────────────────┐     ┌──────────────────┐
│  Ground Truth    │     │   AI Analysis    │
│  Dataset         │     │   Results        │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
         ┌──────────────────┐
         │   Evaluator      │
         │                  │
         │  - Insight Acc   │
         │  - Factual Corr  │
         │  - Completeness  │
         │  - Score Acc     │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  EvalResults     │
         │                  │
         │  Overall: 87.5%  │
         └──────────────────┘
```

**Metrics:**
- **Insight Accuracy**: Did AI identify the key issues?
- **Factual Correctness**: Are observations factually correct?
- **Completeness**: Did AI cover all expected observations?
- **Score Accuracy**: Does quality score match expert opinion?

---

### 7. Human Review (`src/review/`)

Continuous improvement through expert feedback.

```
┌──────────────────────────────────────────────────────────────┐
│                    Review Pipeline                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │  Analysis   │───▶│   Review    │───▶│  Feedback   │       │
│  │  Results    │    │   Queue     │    │   Store     │       │
│  └─────────────┘    └─────────────┘    └─────────────┘       │
│                           │                   │               │
│                           ▼                   ▼               │
│                    ┌─────────────┐    ┌─────────────┐        │
│                    │  Sampling   │    │  Agreement  │        │
│                    │  Strategy   │    │   Rate      │        │
│                    └─────────────┘    └─────────────┘        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**Sampling Strategies:**
- Random: Uniform selection
- Lowest Confidence: AI uncertainty
- Low Score: Poor quality trades
- Newest/Oldest: Temporal selection

---

## Data Flow

### Single Analysis Flow

```
1. Input: Raw FIX message
   "8=FIX.4.4|35=8|37=ORD001|..."

2. Parse: FIX Parser
   → ExecutionReport(order_id="ORD001", symbol="AAPL", ...)

3. Cache Check: Analysis Cache
   → Miss (or Hit → return cached)

4. Prompt: Build prompt with execution data
   → System prompt + user prompt

5. LLM Call: Send to Claude/GPT
   → Raw response text

6. Parse: Extract JSON from response
   → TradeAnalysis(quality_score=7, issues=[...], ...)

7. Store: Cache result for future
   → CacheEntry(key=..., result=..., expires_at=...)

8. Track: Record cost
   → CostRecord(tokens=750, cost=$0.012)

9. Output: Complete result
   → AnalysisResult(execution=..., analysis=..., ...)
```

### Batch Analysis Flow

```
[Execution1, Execution2, Execution3, ...]
              │
              ▼
     ┌────────────────┐
     │ Concurrent     │
     │ Processing     │──────▶ Semaphore (max_concurrent=5)
     └────────┬───────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│Analyze│ │Analyze│ │Analyze│  (parallel)
│ Exec1 │ │ Exec2 │ │ Exec3 │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    └────┬────┴────┬────┘
         ▼         ▼
[Result1, Result2, Result3, ...]
```

---

## Design Decisions

### 1. Why Langfuse for Observability?

**Considered Alternatives:**
- LangSmith: Good but tightly coupled to LangChain
- Custom solution: Too much maintenance overhead
- OpenTelemetry: Generic, not LLM-specific

**Why Langfuse:**
- Purpose-built for LLM applications
- Excellent generation tracking (tokens, latency, cost)
- Clean Python SDK with context managers
- Self-hostable for data privacy
- Supports evaluation integration
- Active development and community

---

### 2. Why This Caching Strategy?

**The Problem:**
- Similar trades should get similar analyses
- Exact matching misses opportunities
- Need to balance hit rate vs. relevance

**Solution: Dual Strategy**

| Strategy | Use Case | Trade-off |
|----------|----------|-----------|
| Exact | Replay detection | Low hit rate, high relevance |
| Semantic | Cost optimization | Higher hit rate, some variance |

**Semantic Key Design:**
```python
key = hash(
    symbol,           # Same asset
    side,             # Same direction
    bucket(price),    # Similar price level
    bucket(quantity), # Similar size
    time_bucket,      # Same time window
)
```

---

### 3. Why These Evaluation Metrics?

**Goal:** Measure AI quality objectively

| Metric | Purpose | Threshold |
|--------|---------|-----------|
| Insight Accuracy | Find the right issues | > 80% |
| Factual Correctness | No wrong statements | > 95% |
| Completeness | Cover all observations | > 70% |
| Score Accuracy | Align with expert | ±1 point |

**Fuzzy Matching:**
- Direct string comparison fails on paraphrasing
- Use keyword overlap + sequence similarity
- Threshold of 0.5 balances precision/recall

---

### 4. Why Pydantic for Data Models?

**Benefits:**
- Automatic validation at boundaries
- Type hints for IDE support
- JSON serialization built-in
- Field constraints (1-10 scores)
- Clear error messages

**Example:**
```python
class TradeAnalysis(BaseModel):
    quality_score: int = Field(ge=1, le=10)  # Validated
    confidence: float = Field(ge=0, le=1)     # Validated
```

---

### 5. Why SQLite for Persistence?

**Used For:**
- Analysis cache
- Cost tracking
- Feedback storage

**Benefits:**
- Zero configuration
- Single file per database
- Atomic transactions
- Good enough for single-user
- Easy backup (copy file)

**Alternative Considered:**
- PostgreSQL: Overkill for local use
- Redis: No persistence by default
- JSON files: No querying capability

---

## Technology Choices

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Language | Python 3.11+ | Ecosystem, type hints, async |
| LLM | Claude/GPT | Best reasoning capability |
| Validation | Pydantic | Type safety, serialization |
| Tracing | Langfuse | LLM-native observability |
| Cache | SQLite | Simple, embedded, queryable |
| Testing | pytest | Standard, fixtures, async |
| Linting | ruff | Fast, comprehensive |
| Types | mypy | Strict type checking |

---

## Extension Points

### 1. Adding a New LLM Provider

```python
# In src/agents/llm_client.py

class LLMClient:
    def _create_client(self):
        if self.provider == "anthropic":
            return AnthropicClient(...)
        elif self.provider == "openai":
            return OpenAIClient(...)
        elif self.provider == "gemini":  # NEW
            return GeminiClient(...)
```

### 2. Adding a New Cache Backend

```python
# In src/agents/cache.py

class RedisCacheBackend(CacheBackend):
    def __init__(self, redis_url: str):
        self.client = redis.from_url(redis_url)

    def get(self, key: str) -> CacheEntry | None:
        data = self.client.get(key)
        return CacheEntry.model_validate_json(data) if data else None

    def set(self, key: str, entry: CacheEntry) -> None:
        ttl = (entry.expires_at - datetime.now()).total_seconds()
        self.client.setex(key, int(ttl), entry.model_dump_json())
```

### 3. Adding a New Evaluation Metric

```python
# In src/evaluation/metrics.py

def recommendation_quality(
    ai_analysis: TradeAnalysis,
    ground_truth: ExpertAnalysis,
) -> float:
    """New metric: Quality of recommendations."""
    if not ground_truth.expected_recommendations:
        return 1.0  # No expectations

    matches = count_matches(
        ai_analysis.recommendations,
        ground_truth.expected_recommendations,
        threshold=0.5,
    )
    return matches / len(ground_truth.expected_recommendations)
```

### 4. Adding a New Prompt Variant

```python
# In src/agents/prompts.py

class PromptVariant(Enum):
    QUICK = "quick"
    DETAILED = "detailed"
    BATCH = "batch"
    REGULATORY = "regulatory"  # NEW

REGULATORY_TEMPLATE = """
Analyze this trade execution for regulatory compliance:

{execution}

Focus on:
1. Best execution obligations
2. Order handling requirements
3. Reporting compliance
4. Client protection rules

Output JSON with:
- compliance_score (1-10)
- violations (list of potential issues)
- required_actions (list of remediation steps)
"""
```

---

## Performance Characteristics

### Latency

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| FIX Parse | < 5ms | In-memory |
| Cache Check | < 10ms | SQLite query |
| LLM Call | 1-3s | Network + generation |
| Response Parse | < 10ms | JSON extraction |
| **Total (uncached)** | 1-4s | Dominated by LLM |
| **Total (cached)** | < 20ms | No LLM call |

### Throughput

| Configuration | Rate | Notes |
|---------------|------|-------|
| Sequential | ~30/min | Single LLM call at a time |
| Concurrent (5) | ~100/min | Parallel processing |
| With cache (50% hit) | ~200/min | Half skip LLM |

### Cost

| Model | Input | Output | Per Analysis |
|-------|-------|--------|--------------|
| Claude Sonnet | $3/1M | $15/1M | ~$0.01-0.02 |
| Claude Opus | $15/1M | $75/1M | ~$0.05-0.10 |
| GPT-4 Turbo | $10/1M | $30/1M | ~$0.02-0.04 |

---

## Security Model

### Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│                    Trust Boundary                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    Validated    ┌─────────────┐            │
│  │  External   │───────────────▶│  Internal   │            │
│  │   Input     │                 │  Processing │            │
│  │ (FIX msgs)  │                 │             │            │
│  └─────────────┘                 └─────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         │                                   │
         │                                   │
         ▼                                   ▼
┌─────────────────┐                 ┌─────────────────┐
│  Pydantic       │                 │  LLM API        │
│  Validation     │                 │  (API Key)      │
└─────────────────┘                 └─────────────────┘
```

### Key Protections

1. **Input Validation**: All FIX messages validated via Pydantic
2. **API Keys**: Environment variables only, never logged
3. **No PII in Logs**: Only order IDs and symbols
4. **Local Storage**: SQLite files with filesystem permissions
5. **Prompt Injection**: Analysis treats execution as data, not instructions

---

## Deployment Considerations

### Local Development

```bash
# Clone and setup
git clone https://github.com/org/ai-trade-execution-analyzer
cd ai-trade-execution-analyzer
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with API keys

# Run
python -m src.pipeline
```

### Production

```yaml
# docker-compose.yml (example)
services:
  analyzer:
    build: .
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
    volumes:
      - ./data:/app/data  # Persist cache and costs
```

### Monitoring

1. **Langfuse Dashboard**: LLM usage, latency, errors
2. **Cost CLI**: Budget tracking and alerts
3. **Logs**: Structured JSON for ingestion

---

## Future Considerations

### Potential Enhancements

1. **Streaming Responses**: Real-time analysis feedback
2. **Model Selection**: Automatic routing based on complexity
3. **Distributed Cache**: Redis for multi-instance deployment
4. **Webhook Integration**: Push results to external systems
5. **Custom Metrics**: User-defined evaluation criteria

### Scalability Path

```
Current: Single instance, local SQLite
    ↓
Step 1: Containerized, shared volume
    ↓
Step 2: PostgreSQL for shared state
    ↓
Step 3: Redis cache, message queue
    ↓
Step 4: Kubernetes with horizontal scaling
```
