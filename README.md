# AI-Powered Trade Execution Analyzer

AI-powered analysis of trade execution quality using LLMs. A learning project demonstrating AI-native development patterns for capital markets.

## Overview

**Difficulty**: Medium | **Time**: 4-6 weeks | **Primary Learning**: AI-native development patterns

This project builds an AI system that analyzes trade execution quality by parsing FIX protocol messages and generating natural language insights. The same patterns apply to fitness data analysis (GPS running data, workout logs).

### What You'll Learn
- Prompt engineering for structured financial data
- Building evaluation pipelines for AI outputs
- Human-in-the-loop verification patterns
- AI-native development workflow

### Why This Matters
Demonstrates understanding of AI + domain integration, evaluation pipelines, and structured data handling - critical skills for AI engineering leadership.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Claude 3.5 Sonnet / GPT-4 | Analysis and insight generation |
| Orchestration | LangGraph | Workflow management |
| Observability | Langfuse | Tracing and monitoring |
| Evals | DeepEval / Custom | Quality measurement |
| Language | Python 3.11+ | Implementation |
| Data | FIX Protocol / GPS data | Input sources |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  FIX Messages          │  GPS Running Data    │  Workout Logs   │
│  (Trade execution)     │  (Pace, HR, cadence) │  (Structured)   │
└───────────┬─────────────────────┬─────────────────────┬─────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PARSING LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  FIX Parser            │  GPX/TCX Parser      │  JSON Parser    │
│  - Tag extraction      │  - Coordinate parse  │  - Schema valid │
│  - Message validation  │  - Time series       │  - Type coerce  │
└───────────┬─────────────────────┬─────────────────────┬─────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STRUCTURED DATA LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  Trade Execution DTO   │  Activity DTO        │  Workout DTO    │
│  - Symbol, qty, price  │  - Splits, zones     │  - Sets, reps   │
│  - Timestamps, venue   │  - Elevation, effort │  - Load, rest   │
└───────────┬─────────────────────┬─────────────────────┬─────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ANALYSIS AGENT                              │
├─────────────────────────────────────────────────────────────────┤
│  LLM (Claude/GPT-4)                                              │
│  - Pattern recognition                                           │
│  - Quality assessment                                            │
│  - Anomaly detection                                             │
│  - Natural language insights                                     │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VERIFICATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  Evaluation Pipeline                                             │
│  - Factual accuracy check                                        │
│  - Source attribution validation                                 │
│  - Hallucination detection                                       │
│  - Human review queue                                            │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  Analysis Report       │  Metrics Dashboard   │  Alerts         │
│  - Executive summary   │  - Quality scores    │  - Anomalies    │
│  - Detailed insights   │  - Trends over time  │  - Thresholds   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Foundation (Week 1-2)

#### Week 1: Data Layer
- [ ] Set up Python project with uv/poetry
- [ ] Implement FIX message parser
  - Parse standard FIX 4.2/4.4 execution reports
  - Extract key fields: symbol, quantity, price, timestamp, venue
  - Handle edge cases and malformed messages
- [ ] Create structured data models (Pydantic)
- [ ] Write unit tests for parser
- [ ] Create sample FIX message dataset (50+ messages)

**Deliverable**: Working parser that converts FIX messages to structured data

#### Week 2: Basic LLM Integration
- [ ] Set up LLM client (Anthropic/OpenAI SDK)
- [ ] Design analysis prompt template
  - Input: Structured execution data
  - Output: Quality assessment with reasoning
- [ ] Implement basic analysis agent
- [ ] Add Langfuse tracing
- [ ] Test with sample data

**Deliverable**: End-to-end pipeline from FIX message to analysis

### Phase 2: Quality & Evaluation (Week 3-4)

#### Week 3: Evaluation Pipeline
- [ ] Create ground truth dataset
  - 50+ FIX messages with expert annotations
  - Quality scores (1-10) for execution
  - Key insights that should be detected
- [ ] Implement evaluation metrics
  - Insight accuracy (did AI find the right issues?)
  - Factual correctness (are claims verifiable?)
  - Completeness (did AI miss obvious patterns?)
- [ ] Set up automated eval runs
- [ ] Create eval dashboard

**Deliverable**: Automated evaluation pipeline with metrics

#### Week 4: Human-in-the-Loop
- [ ] Build simple review interface
  - Show AI analysis alongside raw data
  - Allow human to rate/correct
- [ ] Implement feedback loop
  - Store human corrections
  - Track agreement rate over time
- [ ] Add disagreement alerts
- [ ] Document evaluation criteria

**Deliverable**: Human review workflow integrated

### Phase 3: Production Hardening (Week 5-6)

#### Week 5: Observability & Cost
- [ ] Comprehensive Langfuse instrumentation
  - Token usage per analysis
  - Latency breakdown
  - Error tracking
- [ ] Implement caching layer
  - Cache similar analysis patterns
  - Reduce redundant LLM calls
- [ ] Add cost tracking dashboard
- [ ] Set up alerting for anomalies

**Deliverable**: Full observability with cost tracking

#### Week 6: Documentation & Polish
- [ ] Write API documentation
- [ ] Create usage examples
- [ ] Performance optimization
- [ ] Security review (no PII leakage)
- [ ] Final testing and bug fixes

**Deliverable**: Production-ready system with documentation

---

## Key Files Structure

```
ai-trade-execution-analyzer/
├── README.md
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── fix_parser.py          # FIX protocol parsing
│   │   ├── gpx_parser.py          # GPS data parsing (fitness variant)
│   │   └── models.py              # Pydantic data models
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── analyzer.py            # Main analysis agent
│   │   └── prompts.py             # Prompt templates
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── runner.py              # Eval pipeline runner
│   │   └── datasets/              # Ground truth datasets
│   └── observability/
│       ├── __init__.py
│       └── tracing.py             # Langfuse integration
├── tests/
│   ├── test_parsers.py
│   ├── test_agents.py
│   └── test_evaluation.py
├── examples/
│   ├── sample_fix_messages.txt
│   └── analyze_execution.py
└── docs/
    ├── ARCHITECTURE.md
    └── EVALUATION.md
```

---

## Sample Code Snippets

### FIX Message Parsing
```python
from pydantic import BaseModel
from datetime import datetime

class ExecutionReport(BaseModel):
    order_id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    venue: str
    timestamp: datetime
    fill_type: str  # FULL/PARTIAL

def parse_fix_message(raw: str) -> ExecutionReport:
    """Parse FIX execution report into structured data."""
    fields = dict(pair.split('=') for pair in raw.split('|') if '=' in pair)
    return ExecutionReport(
        order_id=fields.get('37', ''),
        symbol=fields.get('55', ''),
        side='BUY' if fields.get('54') == '1' else 'SELL',
        quantity=float(fields.get('32', 0)),
        price=float(fields.get('31', 0)),
        venue=fields.get('30', 'UNKNOWN'),
        timestamp=datetime.strptime(fields.get('60', ''), '%Y%m%d-%H:%M:%S.%f'),
        fill_type='FULL' if fields.get('39') == '2' else 'PARTIAL'
    )
```

### Analysis Agent
```python
from langfuse.decorators import observe
from anthropic import Anthropic

@observe(name="analyze_execution")
def analyze_execution(execution: ExecutionReport) -> dict:
    """Analyze trade execution quality using LLM."""
    client = Anthropic()

    prompt = f"""Analyze this trade execution for quality issues:

Symbol: {execution.symbol}
Side: {execution.side}
Quantity: {execution.quantity}
Price: {execution.price}
Venue: {execution.venue}
Time: {execution.timestamp}
Fill Type: {execution.fill_type}

Provide:
1. Execution quality score (1-10)
2. Key observations (bullet points)
3. Potential issues or anomalies
4. Recommendations

Base your analysis on execution timing, venue selection, and fill quality."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "analysis": response.content[0].text,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens
    }
```

### Evaluation Metric
```python
def evaluate_analysis(ai_analysis: str, ground_truth: dict) -> dict:
    """Evaluate AI analysis against expert ground truth."""
    scores = {
        "insight_accuracy": 0.0,
        "factual_correctness": 0.0,
        "completeness": 0.0
    }

    # Check if AI identified key issues
    for issue in ground_truth["key_issues"]:
        if issue.lower() in ai_analysis.lower():
            scores["insight_accuracy"] += 1
    scores["insight_accuracy"] /= len(ground_truth["key_issues"])

    # Verify factual claims
    # ... (implement claim extraction and verification)

    return scores
```

---

## Success Criteria

### Functional
- [ ] Parse 95%+ of valid FIX messages correctly
- [ ] Generate coherent analysis for all parsed messages
- [ ] Evaluation pipeline runs in CI/CD

### Quality
- [ ] Insight accuracy > 80% vs expert ground truth
- [ ] Zero factual errors in verifiable claims
- [ ] Human agreement rate > 85%

### Performance
- [ ] Analysis latency < 5 seconds
- [ ] Cost per analysis < $0.05
- [ ] 99% uptime for API

---

## Learning Outcomes

After completing this project, you will be able to:

1. **Explain** AI-native development patterns to your team
2. **Design** evaluation pipelines for AI outputs
3. **Implement** observability for LLM applications
4. **Demonstrate** human-in-the-loop verification patterns
5. **Measure** AI quality with quantitative metrics

---

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Langfuse Quick Start](https://langfuse.com/docs/get-started)
- [DeepEval Documentation](https://docs.deepeval.ai/)
- [FIX Protocol Specification](https://www.fixtrading.org/standards/)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)

---

## License

MIT License - See LICENSE file for details.
