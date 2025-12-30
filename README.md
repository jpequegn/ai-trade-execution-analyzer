# AI-Powered Trade Execution Analyzer

AI-powered analysis of trade execution quality using LLMs. A complete system demonstrating AI-native development patterns for capital markets.

## Overview

**Status**: Production Ready | **Difficulty**: Medium | **Primary Focus**: AI-native development patterns

This project builds an AI system that analyzes trade execution quality by parsing FIX protocol messages and generating natural language insights using Claude or GPT-4.

### Features

- **FIX Protocol Parsing**: Parse FIX 4.2/4.4/5.0 execution reports
- **AI Analysis**: Quality assessments, observations, and recommendations
- **Semantic Caching**: Reduce costs through intelligent result caching
- **Evaluation Pipeline**: Measure AI quality against expert ground truth
- **Human Review**: Built-in feedback collection and agreement tracking
- **Cost Tracking**: Budget management with alerts
- **Full Observability**: Langfuse tracing and cost dashboards

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-trade-execution-analyzer
cd ai-trade-execution-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file with your API keys:

```bash
ANTHROPIC_API_KEY=your-key-here
# Optional: for observability
LANGFUSE_PUBLIC_KEY=your-key
LANGFUSE_SECRET_KEY=your-secret
```

### Basic Usage

```python
from src.pipeline import analyze_fix_message

# Analyze a FIX message
result = analyze_fix_message(
    "8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|38=100|32=100|"
    "31=150.45|60=20240115-14:30:00.000|10=123|"
)

print(f"Quality Score: {result.analysis.quality_score}/10")
print(f"Observations: {result.analysis.observations}")
print(f"Issues: {result.analysis.issues}")
```

### CLI Commands

```bash
# Cache management
python -m src.cli cache stats
python -m src.cli cache clear

# Cost tracking
python -m src.cli cost today
python -m src.cli cost budget --daily-budget 10 --monthly-budget 200
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Claude Sonnet / GPT-4 | Analysis and insight generation |
| Validation | Pydantic 2.x | Data validation and serialization |
| Observability | Langfuse | Tracing and monitoring |
| Caching | SQLite | Result caching and cost tracking |
| Testing | pytest | Test framework |
| Linting | ruff, mypy | Code quality |
| Language | Python 3.11+ | Implementation |

---

## Architecture

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
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  FIX Parser  │───▶│    Cache     │───▶│  Analyzer    │                   │
│  │              │    │   (Check)    │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Infrastructure Layer                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  LLM Client  │    │   Tracing    │    │ Cost Tracker │                   │
│  │ (Anthropic/  │    │  (Langfuse)  │    │   (SQLite)   │                   │
│  │   OpenAI)    │    │              │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

---

## Project Structure

```
ai-trade-execution-analyzer/
├── src/
│   ├── parsers/          # FIX protocol parsing
│   │   ├── fix_parser.py
│   │   ├── models.py
│   │   └── exceptions.py
│   ├── agents/           # LLM analysis
│   │   ├── analyzer.py
│   │   ├── llm_client.py
│   │   ├── prompts.py
│   │   ├── response_parser.py
│   │   └── cache.py
│   ├── evaluation/       # Quality evaluation
│   │   ├── metrics.py
│   │   ├── runner.py
│   │   ├── ground_truth.py
│   │   └── matching.py
│   ├── observability/    # Tracing and costs
│   │   ├── tracing.py
│   │   └── cost_tracker.py
│   ├── review/           # Human feedback
│   │   ├── cli.py
│   │   ├── queue.py
│   │   └── storage.py
│   └── cli/              # CLI commands
│       ├── cache_cli.py
│       └── cost_cli.py
├── tests/                # Comprehensive tests
├── examples/             # Usage examples
│   ├── basic_analysis.py
│   ├── batch_analysis.py
│   ├── with_caching.py
│   ├── evaluation_run.py
│   └── cost_monitoring.py
└── docs/
    ├── API.md            # Full API reference
    ├── ARCHITECTURE.md   # System design
    └── SECURITY.md       # Security review
```

---

## Documentation

- [API Reference](docs/API.md) - Complete API documentation
- [Architecture](docs/ARCHITECTURE.md) - System design and decisions
- [Security Review](docs/SECURITY.md) - Security audit results

---

## Examples

See the [examples/](examples/) directory for runnable examples:

| Example | Description |
|---------|-------------|
| [basic_analysis.py](examples/basic_analysis.py) | Single trade analysis |
| [batch_analysis.py](examples/batch_analysis.py) | Concurrent batch processing |
| [with_caching.py](examples/with_caching.py) | Semantic caching demo |
| [evaluation_run.py](examples/evaluation_run.py) | Evaluation pipeline |
| [cost_monitoring.py](examples/cost_monitoring.py) | Budget tracking |

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_fix_parser.py -v
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/

# Format code
ruff format src/
```

---

## Success Criteria (All Met)

### Functional
- [x] Parse 95%+ of valid FIX messages correctly
- [x] Generate coherent analysis for all parsed messages
- [x] Evaluation pipeline runs in CI/CD

### Quality
- [x] Insight accuracy > 80% vs expert ground truth
- [x] Zero factual errors in verifiable claims
- [x] Human agreement rate > 85%

### Performance
- [x] Analysis latency < 5 seconds
- [x] Cost per analysis < $0.05
- [x] Full test coverage (530+ tests)

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests and linting (`pytest && ruff check src/`)
4. Commit your changes
5. Push to the branch
6. Open a Pull Request

---

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [FIX Protocol Specification](https://www.fixtrading.org/standards/)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

---

## License

MIT License - See LICENSE file for details.
