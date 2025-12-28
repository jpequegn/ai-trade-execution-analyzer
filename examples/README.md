# Sample Datasets

This directory contains sample FIX execution report messages for testing and evaluation.

## Files

| File | Description |
|------|-------------|
| `sample_fix_messages.txt` | Raw FIX messages (55 samples) |
| `sample_fix_messages.json` | Parsed JSON format with metadata |
| `generate_dataset.py` | Script to regenerate JSON from text |

## Dataset Structure

### Categories

The 55 sample messages are organized into categories:

| Category | Message Range | Expected Score | Description |
|----------|--------------|----------------|-------------|
| Good | 1-15 | 8-10 | Optimal executions |
| Average | 16-35 | 5-7 | Acceptable but improvable |
| Poor | 36-50 | 1-4 | Problematic executions |
| Edge Cases | 51-55 | Varies | Special scenarios |

### Good Executions (Score 8-10)

Characteristics:
- Fast execution at market open or high liquidity periods
- Optimal venue selection (primary exchange for listed securities)
- Full fills with no slippage
- Competitive pricing

Example scenarios:
- Market open executions on NASDAQ/NYSE
- Blue chip stocks with high liquidity
- Proper venue routing (NASDAQ for tech, NYSE for financial)

### Average Executions (Score 5-7)

Characteristics:
- Partial fills requiring multiple executions
- Suboptimal venue selection
- Mid-day execution with lower liquidity
- Minor price slippage

Example scenarios:
- Multi-fill orders with price drift
- Execution on alternative venues (EDGX, BATS, IEX)
- Lunch hour trades
- Late day executions

### Poor Executions (Score 1-4)

Characteristics:
- Very slow execution spanning hours
- Significant price slippage
- Poor venue selection for liquid securities
- Execution in low liquidity stocks
- Near market close timing issues

Example scenarios:
- Multi-hour fill sequences
- Meme stocks with volatility
- ETF trades on off-exchange venues
- Trades in final minutes before close

### Edge Cases

Special scenarios for testing parser robustness:
- Very large orders (100,000+ shares)
- Odd lot orders (< 100 shares)
- FIX 4.2 format messages
- After-hours executions
- Pre-market executions

## Symbols Covered

### Large Cap Tech
AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, INTC, CRM, ORCL, ADBE, NFLX

### Financial Services
JPM, V, UNH

### Consumer/Retail
JNJ, DIS, KO, PEP, MCD, HD, WMT, COST, PG

### Energy
XOM, CVX

### Healthcare
LLY

### Industrial
BA

### Telecom
VZ, T

### Volatile/Meme
GME, AMC, BBBY

### ETFs
SPY, QQQ

### Other
IBM, BRK.B, F, SIRI, PLTR

## Venues

| Venue | Type | Typical Use |
|-------|------|-------------|
| NYSE | Primary | NYSE-listed stocks |
| NASDAQ | Primary | NASDAQ-listed stocks |
| ARCA | ECN | ETFs, extended hours |
| EDGX | Alternative | Cost-sensitive orders |
| BATS | Alternative | High-frequency trading |
| IEX | Alternative | Anti-HFT protection |

## Usage

### Loading Raw Messages

```python
from pathlib import Path

def load_messages(file_path: Path) -> list[str]:
    messages = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            messages.append(line)
    return messages

messages = load_messages(Path("examples/sample_fix_messages.txt"))
```

### Loading JSON Dataset

```python
import json
from pathlib import Path

with open(Path("examples/sample_fix_messages.json")) as f:
    dataset = json.load(f)

print(f"Total messages: {dataset['metadata']['total_messages']}")

for entry in dataset["executions"]:
    print(f"{entry['id']}: {entry['parsed']['symbol']} - {entry['category']}")
```

### Regenerating JSON

```bash
uv run python examples/generate_dataset.py
```

## JSON Schema

```json
{
  "metadata": {
    "total_messages": 55,
    "source_file": "sample_fix_messages.txt",
    "description": "...",
    "categories": { ... }
  },
  "executions": [
    {
      "id": "MSG001",
      "category": "good",
      "expected_score_range": "8-10",
      "raw_message": "8=FIX.4.4|35=8|...",
      "parsed": {
        "order_id": "ORD001",
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 100.0,
        "price": 185.50,
        "venue": "NASDAQ",
        "timestamp": "2024-01-15T09:30:05.123000",
        "fill_type": "FULL",
        "fix_version": "FIX.4.4",
        "exec_type": "F",
        "cum_qty": 100.0,
        "avg_px": 185.50
      }
    }
  ]
}
```

## Adding New Samples

1. Add FIX message to `sample_fix_messages.txt`
2. Run `uv run python examples/generate_dataset.py`
3. Verify parsing in JSON output
