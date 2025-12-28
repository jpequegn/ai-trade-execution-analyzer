# Ground Truth Dataset

Expert-annotated dataset for evaluating AI trade execution analysis quality.

## Overview

This dataset contains 55 expert-annotated trade executions with ground truth quality scores, issue identification, and expected observations. It serves as the benchmark for measuring how well the AI analysis matches human expert assessment.

## Dataset Statistics

| Category | Count |
|----------|-------|
| **Total Samples** | 55 |
| Good Executions (8-10) | 19 |
| Average Executions (5-7) | 21 |
| Poor Executions (1-4) | 15 |

### Coverage by Issue Category
- None (no issues): 17
- Timing: 19
- Venue Selection: 5
- Fill Quality: 6
- Price Slippage: 8

### Coverage by Asset Type
- Large Cap Stocks (AAPL, MSFT, GOOGL, etc.): 30
- ETFs (SPY, QQQ, IWM, etc.): 12
- Volatile/Meme Stocks (GME, AMC, MARA, etc.): 8
- Leveraged ETFs (TQQQ, SOXL, etc.): 5

### Coverage by Market Condition
- Market Open (9:30-9:45): 8
- Market Close (15:45-16:00): 6
- After Hours: 3
- Lunch Hour (12:00-13:00): 8
- Optimal Hours (10:00-15:00): 30

## File Structure

```
datasets/
├── README.md              # This file
└── ground_truth.json      # The annotated dataset
```

## Schema

Each sample in `ground_truth.json` follows this structure:

```json
{
  "id": "GT001",
  "fix_message": "8=FIX.4.4|35=8|37=ORD001|55=AAPL|...",
  "execution": {
    "order_id": "ORD001",
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 1000.0,
    "price": 185.50,
    "venue": "NYSE",
    "timestamp": "2024-01-15T10:30:00",
    "fill_type": "FULL"
  },
  "expert_analysis": {
    "quality_score": 9,
    "key_issues": [],
    "expected_observations": [
      "Full fill of 1000 shares",
      "Executed on primary listing exchange (NYSE)"
    ],
    "severity": "none",
    "category": "none"
  },
  "metadata": {
    "annotator": "expert",
    "annotation_date": "2024-01-15",
    "confidence": "high",
    "notes": "Optimal execution example"
  }
}
```

## Quality Score Rubric

### Score 10: Perfect Execution
- No improvements possible
- Optimal venue selection (primary exchange)
- Optimal timing (10:00-15:00, avoiding open/close)
- Full fill achieved
- Best possible price (at or better than VWAP)
- No adverse selection

**Example**: Full fill of 500 MSFT shares on NASDAQ at 11:15 AM

### Score 8-9: Excellent Execution
- Minor optimization possible but not material
- Appropriate venue
- Good timing (within optimal hours)
- Full fill or minimal partial
- Competitive pricing

**Example**: Full fill of AAPL at 10:05 AM (slightly early, minor volatility)

### Score 6-7: Good Execution
- Some clear improvements available
- May have minor venue concerns
- Acceptable timing but not optimal
- Some price slippage or partial fills
- Nothing severely wrong

**Example**: Full fill on BATS instead of primary NYSE, or execution at 12:30 PM during lunch lull

### Score 4-5: Average Execution
- Multiple issues identified
- Timing concerns (early/late)
- Venue selection questionable
- Partial fills or noticeable slippage
- Requires attention

**Example**: Partial fill during lunch hour, or execution on volatile stock near close

### Score 2-3: Poor Execution
- Significant problems
- Very poor timing (first 5 min or last 5 min)
- Wrong venue for asset type
- Major adverse selection likely
- Requires immediate improvement

**Example**: Meme stock execution at market open, or leveraged ETF near close

### Score 1: Failed/Critical
- Severely problematic
- Maximum adverse conditions
- Critical timing errors
- Potential regulatory concerns
- System-level failure

**Example**: AMC buy 1 minute before close, or 3x leveraged ETF at open

## Annotation Guidelines

### What Makes a Good Annotation

1. **Be Specific**: List exact issues, not vague concerns
   - ❌ "Bad timing"
   - ✅ "Execution 2 minutes after open during peak volatility"

2. **Cite Evidence**: Reference concrete data points
   - ❌ "Price seems high"
   - ✅ "Price $0.15 above quoted mid at time of execution"

3. **Consider Context**: Same timing for AAPL vs MARA is different
   - Large caps can tolerate more timing variation
   - Volatile names need precise timing

4. **Match Severity to Score**: Ensure consistency
   - Score ≤ 3 should have severity "high" or "critical"
   - Score ≥ 9 should have severity "none"

5. **Document Rationale**: Use notes field for edge cases

### Categories

| Category | When to Use |
|----------|------------|
| `none` | Score 8-10, no material issues |
| `venue_selection` | Wrong exchange, dark pool concerns |
| `timing` | Open/close, lunch hour, after hours |
| `fill_quality` | Partial fills, unfilled portions |
| `price_slippage` | Above VWAP, adverse selection |
| `order_handling` | Order type issues, algo failures |
| `market_conditions` | Volatility, news events |

### Severity Levels

| Severity | Description | Typical Score |
|----------|-------------|---------------|
| `none` | No issues | 8-10 |
| `low` | Minor improvement possible | 7-8 |
| `medium` | Noticeable impact | 5-7 |
| `high` | Significant problem | 2-4 |
| `critical` | Severe failure | 1-2 |

## Adding New Samples

### Using Python

```python
from src.evaluation import (
    create_sample,
    load_ground_truth,
    save_ground_truth,
    IssueCategory,
    IssueSeverity,
    AnnotatorConfidence,
)

# Load existing dataset
dataset = load_ground_truth()

# Create new sample
sample = create_sample(
    sample_id="GT056",
    fix_message="8=FIX.4.4|35=8|37=ORD056|55=AAPL|54=1|32=100|31=185.00|30=NYSE|60=20240116-10:30:00.000|39=2",
    quality_score=9,
    key_issues=[],
    expected_observations=["Full fill on primary exchange"],
    severity=IssueSeverity.NONE,
    category=IssueCategory.NONE,
    annotator="your_name",
    confidence=AnnotatorConfidence.HIGH,
    notes="New sample for expanded coverage",
)

# Add to dataset
dataset.samples.append(sample)

# Validate and save
from src.evaluation import validate_dataset
result = validate_dataset()
if result.valid:
    save_ground_truth(dataset)
```

### Validation

Always validate after making changes:

```bash
python -m src.evaluation.validator
```

Expected output for a valid dataset:
```
============================================================
Ground Truth Dataset Validation Report
============================================================

Status: PASSED
Total Samples: 55
Errors: 0
Warnings: 0

Statistics:
  score_distribution: {'good (8-10)': 19, 'average (5-7)': 20, 'poor (1-4)': 16}
  average_score: 5.78
  categories: {'none': 15, 'timing': 15, ...}
  ...
============================================================
```

## Usage in Evaluation

```python
from src.evaluation import load_ground_truth

# Load the dataset
dataset = load_ground_truth()

# Get samples by score range
good_samples = dataset.good_executions  # Score 8-10
poor_samples = dataset.poor_executions  # Score 1-4

# Get samples by category
timing_issues = dataset.get_by_category(IssueCategory.TIMING)

# Get statistics
stats = dataset.statistics()
print(f"Average expert score: {stats['average_score']}")
```

## Design Decisions

1. **50+ Samples Minimum**: Provides statistical significance for evaluation
2. **Balanced Distribution**: Roughly equal good/average/poor to test all ranges
3. **Diverse Scenarios**: Covers timing, venue, fill, slippage issues
4. **Real-World Cases**: Based on common execution problems
5. **Expert Confidence**: Each annotation includes confidence level
6. **Parseable FIX**: All messages validate against FIX 4.4 spec
