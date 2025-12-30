# Evaluation Criteria

This document provides comprehensive guidelines for evaluating AI-generated trade execution analyses, including scoring rubrics, issue categories, and review guidelines.

## Table of Contents

- [Quality Score Rubric](#quality-score-rubric)
- [Issue Categories](#issue-categories)
- [Review Guidelines](#review-guidelines)
- [Edge Cases](#edge-cases)
- [Agreement Tracking](#agreement-tracking)
- [Examples](#examples)

---

## Quality Score Rubric

Trade execution quality is scored on a 1-10 scale based on multiple factors.

### Score Definitions

| Score | Rating | Description |
|-------|--------|-------------|
| 9-10 | Excellent | Near-perfect execution with price improvement and optimal timing |
| 7-8 | Good | Solid execution with minor imperfections |
| 5-6 | Acceptable | Reasonable execution but with notable issues |
| 3-4 | Poor | Significant execution problems |
| 1-2 | Very Poor | Major failures in execution |

### Scoring Components

Each component contributes to the overall score:

#### 1. Fill Quality (30%)
- **Full Fill (10)**: 100% of order filled
- **Near Complete (8-9)**: 90-99% filled
- **Partial Fill (5-7)**: 50-89% filled
- **Minimal Fill (2-4)**: 10-49% filled
- **Unfilled (1)**: <10% filled

#### 2. Price Quality (30%)
- **Price Improvement (9-10)**: Better than limit/expected price
- **At Price (7-8)**: Executed at expected price
- **Minor Slippage (5-6)**: <0.5% worse than expected
- **Moderate Slippage (3-4)**: 0.5-2% worse
- **Severe Slippage (1-2)**: >2% worse

#### 3. Timing Quality (20%)
- **Optimal (9-10)**: Executed during high liquidity periods
- **Good (7-8)**: Reasonable timing
- **Suboptimal (5-6)**: Could have been better
- **Poor (3-4)**: Bad timing (near close, during news)
- **Very Poor (1-2)**: Worst possible timing

#### 4. Venue Selection (20%)
- **Optimal (9-10)**: Best available venue for liquidity
- **Good (7-8)**: Reasonable venue choice
- **Acceptable (5-6)**: Not the best but acceptable
- **Poor (3-4)**: Suboptimal venue
- **Very Poor (1-2)**: Wrong venue choice

### Score Calculation

```python
final_score = (
    fill_quality * 0.30 +
    price_quality * 0.30 +
    timing_quality * 0.20 +
    venue_selection * 0.20
)
```

---

## Issue Categories

### Venue Selection Issues

Flag venue selection issues when:

- **Dark Pool Misuse**: Using dark pools for small orders (<1000 shares)
- **Exchange Mismatch**: Not routing to exchange with best liquidity
- **Market Maker Conflicts**: Routing to market makers with poor execution history
- **Regulatory Issues**: Routing that could cause regulatory concerns

**Examples**:
- "Routed to dark pool for 100 shares when lit exchange had better NBBO"
- "Primary exchange had higher liquidity but order routed to secondary"

### Timing Issues

Flag timing issues when:

- **Market Open/Close**: Executing within 5 minutes of open/close without reason
- **Lunch Hour**: Large orders during 11:30-13:30 (low liquidity)
- **News Events**: Executing during known volatility events
- **End of Day**: Executing in final minutes with price impact

**Examples**:
- "Executed 5 minutes before market close with visible price impact"
- "Large order during lunch hour resulted in partial fill"

### Fill Quality Issues

Flag fill quality issues when:

- **Partial Fill Without Reason**: Order not filled despite available liquidity
- **Slow Fill**: Order took longer than expected to fill
- **Fill Price Variance**: Multiple fills at varying prices

**Examples**:
- "Only 40% filled despite available liquidity at quoted price"
- "Fill took 45 seconds for liquid stock"

### Price Impact Issues

Flag price impact issues when:

- **Slippage > 0.5%**: Execution worse than expected by more than 0.5%
- **Market Impact**: Order moved the market visibly
- **Spread Crossing**: Unnecessarily crossed the spread

**Examples**:
- "1.5% slippage on 1000 share order in liquid stock"
- "Order caused 20bp market move"

---

## Review Guidelines

### How to Handle Ambiguous Cases

When evaluating executions that could be scored multiple ways:

1. **Consider Market Conditions**: What was the market doing at execution time?
2. **Consider Order Size**: Large orders naturally have more impact
3. **Consider Stock Characteristics**: Volatile stocks have different expectations
4. **Consider Time of Day**: Liquidity varies throughout the day

**Default to Conservative**: When uncertain, assume reasonable execution unless evidence suggests otherwise.

### When to Disagree with AI

You should disagree with the AI's assessment when:

1. **Factual Errors**: AI states incorrect facts about the execution
2. **Missing Critical Issues**: AI missed an obvious problem
3. **Incorrect Severity**: AI scored an issue too high or too low
4. **Context Misunderstanding**: AI didn't understand market conditions
5. **Score Inconsistency**: Score doesn't match the identified issues

### How to Document Corrections

When correcting AI analysis:

1. **Be Specific**: "AI missed timing issue" is not enough
2. **Provide Context**: Explain why the AI was wrong
3. **Suggest Improvement**: How could the AI have done better?
4. **Rate Severity**: Was this a minor or major miss?

**Example Correction**:
```
Issue: AI gave score of 8/10 but should be 5/10
Reason: AI missed that execution occurred during a known news event
  for this stock (earnings release). The 2% slippage was actually
  expected given the volatility.
Suggestion: AI should check for corporate events before scoring.
```

---

## Edge Cases

### Volatile Stocks (Meme Stocks, Biotech)

- Accept higher slippage as normal (up to 3%)
- Weight timing more heavily (30%)
- Consider news/social media context

### Large Institutional Orders

- Partial fills are expected for orders >1% of ADV
- Market impact is expected
- Evaluate VWAP/TWAP vs. single execution

### Pre-Market/After-Hours

- Lower liquidity expectations
- Wider spreads acceptable
- Evaluate against pre/post market conditions, not regular hours

### IPO Day Trading

- Very wide spreads acceptable
- No historical reference point
- Focus on fill quality over price

### Illiquid Securities

- Partial fills expected
- Accept wider spreads
- Evaluate effort to find liquidity

---

## Agreement Tracking

### Metrics Tracked

| Metric | Definition | Target |
|--------|------------|--------|
| Score Agreement | AI and human scores within 1 point | >85% |
| Issue Agreement | Same issues identified | >80% |
| Overall Agreement | Weighted combination | >85% |

### Alert Thresholds

| Condition | Alert Level |
|-----------|-------------|
| Agreement < 75% | Critical |
| Agreement < 85% | Warning |
| 5+ consecutive disagreements | Warning |
| New disagreement pattern | Warning |

### Trend Analysis

Agreement is tracked over time with three trend states:

- **Improving**: Recent agreement 5%+ higher than earlier
- **Stable**: Within ±5% of historical average
- **Declining**: Recent agreement 5%+ lower than earlier

---

## Examples

### Example 1: Excellent Execution (Score: 9/10)

**FIX Message**:
```
8=FIX.4.4|35=8|55=AAPL|54=1|38=100|32=100|31=150.45|44=150.50|60=20240115-09:30:05.000
```

**Analysis**:
- Full fill achieved (100%)
- Price improvement: Bought at $150.45 vs limit of $150.50
- Timing: Just after market open, good liquidity
- No issues identified

**Key Observations**:
- "Full fill achieved with $0.05 price improvement"
- "Executed in first 5 minutes with optimal liquidity"

---

### Example 2: Good Execution (Score: 7/10)

**FIX Message**:
```
8=FIX.4.4|35=8|55=MSFT|54=2|38=500|32=500|31=380.15|44=380.00|60=20240115-14:30:00.000
```

**Analysis**:
- Full fill achieved (100%)
- Minor slippage: Sold at $380.15 vs limit of $380.00 (worse)
- Timing: Mid-afternoon, decent liquidity

**Issues**:
- "0.04% slippage on execution"

**Key Observations**:
- "Full fill with minor slippage"
- "Mid-afternoon execution during normal trading"

---

### Example 3: Poor Execution (Score: 4/10)

**FIX Message**:
```
8=FIX.4.4|35=8|55=GME|54=1|38=1000|32=600|31=26.50|44=25.00|60=20240115-15:55:00.000
```

**Analysis**:
- Partial fill: Only 60% filled
- Significant slippage: 6% worse than limit
- Bad timing: 5 minutes before close

**Issues**:
- "Partial fill of only 60%"
- "6% slippage significantly above acceptable threshold"
- "Executed near market close with visible price impact"
- "High volatility stock without protective measures"

**Recommendations**:
- "Use limit orders for volatile stocks"
- "Avoid executing large orders near market close"
- "Consider TWAP/VWAP for size"

---

### Example 4: Edge Case - Earnings Day (Score: 6/10)

**Context**: Stock had earnings release 30 minutes before execution.

**FIX Message**:
```
8=FIX.4.4|35=8|55=NVDA|54=1|38=200|32=200|31=485.00|44=480.00|60=20240115-17:00:00.000
```

**Analysis**:
- Full fill achieved
- 1% slippage (normally would be issue)
- Post-earnings volatility explains spread

**Reviewer Notes**:
- "1% slippage acceptable given earnings volatility"
- "Full fill during volatile period is positive"
- "Score adjusted upward due to context"

---

## Review Session Workflow

### Before Starting

1. Review recent AI analyses for context
2. Check for any known market events
3. Set your reviewer ID consistently

### During Review

1. Read the execution details carefully
2. Review AI's score and reasoning
3. Decide if you agree with the score (within 1 point)
4. If disagreeing, note the specific issues
5. Provide corrections for any missed issues
6. Add notes for unusual circumstances

### After Session

1. Review session statistics
2. Check for patterns in your disagreements
3. Update this documentation if new edge cases found

---

## Appendix: Quick Reference

### Score Quick Guide

| Condition | Suggested Score |
|-----------|-----------------|
| Full fill + price improvement + good timing | 9-10 |
| Full fill + at price + acceptable timing | 7-8 |
| Full fill + minor slippage OR partial fill | 5-6 |
| Partial fill + slippage OR multiple issues | 3-4 |
| Multiple severe issues | 1-2 |

### Issue Severity

| Issue Type | Severity | Score Impact |
|------------|----------|--------------|
| >2% slippage | High | -3 to -4 |
| 1-2% slippage | Medium | -1 to -2 |
| Near close execution | Medium | -1 to -2 |
| Partial fill <75% | Medium | -2 to -3 |
| Wrong venue | Low | -1 |

### Common AI Mistakes

Based on feedback tracking, AI commonly:

1. Misses timing issues during corporate events
2. Over-scores volatile stocks
3. Under-weights venue selection importance
4. Doesn't adjust for market conditions

These are areas where human reviewers should be especially attentive.
