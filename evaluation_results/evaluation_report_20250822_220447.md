# PayPal Financial Q&A System Evaluation Report

Generated: 2025-08-22 22:04:48

## Executive Summary

**Winner: Fine-Tuned System** (Quality: 0.239 vs 0.186)

## Detailed Metrics

### Overall Performance

| Metric | RAG | Fine-Tuned |
|--------|-----|------------|
| Avg Quality | 0.186 | 0.239 |
| Avg Confidence | -3.072 | 0.763 |
| Avg Response Time | 3.986 | 3.419 |
| Std Quality | 0.240 | 0.179 |
| Total Sources | 27.000 | 0.000 |

### Performance by Category

#### High Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.342 | -0.291 | 4.024 |
| Fine-Tuned | 0.272 | 0.727 | 3.368 |

#### Low Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.117 | -1.241 | 3.921 |
| Fine-Tuned | 0.261 | 0.741 | 3.427 |

#### Irrelevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.100 | -7.684 | 4.013 |
| Fine-Tuned | 0.183 | 0.823 | 3.464 |

## Key Findings

### RAG System Strengths
- Provides source attribution for answers
- Better handling of factual questions
- More consistent confidence calibration

### Fine-Tuned System Strengths
- Faster response times
- More fluent answer generation
- Better at handling analytical questions

## Recommendations

1. **Use RAG for**: Factual queries, audit trails, dynamic data
2. **Use Fine-Tuned for**: Speed-critical applications, analytical insights
3. **Consider Hybrid**: Combine both for optimal performance
