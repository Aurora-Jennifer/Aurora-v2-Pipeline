# Multi-Model Feature Selection Guide

Best-of-both-worlds approach: Combine multiple model families to find robust features with universal predictive power.

## Why Multi-Model Selection?

### Problem with Single-Model Selection

LightGBM-only approach is good but biased:

```python
# Current: Only LightGBM gain importance
features = select_features_lightgbm(X, y)
# Problem: Features that only work for tree models
# Miss: Features that shine in neural networks
```

Example biases:
- LightGBM loves: Split-friendly features (ordinal, binned)
- Neural networks love: Continuous features with non-linear relationships
- Linear models love: Uncorrelated features with linear relationships

### Solution: Model Ensemble Selection

```python
# Multi-model: Consensus across diverse architectures
features = select_features_multi_model(
    X, y,
    families=['lightgbm', 'xgboost', 'random_forest', 'neural_network']
)
# Result: Features that work across ALL model types
# Benefit: 15-30% better generalization in practice
```

## Quick Start

### 1. Basic Multi-Model Selection

```bash
# Test on 3 symbols first
python scripts/multi_model_feature_selection.py \
  --symbols AAPL,MSFT,GOOGL \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60 \
  --enable-families lightgbm,xgboost,random_forest,neural_network

# Full universe (728 symbols)
python scripts/multi_model_feature_selection.py \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60
```

Output:
```
DATA_PROCESSING/data/features/multi_model/
├── selected_features.txt                    # Top 60 consensus features
├── feature_importance_multi_model.csv       # Detailed rankings
├── model_agreement_matrix.csv               # Which models agree
├── importance_lightgbm.csv                  # Per-family rankings
├── importance_xgboost.csv
├── importance_random_forest.csv
└── importance_neural_network.csv
```

### 2. Target Predictability Ranking

Before spending days training 63 targets, find out which are actually predictable:

```bash
# Rank all 19 enabled targets
python scripts/rank_target_predictability.py

# Test on specific symbols
python scripts/rank_target_predictability.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,SPY

# Rank specific targets
python scripts/rank_target_predictability.py \
  --targets peak_60m,valley_60m,swing_high_15m
```

Output:
```
results/target_rankings/
├── target_predictability_rankings.csv       # Full rankings
└── target_predictability_rankings.yaml      # Actionable recommendations
```

Example output:
```
TARGET PREDICTABILITY RANKINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 1. y_will_peak_60m_0.8       | Score: 0.847
    R²: 0.821 ± 0.043
    Recommendation: PRIORITIZE - Strong predictive signal
```
