# Comprehensive Feature Ranking System

## Overview

The comprehensive feature ranking system combines target-dependent (predictive) and target-independent (data quality) metrics to identify features with the best "edge" across multiple dimensions.

## Two Types of Edge

### 1. Target-Dependent Edge (Predictive Power)

What it measures: How well a feature predicts the target variable.

Metrics:
- Model Importance: Feature importance from trained models (LightGBM, XGBoost, Random Forest, Neural Networks)
- Cross-Model Consensus: Fraction of models that agree the feature is important
- Consistency: Stability of importance across different symbols

Why it matters: Features with high predictive edge directly contribute to model performance.

### 2. Target-Independent Edge (Data Quality)

What it measures: Intrinsic quality and information content of the feature.

Metrics:
- Completeness: Low missing value rate (more complete = better)
- Variance: High variance = more informative (zero variance = useless)
- Distribution Quality: Skewness and kurtosis (normal-like distributions are often preferred)
- Redundancy: Low correlation with other features (high correlation = redundant)

Why it matters: High-quality features are more stable, require less imputation, and provide unique information.

## Composite Edge Score

The final ranking combines both types of edge:

```
Composite Edge =
  0.50 × Predictive Edge +
  0.30 × Quality Edge +
  (-0.20) × Redundancy Penalty
```

Interpretation:
- 0.8-1.0: Exceptional features (high predictive power + high quality + low redundancy)
- 0.6-0.8: Excellent features
- 0.4-0.6: Good features
- 0.2-0.4: Decent features (may need improvement)
- < 0.2: Weak features (consider excluding)

## Usage

### Basic Usage (Target-Independent Only)

Rank features by data quality only (no target needed):

```bash
conda activate trader_env
cd /home/Jennifer/trader

python scripts/rank_features_comprehensive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/feature_quality_ranking
```

This is useful for:
- Initial feature audit: Find features with data quality issues
- Feature engineering: Identify which features need cleaning/imputation
- Multicollinearity detection: Find redundant features

### With Target (Full Ranking)

Rank features by both predictive power and quality:

```bash
python scripts/rank_features_comprehensive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target y_will_peak_60m_0.8 \
  --output-dir results/peak_60m_feature_ranking
```

This provides:
- Predictive ranking: Which features best predict the target
- Quality ranking: Which features have best data quality
- Composite ranking: Best overall features considering both

## Output Files

### 1. `feature_rankings_comprehensive.csv`

CSV file with all metrics for each feature:

| Column | Description |
|--------|-------------|
| `feature_name` | Feature name |
| `composite_edge` | Final ranking score (0-1) |
| `predictive_edge` | Model-based importance (0-1) |
| `quality_edge` | Data quality score (0-1) |
| `model_importance_mean` | Average importance across models |
| `model_consensus` | Fraction of models that agree |
| `missing_rate` | Fraction of missing values |
| `variance` | Feature variance |
| `max_correlation` | Highest correlation with another feature |
| `n_high_corr` | Number of features with |r| > 0.9 |

### 2. `feature_rankings_comprehensive.json`

JSON file for programmatic access (same data as CSV).

### 3. `feature_ranking_report.md`

Markdown report with:
- Top 50 features by composite edge
- Metrics explanation
- Summary statistics

## Example Workflow

### Step 1: Initial Quality Audit

```bash
# Rank all features by quality (no target needed)
python scripts/rank_features_comprehensive.py \
  --symbols AAPL,MSFT,GOOGL \
  --output-dir results/quality_audit
```

Action items:
- Identify features with > 50% missing values
- Find features with zero/near-zero variance
- Detect highly correlated feature pairs (|r| > 0.9)

### Step 2: Target-Specific Ranking

```bash
# Rank features for specific target
python scripts/rank_features_comprehensive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target y_will_peak_60m_0.8 \
  --output-dir results/peak_60m_features
```

Action items:
- Select top 50-100 features by composite edge
- Verify predictive edge > 0.3 (good predictive power)
- Verify quality edge > 0.5 (good data quality)
- Verify max_correlation < 0.8 (low redundancy)

### Step 3: Feature Selection

Use the rankings to:
1. Keep: Top features with high composite edge
2. Remove: Features with high redundancy (max_correlation > 0.9)
3. Improve: Features with low quality edge (missing data, low variance)

## Interpreting Results

### High Predictive Edge, Low Quality Edge

Example: `feature_A` has predictive_edge=0.8, quality_edge=0.2

Meaning: Feature is very predictive but has data quality issues (missing values, low variance).

Action:
- Keep the feature
- Improve data quality (imputation, feature engineering)
- Consider creating a cleaned version

### Low Predictive Edge, High Quality Edge

Example: `feature_B` has predictive_edge=0.2, quality_edge=0.9

Meaning: Feature has excellent data quality but low predictive power.

Action:
- Keep for now (good data quality is valuable)
- Consider feature engineering (interactions, transformations)
- May be useful as a control/context feature

### High Redundancy

Example: `feature_C` has max_correlation=0.95 with `feature_D`

Meaning: Features are nearly identical (redundant).

Action:
- Keep the one with higher composite edge
- Remove the other
- Or create a single combined feature

### High Composite Edge

Example: `feature_E` has composite_edge=0.85

Meaning: Excellent feature with high predictive power, good quality, and low redundancy.

Action:
- Definitely keep in your feature set
- Consider using as a primary feature in models
- Monitor for stability over time

## Advanced Configuration

Edit `CONFIG/comprehensive_feature_ranking.yaml` to customize:

```yaml
# Adjust weights
composite_weights:
  predictive: 0.60  # Increase if predictive power is more important
  quality: 0.25     # Decrease if quality is less important
  redundancy: -0.15 # Adjust redundancy penalty

# Change model families
model_families:
  - lightgbm
  - xgboost
  - random_forest
  - neural_network
```

## Integration with Multi-Model Feature Selection

The comprehensive ranking complements the existing `multi_model_feature_selection.py`:

1. Comprehensive Ranking: Use for initial feature audit and quality assessment
2. Multi-Model Selection: Use for target-specific feature selection with multiple models
3. Combine Results: Use both rankings to make final feature selection decisions

Example:
```bash
# Step 1: Comprehensive ranking (quality + predictive)
python scripts/rank_features_comprehensive.py \
  --target y_will_peak_60m_0.8 \
  --output-dir results/comprehensive

# Step 2: Multi-model selection (detailed model importance)
python scripts/multi_model_feature_selection.py \
  --target y_will_peak_60m_0.8 \
  --output-dir results/multi_model

# Step 3: Combine insights from both
# - Use comprehensive ranking to filter low-quality features
# - Use multi-model ranking to select top predictive features
```

## Best Practices

1. Start with quality audit: Run without target first to identify data issues
2. Rank per target: Different targets may need different features
3. Check redundancy: Remove highly correlated features (|r| > 0.9)
4. Balance metrics: Don't ignore quality for predictive power (or vice versa)
5. Validate stability: Re-run rankings periodically to check for drift

## Troubleshooting

### "No safe features found"

Cause: All features filtered out by leakage detection.

Fix: Check `CONFIG/excluded_features.yaml` - may be too restrictive.

### "All features have low composite edge"

Cause: Either data quality is poor or target is hard to predict.

Fix:
- Check data quality metrics (missing rates, variance)
- Verify target is not degenerate (single class)
- Consider feature engineering

### "High redundancy scores"

Cause: Many features are highly correlated.

Fix:
- Use rankings to identify redundant pairs
- Keep one feature per highly correlated group
- Consider dimensionality reduction (PCA, feature selection)

## Summary

The comprehensive feature ranking system provides:
- Predictive Edge: Which features best predict your target
- Quality Edge: Which features have best data quality
- Redundancy Detection: Which features are redundant
- Composite Ranking: Best overall features considering all factors

Use it to make informed feature selection decisions.
