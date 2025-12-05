# Quick Start: Comprehensive Feature Ranking

## Run in Your Terminal

### Step 1: Activate Environment

```bash
conda activate trader_env
cd /home/Jennifer/trader
```

## Option 1: Quality Audit (No Target Needed)

Best for: Initial feature audit, finding data quality issues, detecting redundancy

```bash
python scripts/rank_features_comprehensive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/feature_quality_audit
```

What it does:
- Ranks features by data quality (missing rate, variance)
- Detects redundant features (high correlation)
- Identifies features with distribution issues
- No target needed - works on any dataset

Output: `results/feature_quality_audit/feature_rankings_comprehensive.csv`

## Option 2: Full Ranking with Target

Best for: Finding best features for a specific prediction task

```bash
python scripts/rank_features_comprehensive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target y_will_peak_60m_0.8 \
  --output-dir results/peak_60m_feature_ranking
```

What it does:
- Ranks features by predictive power (model importance)
- Ranks features by data quality (completeness, variance)
- Detects redundancy (multicollinearity)
- Combines all metrics into composite edge score

Output: `results/peak_60m_feature_ranking/feature_rankings_comprehensive.csv`

## Option 3: Use Your Own Target

Replace `y_will_peak_60m_0.8` with any target column:

```bash
python scripts/rank_features_comprehensive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target <YOUR_TARGET_COLUMN> \
  --output-dir results/custom_feature_ranking
```

Example targets:
- `y_will_valley_60m_0.8`
- `y_will_peak_30m_0.6`
- Any target from your `target_configs.yaml`

## Understanding the Output

### CSV Columns

| Column | Meaning | Good Value |
|--------|---------|------------|
| `composite_edge` | Final ranking score | > 0.6 = excellent |
| `predictive_edge` | Model-based importance | > 0.4 = good |
| `quality_edge` | Data quality score | > 0.5 = good |
| `max_correlation` | Highest correlation with another feature | < 0.8 = low redundancy |
| `missing_rate` | Fraction of missing values | < 0.1 = good |
| `variance` | Feature variance | Higher = more informative |

### Top Features

Features with high composite_edge are your best bets:
- High predictive power
- Good data quality
- Low redundancy

## Example Workflow

### 1. Start with Quality Audit

```bash
# Find data quality issues
python scripts/rank_features_comprehensive.py \
  --symbols AAPL,MSFT,GOOGL \
  --output-dir results/quality_audit
```

Action: Review `quality_audit/feature_rankings_comprehensive.csv`
- Identify features with `missing_rate > 0.5` (consider excluding)
- Find features with `max_correlation > 0.9` (redundant pairs)
- Check `variance` (zero variance = constant feature)

### 2. Rank for Your Best Target

```bash
# Rank features for your top target
python scripts/rank_features_comprehensive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target y_will_peak_60m_0.8 \
  --output-dir results/peak_60m_features
```

Action: Select top 50-100 features with:
- `composite_edge > 0.5`
- `predictive_edge > 0.3`
- `max_correlation < 0.8`

### 3. Use in Feature Selection

Use the rankings to:
- Keep: Top features by composite_edge
- Remove: Highly redundant features (max_correlation > 0.9)
- Improve: Features with low quality_edge (missing data, low variance)

## All Options

```bash
python scripts/rank_features_comprehensive.py --help
```

Available options:
- `--symbols`: Comma-separated list (default: AAPL,MSFT,GOOGL,TSLA,JPM)
- `--target`: Target column name (optional)
- `--data-dir`: Data directory path (default: data/data_labeled)
- `--output-dir`: Output directory (default: results/feature_rankings)
- `--model-families`: Model families to use (default: lightgbm,random_forest,neural_network)
- `--max-samples`: Max samples per symbol (default: 50000)

## Expected Runtime

- Quality audit (no target): ~5-10 minutes
- Full ranking (with target): ~15-30 minutes

Depends on:
- Number of symbols
- Number of features
- Model families used

## Output Files

Each run creates:
1. `feature_rankings_comprehensive.csv`: Full rankings (all metrics)
2. `feature_rankings_comprehensive.json`: JSON format (for scripts)
3. `feature_ranking_report.md`: Human-readable report (top 50 features)

## Next Steps

After ranking:
1. Review top 50 features in the report
2. Check for redundancy (remove highly correlated pairs)
3. Use top features in your models
4. Re-run periodically to check for drift

See `docs/COMPREHENSIVE_FEATURE_RANKING.md` for detailed documentation.

## Quick Reference

```bash
# Quality audit
python scripts/rank_features_comprehensive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/quality_audit

# Full ranking with target
python scripts/rank_features_comprehensive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target y_will_peak_60m_0.8 \
  --output-dir results/peak_60m_features
```
