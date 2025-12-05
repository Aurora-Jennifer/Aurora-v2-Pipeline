# Feature Selection Guide

Comprehensive guide to feature selection for predictive modeling.

## Overview

Feature selection is critical for:
- Reducing overfitting by eliminating noise
- Faster training with fewer dimensions
- Better interpretability by focusing on what matters
- Improved generalization to new data

## Quick Start

### 1. Basic Feature Selection (Single Target)

```bash
# Select top 60 features for peak prediction
python scripts/select_features.py \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60

# Test on specific symbols first
python scripts/select_features.py \
  --symbols AAPL,MSFT,GOOGL \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60
```

Output:
```
DATA_PROCESSING/data/features/
├── selected_features.txt              # Use this in training!
├── feature_importance_summary.csv     # Detailed scores
└── feature_selection_metadata.json    # Run metadata
```

### 2. Aggregate by Feature Concepts

Instead of individual features, see which concepts (RSI, momentum, volatility) are most important:

```bash
python scripts/aggregate_feature_groups.py \
  --input DATA_PROCESSING/data/features/feature_importance_summary.csv
```

Example Output:
```
Top Feature Concept Groups:
1. returns_zscore     Score: 639,012  (5 features)
2. excursion          Score: 314,073  (10 features)
3. time_to_hit        Score: 274,600  (9 features)
4. time_metrics       Score: 262,684  (5 features)
5. returns_ordinal    Score: 157,939  (5 features)
```

Why This Matters:
- Related features (like `ret_zscore_5m`, `ret_zscore_10m`, `ret_zscore_15m`) split their importance
- Grouping reveals that "returns_zscore" as a concept is most predictive
- You can then select the best 1-2 representatives from each top group

### 3. Target-Specific Feature Sets

Different targets need different features. Run selection for all your important targets:

```bash
# Run feature selection for 7 different targets
python scripts/run_multi_target_selection.py

# Test mode (3 symbols only)
python scripts/run_multi_target_selection.py --symbols AAPL,MSFT,GOOGL

# Run specific targets only
python scripts/run_multi_target_selection.py --targets peak_60m,valley_60m
```

Available Targets:
- `peak_60m`: Upward barrier hits (long signals)
- `valley_60m`: Downward barrier hits (short signals)
- `swing_high_15m`: Short-term reversal highs
- `swing_low_15m`: Short-term reversal lows
- `mfe_15m`: Max favorable excursion (profit targets)
- `mdd_15m`: Max drawdown (stop loss sizing)
- `first_touch_60m`: Direction prediction

Output Structure:
```
DATA_PROCESSING/data/features/
├── peak_60m/
│   ├── selected_features.txt
│   └── feature_importance_summary.csv
├── valley_60m/
│   ├── selected_features.txt
│   └── feature_importance_summary.csv
├── swing_high_15m/
│   └── ...
└── multi_target_summary.json
```

## How It Works

### Algorithm

1. Per-Symbol Training: Train a LightGBM model for each symbol independently
2. Extract Importance: Get feature importance (gain) from each model
3. Aggregate Across Universe: Average/sum importance across all symbols
4. Rank & Select: Select top N features by aggregated score

### Why Per-Symbol?

Training on each symbol separately prevents:
- Dominant symbol bias (AAPL/MSFT dominating the signal)
- Scale differences (high-price vs low-price stocks)
- Sector-specific patterns (tech vs energy)

It finds features that are universally predictive across your universe.

## Configuration

All feature selection parameters are centralized in config files. No hardcoded values.

### Feature Selection Config (`CONFIG/feature_selection_config.yaml`)

Controls all aspects of the feature selection process:

```yaml
# Paths
paths:
  data_dir: "data/data_labeled/interval=5m"
  output_dir: "DATA_PROCESSING/data/features"

# Defaults
defaults:
  top_n: 60
  method: "mean"
  num_workers: 12
  target_column: "y_will_peak_60m_0.8"

# LightGBM parameters (17 tunable hyperparameters)
lightgbm:
  objective: "regression_l1"
  n_estimators: 500
  learning_rate: 0.05
  num_leaves: 31
  max_depth: -1
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1
  reg_lambda: 0.1
  # ... and more
```

To customize: Edit `CONFIG/feature_selection_config.yaml` or use `--config your_config.yaml`

### Target Configs (`CONFIG/target_configs.yaml`)

Defines all targets for multi-target selection:

```yaml
targets:
  peak_60m:
    target_column: "y_will_peak_60m_0.8"
    description: "Predict upward barrier hits (peaks) at 60m horizon"
    use_case: "Long entry signals, profit target optimization"
    top_n: 60
    method: "mean"
    enabled: true  # ← Turn on/off

  valley_60m:
    target_column: "y_will_valley_60m_0.8"
    # ...
    enabled: true
```

To enable/disable targets: Edit `enabled: true/false` in the config.

### Feature Groups (`CONFIG/feature_groups.yaml`)

Defines how to group related features:

```yaml
returns_zscore:
  - ret_zscore_5m
  - ret_zscore_10m
  - ret_zscore_15m
  - ret_zscore_30m
  - ret_zscore_60m

rsi:
  - rsi_14
  - rsi_21
  - rsi_28

momentum:
  - mom_5
  - mom_10
  - mom_20
  - roc
```

Edit this file to add your own feature families.

### Adding New Targets

Edit `CONFIG/target_configs.yaml` to add new targets:

```yaml
targets:
  your_target:
    target_column: "y_your_column"
    description: "What this predicts"
    use_case: "When to use this"
    top_n: 60
    method: "mean"
    enabled: true
```

No code changes needed.

## Best Practices

### 1. Start with Concept Groups

Don't do this:
```python
# Blindly use top 60 individual features
X = df[selected_features]  # Might have 5 RSI variants, 0 volume features
```

Do this:
```python
# 1. Aggregate by groups
python scripts/aggregate_feature_groups.py --input <csv>

# 2. Review grouped results
# Top groups: returns_zscore (639k), excursion (314k), time_to_hit (275k)

# 3. Build balanced feature set:
#    - Pick 2 best from returns_zscore
#    - Pick 2 best from excursion
#    - Pick 1-2 from time_to_hit
#    - Add 1 from each other high-scoring group
```

### 2. Use Target-Specific Features

Don't do this:
```python
# Use same features for all models
features = load_features("selected_features.txt")
peak_model.train(X[features], y_peak)
valley_model.train(X[features], y_valley)  # Wrong!
```

Do this:
```python
# Use specialized features for each target
peak_features = load_features("features/peak_60m/selected_features.txt")
valley_features = load_features("features/valley_60m/selected_features.txt")

peak_model.train(X[peak_features], y_peak)
valley_model.train(X[valley_features], y_valley)
```

### 3. Handle Timescale Redundancy

If you have features at multiple timescales, don't select all of them:

```python
# Bad: All 5 timescales of same concept
['ret_zscore_5m', 'ret_zscore_10m', 'ret_zscore_15m',
 'ret_zscore_30m', 'ret_zscore_60m']

# Good: Select 2-3 diverse timescales
['ret_zscore_5m', 'ret_zscore_30m']  # Short + medium term
```

## Usage in Training

### Load and Use Selected Features

```python
from pathlib import Path

# Load feature list
feature_file = Path("DATA_PROCESSING/data/features/peak_60m/selected_features.txt")
with open(feature_file) as f:
    selected_features = [line.strip() for line in f]

# Filter your data
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# Train model
model.train(X_train, y_train)
```

### Validate Feature Availability

```python
# Check that all selected features exist in your data
missing_features = set(selected_features) - set(X_train.columns)
if missing_features:
    print(f"Missing features: {missing_features}")
    selected_features = [f for f in selected_features if f in X_train.columns]
```

## Advanced Options

### Aggregation Methods

```bash
# Mean (robust to outliers)
python scripts/select_features.py --method mean

# Median (very robust)
python scripts/select_features.py --method median

# Frequency (features that appear in top N for most symbols)
python scripts/select_features.py --method frequency
```

### Parallelization

```bash
# Use all CPUs minus 2 (default)
python scripts/select_features.py --num-workers 14

# Single-threaded (debugging)
python scripts/select_features.py --num-workers 1
```

### Custom LightGBM Config

Option 1: Edit the main config

```bash
# Edit CONFIG/feature_selection_config.yaml
vim CONFIG/feature_selection_config.yaml

# Change:
lightgbm:
  n_estimators: 1000     # Increase from 500 for more stable importance
  learning_rate: 0.03    # Decrease from 0.05 for better accuracy
```

Option 2: Create a custom config

```yaml
# my_fast_config.yaml
lightgbm:
  n_estimators: 100      # Fast testing
  learning_rate: 0.1
  subsample: 0.5

# Use it:
python scripts/select_features.py --config my_fast_config.yaml
```

## Interpreting Results

### Feature Importance Summary CSV

```csv
feature,score,frequency,frequency_pct
time_in_profit_60m,233598.42,3,100.0
ret_zscore_15m,182412.45,3,100.0
mfe_share_60m,181874.33,3,100.0
```

- score: Aggregated importance (higher = more predictive)
- frequency: How many symbols had this feature in top N
- frequency_pct: Percentage of symbols (100% = universal)

### Grouped Importance CSV

```csv
group,score,num_features,avg_feature_score,coverage_pct,top_feature
returns_zscore,639012,5,127802,8.3,ret_zscore_15m
excursion,314073,10,31407,16.7,mfe_share_60m
time_to_hit,274600,9,30511,15.0,tth_60m_0.8
```

- score: Total importance for this concept
- num_features: How many features in this group
- avg_feature_score: Average per feature (high = consistently important)
- top_feature: Best representative from this group

## Troubleshooting

### Issue: "No feature importances collected"

Cause: Target column not found or all data is NaN

Fix:
```bash
# Check available targets
python -c "import pandas as pd; df = pd.read_parquet('data/data_labeled/interval=5m/symbol=AAPL/AAPL.parquet'); print([c for c in df.columns if c.startswith('y_')])"

# Use correct target name
python scripts/select_features.py --target-column y_will_peak_60m_0.8
```

### Issue: "pandas dtypes must be int, float or bool"

Cause: Object dtype columns in data (like `symbol`, `interval`)

Fix: Already handled automatically by the script. If you see this, check the error details.

### Issue: Feature selection is slow

Tip: Test on a few symbols first:
```bash
python scripts/select_features.py --symbols AAPL,MSFT,GOOGL --top-n 60
```

## Example Workflow

### Day 1: Initial Exploration

```bash
# 1. Test on 3 symbols
python scripts/select_features.py --symbols AAPL,MSFT,GOOGL --target-column y_will_peak_60m_0.8 --top-n 60

# 2. Aggregate by concepts
python scripts/aggregate_feature_groups.py --input DATA_PROCESSING/data/features/feature_importance_summary.csv

# 3. Review results, understand which concepts matter
```

### Day 2: Full Universe Selection

```bash
# 1. Run on full universe (728 symbols)
python scripts/select_features.py --target-column y_will_peak_60m_0.8 --top-n 60

# 2. Aggregate again
python scripts/aggregate_feature_groups.py --input DATA_PROCESSING/data/features/feature_importance_summary.csv

# 3. Build balanced feature set from top groups
```

### Day 3: Multi-Target Selection

```bash
# Run for all important targets
python scripts/run_multi_target_selection.py

# This creates specialized feature sets for:
# - Peaks, valleys, swings, MFE, MDD, etc.
```

### Day 4: Train Models

```python
# Use specialized features for each model
peak_features = load_features("features/peak_60m/selected_features.txt")
valley_features = load_features("features/valley_60m/selected_features.txt")

# Train with reduced feature set
lgbm_peak.train(X[peak_features], y_peak)
lgbm_valley.train(X[valley_features], y_valley)

# Expect: Faster training, less overfitting, better performance
```

## Key Insights from Your Data

From the test run (3 symbols, 60 features):

### Top 5 Individual Features
1. `time_in_profit_60m` (233,598) - Time spent above entry price
2. `ret_zscore_15m` (182,412) - Standardized 15-min return
3. `mfe_share_60m` (181,874) - Max favorable excursion share
4. `ret_zscore_60m` (141,209) - Standardized 60-min return
5. `ret_zscore_10m` (127,246) - Standardized 10-min return

### Top 5 Concept Groups
1. returns_zscore (639,012) - Standardized returns at various timescales
2. excursion (314,073) - Max favorable/adverse movement metrics
3. time_to_hit (274,600) - Barrier crossing times
4. time_metrics (262,684) - Time in profit/drawdown
5. returns_ordinal (157,939) - Rank-based returns

Insight: Your most predictive features are:
- Relative returns (z-scored, ordinal) rather than raw returns
- Path-dependent metrics (time in profit, excursion shares)
- Barrier dynamics (time-to-hit, hit direction)

Recommendation: Focus on these concepts and reduce generic technical indicators (RSI, MACD) which ranked low.

## References

- LightGBM Feature Importance: Uses "gain" (total reduction in loss)
- Aggregation Strategy: Per-symbol training avoids dominant stock bias
- Supervised Selection: Target-based ranking ensures predictive power
