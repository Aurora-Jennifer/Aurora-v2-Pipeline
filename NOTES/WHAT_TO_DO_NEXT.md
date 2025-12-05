# What To Do Next: Clear Action Plan

Step-by-step workflow.

## Step 1: Rank Features by IC and Predictive Power

Main feature selection script - combines:
- IC (correlation with targets) - simple, interpretable
- Predictive power (model importance) - what actually works
- Uses your target rankings - focuses on best targets

### Run It:

```bash
conda activate trader_env
cd /home/Jennifer/trader

python scripts/rank_features_by_ic_and_predictive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/feature_selection
```

What it does:
- Automatically uses your top 10 targets from `results/final_clean/target_predictability_rankings.yaml`
- Ranks features by IC (correlation) and predictive power (model importance)
- Combines both metrics into a final ranking
- Weights features by target quality (better targets = more weight)

Output: `results/feature_selection/feature_rankings_ic_predictive.csv`

## Step 2: Review the Rankings

Open the CSV file and look for:

### Top Features (Keep These!)
- Combined score > 0.5: Excellent features
- IC (abs) > 0.1: Good correlation with targets
- Predictive > 0.3: Models find them useful

### What to Do:
1. Select top 50-100 features with highest combined score
2. Check redundancy: If two features have similar names and both rank high, pick one
3. Verify target: Make sure the "Best Target" column shows a good target (R² > 0.1)

## Step 3: Use Selected Features

### Option A: Update Your Feature Config

Create a feature list file:

```bash
# Extract top 50 features
python -c "
import pandas as pd
df = pd.read_csv('results/feature_selection/feature_rankings_ic_predictive.csv')
top_features = df.head(50)['feature_name'].tolist()
print('\n'.join(top_features))
" > CONFIG/selected_features.txt
```

### Option B: Use in Multi-Model Feature Selection

Run multi-model selection on your best target:

```bash
python scripts/multi_model_feature_selection.py \
  --target y_will_peak_60m_0.8 \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/peak_60m_multi_model
```

This gives you detailed model-specific rankings.

## Step 4: Build Your Strategy

### With Your Top Features:

1. Train models using top 50-100 features
2. Backtest on historical data
3. Validate on out-of-sample data
4. Paper trade before going live

## Quick Reference: All Your Scripts

### 1. Target Ranking (Already Done)
```bash
python scripts/rank_target_predictability.py --discover-all
```
Purpose: Find which targets are most predictable

### 2. Feature Ranking (Step 1)
```bash
python scripts/rank_features_by_ic_and_predictive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM
```
Purpose: Rank features by IC + predictive power

### 3. Multi-Model Selection (Optional)
```bash
python scripts/multi_model_feature_selection.py \
  --target y_will_peak_60m_0.8 \
  --symbols AAPL,MSFT,GOOGL
```
Purpose: Detailed model-specific feature rankings

### 4. Comprehensive Ranking (Optional)
```bash
python scripts/rank_features_comprehensive.py \
  --target y_will_peak_60m_0.8 \
  --symbols AAPL,MSFT,GOOGL
```
Purpose: Quality + predictive ranking

## Expected Timeline

- Step 1 (Feature Ranking): ~2-3 hours for 5 symbols
- Step 2 (Review): ~30 minutes
- Step 3 (Use Features): ~1 hour
- Step 4 (Build Strategy): Ongoing

Total: ~4-5 hours to get from feature selection to model training

## Next Steps After Feature Selection

1. Train models with selected features
2. Validate on out-of-sample data
3. Backtest trading strategy
4. Paper trade before going live
