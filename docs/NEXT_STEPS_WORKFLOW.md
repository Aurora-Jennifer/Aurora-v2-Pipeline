# Next Steps - Complete Workflow

## Step 1: Run Target Ranking

Rank all targets by their predictability across all 11 models.

```bash
conda activate trader_env
cd /home/Jennifer/trader

python scripts/rank_target_predictability.py \
  --discover-all \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/target_rankings
```

What this does:
- Discovers ALL targets from your data
- Evaluates each target with all 11 models
- Ranks targets by composite score (R² + importance)
- Outputs: `target_rankings.csv` with scores

Time: ~2-3 hours (depends on number of targets)

Output location: `results/target_rankings/target_rankings.csv`

## Step 2: Review Target Rankings

After Step 1 completes, review the results:

```bash
# View top targets
head -20 results/target_rankings/target_rankings.csv

# Or open in your editor
code results/target_rankings/target_rankings.csv
```

Look for:
- Targets with high composite scores (>0.5)
- Targets with positive R² (actually predictable)
- Targets important in 6+ models (robust)

Action: Identify top 5-10 targets to focus on.

## Step 3: Run Feature Selection

Once you know which targets are best, rank features for those targets.

### Option A: Use Top Targets from Rankings

```bash
# First, check what targets ranked highest
python -c "
import pandas as pd
df = pd.read_csv('results/target_rankings/target_rankings.csv')
print('Top 10 targets:')
print(df.head(10)[['target_name', 'composite_score', 'mean_r2']].to_string())
"

# Then run feature ranking for top targets
python scripts/rank_features_by_ic_and_predictive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/feature_selection
```

### Option B: Use Specific Targets

```bash
# If you know which targets you want
python scripts/rank_features_by_ic_and_predictive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --targets y_will_peak_60m_0.8,y_will_valley_60m_0.8 \
  --output-dir results/feature_selection
```

What this does:
- Computes IC (Information Coefficient) for each feature
- Computes predictive power using all 11 models
- Combines IC + predictive power into final score
- Outputs: `feature_rankings.csv` and `feature_rankings_by_model.csv`

Time: ~2-3 hours per symbol (or ~10-15 hours for 5 symbols)

Output location: `results/feature_selection/`

## Step 4: Review Feature Rankings

After Step 3 completes, review the results:

```bash
# View top features
head -30 results/feature_selection/feature_rankings.csv

# View per-model breakdown
head -30 results/feature_selection/feature_rankings_by_model.csv
```

Look for:
- Features with high combined scores (IC + predictive)
- Features important in 8+ models (maximum consensus)
- Features that work across multiple targets

Action: Select top 50-100 features for production.

## Step 5: Validate Selected Features

Optional but recommended - validate your selected features:

```bash
# Compare different feature sets
python scripts/compare_feature_sets.py \
  --old-features results/feature_selection/feature_rankings.csv \
  --new-features results/feature_selection/feature_rankings.csv \
  --top-n 50
```

## Quick Start (Fast Path)

If you want to test quickly first:

### 1. Quick Target Ranking (5 symbols, sample)
```bash
python scripts/rank_target_predictability.py \
  --discover-all \
  --symbols AAPL,MSFT \
  --output-dir results/target_rankings_quick
```

### 2. Quick Feature Ranking (1 symbol, 1 target)
```bash
python scripts/rank_features_by_ic_and_predictive.py \
  --symbols AAPL \
  --targets y_will_peak_60m_0.8 \
  --output-dir results/feature_selection_quick
```

Time: ~30-60 minutes total

## Expected Results

### Target Rankings:
- Top targets: Composite score > 0.5, R² > 0.1
- Medium targets: Composite score 0.3-0.5, R² -0.1 to 0.1
- Poor targets: Composite score < 0.3, R² < -0.1

### Feature Rankings:
- Top features: Combined score > 0.7, important in 8+ models
- Good features: Combined score 0.5-0.7, important in 5-7 models
- Marginal features: Combined score < 0.5, important in < 5 models

## Important Notes

### Time Estimates:
- Target ranking: ~2-3 hours (all targets, 5 symbols, 11 models)
- Feature ranking: ~2-3 hours per symbol (all features, 11 models)
- Total: ~12-18 hours for complete analysis

### If It's Too Slow:
You can temporarily disable slower models in `CONFIG/multi_model_feature_selection.yaml`:
- Set `boruta: enabled: false` (saves ~5-10 min per symbol)
- Set `stability_selection: enabled: false` (saves ~3-5 min per symbol)
- Set `rfe: enabled: false` (saves ~2-3 min per symbol)

### Dependencies:
Make sure you have:
```bash
conda activate trader_env
pip install catboost Boruta  # If not already installed
```

## Recommended Order

1. Run target ranking (Step 1) - This tells you which targets are worth pursuing
2. Review target results (Step 2) - Pick your top targets
3. Run feature ranking (Step 3) - Find best features for your top targets
4. Review feature results (Step 4) - Select production features
5. Validate (Step 5) - Optional but recommended

## What You'll Get

### From Target Ranking:
- List of all targets ranked by predictability
- R² scores per model
- Importance scores per model
- Composite scores (overall ranking)
- Recommendations (PRIORITIZE/DEPRIORITIZE)

### From Feature Ranking:
- Features ranked by IC + predictive power
- Per-model importance breakdown
- Combined scores
- Which features work across all models
