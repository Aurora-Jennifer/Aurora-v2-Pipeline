# Example: Multi-Model Feature Selection Workflow

Real-world example of using multi-model feature selection to improve a trading system.

---

## Scenario

You're training models for **60-minute peak prediction** (`y_will_peak_60m_0.8`).

Current approach:
- Single-model (LightGBM) feature selection
- 60 features selected
- Train LightGBM + XGBoost + Neural Network ensemble

**Problem:** Features selected by LightGBM don't work well in the neural network.

**Solution:** Multi-model feature selection to find universal features.

---

## Workflow

### Step 1: Baseline (Single-Model Selection)

```bash
# Current approach: LightGBM only
python scripts/select_features.py \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60 \
  --data-dir data/data_labeled/interval=5m \
  --output-dir DATA_PROCESSING/data/features/baseline
```

**Output:**
```
✅ Selected 60 features
Top 5:
  1. time_in_profit_60m (importance: 1247.8)
  2. ret_zscore_15m (importance: 1108.2)
  3. mfe_share_60m (importance: 982.4)
  4. volume_5m (importance: 871.3)
  5. rsi_14 (importance: 764.5)
```

**Time:** 2 hours for 728 symbols

### Step 2: Multi-Model Selection (Test)

First, test on small sample to validate approach:

```bash
# Test on 5 symbols
python scripts/multi_model_feature_selection.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,SPY \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60 \
  --enable-families lightgbm,xgboost,random_forest,neural_network \
  --output-dir DATA_PROCESSING/data/features/multi_model_test
```

**Output:**
```
✅ Completed 20/20 model runs (5 symbols × 4 families)

Top 10 consensus features:
  1. time_in_profit_60m    | score=923.4 | agree=4/4 | std=41.2
  2. ret_zscore_15m        | score=856.7 | agree=4/4 | std=38.1
  3. mfe_share_60m         | score=798.2 | agree=4/4 | std=52.3
  4. excursion_up_60m      | score=687.5 | agree=3/4 | std=89.1
  5. volume_5m             | score=412.8 | agree=2/4 | std=142.3  ⚠️
  6. momentum_10m          | score=398.6 | agree=4/4 | std=31.2
  7. volatility_30m        | score=376.4 | agree=4/4 | std=28.7
  8. rsi_14                | score=298.1 | agree=2/4 | std=156.8  ⚠️
  9. returns_ordinal_15m   | score=287.3 | agree=4/4 | std=24.1
 10. time_to_hit_60m_0.8   | score=274.9 | agree=4/4 | std=35.6
```

**Key insights:**
- `volume_5m` and `rsi_14` have high std → models disagree
- LightGBM loves them, but neural networks don't
- New features appear: `momentum_10m`, `volatility_30m`

**Time:** 10 minutes

### Step 3: Compare Features

```bash
python scripts/compare_feature_sets.py \
  --set1 DATA_PROCESSING/data/features/baseline/selected_features.txt \
  --set2 DATA_PROCESSING/data/features/multi_model_test/selected_features.txt \
  --name1 "LightGBM-only" \
  --name2 "Multi-model" \
  --output results/feature_comparison.csv
```

**Output:**
```
Comparison: LightGBM-only vs Multi-model
═══════════════════════════════════════════════════════════════════
LightGBM-only: 60 features
Multi-model: 60 features
Overlap: 42 features (70.0% of LightGBM-only)
Only in LightGBM-only: 18 features
Only in Multi-model: 18 features

Jaccard similarity: 0.583
⚠️  Moderate agreement - some differences

🔵 Only in LightGBM-only (top 10):
  1. volume_5m              ← High in LightGBM, low in others
  2. rsi_14                 ← Same issue
  3. macd_signal
  4. obv_5m
  5. vwap_distance
  ...

🟢 Only in Multi-model (top 10):
  1. momentum_10m           ← Neural networks love this
  2. volatility_30m         ← Same
  3. returns_ordinal_15m    ← More robust across models
  4. regime_trend
  5. autocorr_5
  ...
```

**Insight:** 30% of features are different! Multi-model finds features that work universally.

### Step 4: Full Universe Multi-Model Selection

Now run on all symbols:

```bash
# Full run: 728 symbols, 4 model families
python scripts/multi_model_feature_selection.py \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60 \
  --enable-families lightgbm,xgboost,random_forest,neural_network \
  --output-dir DATA_PROCESSING/data/features/multi_model
```

**Time:** 8-10 hours (can run overnight)

**Output:**
```
✅ Multi-Model Feature Selection Complete!

📊 Top 10 Features by Consensus:
  1. time_in_profit_60m        | score=1847.23 | agree=4/4
  2. ret_zscore_15m            | score=1792.45 | agree=4/4
  3. mfe_share_60m             | score=1761.88 | agree=4/4
  4. excursion_up_60m          | score=1658.12 | agree=4/4
  5. momentum_10m              | score=1587.91 | agree=4/4
  6. volatility_30m            | score=1543.76 | agree=4/4
  7. returns_ordinal_15m       | score=1492.34 | agree=4/4
  8. time_to_hit_60m_0.8       | score=1467.82 | agree=4/4
  9. autocorr_5                | score=1421.56 | agree=4/4
 10. regime_trend              | score=1398.43 | agree=3/4

📁 Output files:
  • selected_features.txt (60 consensus features)
  • feature_importance_multi_model.csv (detailed rankings)
  • model_agreement_matrix.csv
  • importance_<family>.csv (per-family rankings)
```

### Step 5: Train Models with New Features

```python
# training_script.py

# Load consensus features
with open("DATA_PROCESSING/data/features/multi_model/selected_features.txt") as f:
    consensus_features = [line.strip() for line in f]

# Load per-family features for specialized models
import pandas as pd

lightgbm_df = pd.read_csv("DATA_PROCESSING/data/features/multi_model/importance_lightgbm.csv")
lightgbm_features = lightgbm_df['feature'].head(60).tolist()

neural_df = pd.read_csv("DATA_PROCESSING/data/features/multi_model/importance_neural_network.csv")
neural_features = neural_df['feature'].head(60).tolist()

# Train ensemble
ensemble_models = {
    'lightgbm': train_lightgbm(X[lightgbm_features], y),
    'xgboost': train_xgboost(X[consensus_features], y),
    'neural_network': train_neural_network(X[neural_features], y)
}
```

### Step 6: Evaluate Performance

**Backtesting results (walk-forward, 252-day folds):**

| Feature Set | LightGBM | XGBoost | Neural Net | Ensemble |
|-------------|----------|---------|------------|----------|
| **Baseline (LightGBM-only)** | 0.653 | 0.641 | 0.592 | 0.687 |
| **Multi-model** | 0.649 | 0.658 | 0.641 | 0.714 |
| **Improvement** | -0.6% | +2.7% | +8.3% | +3.9% |

**Key findings:**
- LightGBM: Slightly worse (expected, optimized for itself)
- XGBoost: Better (+2.7%)
- Neural Network: **Much better (+8.3%)** ⭐
- Ensemble: Better (+3.9%)

**Sharpe ratio improvement:**
- Baseline: 1.82
- Multi-model: 1.96 (+7.7%)

**Conclusion:** Multi-model features generalize better across model families.

---

## Step 7: Target Ranking (Bonus)

Which of your 63 targets are actually predictable?

```bash
python scripts/rank_target_predictability.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,SPY \
  --model-families lightgbm,random_forest,neural_network
```

**Output:**
```
TARGET PREDICTABILITY RANKINGS
═══════════════════════════════════════════════════════════════════

 1. y_will_peak_60m_0.8           | Score: 0.847
    R²: 0.821 ± 0.043
    Recommendation: PRIORITIZE - Strong predictive signal

 2. y_first_touch_60m_0.8         | Score: 0.792
    R²: 0.754 ± 0.062
    Recommendation: PRIORITIZE - Strong predictive signal

 3. y_will_valley_60m_0.8         | Score: 0.761
    R²: 0.735 ± 0.071
    Recommendation: ENABLE - Good predictive signal

 ...

15. y_will_swing_high_15m_0.10    | Score: 0.512
    R²: 0.468 ± 0.098
    Recommendation: TEST - Moderate signal

...

45. y_will_swing_low_5m_0.05      | Score: 0.283
    R²: 0.241 ± 0.152
    Recommendation: DEPRIORITIZE - Weak signal, low ROI
```

**Action:** Disable bottom 20 targets in `CONFIG/target_configs.yaml`

**Compute saved:** ~40 hours per training run (60% reduction)

---

## Summary: What You Gained

### Computational Cost

| Task | Baseline | Multi-Model | Additional Time |
|------|----------|-------------|-----------------|
| Feature selection (1 target) | 2h | 10h | +8h |
| Total targets enabled | 63 | 15 (top ranked) | -48 targets |
| Training time per run | 80h | 25h | -55h (68% faster) |

**Net result:** Initial +8h investment, but save 55h per training run.

### Performance Gains

| Metric | Baseline | Multi-Model | Improvement |
|--------|----------|-------------|-------------|
| Neural network R² | 0.592 | 0.641 | +8.3% |
| Ensemble R² | 0.687 | 0.714 | +3.9% |
| Sharpe ratio | 1.82 | 1.96 | +7.7% |
| Feature robustness | Low | High | ✅ |

### Insights Gained

1. **Model-specific biases revealed:**
   - LightGBM overvalues: `volume_5m`, `rsi_14`, `obv_5m`
   - Neural networks prefer: `momentum_10m`, `volatility_30m`, `autocorr_5`

2. **Universal features identified:**
   - `time_in_profit_60m`: Works across ALL models
   - `ret_zscore_15m`: High consensus, low std
   - `mfe_share_60m`: Robust predictor

3. **Target predictability ranked:**
   - Top 15 targets: High R², worth training
   - Bottom 20 targets: Low R², deprioritize

4. **Computational efficiency:**
   - Focus on predictable targets only
   - Save 55h per training run

---

## Lessons Learned

### What Worked Well

✅ **Multi-model consensus:** Found robust features  
✅ **Target ranking:** Identified unpredictable targets early  
✅ **Per-family features:** Improved specialized models  
✅ **Agreement matrix:** Revealed model-specific biases  

### What Didn't Work

❌ **SHAP on all models:** Too slow, permutation was enough  
❌ **100k samples per symbol:** Overkill, 50k was sufficient  
❌ **6 model families:** Diminishing returns after 4  

### Best Configuration (Lessons Applied)

```yaml
# CONFIG/multi_model_feature_selection.yaml
model_families:
  lightgbm:
    enabled: true
    importance_method: "native"
  
  xgboost:
    enabled: true
    importance_method: "native"
  
  random_forest:
    enabled: true
    importance_method: "native"
  
  neural_network:
    enabled: true
    importance_method: "permutation"  # Not SHAP (too slow)

sampling:
  max_samples_per_symbol: 50000  # Sweet spot

aggregation:
  require_min_models: 3  # Need 3/4 models to agree
```

---

## Quick Reference Commands

```bash
# 1. Rank targets (30 min)
python scripts/rank_target_predictability.py

# 2. Disable weak targets
vim CONFIG/target_configs.yaml  # Set enabled: false for bottom 20

# 3. Multi-model selection for top targets (8h overnight)
python scripts/multi_model_feature_selection.py \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60

# 4. Compare with baseline
python scripts/compare_feature_sets.py \
  --set1 DATA_PROCESSING/data/features/baseline/selected_features.txt \
  --set2 DATA_PROCESSING/data/features/multi_model/selected_features.txt

# 5. Use in training
FEATURES=$(cat DATA_PROCESSING/data/features/multi_model/selected_features.txt | tr '\n' ',')
python train_models.py --features $FEATURES
```

---

**Result:** Better features, faster training, higher Sharpe ratio. Worth the initial investment.

